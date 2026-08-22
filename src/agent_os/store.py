from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import JsonValue
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

from agent_os.models import (
    AgentExecutionView,
    ExecutionStatus,
    RunInput,
    RunStatus,
    RunView,
    TaskStatus,
    TaskView,
    TranscriptEvent,
)

SCHEMA = "agent_os"
metadata = MetaData(schema=SCHEMA)
json_type = JSON().with_variant(JSONB(), "postgresql")

runs = Table(
    "runs",
    metadata,
    Column("id", String, primary_key=True),
    Column("target_id", String(64), nullable=False, index=True),
    Column("active_target_id", String(64), unique=True),
    Column("idempotency_key", String, unique=True),
    Column("repository_path", Text, nullable=False),
    Column("plan_path", Text, nullable=False),
    Column("plan_id", String(64), nullable=False),
    Column("base_commit", String(64), nullable=False),
    Column("integration_branch", Text, nullable=False),
    Column("integration_head", String(64)),
    Column("status", String(16), nullable=False),
    Column("current_round", Integer, nullable=False, default=0),
    Column("failure_reason", Text),
    Column("workflow_id", String, nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

tasks = Table(
    "tasks",
    metadata,
    Column("id", String, primary_key=True),
    Column("run_id", String, ForeignKey(f"{SCHEMA}.runs.id", ondelete="CASCADE"), nullable=False),
    Column("round_number", Integer, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("description", Text, nullable=False),
    Column("acceptance_criteria", json_type, nullable=False),
    Column("status", String(16), nullable=False),
    Column("developer_workflow_id", String),
)
Index("ix_tasks_run_round", tasks.c.run_id, tasks.c.round_number, tasks.c.ordinal)

developer_sessions = Table(
    "developer_sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("task_id", String, ForeignKey(f"{SCHEMA}.tasks.id", ondelete="CASCADE"), nullable=False),
    Column("message_history", json_type, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

executions = Table(
    "agent_executions",
    metadata,
    Column("id", String, primary_key=True),
    Column("run_id", String, ForeignKey(f"{SCHEMA}.runs.id", ondelete="CASCADE"), nullable=False),
    Column("role", String(16), nullable=False),
    Column("workflow_id", String, nullable=False),
    Column("session_id", String),
    Column("task_id", String, ForeignKey(f"{SCHEMA}.tasks.id", ondelete="CASCADE")),
    Column("status", String(16), nullable=False),
    Column("final_output", json_type),
    Column("next_sequence", Integer, nullable=False, default=0),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
)

events = Table(
    "transcript_events",
    metadata,
    Column("execution_id", String, ForeignKey(f"{SCHEMA}.agent_executions.id", ondelete="CASCADE"), primary_key=True),
    Column("sequence", Integer, primary_key=True),
    Column("event_key", String),
    Column("event_type", String, nullable=False),
    Column("payload", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("execution_id", "event_key", name="uq_transcript_event_key"),
)


class ActiveRunError(RuntimeError):
    pass


class IdempotencyKeyError(ActiveRunError):
    pass


class ReplayConflictError(RuntimeError):
    pass


class StateTransitionError(RuntimeError):
    pass


ACTIVE_RUN_STATUSES = frozenset({RunStatus.QUEUED, RunStatus.RUNNING})
TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED}
)
TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.INTEGRATED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)
TERMINAL_EXECUTION_STATUSES = frozenset(
    {
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    }
)

RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset(
        {RunStatus.RUNNING, RunStatus.CANCELLING, RunStatus.COMPLETE, RunStatus.FAILED}
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.CANCELLING,
            RunStatus.FINALIZING,
            RunStatus.COMPLETE,
            RunStatus.FAILED,
        }
    ),
    RunStatus.CANCELLING: frozenset({RunStatus.CANCELLED}),
    RunStatus.FINALIZING: frozenset({RunStatus.COMPLETE, RunStatus.FAILED}),
}

TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset(
        {TaskStatus.DEVELOPING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.DEVELOPING: frozenset(
        {
            TaskStatus.DEVELOPING,
            TaskStatus.STAGING,
            TaskStatus.FIXING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.STAGING: frozenset(
        {
            TaskStatus.REVIEWING,
            TaskStatus.FIXING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.REVIEWING: frozenset(
        {
            TaskStatus.REVIEWING,
            TaskStatus.FIXING,
            TaskStatus.INTEGRATED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.FIXING: frozenset(
        {
            TaskStatus.FIXING,
            TaskStatus.STAGING,
            TaskStatus.REVIEWING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
}


class StateStore:
    def __init__(self, database_url: str):
        base_engine = (
            create_engine(database_url, poolclass=NullPool)
            if database_url.startswith("sqlite")
            else create_engine(database_url)
        )
        if base_engine.dialect.name == "sqlite":
            event.listen(base_engine, "connect", self._enable_sqlite_foreign_keys)
        self.engine: Engine = (
            base_engine.execution_options(schema_translate_map={SCHEMA: None})
            if base_engine.dialect.name == "sqlite"
            else base_engine
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    def bootstrap(self) -> None:
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as connection:
                connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        metadata.create_all(self.engine)

    def health(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(select(1)).scalar_one()

    def assert_target_lease(
        self,
        run_id: str,
        target_id: str,
        allowed_statuses: frozenset[RunStatus],
    ) -> None:
        """Fence a target mutation to its active run generation and state."""
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    runs.c.target_id,
                    runs.c.active_target_id,
                    runs.c.status,
                ).where(runs.c.id == run_id)
            ).mappings().one_or_none()
        if (
            row is None
            or row["target_id"] != target_id
            or row["active_target_id"] != target_id
        ):
            raise StateTransitionError(
                f"run {run_id} does not own the active target lease for {target_id}"
            )
        status = RunStatus(row["status"])
        if status not in allowed_statuses:
            allowed = ", ".join(sorted(value.value for value in allowed_statuses))
            raise StateTransitionError(
                f"run {run_id} target lease has status {status.value}; "
                f"expected one of: {allowed}"
            )

    def create_run(
        self,
        value: RunInput,
        idempotency_key: str | None,
        enqueue: Callable[[Connection, RunInput], None],
    ) -> tuple[RunView, bool]:
        now = datetime.now(UTC)
        try:
            with self.engine.begin() as connection:
                if idempotency_key:
                    existing = connection.execute(
                        select(runs).where(runs.c.idempotency_key == idempotency_key)
                    ).mappings().one_or_none()
                    if existing is not None:
                        self._validate_idempotency_replay(existing, value, idempotency_key)
                        return self.get_run(existing["id"], connection=connection), False
                active = connection.execute(
                    select(runs.c.id).where(runs.c.active_target_id == value.target_id)
                ).scalar_one_or_none()
                if active:
                    raise ActiveRunError(f"target already has active run {active}")
                connection.execute(
                    insert(runs).values(
                        id=value.run_id,
                        target_id=value.target_id,
                        active_target_id=value.target_id,
                        idempotency_key=idempotency_key,
                        repository_path=value.repository_path,
                        plan_path=value.plan_path,
                        plan_id=value.plan_id,
                        base_commit=value.base_commit,
                        integration_branch=value.integration_branch,
                        integration_head=None,
                        status=RunStatus.QUEUED.value,
                        current_round=0,
                        failure_reason=None,
                        workflow_id=value.workflow_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
                enqueue(connection, value)
                return self.get_run(value.run_id, connection=connection), True
        except IntegrityError as exc:
            return self._resolve_run_insert_conflict(value, idempotency_key, exc)

    def _resolve_run_insert_conflict(
        self,
        value: RunInput,
        idempotency_key: str | None,
        cause: IntegrityError,
    ) -> tuple[RunView, bool]:
        with self.engine.connect() as connection:
            if idempotency_key:
                existing = connection.execute(
                    select(runs).where(runs.c.idempotency_key == idempotency_key)
                ).mappings().one_or_none()
                if existing is not None:
                    self._validate_idempotency_replay(existing, value, idempotency_key)
                    return self.get_run(existing["id"], connection=connection), False
            active = connection.execute(
                select(runs.c.id).where(runs.c.active_target_id == value.target_id)
            ).scalar_one_or_none()
            if active:
                raise ActiveRunError(f"target already has active run {active}")
        raise cause

    @staticmethod
    def _validate_idempotency_replay(row, value: RunInput, idempotency_key: str) -> None:
        expected = {
            "target_id": value.target_id,
            "repository_path": value.repository_path,
            "plan_path": value.plan_path,
            "plan_id": value.plan_id,
            "base_commit": value.base_commit,
            "integration_branch": value.integration_branch,
        }
        if any(row[name] != expected_value for name, expected_value in expected.items()):
            raise IdempotencyKeyError(
                f"idempotency key {idempotency_key!r} was used for a different target"
            )

    def get_run(self, run_id: str, *, connection: Connection | None = None) -> RunView:
        if connection is None:
            with self.engine.connect() as own_connection:
                return self.get_run(run_id, connection=own_connection)
        row = connection.execute(select(runs).where(runs.c.id == run_id)).mappings().one_or_none()
        if row is None:
            raise KeyError(run_id)
        task_rows = connection.execute(
            select(tasks)
            .where(tasks.c.run_id == run_id)
            .order_by(tasks.c.round_number, tasks.c.ordinal)
        ).mappings()
        task_views = [self._task_view(task) for task in task_rows]
        return RunView(
            id=row["id"],
            target_id=row["target_id"],
            repository_path=row["repository_path"],
            plan_path=row["plan_path"],
            plan_id=row["plan_id"],
            base_commit=row["base_commit"],
            integration_branch=row["integration_branch"],
            integration_head=row["integration_head"],
            status=RunStatus(row["status"]),
            current_round=row["current_round"],
            failure_reason=row["failure_reason"],
            workflow_id=row["workflow_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            tasks=task_views,
        )

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        current_round: int | None = None,
        integration_head: str | None = None,
        failure_reason: str | None = None,
    ) -> RunView:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(runs).where(runs.c.id == run_id).with_for_update()
            ).mappings().one_or_none()
            if row is None:
                raise KeyError(run_id)
            current = RunStatus(row["status"])
            requested = status or current
            if current in TERMINAL_RUN_STATUSES:
                self._validate_terminal_run_replay(
                    row,
                    requested=requested,
                    current_round=current_round,
                    integration_head=integration_head,
                    failure_reason=failure_reason,
                )
                return self.get_run(run_id, connection=connection)
            if requested != current and requested not in RUN_TRANSITIONS[current]:
                raise StateTransitionError(
                    f"run {run_id} cannot transition from {current.value} to {requested.value}"
                )
            values: dict[str, object] = {
                "status": requested.value,
                "updated_at": datetime.now(UTC),
            }
            if requested in TERMINAL_RUN_STATUSES:
                values["active_target_id"] = None
            if current_round is not None:
                values["current_round"] = current_round
            if integration_head is not None:
                values["integration_head"] = integration_head
            if failure_reason is not None:
                values["failure_reason"] = failure_reason
            if connection.execute(
                update(runs)
                .where(runs.c.id == run_id, runs.c.status == current.value)
                .values(**values)
            ).rowcount == 0:
                raise StateTransitionError(  # pragma: no cover - compare-and-set race
                    f"run {run_id} changed concurrently"
                )
            if requested in {RunStatus.FAILED, RunStatus.CANCELLED}:
                self._terminalize_run_projections(
                    connection,
                    run_id,
                    requested,
                    failure_reason=failure_reason,
                )
            return self.get_run(run_id, connection=connection)

    @staticmethod
    def _validate_terminal_run_replay(
        row,
        *,
        requested: RunStatus,
        current_round: int | None,
        integration_head: str | None,
        failure_reason: str | None,
    ) -> None:
        current = RunStatus(row["status"])
        values_match = (
            (current_round is None or current_round == row["current_round"])
            and (integration_head is None or integration_head == row["integration_head"])
            and (failure_reason is None or failure_reason == row["failure_reason"])
        )
        if requested is not current or not values_match:
            raise StateTransitionError(
                f"terminal run {row['id']} cannot transition from {current.value} "
                f"to {requested.value}"
            )

    def request_cancellation(
        self,
        run_id: str,
        enqueue_finalizer: Callable[[Connection], None] | None = None,
    ) -> tuple[RunView, bool]:
        """Request cancellation and durably enqueue its finalizer atomically.

        A repeated request while cancellation is in progress re-enqueues the same
        fixed workflow ID. DBOS treats that as idempotent, which lets callers
        recover from a failure after the database transaction commits.
        """
        with self.engine.begin() as connection:
            changed = (
                connection.execute(
                    update(runs)
                    .where(
                        runs.c.id == run_id,
                        runs.c.status.in_(
                            [RunStatus.QUEUED.value, RunStatus.RUNNING.value]
                        ),
                    )
                    .values(
                        status=RunStatus.CANCELLING.value,
                        updated_at=datetime.now(UTC),
                    )
                ).rowcount
                == 1
            )
            row = connection.execute(
                select(runs.c.status).where(runs.c.id == run_id)
            ).one_or_none()
            if row is None:
                raise KeyError(run_id)
            current = RunStatus(row.status)
            if current is RunStatus.CANCELLING:
                if enqueue_finalizer is not None:
                    enqueue_finalizer(connection)
                return self.get_run(run_id, connection=connection), changed
            if current in TERMINAL_RUN_STATUSES or current is RunStatus.FINALIZING:
                return self.get_run(run_id, connection=connection), False
            raise StateTransitionError(  # pragma: no cover - database invariant
                f"run {run_id} cannot request cancellation from {current.value}"
            )

    def finalize_cancellation(self, run_id: str) -> tuple[RunView, bool]:
        run = self.get_run(run_id)
        if run.status in TERMINAL_RUN_STATUSES:
            return run, False
        if run.status is not RunStatus.CANCELLING:
            raise StateTransitionError(
                f"run {run_id} cannot finalize cancellation from {run.status.value}"
            )
        try:
            return self.update_run(run_id, status=RunStatus.CANCELLED), True
        except StateTransitionError:
            current = self.get_run(run_id)
            if current.status in TERMINAL_RUN_STATUSES:
                return current, False
            raise

    def _terminalize_run_projections(
        self,
        connection: Connection,
        run_id: str,
        run_status: RunStatus,
        *,
        failure_reason: str | None,
    ) -> None:
        now = datetime.now(UTC)
        if run_status is RunStatus.CANCELLED:
            task_status = TaskStatus.CANCELLED
            execution_status = ExecutionStatus.CANCELLED
            event_type = "lifecycle.cancelled"
            event_key = "run-cancelled"
            payload: JsonValue = {"reason": failure_reason or "run cancelled"}
        else:
            task_status = TaskStatus.FAILED
            execution_status = ExecutionStatus.FAILED
            event_type = "lifecycle.failed"
            event_key = "run-failed"
            payload = {"error": failure_reason or "run failed"}
        connection.execute(
            update(tasks)
            .where(
                tasks.c.run_id == run_id,
                tasks.c.status.not_in(
                    [status.value for status in TERMINAL_TASK_STATUSES]
                ),
            )
            .values(status=task_status.value)
        )
        execution_ids = connection.execute(
            select(executions.c.id).where(
                executions.c.run_id == run_id,
                executions.c.status == ExecutionStatus.RUNNING.value,
            )
        ).scalars()
        for execution_id in execution_ids:
            self._append_event_in_transaction(
                connection,
                execution_id,
                event_type,
                payload,
                event_key=event_key,
            )
        connection.execute(
            update(executions)
            .where(
                executions.c.run_id == run_id,
                executions.c.status == ExecutionStatus.RUNNING.value,
            )
            .values(
                status=execution_status.value,
                final_output=payload,
                completed_at=now,
            )
        )

    def create_task(
        self,
        *,
        run_id: str,
        task_id: str,
        round_number: int,
        ordinal: int,
        description: str,
        acceptance_criteria: list[str],
        developer_workflow_id: str | None = None,
    ) -> TaskView:
        expected = {
            "run_id": run_id,
            "round_number": round_number,
            "ordinal": ordinal,
            "description": description,
            "acceptance_criteria": acceptance_criteria,
        }
        try:
            with self.engine.begin() as connection:
                existing = connection.execute(
                    select(tasks).where(tasks.c.id == task_id)
                ).mappings().one_or_none()
                if existing is not None:
                    self._validate_replay("task", task_id, existing, expected)
                    return self._task_view(existing)
                self._require_active_run(connection, run_id)
                connection.execute(
                    insert(tasks).values(
                        id=task_id,
                        run_id=run_id,
                        round_number=round_number,
                        ordinal=ordinal,
                        description=description,
                        acceptance_criteria=acceptance_criteria,
                        status=TaskStatus.QUEUED.value,
                        developer_workflow_id=developer_workflow_id,
                    )
                )
                row = connection.execute(
                    select(tasks).where(tasks.c.id == task_id)
                ).mappings().one()
                return self._task_view(row)
        except IntegrityError:  # pragma: no cover - exact insert race is database-scheduled
            with self.engine.connect() as connection:
                existing = connection.execute(
                    select(tasks).where(tasks.c.id == task_id)
                ).mappings().one_or_none()
                if existing is None:
                    raise
                self._validate_replay("task", task_id, existing, expected)
                return self._task_view(existing)

    @staticmethod
    def _validate_replay(kind: str, identifier: str, row, expected: dict) -> None:
        if any(row[name] != value for name, value in expected.items()):
            raise ReplayConflictError(f"conflicting replay for {kind} {identifier}")

    @staticmethod
    def _require_active_run(connection: Connection, run_id: str) -> None:
        value = connection.execute(
            select(runs.c.status).where(runs.c.id == run_id)
        ).scalar_one_or_none()
        if value is None:
            raise KeyError(run_id)
        if RunStatus(value) not in ACTIVE_RUN_STATUSES:
            raise StateTransitionError(f"run {run_id} is terminal")

    @staticmethod
    def _task_view(row) -> TaskView:
        return TaskView(
            id=row["id"],
            run_id=row["run_id"],
            round_number=row["round_number"],
            ordinal=row["ordinal"],
            description=row["description"],
            acceptance_criteria=list(row["acceptance_criteria"]),
            status=TaskStatus(row["status"]),
            developer_workflow_id=row["developer_workflow_id"],
        )

    def update_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        developer_workflow_id: str | None = None,
    ) -> None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(tasks).where(tasks.c.id == task_id).with_for_update()
            ).mappings().one_or_none()
            if row is None:
                raise KeyError(task_id)
            current = TaskStatus(row["status"])
            if current in TERMINAL_TASK_STATUSES:
                same_workflow = (
                    developer_workflow_id is None
                    or developer_workflow_id == row["developer_workflow_id"]
                )
                if status is current and same_workflow:
                    return
                raise StateTransitionError(
                    f"terminal task {task_id} cannot transition from "
                    f"{current.value} to {status.value}"
                )
            if status != current and status not in TASK_TRANSITIONS[current]:
                raise StateTransitionError(
                    f"task {task_id} cannot transition from {current.value} to {status.value}"
                )
            values: dict[str, object] = {"status": status.value}
            if developer_workflow_id is not None:
                values["developer_workflow_id"] = developer_workflow_id
            if connection.execute(
                update(tasks)
                .where(tasks.c.id == task_id, tasks.c.status == current.value)
                .values(**values)
            ).rowcount == 0:
                raise StateTransitionError(  # pragma: no cover - compare-and-set race
                    f"task {task_id} changed concurrently"
                )

    def create_execution(
        self,
        *,
        execution_id: str,
        run_id: str,
        role: Literal["planner", "developer", "reviewer"],
        workflow_id: str,
        session_id: str | None = None,
        task_id: str | None = None,
        started_at: datetime | None = None,
    ) -> AgentExecutionView:
        actual_started_at = started_at or datetime.now(UTC)
        expected = {
            "run_id": run_id,
            "role": role,
            "workflow_id": workflow_id,
            "session_id": session_id,
            "task_id": task_id,
        }
        try:
            with self.engine.begin() as connection:
                existing = connection.execute(
                    select(executions).where(executions.c.id == execution_id)
                ).mappings().one_or_none()
                if existing is not None:
                    self._validate_replay(
                        "execution", execution_id, existing, expected
                    )
                    return self._execution_view(existing)
                self._require_active_run(connection, run_id)
                connection.execute(
                    insert(executions).values(
                        id=execution_id,
                        run_id=run_id,
                        role=role,
                        workflow_id=workflow_id,
                        session_id=session_id,
                        task_id=task_id,
                        status=ExecutionStatus.RUNNING.value,
                        final_output=None,
                        next_sequence=0,
                        started_at=actual_started_at,
                        completed_at=None,
                    )
                )
                row = connection.execute(
                    select(executions).where(executions.c.id == execution_id)
                ).mappings().one()
                return self._execution_view(row)
        except IntegrityError:  # pragma: no cover - exact insert race is database-scheduled
            with self.engine.connect() as connection:
                existing = connection.execute(
                    select(executions).where(executions.c.id == execution_id)
                ).mappings().one_or_none()
                if existing is None:
                    raise
                self._validate_replay("execution", execution_id, existing, expected)
                return self._execution_view(existing)

    def get_execution(self, execution_id: str) -> AgentExecutionView:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(executions).where(executions.c.id == execution_id)
            ).mappings().one_or_none()
        if row is None:
            raise KeyError(execution_id)
        return self._execution_view(row)

    def list_executions(self, run_id: str) -> list[AgentExecutionView]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(executions)
                .where(executions.c.run_id == run_id)
                .order_by(executions.c.started_at, executions.c.id)
            ).mappings()
            return [self._execution_view(row) for row in rows]

    @staticmethod
    def _execution_view(row) -> AgentExecutionView:
        return AgentExecutionView(
            id=row["id"],
            run_id=row["run_id"],
            role=row["role"],
            workflow_id=row["workflow_id"],
            session_id=row["session_id"],
            task_id=row["task_id"],
            status=ExecutionStatus(row["status"]),
            final_output=row["final_output"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def finish_execution(
        self, execution_id: str, status: ExecutionStatus, final_output: JsonValue
    ) -> AgentExecutionView:
        if status is ExecutionStatus.RUNNING:
            raise StateTransitionError("finish_execution requires a terminal status")
        with self.engine.begin() as connection:
            return self._finish_execution_in_transaction(
                connection, execution_id, status, final_output
            )

    def _finish_execution_in_transaction(
        self,
        connection: Connection,
        execution_id: str,
        status: ExecutionStatus,
        final_output: JsonValue,
    ) -> AgentExecutionView:
        row = connection.execute(
            select(executions)
            .where(executions.c.id == execution_id)
            .with_for_update()
        ).mappings().one_or_none()
        if row is None:
            raise KeyError(execution_id)
        current = ExecutionStatus(row["status"])
        if current in TERMINAL_EXECUTION_STATUSES:
            if current is status and row["final_output"] == final_output:
                return self._execution_view(row)
            raise StateTransitionError(
                f"terminal execution {execution_id} cannot transition from "
                f"{current.value} to {status.value}"
            )
        if connection.execute(
            update(executions)
            .where(
                executions.c.id == execution_id,
                executions.c.status == ExecutionStatus.RUNNING.value,
            )
            .values(
                status=status.value,
                final_output=final_output,
                completed_at=datetime.now(UTC),
            )
        ).rowcount == 0:
            raise StateTransitionError(  # pragma: no cover - compare-and-set race
                f"execution {execution_id} changed concurrently"
            )
        finished = dict(row)
        finished.update(
            status=status.value,
            final_output=final_output,
            completed_at=connection.execute(
                select(executions.c.completed_at).where(executions.c.id == execution_id)
            ).scalar_one(),
        )
        return self._execution_view(finished)

    def append_event(
        self,
        execution_id: str,
        event_type: str,
        payload: JsonValue,
        *,
        event_key: str | None = None,
    ) -> TranscriptEvent:
        if event_key == "":
            raise ValueError("event_key must not be empty")
        try:
            with self.engine.begin() as connection:
                return self._append_event_in_transaction(
                    connection,
                    execution_id,
                    event_type,
                    payload,
                    event_key=event_key,
                )
        except IntegrityError:  # pragma: no cover - exact insert race is database-scheduled
            if event_key is None:
                raise
            with self.engine.connect() as connection:
                row = connection.execute(
                    select(events).where(
                        events.c.execution_id == execution_id,
                        events.c.event_key == event_key,
                    )
                ).mappings().one_or_none()
            if row is None:
                raise
            self._validate_event_replay(row, event_type, payload, event_key)
            return self._event_view(row)

    def _append_event_in_transaction(
        self,
        connection: Connection,
        execution_id: str,
        event_type: str,
        payload: JsonValue,
        *,
        event_key: str | None,
    ) -> TranscriptEvent:
        if event_key is not None:
            existing = connection.execute(
                select(events).where(
                    events.c.execution_id == execution_id,
                    events.c.event_key == event_key,
                )
            ).mappings().one_or_none()
            if existing is not None:
                self._validate_event_replay(existing, event_type, payload, event_key)
                return self._event_view(existing)
        execution_status = connection.execute(
            select(executions.c.status).where(executions.c.id == execution_id)
        ).scalar_one_or_none()
        if execution_status is None:
            raise KeyError(execution_id)
        if ExecutionStatus(execution_status) is not ExecutionStatus.RUNNING:
            raise StateTransitionError(f"execution {execution_id} is terminal")
        sequence = connection.execute(
            update(executions)
            .where(
                executions.c.id == execution_id,
                executions.c.status == ExecutionStatus.RUNNING.value,
            )
            .values(next_sequence=executions.c.next_sequence + 1)
            .returning(executions.c.next_sequence)
        ).scalar_one_or_none()
        if sequence is None:
            raise StateTransitionError(  # pragma: no cover - compare-and-set race
                f"execution {execution_id} changed concurrently"
            )
        now = datetime.now(UTC)
        connection.execute(
            insert(events).values(
                execution_id=execution_id,
                sequence=sequence,
                event_key=event_key,
                event_type=event_type,
                payload=payload,
                created_at=now,
            )
        )
        row = connection.execute(
            select(events).where(
                events.c.execution_id == execution_id,
                events.c.sequence == sequence,
            )
        ).mappings().one()
        return self._event_view(row)

    @staticmethod
    def _validate_event_replay(
        row, event_type: str, payload: JsonValue, event_key: str
    ) -> None:
        if row["event_type"] != event_type or row["payload"] != payload:
            raise ReplayConflictError(f"conflicting replay for event {event_key}")

    @staticmethod
    def _event_view(row) -> TranscriptEvent:
        return TranscriptEvent(
            execution_id=row["execution_id"],
            sequence=row["sequence"],
            event_key=row["event_key"],
            event_type=row["event_type"],
            payload=row["payload"],
            created_at=row["created_at"],
        )

    def finish_execution_with_event(
        self,
        execution_id: str,
        status: ExecutionStatus,
        final_output: JsonValue,
        *,
        event_key: str,
        event_type: str = "agent.final_output",
    ) -> tuple[TranscriptEvent, AgentExecutionView]:
        if status is ExecutionStatus.RUNNING:
            raise StateTransitionError(
                "finish_execution_with_event requires a terminal status"
            )
        if not event_key:
            raise ValueError("event_key must not be empty")
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(executions).where(executions.c.id == execution_id).with_for_update()
            ).mappings().one_or_none()
            if existing is None:
                raise KeyError(execution_id)
            current = ExecutionStatus(existing["status"])
            if current in TERMINAL_EXECUTION_STATUSES:
                if current is not status or existing["final_output"] != final_output:
                    raise StateTransitionError(
                        f"terminal execution {execution_id} conflicts with completion replay"
                    )
                event = self._get_replayed_event(
                    connection, execution_id, event_key, event_type, final_output
                )
                return event, self._execution_view(existing)
            final_event = self._append_event_in_transaction(
                connection,
                execution_id,
                event_type,
                final_output,
                event_key=event_key,
            )
            finished = self._finish_execution_in_transaction(
                connection, execution_id, status, final_output
            )
            return final_event, finished

    def _get_replayed_event(
        self,
        connection: Connection,
        execution_id: str,
        event_key: str,
        event_type: str,
        payload: JsonValue,
    ) -> TranscriptEvent:
        row = connection.execute(
            select(events).where(
                events.c.execution_id == execution_id,
                events.c.event_key == event_key,
            )
        ).mappings().one_or_none()
        if row is None:
            raise ReplayConflictError(
                f"terminal execution {execution_id} has no event {event_key}"
            )
        self._validate_event_replay(row, event_type, payload, event_key)
        return self._event_view(row)

    def list_events(
        self, execution_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TranscriptEvent]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(events)
                .where(
                    events.c.execution_id == execution_id,
                    events.c.sequence > after_sequence,
                )
                .order_by(events.c.sequence)
                .limit(limit)
            ).mappings()
            return [
                self._event_view(row)
                for row in rows
            ]

    def execution_terminal(self, execution_id: str) -> bool:
        return self.get_execution(execution_id).status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }

    def save_history(self, session_id: str, task_id: str, history: list[JsonValue]) -> None:
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(developer_sessions.c.id).where(developer_sessions.c.id == session_id)
            ).scalar_one_or_none()
            if existing:
                connection.execute(
                    update(developer_sessions)
                    .where(developer_sessions.c.id == session_id)
                    .values(message_history=history, updated_at=now)
                )
            else:
                connection.execute(
                    insert(developer_sessions).values(
                        id=session_id,
                        task_id=task_id,
                        message_history=history,
                        updated_at=now,
                    )
                )

    def load_history(self, session_id: str) -> list[JsonValue]:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(developer_sessions.c.message_history).where(
                    developer_sessions.c.id == session_id
                )
            ).scalar_one_or_none()
        return list(value or [])

    def clear(self) -> None:
        with self.engine.begin() as connection:
            for table in (events, executions, developer_sessions, tasks, runs):
                connection.execute(delete(table))
