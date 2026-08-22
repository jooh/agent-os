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
    create_engine,
    delete,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine
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
    Column("task_id", String),
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
    Column("event_type", String, nullable=False),
    Column("payload", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


class ActiveRunError(RuntimeError):
    pass


class StateStore:
    def __init__(self, database_url: str):
        base_engine = (
            create_engine(database_url, poolclass=NullPool)
            if database_url.startswith("sqlite")
            else create_engine(database_url)
        )
        self.engine: Engine = (
            base_engine.execution_options(schema_translate_map={SCHEMA: None})
            if base_engine.dialect.name == "sqlite"
            else base_engine
        )

    def bootstrap(self) -> None:
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as connection:
                connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        metadata.create_all(self.engine)

    def health(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(select(1)).scalar_one()

    def create_run(
        self,
        value: RunInput,
        idempotency_key: str | None,
        enqueue: Callable[[Connection, RunInput], None],
    ) -> tuple[RunView, bool]:
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            if idempotency_key:
                existing = connection.execute(
                    select(runs.c.id).where(runs.c.idempotency_key == idempotency_key)
                ).scalar_one_or_none()
                if existing:
                    return self.get_run(existing, connection=connection), False
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
        values: dict[str, object] = {"updated_at": datetime.now(UTC)}
        if status is not None:
            values["status"] = status.value
            if status in {RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED}:
                values["active_target_id"] = None
        if current_round is not None:
            values["current_round"] = current_round
        if integration_head is not None:
            values["integration_head"] = integration_head
        if failure_reason is not None:
            values["failure_reason"] = failure_reason
        with self.engine.begin() as connection:
            if connection.execute(
                update(runs).where(runs.c.id == run_id).values(**values)
            ).rowcount == 0:
                raise KeyError(run_id)
        return self.get_run(run_id)

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
        with self.engine.begin() as connection:
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
            row = connection.execute(select(tasks).where(tasks.c.id == task_id)).mappings().one()
        return self._task_view(row)

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
        values: dict[str, object] = {"status": status.value}
        if developer_workflow_id is not None:
            values["developer_workflow_id"] = developer_workflow_id
        with self.engine.begin() as connection:
            if connection.execute(
                update(tasks).where(tasks.c.id == task_id).values(**values)
            ).rowcount == 0:
                raise KeyError(task_id)

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
        started_at = started_at or datetime.now(UTC)
        with self.engine.begin() as connection:
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
                    started_at=started_at,
                    completed_at=None,
                )
            )
        return self.get_execution(execution_id)

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
        with self.engine.begin() as connection:
            if connection.execute(
                update(executions)
                .where(executions.c.id == execution_id)
                .values(
                    status=status.value,
                    final_output=final_output,
                    completed_at=datetime.now(UTC),
                )
            ).rowcount == 0:
                raise KeyError(execution_id)
        return self.get_execution(execution_id)

    def append_event(
        self, execution_id: str, event_type: str, payload: JsonValue
    ) -> TranscriptEvent:
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            sequence = connection.execute(
                update(executions)
                .where(executions.c.id == execution_id)
                .values(next_sequence=executions.c.next_sequence + 1)
                .returning(executions.c.next_sequence)
            ).scalar_one_or_none()
            if sequence is None:
                raise KeyError(execution_id)
            connection.execute(
                insert(events).values(
                    execution_id=execution_id,
                    sequence=sequence,
                    event_type=event_type,
                    payload=payload,
                    created_at=now,
                )
            )
        return TranscriptEvent(
            execution_id=execution_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            created_at=now,
        )

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
                TranscriptEvent(
                    execution_id=row["execution_id"],
                    sequence=row["sequence"],
                    event_type=row["event_type"],
                    payload=row["payload"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def execution_terminal(self, execution_id: str) -> bool:
        return self.get_execution(execution_id).status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
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
