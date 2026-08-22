from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

from agent_os.models import ExecutionStatus, RunInput, RunStatus, TaskStatus
from agent_os.store import (
    ActiveRunError,
    IdempotencyKeyError,
    ReplayConflictError,
    StateStore,
    StateTransitionError,
    developer_sessions,
    events,
    executions,
    runs,
    tasks,
)


def run_input(
    tmp_path: Path, run_id: str = "run-1", *, target_id: str = "target-1"
) -> RunInput:
    return RunInput(
        run_id=run_id,
        workflow_id=f"engineering-run:{run_id}",
        target_id=target_id,
        repository_path=str(tmp_path),
        plan_path="PLAN.md",
        plan_id="a" * 64,
        base_commit="b" * 40,
        integration_branch="agent/aaaaaaaaaaaa-bbbbbbbbbbbb/integration",
        database_url=f"sqlite:///{tmp_path / 'state.db'}",
        state_dir=str(tmp_path / "state"),
        planner_model="test:model",
        developer_model="test:model",
        reviewer_model="test:model",
        max_rounds=10,
        max_review_cycles=3,
        max_developer_turns=6,
        model_request_limit=50,
        shell_timeout_seconds=900,
    )


def test_store_bootstrap_run_idempotency_and_active_conflict(tmp_path: Path) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    assert isinstance(store.engine.pool, NullPool)
    store.bootstrap()
    enqueued: list[str] = []

    def enqueue(_connection: Connection, value: RunInput) -> None:
        enqueued.append(value.workflow_id)

    first, created = store.create_run(run_input(tmp_path), "request-1", enqueue)
    repeated, repeated_created = store.create_run(run_input(tmp_path, "ignored"), "request-1", enqueue)

    assert created is True
    assert repeated_created is False
    assert repeated.id == first.id
    assert enqueued == ["engineering-run:run-1"]

    with pytest.raises(ActiveRunError):
        store.create_run(run_input(tmp_path, "run-2"), None, enqueue)

    store.update_run("run-1", status=RunStatus.COMPLETE, integration_head="c" * 40)
    second, second_created = store.create_run(run_input(tmp_path, "run-2"), None, enqueue)
    assert second_created is True
    assert second.status is RunStatus.QUEUED
    store.update_run(
        "run-2", status=RunStatus.RUNNING, current_round=2, failure_reason="diagnostic"
    )
    assert store.get_run("run-2").failure_reason == "diagnostic"
    assert store.update_run("run-2", current_round=3).current_round == 3

    with pytest.raises(KeyError):
        store.get_run("missing")
    with pytest.raises(KeyError):
        store.update_run("missing", status=RunStatus.FAILED)


def test_idempotency_key_is_atomic_and_bound_to_target(tmp_path: Path) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    ready = Barrier(2)
    enqueued: list[str] = []

    def create(value: RunInput) -> tuple[str, bool]:
        ready.wait()
        run, created = store.create_run(
            value,
            "shared-request",
            lambda _connection, input_value: enqueued.append(input_value.workflow_id),
        )
        return run.id, created

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(create, [run_input(tmp_path), run_input(tmp_path, "run-2")])
        )

    assert len({run_id for run_id, _created in results}) == 1
    assert sorted(created for _run_id, created in results) == [False, True]
    assert len(enqueued) == 1

    with pytest.raises(IdempotencyKeyError, match="different target"):
        store.create_run(
            run_input(tmp_path, "run-3", target_id="target-2"),
            "shared-request",
            lambda _connection, _value: None,
        )

    cause = IntegrityError("insert", {}, RuntimeError("constraint"))
    with pytest.raises(ActiveRunError, match="active run"):
        store._resolve_run_insert_conflict(
            run_input(tmp_path, "run-4"), "unknown-request", cause
        )
    store.update_run(results[0][0], status=RunStatus.COMPLETE)
    with pytest.raises(IntegrityError) as raised:
        store._resolve_run_insert_conflict(
            run_input(tmp_path, "run-5", target_id="unused-target"), None, cause
        )
    assert raised.value is cause


def test_cancellation_request_preserves_a_terminal_transition(tmp_path: Path) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.update_run("run-1", status=RunStatus.COMPLETE)
    enqueued = False

    def enqueue(_connection) -> None:
        nonlocal enqueued
        enqueued = True

    run, changed = store.request_cancellation("run-1", enqueue)
    assert run.status is RunStatus.COMPLETE
    assert changed is False
    assert enqueued is False


def test_cancellation_request_atomically_enqueues_and_retries_finalizer(
    tmp_path: Path,
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    attempts: list[str] = []

    def fail_enqueue(_connection) -> None:
        attempts.append("failed")
        raise RuntimeError("enqueue failed")

    with pytest.raises(RuntimeError, match="enqueue failed"):
        store.request_cancellation("run-1", fail_enqueue)
    assert store.get_run("run-1").status is RunStatus.QUEUED

    def enqueue(_connection) -> None:
        attempts.append("enqueued")

    cancelling, changed = store.request_cancellation("run-1", enqueue)
    assert cancelling.status is RunStatus.CANCELLING
    assert changed is True

    repeated, repeated_changed = store.request_cancellation("run-1", enqueue)
    assert repeated.status is RunStatus.CANCELLING
    assert repeated_changed is False
    assert attempts == ["failed", "enqueued", "enqueued"]

    with pytest.raises(KeyError, match="missing-run"):
        store.request_cancellation("missing-run", enqueue)


def test_store_tasks_executions_history_and_monotonic_events(tmp_path: Path) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    task = store.create_task(
        run_id="run-1",
        task_id="r01-t01-deadbeef",
        round_number=1,
        ordinal=1,
        description="Implement the feature",
        acceptance_criteria=["It works"],
    )
    assert store.get_run("run-1").tasks == [task]

    started = datetime.now(UTC)
    store.create_execution(
        execution_id="execution-1",
        run_id="run-1",
        role="developer",
        workflow_id="developer-1",
        session_id="session-1",
        task_id=task.id,
        started_at=started,
    )
    event_1 = store.append_event("execution-1", "lifecycle.started", {"ok": True})
    event_2 = store.append_event("execution-1", "model.output", {"text": "done"})
    assert [event.sequence for event in store.list_events("execution-1")] == [1, 2]
    assert event_1.sequence == 1
    assert event_2.sequence == 2

    store.save_history("session-1", task.id, [{"kind": "message"}])
    assert store.load_history("session-1") == [{"kind": "message"}]
    store.save_history("session-1", task.id, [{"kind": "updated"}])
    assert store.load_history("session-1") == [{"kind": "updated"}]
    assert store.load_history("missing") == []
    store.update_task(
        task.id, TaskStatus.DEVELOPING, developer_workflow_id="developer-1"
    )
    store.update_task(task.id, TaskStatus.FIXING)
    assert store.get_run("run-1").tasks[0].developer_workflow_id == "developer-1"
    store.finish_execution("execution-1", ExecutionStatus.SUCCEEDED, {"summary": "done"})
    execution = store.list_executions("run-1")[0]
    assert execution.status is ExecutionStatus.SUCCEEDED
    assert store.execution_terminal("execution-1") is True

    with pytest.raises(KeyError):
        store.update_task("missing", TaskStatus.FAILED)
    with pytest.raises(KeyError):
        store.get_execution("missing")
    with pytest.raises(KeyError):
        store.finish_execution("missing", ExecutionStatus.FAILED, {"error": "missing"})
    with pytest.raises(KeyError):
        store.append_event("missing", "event", {})

    store.clear()
    with pytest.raises(KeyError):
        store.get_run("run-1")


def test_creation_replays_are_idempotent_but_conflicts_are_rejected(
    tmp_path: Path,
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    first_task = store.create_task(
        run_id="run-1",
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Implement it",
        acceptance_criteria=["It works"],
    )
    assert (
        store.create_task(
            run_id="run-1",
            task_id="task-1",
            round_number=1,
            ordinal=1,
            description="Implement it",
            acceptance_criteria=["It works"],
        )
        == first_task
    )
    with pytest.raises(ReplayConflictError, match="task task-1"):
        store.create_task(
            run_id="run-1",
            task_id="task-1",
            round_number=1,
            ordinal=1,
            description="Something else",
            acceptance_criteria=["It works"],
        )

    started = datetime.now(UTC)
    first_execution = store.create_execution(
        execution_id="execution-1",
        run_id="run-1",
        role="developer",
        workflow_id="developer-1",
        session_id="session-1",
        task_id="task-1",
        started_at=started,
    )
    assert (
        store.create_execution(
            execution_id="execution-1",
            run_id="run-1",
            role="developer",
            workflow_id="developer-1",
            session_id="session-1",
            task_id="task-1",
            started_at=started,
        )
        == first_execution
    )
    with pytest.raises(ReplayConflictError, match="execution execution-1"):
        store.create_execution(
            execution_id="execution-1",
            run_id="run-1",
            role="developer",
            workflow_id="developer-2",
            session_id="session-1",
            task_id="task-1",
            started_at=started,
        )


def test_guarded_transitions_and_cancellation_terminalize_projections(
    tmp_path: Path,
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.update_run("run-1", status=RunStatus.RUNNING, current_round=1)
    store.create_task(
        run_id="run-1",
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Implement it",
        acceptance_criteria=["It works"],
    )
    store.update_task("task-1", TaskStatus.DEVELOPING)
    store.create_execution(
        execution_id="execution-1",
        run_id="run-1",
        role="developer",
        workflow_id="developer-1",
        task_id="task-1",
    )

    cancelling, changed = store.request_cancellation("run-1")

    assert changed is True
    assert cancelling.status is RunStatus.CANCELLING
    assert cancelling.tasks[0].status is TaskStatus.DEVELOPING
    assert store.get_execution("execution-1").status is ExecutionStatus.RUNNING
    with pytest.raises(ActiveRunError):
        store.create_run(
            run_input(tmp_path, "run-2"), None, lambda _connection, _value: None
        )
    repeated_request, request_changed = store.request_cancellation("run-1")
    assert repeated_request.status is RunStatus.CANCELLING
    assert request_changed is False

    cancelled, finalized = store.finalize_cancellation("run-1")

    assert finalized is True
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.tasks[0].status is TaskStatus.CANCELLED
    execution = store.get_execution("execution-1")
    assert execution.status is ExecutionStatus.CANCELLED
    assert execution.completed_at is not None
    assert store.execution_terminal("execution-1") is True
    cancellation_events = store.list_events("execution-1")
    assert [(event.event_key, event.event_type) for event in cancellation_events] == [
        ("run-cancelled", "lifecycle.cancelled")
    ]

    repeated, repeated_changed = store.finalize_cancellation("run-1")
    assert repeated == cancelled
    assert repeated_changed is False
    terminal_request, terminal_request_changed = store.request_cancellation("run-1")
    assert terminal_request == cancelled
    assert terminal_request_changed is False
    with pytest.raises(StateTransitionError):
        store.update_run("run-1", status=RunStatus.FAILED, failure_reason="late")
    with pytest.raises(StateTransitionError):
        store.update_run("run-1", integration_head="c" * 40)
    with pytest.raises(StateTransitionError):
        store.update_task("task-1", TaskStatus.INTEGRATED)
    with pytest.raises(StateTransitionError):
        store.finish_execution(
            "execution-1", ExecutionStatus.SUCCEEDED, {"summary": "late"}
        )
    with pytest.raises(StateTransitionError):
        store.append_event("execution-1", "model.output", {"text": "late"})

    next_run, created = store.create_run(
        run_input(tmp_path, "run-2"), None, lambda _connection, _value: None
    )
    assert created is True
    assert next_run.status is RunStatus.QUEUED


def test_cancellation_finalization_guards_and_concurrent_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    with pytest.raises(StateTransitionError, match="cannot transition"):
        store.update_run("run-1", status=RunStatus.CANCELLED)
    with pytest.raises(StateTransitionError, match="cannot finalize"):
        store.finalize_cancellation("run-1")

    store.request_cancellation("run-1")
    original_update = store.update_run

    def cancelled_then_conflict(*_args, **_kwargs):
        with store.engine.begin() as connection:
            connection.execute(
                update(runs)
                .where(runs.c.id == "run-1")
                .values(status=RunStatus.CANCELLED.value, active_target_id=None)
            )
        raise StateTransitionError("lost race")

    monkeypatch.setattr(store, "update_run", cancelled_then_conflict)
    run, changed = store.finalize_cancellation("run-1")
    assert run.status is RunStatus.CANCELLED
    assert changed is False

    second_store = StateStore(f"sqlite:///{tmp_path / 'second-finalize.db'}")
    second_store.bootstrap()
    second_store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    second_store.request_cancellation("run-1")

    def conflict(*_args, **_kwargs):
        raise StateTransitionError("lost race")

    monkeypatch.setattr(second_store, "update_run", conflict)
    with pytest.raises(StateTransitionError, match="lost race"):
        second_store.finalize_cancellation("run-1")
    monkeypatch.setattr(store, "update_run", original_update)


def test_finalizing_run_holds_target_lease_and_completion_wins_cancellation(
    tmp_path: Path,
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.update_run("run-1", status=RunStatus.RUNNING)

    finalizing = store.update_run("run-1", status=RunStatus.FINALIZING)
    assert finalizing.status is RunStatus.FINALIZING
    unchanged, changed = store.request_cancellation("run-1")
    assert unchanged.status is RunStatus.FINALIZING
    assert changed is False
    with pytest.raises(ActiveRunError):
        store.create_run(
            run_input(tmp_path, "run-2"), None, lambda _connection, _value: None
        )

    completed = store.update_run("run-1", status=RunStatus.COMPLETE)
    assert completed.status is RunStatus.COMPLETE
    replacement, created = store.create_run(
        run_input(tmp_path, "run-2"), None, lambda _connection, _value: None
    )
    assert created is True
    assert replacement.status is RunStatus.QUEUED


def test_target_lease_assertion_requires_owner_and_explicit_status(
    tmp_path: Path,
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    value = run_input(tmp_path, target_id="a" * 64)
    store.create_run(value, None, lambda _connection, _value: None)
    store.update_run(value.run_id, status=RunStatus.RUNNING)

    store.assert_target_lease(
        value.run_id, value.target_id, frozenset({RunStatus.RUNNING})
    )
    with pytest.raises(StateTransitionError, match="status running"):
        store.assert_target_lease(
            value.run_id, value.target_id, frozenset({RunStatus.FINALIZING})
        )
    with pytest.raises(StateTransitionError, match="target lease"):
        store.assert_target_lease(
            value.run_id, "b" * 64, frozenset({RunStatus.RUNNING})
        )

    store.update_run(value.run_id, status=RunStatus.FINALIZING)
    store.assert_target_lease(
        value.run_id, value.target_id, frozenset({RunStatus.FINALIZING})
    )
    store.update_run(value.run_id, status=RunStatus.COMPLETE)
    with pytest.raises(StateTransitionError, match="target lease"):
        store.assert_target_lease(
            value.run_id, value.target_id, frozenset({RunStatus.RUNNING})
        )

def test_failed_run_terminalizes_projections_and_terminal_replays_are_exact(
    tmp_path: Path,
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.create_task(
        run_id="run-1",
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Task",
        acceptance_criteria=["Done"],
    )
    store.create_execution(
        execution_id="execution-1",
        run_id="run-1",
        role="developer",
        workflow_id="developer-1",
        task_id="task-1",
    )

    failed = store.update_run(
        "run-1", status=RunStatus.FAILED, current_round=1, failure_reason="boom"
    )
    assert failed.tasks[0].status is TaskStatus.FAILED
    assert store.get_execution("execution-1").status is ExecutionStatus.FAILED
    assert store.list_events("execution-1")[0].event_type == "lifecycle.failed"
    assert (
        store.update_run(
            "run-1", status=RunStatus.FAILED, current_round=1, failure_reason="boom"
        )
        == failed
    )


def test_store_rejects_invalid_projection_operations(tmp_path: Path) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    with pytest.raises(KeyError):
        store.create_task(
            run_id="missing",
            task_id="task-missing-run",
            round_number=1,
            ordinal=1,
            description="Task",
            acceptance_criteria=["Done"],
        )
    with pytest.raises(KeyError):
        store.create_execution(
            execution_id="execution-missing-run",
            run_id="missing",
            role="planner",
            workflow_id="planner-missing",
        )

    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    task = store.create_task(
        run_id="run-1",
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Task",
        acceptance_criteria=["Done"],
    )
    with pytest.raises(StateTransitionError, match="queued to reviewing"):
        store.update_task(task.id, TaskStatus.REVIEWING)
    store.update_task(task.id, TaskStatus.FAILED)
    store.update_task(task.id, TaskStatus.FAILED)
    with pytest.raises(StateTransitionError, match="terminal task"):
        store.update_task(
            task.id, TaskStatus.FAILED, developer_workflow_id="late-developer"
        )

    store.update_run("run-1", status=RunStatus.FAILED)
    with pytest.raises(StateTransitionError, match="terminal"):
        store.create_task(
            run_id="run-1",
            task_id="late-task",
            round_number=2,
            ordinal=1,
            description="Late",
            acceptance_criteria=["Rejected"],
        )
    with pytest.raises(StateTransitionError, match="terminal"):
        store.create_execution(
            execution_id="late-execution",
            run_id="run-1",
            role="planner",
            workflow_id="late-planner",
        )


def test_stable_events_and_final_event_completion_are_idempotent(
    tmp_path: Path,
) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.create_execution(
        execution_id="execution-1",
        run_id="run-1",
        role="planner",
        workflow_id="planner-1",
    )

    first = store.append_event(
        "execution-1", "model.output", {"text": "hello"}, event_key="output-1"
    )
    assert (
        store.append_event(
            "execution-1", "model.output", {"text": "hello"}, event_key="output-1"
        )
        == first
    )
    with pytest.raises(ReplayConflictError, match="event output-1"):
        store.append_event(
            "execution-1", "model.output", {"text": "different"}, event_key="output-1"
        )

    final_event, execution = store.finish_execution_with_event(
        "execution-1",
        ExecutionStatus.SUCCEEDED,
        {"complete": True},
        event_key="final-output",
    )
    assert final_event.sequence == 2
    assert execution.status is ExecutionStatus.SUCCEEDED
    assert execution.final_output == {"complete": True}
    repeated_event, repeated_execution = store.finish_execution_with_event(
        "execution-1",
        ExecutionStatus.SUCCEEDED,
        {"complete": True},
        event_key="final-output",
    )
    assert repeated_event == final_event
    assert repeated_execution == execution
    assert len(store.list_events("execution-1")) == 2


def test_execution_and_event_completion_validation(tmp_path: Path) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.create_execution(
        execution_id="execution-1",
        run_id="run-1",
        role="planner",
        workflow_id="planner-1",
    )
    with pytest.raises(StateTransitionError, match="terminal status"):
        store.finish_execution("execution-1", ExecutionStatus.RUNNING, None)
    with pytest.raises(ValueError, match="event_key"):
        store.append_event("execution-1", "model.output", {}, event_key="")
    with pytest.raises(StateTransitionError, match="terminal status"):
        store.finish_execution_with_event(
            "execution-1", ExecutionStatus.RUNNING, None, event_key="final"
        )
    with pytest.raises(ValueError, match="event_key"):
        store.finish_execution_with_event(
            "execution-1", ExecutionStatus.SUCCEEDED, {}, event_key=""
        )
    with pytest.raises(KeyError):
        store.finish_execution_with_event(
            "missing", ExecutionStatus.SUCCEEDED, {}, event_key="final"
        )

    finished = store.finish_execution(
        "execution-1", ExecutionStatus.SUCCEEDED, {"complete": True}
    )
    assert (
        store.finish_execution(
            "execution-1", ExecutionStatus.SUCCEEDED, {"complete": True}
        )
        == finished
    )
    with pytest.raises(StateTransitionError, match="terminal execution"):
        store.finish_execution("execution-1", ExecutionStatus.FAILED, {"error": "late"})
    with pytest.raises(StateTransitionError, match="completion replay"):
        store.finish_execution_with_event(
            "execution-1",
            ExecutionStatus.FAILED,
            {"error": "late"},
            event_key="final",
        )
    with pytest.raises(ReplayConflictError, match="has no event"):
        store.finish_execution_with_event(
            "execution-1",
            ExecutionStatus.SUCCEEDED,
            {"complete": True},
            event_key="missing-final",
        )


def test_sqlite_foreign_keys_reject_orphans_and_cascade(tmp_path: Path) -> None:
    store = StateStore(f"sqlite:///{tmp_path / 'state.db'}")
    store.bootstrap()
    with pytest.raises(IntegrityError), store.engine.begin() as connection:
        connection.execute(
            insert(tasks).values(
                id="orphan",
                run_id="missing",
                round_number=1,
                ordinal=1,
                description="Orphan",
                acceptance_criteria=["Rejected"],
                status=TaskStatus.QUEUED.value,
                developer_workflow_id=None,
            )
        )

    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.create_task(
        run_id="run-1",
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Task",
        acceptance_criteria=["Done"],
    )
    store.save_history("session-1", "task-1", [])
    store.create_execution(
        execution_id="execution-1",
        run_id="run-1",
        role="developer",
        workflow_id="developer-1",
        task_id="task-1",
    )
    store.append_event("execution-1", "lifecycle.started", {})

    with store.engine.begin() as connection:
        connection.execute(delete(runs).where(runs.c.id == "run-1"))
    with store.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(tasks)).scalar_one() == 0
        assert (
            connection.execute(
                select(func.count()).select_from(developer_sessions)
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(select(func.count()).select_from(executions)).scalar_one()
            == 0
        )
        assert connection.execute(select(func.count()).select_from(events)).scalar_one() == 0

    with pytest.raises(IntegrityError), store.engine.begin() as connection:
        connection.execute(
            insert(events).values(
                execution_id="missing",
                sequence=1,
                event_key=None,
                event_type="orphan",
                payload={},
                created_at=datetime.now(UTC),
            )
        )


def test_postgresql_bootstrap_creates_application_schema(monkeypatch) -> None:
    statements: list[str] = []

    class ConnectionContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement):
            statements.append(str(statement))

    class Engine:
        dialect = type("Dialect", (), {"name": "postgresql"})()

        def begin(self):
            return ConnectionContext()

    engine = Engine()
    monkeypatch.setattr("agent_os.store.create_engine", lambda _url: engine)
    monkeypatch.setattr("agent_os.store.metadata.create_all", lambda value: statements.append("tables"))
    StateStore("postgresql+psycopg://localhost/test").bootstrap()
    assert statements == ["CREATE SCHEMA IF NOT EXISTS agent_os", "tables"]
