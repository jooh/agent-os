from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.engine import Connection
from sqlalchemy.pool import NullPool

from agent_os.models import ExecutionStatus, RunInput, RunStatus, TaskStatus
from agent_os.store import ActiveRunError, StateStore


def run_input(tmp_path: Path, run_id: str = "run-1") -> RunInput:
    return RunInput(
        run_id=run_id,
        workflow_id=f"engineering-run:{run_id}",
        target_id="target-1",
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
    store.update_task(task.id, TaskStatus.FIXING, developer_workflow_id="developer-1")
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
