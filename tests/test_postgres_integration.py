import os
import uuid
from pathlib import Path

import pytest

from agent_os.models import ExecutionStatus, RunInput
from agent_os.runtime import DBOSEnqueuer
from agent_os.store import StateStore

POSTGRES_URL = os.environ.get("AGENT_OS_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="AGENT_OS_TEST_POSTGRES_URL is not configured"
)


def test_postgresql_schema_jsonb_sequence_and_transactional_enqueue(tmp_path: Path) -> None:
    assert POSTGRES_URL is not None
    suffix = uuid.uuid4().hex
    value = RunInput(
        run_id=f"postgres-{suffix}",
        workflow_id=f"engineering-run:postgres-{suffix}",
        target_id=suffix.ljust(64, "0"),
        repository_path=str(tmp_path),
        plan_path="PLAN.md",
        plan_id="a" * 64,
        base_commit="b" * 40,
        integration_branch="agent/aaaaaaaaaaaa-bbbbbbbbbbbb/integration",
        database_url=POSTGRES_URL,
        state_dir=str(tmp_path / "state"),
        planner_model="test",
        developer_model="test",
        reviewer_model="test",
        max_rounds=2,
        max_review_cycles=2,
        max_developer_turns=2,
        model_request_limit=10,
        shell_timeout_seconds=30,
    )
    store = StateStore(POSTGRES_URL)
    store.bootstrap()
    enqueuer = DBOSEnqueuer(POSTGRES_URL)
    enqueuer.register_queues(2)

    run, created = store.create_run(value, f"request-{suffix}", enqueuer.enqueue)
    assert created and run.id == value.run_id
    task = store.create_task(
        run_id=value.run_id,
        task_id=f"task-{suffix}",
        round_number=1,
        ordinal=1,
        description="PostgreSQL integration",
        acceptance_criteria=["JSONB round-trips"],
    )
    store.save_history("session-1", task.id, [{"kind": "request", "parts": []}])
    assert store.load_history("session-1") == [{"kind": "request", "parts": []}]
    store.create_execution(
        execution_id=f"execution-{suffix}",
        run_id=value.run_id,
        role="developer",
        workflow_id=f"developer-{suffix}",
        task_id=task.id,
    )
    first = store.append_event(f"execution-{suffix}", "model.stream", {"delta": "a"})
    second = store.append_event(
        f"execution-{suffix}",
        "tool.result",
        {"ok": True},
        event_key="tool-result-1",
    )
    assert (
        store.append_event(
            f"execution-{suffix}",
            "tool.result",
            {"ok": True},
            event_key="tool-result-1",
        )
        == second
    )
    assert (first.sequence, second.sequence) == (1, 2)
    store.finish_execution(
        f"execution-{suffix}", ExecutionStatus.SUCCEEDED, {"summary": "done"}
    )
