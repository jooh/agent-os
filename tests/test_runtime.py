from pathlib import Path
from typing import ClassVar

from sqlalchemy import create_engine

from agent_os.models import RunInput
from agent_os.runtime import DBOSEnqueuer


class FakeClient:
    instances: ClassVar[list[FakeClient]] = []

    def __init__(self, system_database_url: str):
        self.url = system_database_url
        self.enqueues: list[tuple[dict, tuple]] = []
        self.cancellations: list[tuple[str, bool]] = []
        self.queues: list[tuple[str, dict]] = []
        self.destroyed = False
        self.instances.append(self)

    def enqueue_in_transaction(self, _connection, options, *args):
        self.enqueues.append((options, args))

    def cancel_workflow(self, workflow_id: str, *, cancel_children: bool):
        self.cancellations.append((workflow_id, cancel_children))

    def register_queue(self, name: str, **options):
        self.queues.append((name, options))

    def destroy(self):
        self.destroyed = True


def test_dbos_enqueuer_uses_named_queue_and_cancels_children(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("agent_os.runtime.DBOSClient", FakeClient)
    value = RunInput(
        run_id="run-1",
        workflow_id="engineering-run:run-1",
        target_id="a" * 64,
        repository_path=str(tmp_path),
        plan_path="PLAN.md",
        plan_id="b" * 64,
        base_commit="c" * 40,
        integration_branch="agent/bbbbbbbbbbbb-cccccccccccc/integration",
        database_url="sqlite:///state.db",
        state_dir=str(tmp_path),
        planner_model="test:model",
        developer_model="test:model",
        reviewer_model="test:model",
        max_rounds=10,
        max_review_cycles=3,
        max_developer_turns=6,
        model_request_limit=50,
        shell_timeout_seconds=900,
    )
    enqueuer = DBOSEnqueuer("sqlite:///state.db")
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        enqueuer.enqueue(connection, value)
    enqueue_client = FakeClient.instances[-1]
    options, args = enqueue_client.enqueues[0]
    assert options["workflow_name"] == "agent_os.engineering_run"
    assert options["queue_name"] == "agent_os.orchestrator"
    assert args == (value,)
    assert enqueue_client.destroyed

    enqueuer.cancel(value.workflow_id)
    cancel_client = FakeClient.instances[-1]
    assert cancel_client.cancellations == [(value.workflow_id, True)]
    assert cancel_client.destroyed

    enqueuer.register_queues(3)
    queue_client = FakeClient.instances[-1]
    assert [name for name, _options in queue_client.queues] == [
        "agent_os.orchestrator",
        "agent_os.planner",
        "agent_os.developer",
        "agent_os.reviewer",
    ]
    assert queue_client.queues[2][1]["concurrency"] == 3
    assert queue_client.destroyed
