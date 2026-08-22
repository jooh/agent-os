from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from sqlalchemy import create_engine

from agent_os.models import CancellationFinalizerInput, RunInput
from agent_os.runtime import DBOSEnqueuer


class FakeClient:
    instances: ClassVar[list[FakeClient]] = []

    def __init__(self, system_database_url: str):
        self.url = system_database_url
        self.enqueues: list[tuple[dict, tuple]] = []
        self.cancellations: list[tuple[str, bool]] = []
        self.queues: list[tuple[str, dict]] = []
        self.workflow_queries: list[dict] = []
        self.destroyed = False
        self.instances.append(self)

    def enqueue_in_transaction(self, _connection, options, *args):
        self.enqueues.append((options, args))

    def cancel_workflow(self, workflow_id: str, *, cancel_children: bool):
        self.cancellations.append((workflow_id, cancel_children))

    def list_workflows(self, **query):
        self.workflow_queries.append(query)
        if query.get("offset"):
            return []
        return [
            SimpleNamespace(
                workflow_id="engineering-run:run-1", status="CANCELLED"
            ),
            SimpleNamespace(
                workflow_id="engineering-run:run-1:planner:1", status="SUCCESS"
            ),
            SimpleNamespace(
                workflow_id=(
                    "engineering-run:run-1:cancellation-finalizer:recovery-attempt"
                ),
                status="PENDING",
            ),
        ]

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

    cancellation = CancellationFinalizerInput(
        run_id=value.run_id,
        root_workflow_id=value.workflow_id,
        target_id=value.target_id,
        database_url=value.database_url,
        state_dir=value.state_dir,
    )
    with engine.begin() as connection:
        enqueuer.enqueue_cancellation_finalizer(connection, cancellation)
    finalizer_client = FakeClient.instances[-1]
    options, args = finalizer_client.enqueues[0]
    assert options == {
        "workflow_name": "agent_os.cancellation_finalizer",
        "queue_name": "agent_os.orchestrator",
        "workflow_id": cancellation.finalizer_workflow_id,
    }
    assert args == (cancellation,)
    assert finalizer_client.destroyed

    enqueuer.cancel(value.workflow_id)
    cancel_client = FakeClient.instances[-1]
    assert cancel_client.cancellations == [(value.workflow_id, True)]
    assert cancel_client.destroyed

    assert enqueuer.cancellation_confirmed(value.workflow_id) is True
    confirmation_client = FakeClient.instances[-1]
    assert confirmation_client.workflow_queries == [
        {
            "workflow_id_prefix": value.workflow_id,
            "load_input": False,
            "load_output": False,
            "limit": 100,
            "offset": 0,
        }
    ]
    assert confirmation_client.destroyed

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


def test_cancellation_confirmation_is_conservative_and_paginated(monkeypatch) -> None:
    class PendingClient(FakeClient):
        def list_workflows(self, **query):
            return [
                SimpleNamespace(
                    workflow_id="engineering-run:run-1:developer:1", status="PENDING"
                )
            ]

    monkeypatch.setattr("agent_os.runtime.DBOSClient", PendingClient)
    enqueuer = DBOSEnqueuer("sqlite:///state.db")
    assert enqueuer.cancellation_confirmed("engineering-run:run-1") is False
    assert PendingClient.instances[-1].destroyed

    class PaginatedClient(FakeClient):
        def list_workflows(self, **query):
            if query["offset"] == 0:
                return [
                    SimpleNamespace(workflow_id=f"unrelated-{index}", status="SUCCESS")
                    for index in range(100)
                ]
            return [
                SimpleNamespace(
                    workflow_id="engineering-run:run-1", status="CANCELLED"
                )
            ]

    monkeypatch.setattr("agent_os.runtime.DBOSClient", PaginatedClient)
    assert enqueuer.cancellation_confirmed("engineering-run:run-1") is True
    assert PaginatedClient.instances[-1].destroyed
