import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from agent_os.api import RunCoordinator, create_app
from agent_os.config import Settings
from agent_os.git import GitRepository, _git
from agent_os.models import (
    CancellationFinalizerInput,
    ExecutionStatus,
    RunInput,
    TaskStatus,
)
from agent_os.store import StateStore


class FakeEnqueuer:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.enqueued_inputs: list[RunInput] = []
        self.cancellation_finalizers: list[CancellationFinalizerInput] = []
        self.cancelled: list[str] = []
        self.confirmed = True

    def enqueue(self, _connection, run_input: RunInput) -> None:
        self.enqueued.append(run_input.workflow_id)
        self.enqueued_inputs.append(run_input)

    def cancel(self, workflow_id: str) -> None:
        self.cancelled.append(workflow_id)

    def enqueue_cancellation_finalizer(
        self, _connection, finalizer_input: CancellationFinalizerInput
    ) -> None:
        self.cancellation_finalizers.append(finalizer_input)

    def cancellation_confirmed(self, _workflow_id: str) -> bool:
        return self.confirmed


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def make_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "PLAN.md").write_text("Create value.txt\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root


def test_api_starts_reads_and_cancels_run(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings(
        database_url=database_url,
        model="test:model",
        planner_model_override=None,
        developer_model_override=None,
        reviewer_model_override=None,
        state_dir=tmp_path / "state",
        application_version="0.1.0",
        developer_parallelism=2,
        planner_task_limit=1,
        max_rounds=10,
        max_review_cycles=3,
        max_developer_turns=6,
        model_request_limit=50,
        shell_timeout_seconds=900,
        api_host="127.0.0.1",
        api_port=8000,
    )
    store = StateStore(database_url)
    store.bootstrap()
    enqueuer = FakeEnqueuer()
    coordinator = RunCoordinator(settings, store, enqueuer)
    client = TestClient(create_app(settings, store=store, coordinator=coordinator))

    response = client.post(
        "/v1/runs",
        headers={"Idempotency-Key": "request-1"},
        json={"repository_path": str(repository)},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["integration_branch"].endswith("/integration")
    assert enqueuer.enqueued == [body["workflow_id"]]
    assert enqueuer.enqueued_inputs[0].planner_task_limit == 1
    assert client.get(f"/v1/runs/{body['id']}").status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}

    repeated = client.post(
        "/v1/runs",
        headers={"Idempotency-Key": "request-1"},
        json={"repository_path": str(repository)},
    )
    assert repeated.json()["id"] == body["id"]

    other_repository = tmp_path / "other-repo"
    other_repository.mkdir()
    git(other_repository, "init", "-b", "main")
    git(other_repository, "config", "user.name", "Test")
    git(other_repository, "config", "user.email", "test@example.com")
    (other_repository / "PLAN.md").write_text("A different plan\n")
    git(other_repository, "add", ".")
    git(other_repository, "commit", "-m", "initial")
    reused_for_other_target = client.post(
        "/v1/runs",
        headers={"Idempotency-Key": "request-1"},
        json={"repository_path": str(other_repository)},
    )
    assert reused_for_other_target.status_code == 409

    conflict = client.post(
        "/v1/runs",
        headers={"Idempotency-Key": "request-2"},
        json={"repository_path": str(repository)},
    )
    assert conflict.status_code == 409

    invalid = client.post(
        "/v1/runs", json={"repository_path": str(tmp_path / "not-a-repository")}
    )
    assert invalid.status_code == 422

    cancelled = client.post(f"/v1/runs/{body['id']}/cancel")
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    assert enqueuer.cancelled == [body["workflow_id"]]
    assert [value.finalizer_workflow_id for value in enqueuer.cancellation_finalizers] == [
        f"{body['workflow_id']}:cancellation-finalizer"
    ]
    assert client.post(f"/v1/runs/{body['id']}/cancel").json()["status"] == "cancelled"

    assert client.get("/v1/runs/missing").status_code == 404
    assert client.post("/v1/runs/missing/cancel").status_code == 404
    assert client.get("/v1/runs/missing/executions").status_code == 404
    assert client.get("/v1/executions/missing/events").status_code == 404
    assert client.get("/v1/executions/missing/events/stream").status_code == 404


def test_api_concurrent_idempotent_requests_enqueue_once(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()
    enqueuer = FakeEnqueuer()
    coordinator = RunCoordinator(settings, store, enqueuer)
    client = TestClient(create_app(settings, store=store, coordinator=coordinator))

    def start(_index: int):
        return client.post(
            "/v1/runs",
            headers={"Idempotency-Key": "same-request"},
            json={"repository_path": str(repository)},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(start, range(2)))

    assert [response.status_code for response in responses] == [202, 202]
    assert len({response.json()["id"] for response in responses}) == 1
    assert len(enqueuer.enqueued) == 1


def test_coordinator_requires_all_role_models(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = replace(
        Settings.from_values_for_test(database_url, tmp_path / "state"), model=None
    )
    store = StateStore(database_url)
    store.bootstrap()
    coordinator = RunCoordinator(settings, store, FakeEnqueuer())

    try:
        coordinator.start_run(
            repository_path=str(repository),
            plan_path="PLAN.md",
            base_ref="HEAD",
            idempotency_key=None,
        )
    except ValueError as exc:
        assert "models are required" in str(exc)
    else:
        raise AssertionError("missing models must be rejected")


def test_failed_workflow_cancellation_remains_retryable(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()

    class FlakyEnqueuer(FakeEnqueuer):
        attempts = 0

        def cancel(self, workflow_id: str) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("DBOS unavailable")
            super().cancel(workflow_id)

    enqueuer = FlakyEnqueuer()
    coordinator = RunCoordinator(settings, store, enqueuer)
    run = coordinator.start_run(
        repository_path=str(repository),
        plan_path="PLAN.md",
        base_ref="HEAD",
        idempotency_key=None,
    )
    task = store.create_task(
        run_id=run.id,
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Task",
        acceptance_criteria=["Done"],
    )
    store.update_task(task.id, TaskStatus.DEVELOPING)
    store.create_execution(
        execution_id="execution-1",
        run_id=run.id,
        role="developer",
        workflow_id="developer-1",
        task_id=task.id,
    )

    with pytest.raises(RuntimeError, match="DBOS unavailable"):
        coordinator.cancel_run(run.id)
    assert store.get_run(run.id).status.value == "cancelling"
    assert store.get_run(run.id).tasks[0].status is TaskStatus.DEVELOPING
    assert store.get_execution("execution-1").status is ExecutionStatus.RUNNING

    cancelled = coordinator.cancel_run(run.id)
    assert cancelled.status.value == "cancelled"
    assert cancelled.tasks[0].status is TaskStatus.CANCELLED
    assert store.get_execution("execution-1").status is ExecutionStatus.CANCELLED
    assert enqueuer.attempts == 2
    assert enqueuer.cancelled == [run.workflow_id]


def test_failed_finalizer_enqueue_rolls_back_cancellation_request(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()

    class FailingFinalizerEnqueuer(FakeEnqueuer):
        def enqueue_cancellation_finalizer(
            self, _connection, finalizer_input: CancellationFinalizerInput
        ) -> None:
            _ = finalizer_input
            raise RuntimeError("finalizer enqueue failed")

    enqueuer = FailingFinalizerEnqueuer()
    coordinator = RunCoordinator(settings, store, enqueuer)
    run = coordinator.start_run(
        repository_path=str(repository),
        plan_path="PLAN.md",
        base_ref="HEAD",
        idempotency_key=None,
    )

    with pytest.raises(RuntimeError, match="finalizer enqueue failed"):
        coordinator.cancel_run(run.id)

    assert store.get_run(run.id).status.value == "queued"
    assert enqueuer.cancelled == []


def test_cancelling_run_holds_target_lease_until_dbos_confirmation(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()
    enqueuer = FakeEnqueuer()
    enqueuer.confirmed = False
    coordinator = RunCoordinator(settings, store, enqueuer)
    client = TestClient(create_app(settings, store=store, coordinator=coordinator))
    started = client.post("/v1/runs", json={"repository_path": str(repository)})
    run_id = started.json()["id"]

    cancelling = client.post(f"/v1/runs/{run_id}/cancel")
    assert cancelling.status_code == 202
    assert cancelling.json()["status"] == "cancelling"
    assert (
        client.post("/v1/runs", json={"repository_path": str(repository)}).status_code
        == 409
    )
    assert client.get(f"/v1/runs/{run_id}").json()["status"] == "cancelling"

    enqueuer.confirmed = True
    assert client.get(f"/v1/runs/{run_id}").json()["status"] == "cancelled"
    replacement = client.post(
        "/v1/runs", json={"repository_path": str(repository)}
    )
    assert replacement.status_code == 202
    assert replacement.json()["id"] != run_id


def test_cancellation_waits_for_physical_git_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()
    enqueuer = FakeEnqueuer()
    coordinator = RunCoordinator(settings, store, enqueuer)
    client = TestClient(create_app(settings, store=store, coordinator=coordinator))
    started = client.post("/v1/runs", json={"repository_path": str(repository)})
    run_id = started.json()["id"]

    managed = GitRepository.inspect(repository, "PLAN.md", "HEAD", settings.state_dir)
    managed.prepare_integration_worktree()
    operation_started = Event()
    release_operation = Event()
    original_git = _git

    def block_cleanup(path: Path, *args: str, check: bool = True):
        if args[:3] == ("worktree", "remove", "--force"):
            operation_started.set()
            if not release_operation.wait(timeout=10):
                raise TimeoutError("test did not release cleanup")
        return original_git(path, *args, check=check)

    monkeypatch.setattr("agent_os.git._git", block_cleanup)
    cleanup = Thread(target=managed.cleanup_worktrees)
    cleanup.start()
    try:
        assert operation_started.wait(timeout=10)
        cancelling = client.post(f"/v1/runs/{run_id}/cancel")
        assert cancelling.status_code == 202
        assert cancelling.json()["status"] == "cancelling"
        assert (
            client.post(
                "/v1/runs", json={"repository_path": str(repository)}
            ).status_code
            == 409
        )
    finally:
        release_operation.set()
        cleanup.join(timeout=10)
    assert not cleanup.is_alive()
    assert client.get(f"/v1/runs/{run_id}").json()["status"] == "cancelled"


def test_api_pages_and_streams_transcripts(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()
    coordinator = RunCoordinator(settings, store, FakeEnqueuer())
    run = coordinator.start_run(
        repository_path=str(repository),
        plan_path="PLAN.md",
        base_ref="HEAD",
        idempotency_key=None,
    )
    store.create_execution(
        execution_id="execution-1",
        run_id=run.id,
        role="planner",
        workflow_id="planner-1",
    )
    store.append_event("execution-1", "model.output", {"text": "complete"})
    store.finish_execution("execution-1", ExecutionStatus.SUCCEEDED, {"complete": True})
    client = TestClient(create_app(settings, store=store, coordinator=coordinator))

    page = client.get("/v1/executions/execution-1/events?after_sequence=0&limit=10")
    assert page.status_code == 200
    assert page.json()[0]["sequence"] == 1
    assert client.get(f"/v1/runs/{run.id}/executions").json()[0]["id"] == "execution-1"

    with client.stream(
        "GET",
        "/v1/executions/execution-1/events/stream",
        headers={"Last-Event-ID": "0"},
    ) as stream:
        content = "".join(stream.iter_text())
    assert "id: 1" in content
    assert "event: model.output" in content
    assert '"text":"complete"' in content

    bad_cursor = client.get(
        "/v1/executions/execution-1/events/stream",
        headers={"Last-Event-ID": "not-a-number"},
    )
    assert bad_cursor.status_code == 400


def test_cancellation_terminals_are_visible_and_sse_drains(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()
    coordinator = RunCoordinator(settings, store, FakeEnqueuer())
    run = coordinator.start_run(
        repository_path=str(repository),
        plan_path="PLAN.md",
        base_ref="HEAD",
        idempotency_key=None,
    )
    task = store.create_task(
        run_id=run.id,
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Task",
        acceptance_criteria=["Done"],
    )
    store.update_task(task.id, TaskStatus.DEVELOPING)
    store.create_execution(
        execution_id="execution-running",
        run_id=run.id,
        role="developer",
        workflow_id="developer-1",
        task_id=task.id,
    )
    client = TestClient(create_app(settings, store=store, coordinator=coordinator))

    assert client.post(f"/v1/runs/{run.id}/cancel").status_code == 202
    assert store.get_execution("execution-running").status is ExecutionStatus.CANCELLED
    with client.stream(
        "GET", "/v1/executions/execution-running/events/stream"
    ) as stream:
        content = "".join(stream.iter_text())
    assert "event: lifecycle.cancelled" in content


def test_sse_heartbeats_and_stops_when_client_disconnects(
    tmp_path: Path, monkeypatch
) -> None:
    repository = make_repository(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    settings = Settings.from_values_for_test(database_url, tmp_path / "state")
    store = StateStore(database_url)
    store.bootstrap()
    coordinator = RunCoordinator(settings, store, FakeEnqueuer())
    run = coordinator.start_run(
        repository_path=str(repository),
        plan_path="PLAN.md",
        base_ref="HEAD",
        idempotency_key=None,
    )
    store.create_execution(
        execution_id="execution-running",
        run_id=run.id,
        role="developer",
        workflow_id="developer-1",
    )
    app = create_app(settings, store=store, coordinator=coordinator)
    route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path == "/v1/executions/{execution_id}/events/stream"
    )
    endpoint = route.endpoint

    class Request:
        calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            return self.calls > 1

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("agent_os.api.asyncio.sleep", no_sleep)
    response = endpoint("execution-running", Request(), None)

    async def consume() -> list[str]:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return chunks

    assert asyncio.run(consume()) == [": heartbeat\n\n"]

    store.append_event("execution-running", "model.delta", {"text": "working"})
    response_with_event = endpoint("execution-running", Request(), None)

    async def consume_event() -> list[str]:
        chunks = []
        async for chunk in response_with_event.body_iterator:
            chunks.append(chunk)
        return chunks

    assert "event: model.delta" in asyncio.run(consume_event())[0]
