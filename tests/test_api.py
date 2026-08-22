import asyncio
import subprocess
from dataclasses import replace
from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from agent_os.api import RunCoordinator, create_app
from agent_os.config import Settings
from agent_os.models import ExecutionStatus
from agent_os.store import StateStore


class FakeEnqueuer:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.cancelled: list[str] = []

    def enqueue(self, _connection, run_input) -> None:
        self.enqueued.append(run_input.workflow_id)

    def cancel(self, workflow_id: str) -> None:
        self.cancelled.append(workflow_id)


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
    assert client.get(f"/v1/runs/{body['id']}").status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}

    repeated = client.post(
        "/v1/runs",
        headers={"Idempotency-Key": "request-1"},
        json={"repository_path": str(repository)},
    )
    assert repeated.json()["id"] == body["id"]

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
    assert client.post(f"/v1/runs/{body['id']}/cancel").json()["status"] == "cancelled"

    assert client.get("/v1/runs/missing").status_code == 404
    assert client.post("/v1/runs/missing/cancel").status_code == 404
    assert client.get("/v1/runs/missing/executions").status_code == 404
    assert client.get("/v1/executions/missing/events").status_code == 404
    assert client.get("/v1/executions/missing/events/stream").status_code == 404


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
