import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENT_OS_PROCESS_SMOKE") != "1",
    reason="five-process smoke is not enabled",
)

def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def request(url: str, *, method: str = "GET", payload: dict | None = None) -> dict | list:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers, method=method), timeout=5
    ) as response:
        return json.loads(response.read())


def wait_for_api(base_url: str) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            if request(f"{base_url}/healthz") == {"status": "ok"}:
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise TimeoutError("API did not become healthy")


def wait_for_run(base_url: str, run_id: str) -> dict:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        run = request(f"{base_url}/v1/runs/{run_id}")
        assert isinstance(run, dict)
        if run["status"] in {"complete", "failed", "cancelled"}:
            return run
        time.sleep(0.25)
    raise TimeoutError(f"run {run_id} did not finish")


def test_api_and_four_worker_processes_complete_noop_rerun(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'agent-os.sqlite3'}"
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.com")
    (repository / "PLAN.md").write_text("The repository is already complete.\n")
    (repository / "README.md").write_text("complete\n")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "initial")
    base_commit = git(repository, "rev-parse", "HEAD")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        {
            "DBOS_SYSTEM_DATABASE_URL": database_url,
            "AGENT_OS_MODEL": "test",
            "AGENT_OS_API_PORT": str(port),
            "AGENT_OS_STATE_DIR": str(tmp_path / "state"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    processes: list[subprocess.Popen[bytes]] = []

    def start(*arguments: str) -> None:
        processes.append(
            subprocess.Popen(
                [sys.executable, "-m", "agent_os", *arguments],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    try:
        start("api")
        wait_for_api(base_url)
        for role in ("orchestrator", "planner", "developer", "reviewer"):
            start("worker", "--role", role)

        first = request(
            f"{base_url}/v1/runs",
            method="POST",
            payload={"repository_path": str(repository)},
        )
        assert isinstance(first, dict)
        completed = wait_for_run(base_url, first["id"])
        assert completed["status"] == "complete", completed
        assert completed["integration_head"] == base_commit
        branch = completed["integration_branch"]

        executions = request(f"{base_url}/v1/runs/{first['id']}/executions")
        assert isinstance(executions, list) and executions
        execution_id = executions[0]["id"]
        events = request(f"{base_url}/v1/executions/{execution_id}/events")
        assert isinstance(events, list) and events
        with urllib.request.urlopen(
            f"{base_url}/v1/executions/{execution_id}/events/stream", timeout=5
        ) as stream:
            assert b"agent.final_output" in stream.read()

        second = request(
            f"{base_url}/v1/runs",
            method="POST",
            payload={"repository_path": str(repository)},
        )
        assert isinstance(second, dict)
        rerun = wait_for_run(base_url, second["id"])
        assert rerun["status"] == "complete", rerun
        assert rerun["tasks"] == []
        assert git(repository, "rev-parse", branch) == base_commit
        assert git(repository, "rev-parse", "main") == base_commit
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
