import asyncio
import subprocess
from pathlib import Path

from dbos import DBOS, DBOSConfig

from agent_os.config import Settings
from agent_os.models import (
    DeveloperTurnResult,
    ImplementationTask,
    PlanComparison,
    ReviewResult,
    RunInput,
    RunStatus,
    TaskStatus,
)
from agent_os.runtime import register_worker_queues
from agent_os.store import StateStore
from agent_os.workflows import engineering_run


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "PLAN.md").write_text("Set value.txt to new\n")
    (root / "value.txt").write_text("old\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root, git(root, "rev-parse", "HEAD")


def make_input(tmp_path: Path, root: Path, base: str, run_id: str) -> RunInput:
    return RunInput(
        run_id=run_id,
        workflow_id=f"engineering-run:{run_id}",
        target_id="a" * 64,
        repository_path=str(root),
        plan_path="PLAN.md",
        plan_id="b" * 64,
        base_commit=base,
        integration_branch="agent/bbbbbbbbbbbb-cccccccccccc/integration",
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        state_dir=str(tmp_path / "state"),
        planner_model="test",
        developer_model="test",
        reviewer_model="test",
        max_rounds=3,
        max_review_cycles=3,
        max_developer_turns=6,
        model_request_limit=50,
        shell_timeout_seconds=30,
    )


def test_engineering_run_converges_and_rerun_is_noop(tmp_path: Path, monkeypatch) -> None:
    root, base = make_repository(tmp_path)
    value = make_input(tmp_path, root, base, "run-1")
    config: DBOSConfig = {
        "name": "agent-os-test",
        "application_version": "0.1.0",
        "system_database_url": value.database_url,
        "run_admin_server": False,
    }
    DBOS.destroy()
    DBOS(config=config)
    DBOS.reset_system_database()
    DBOS.launch()
    settings = Settings.from_values_for_test(
        value.database_url, Path(value.state_dir)
    )
    register_worker_queues(settings)
    store = StateStore(value.database_url)
    store.bootstrap()
    store.create_run(value, None, lambda _connection, _value: None)
    planner_calls = 0

    async def fake_planner(_prompt, _deps, _model, _limit):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return PlanComparison(
                complete=False,
                tasks=[
                    ImplementationTask(
                        id="set-value",
                        description="Set value.txt to new",
                        acceptance_criteria=["value.txt contains new"],
                    )
                ],
            )
        return PlanComparison(complete=True)

    async def fake_developer(_prompt, deps, _model, _limit, history):
        Path(deps.worktree, "value.txt").write_text("new\n")
        return (
            DeveloperTurnResult(
                summary="updated value", validation=["checked file"], ready_for_review=True
            ),
            history,
        )

    async def fake_reviewer(_prompt, _deps, _model, _limit):
        return ReviewResult(approved=True)

    monkeypatch.setattr("agent_os.workflows.run_planner_agent", fake_planner)
    monkeypatch.setattr("agent_os.workflows.run_developer_agent", fake_developer)
    monkeypatch.setattr("agent_os.workflows.run_reviewer_agent", fake_reviewer)

    try:
        async def scenario() -> None:
            head = await engineering_run(value)
            completed = store.get_run("run-1")
            assert completed.status is RunStatus.COMPLETE
            assert completed.integration_head == head
            assert completed.tasks[0].status is TaskStatus.INTEGRATED
            assert {
                event.event_type
                for execution in store.list_executions("run-1")
                for event in store.list_events(execution.id)
            } >= {"lifecycle.started", "agent.final_output"}
            assert git(root, "show", f"{value.integration_branch}:value.txt") == "new"
            assert git(root, "rev-parse", "main") == base

            second = value.model_copy(
                update={"run_id": "run-2", "workflow_id": "engineering-run:run-2"}
            )
            store.create_run(second, None, lambda _connection, _value: None)

            async def complete_planner(_prompt, _deps, _model, _limit):
                return PlanComparison(complete=True)

            monkeypatch.setattr("agent_os.workflows.run_planner_agent", complete_planner)
            second_head = await engineering_run(second)
            assert second_head == head
            assert store.get_run("run-2").tasks == []

        asyncio.run(scenario())
    finally:
        store.engine.dispose()
        DBOS.destroy()
