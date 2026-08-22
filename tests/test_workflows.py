import subprocess
from pathlib import Path

from agent_os.models import ImplementationTask, RunInput
from agent_os.workflows import (
    commit_candidate_step,
    integrate_step,
    prepare_integration_step,
    prepare_staging_step,
    prepare_task_step,
    task_identifier,
)


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_input(tmp_path: Path) -> RunInput:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "PLAN.md").write_text("Create a value\n")
    (root / "value.txt").write_text("old\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    base = git(root, "rev-parse", "HEAD")
    return RunInput(
        run_id="run-1",
        workflow_id="engineering-run:run-1",
        target_id="1" * 64,
        repository_path=str(root),
        plan_path="PLAN.md",
        plan_id="2" * 64,
        base_commit=base,
        integration_branch="agent/222222222222-333333333333/integration",
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


def test_task_identifier_is_deterministic_and_safe() -> None:
    task = ImplementationTask(
        id="model-task",
        description="Implement retry handling",
        acceptance_criteria=["Tests pass"],
    )
    first = task_identifier(1, 2, task)
    assert first == task_identifier(1, 2, task)
    assert first.startswith("r01-t02-")


def test_workflow_git_steps_stage_and_integrate(tmp_path: Path) -> None:
    value = make_input(tmp_path)
    integration = Path(prepare_integration_step(value))
    task = Path(prepare_task_step(value, "r01-t01-deadbeef", value.base_commit))
    (task / "value.txt").write_text("new\n")
    candidate = commit_candidate_step(value, str(task), "implement", value.base_commit)
    assert candidate.head is not None

    stage = prepare_staging_step(value, "r01-t01-deadbeef", str(integration), str(task))
    assert stage.conflicts == []
    head = integrate_step(value, stage.path)
    assert git(integration, "rev-parse", "HEAD") == head
    assert git(integration, "show", "HEAD:value.txt") == "new"
