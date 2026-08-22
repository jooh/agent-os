import subprocess
from pathlib import Path
from threading import Event, Thread

from agent_os.git import GitRepository
from agent_os.models import (
    CancellationFinalizerInput,
    ImplementationTask,
    RunInput,
    RunStatus,
)
from agent_os.store import StateStore, StateTransitionError
from agent_os.workflows import (
    commit_candidate_step,
    finalize_cancellation_step,
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

    first_run = task_identifier(1, 2, task, run_id="shared--first-run")
    second_run = task_identifier(1, 2, task, run_id="shared--second-run")
    assert first_run == task_identifier(1, 2, task, run_id="shared--first-run")
    assert first_run != second_run


def test_workflow_git_steps_stage_and_integrate(tmp_path: Path) -> None:
    value = make_input(tmp_path)
    store = StateStore(value.database_url)
    store.bootstrap()
    store.create_run(value, None, lambda _connection, _value: None)
    store.update_run(value.run_id, status=RunStatus.RUNNING)
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
    store.engine.dispose()


def test_stale_integration_cannot_mutate_after_successor_takes_lease(
    tmp_path: Path,
) -> None:
    value = make_input(tmp_path)
    store = StateStore(value.database_url)
    store.bootstrap()
    store.create_run(value, None, lambda _connection, _value: None)
    store.update_run(value.run_id, status=RunStatus.RUNNING)
    repository = GitRepository(
        root=Path(value.repository_path),
        plan_path=value.plan_path,
        base_commit=value.base_commit,
        plan_id=value.plan_id,
        target_id=value.target_id,
        state_root=Path(value.state_dir),
        namespace=value.integration_branch.removesuffix("/integration"),
    )
    integration = Path(prepare_integration_step(value))
    original_head = git(integration, "rev-parse", "HEAD")
    task = Path(prepare_task_step(value, "old-run-task", original_head))
    (task / "value.txt").write_text("stale\n")
    candidate = commit_candidate_step(value, str(task), "stale", value.base_commit)
    stage = prepare_staging_step(value, "old-run-task", str(integration), str(task))
    assert stage.head is not None
    assert candidate.head is not None

    started = Event()
    finished = Event()
    failures: list[BaseException] = []

    def integrate_stale_candidate() -> None:
        started.set()
        try:
            integrate_step(value, stage.path, candidate.head)
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)
        finally:
            finished.set()

    with repository.operation_lock():
        thread = Thread(target=integrate_stale_candidate)
        thread.start()
        assert started.wait(timeout=5)
        assert not finished.wait(timeout=0.1)

        store.request_cancellation(value.run_id)
        finalize_cancellation_step(
            CancellationFinalizerInput(
                run_id=value.run_id,
                root_workflow_id=value.workflow_id,
                target_id=value.target_id,
                database_url=value.database_url,
                state_dir=value.state_dir,
            )
        )
        successor = value.model_copy(
            update={
                "run_id": "run-2",
                "workflow_id": "engineering-run:run-2",
            }
        )
        store.create_run(successor, None, lambda _connection, _value: None)
        store.update_run(successor.run_id, status=RunStatus.RUNNING)

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], StateTransitionError)
    assert "target lease" in str(failures[0])
    assert git(integration, "rev-parse", "HEAD") == original_head
    assert git(integration, "show", "HEAD:value.txt") == "old"
    store.engine.dispose()
