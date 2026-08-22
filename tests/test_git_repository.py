import hashlib
import subprocess
from pathlib import Path

import pytest

from agent_os.git import GitError, GitRepository, _git, _git_bytes


def git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture()
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test User")
    git(root, "config", "user.email", "test@example.com")
    (root / "PLAN.md").write_text("Build the feature.\n")
    (root / "app.py").write_text("VALUE = 1\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root


def test_inspect_computes_plan_base_and_coexisting_branch_names(
    repository: Path, tmp_path: Path
) -> None:
    repo = GitRepository.inspect(repository, "PLAN.md", "HEAD", tmp_path / "state")

    assert len(repo.plan_id) == 64
    assert len(repo.target_id) == 64
    assert repo.integration_branch.endswith("/integration")
    assert repo.task_branch("r01-t01-deadbeef").endswith("/tasks/r01-t01-deadbeef")
    assert repo.staging_branch("r01-t01-deadbeef").endswith("/staging/r01-t01-deadbeef")
    assert repo.integration_branch != repo.task_branch("r01-t01-deadbeef")


def test_inspect_rejects_dirty_or_uncommitted_plan(repository: Path, tmp_path: Path) -> None:
    (repository / "dirty.txt").write_text("dirty")
    with pytest.raises(GitError, match="clean"):
        GitRepository.inspect(repository, "PLAN.md", "HEAD", tmp_path / "state")

    (repository / "dirty.txt").unlink()
    git(repository, "rm", "PLAN.md")
    git(repository, "commit", "-m", "remove plan")
    with pytest.raises(GitError, match="PLAN.md"):
        GitRepository.inspect(repository, "PLAN.md", "HEAD", tmp_path / "state")


def test_inspect_rejects_non_repository_and_unsafe_task_ids(tmp_path: Path) -> None:
    with pytest.raises(GitError, match="not a Git repository"):
        GitRepository.inspect(tmp_path, "PLAN.md", "HEAD", tmp_path / "state")

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    (root / "PLAN.md").write_text("plan\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init")
    repo = GitRepository.inspect(root, "PLAN.md", "HEAD", tmp_path / "state")
    with pytest.raises(GitError, match="unsafe task id"):
        repo.task_branch("../escape")


def test_inspect_hashes_exact_plan_bytes_and_rejects_state_inside_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    plan = b"binary plan: \xff\x00\r\n"
    (root / "PLAN.md").write_bytes(plan)
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init")

    repo = GitRepository.inspect(root, "PLAN.md", "HEAD", tmp_path / "state")
    assert repo.plan_id == hashlib.sha256(plan).hexdigest()
    with pytest.raises(GitError, match="state directory"):
        GitRepository.inspect(root, "PLAN.md", "HEAD", root / ".agent-os-state")


def test_git_helper_and_existing_worktree_errors(repository: Path, tmp_path: Path) -> None:
    with pytest.raises(GitError):
        _git(repository, "rev-parse", "not-a-ref")
    with pytest.raises(GitError):
        _git_bytes(repository, "show", "not-a-ref")

    repo = GitRepository.inspect(repository, "PLAN.md", "HEAD", tmp_path / "state")
    repo.cleanup_worktrees()
    integration = repo.prepare_integration_worktree()
    git(integration, "branch", "-m", "unexpected")
    with pytest.raises(GitError, match="expected"):
        repo.prepare_integration_worktree()


def test_integration_task_stage_and_plan_protection(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = GitRepository.inspect(repository, "PLAN.md", "HEAD", tmp_path / "state")
    integration = repo.prepare_integration_worktree()
    task = repo.prepare_task_worktree("r01-t01-deadbeef", repo.base_commit)
    (task / "app.py").write_text("VALUE = 2\n")
    head = repo.commit_changes(task, "implement task")
    assert head is not None
    assert repo.commit_changes(task, "no-op") is None

    stage = repo.prepare_staging_worktree("r01-t01-deadbeef", integration, task)
    assert stage.conflicts == []
    stage_path = Path(stage.path)
    assert git(stage_path, "show", "HEAD:app.py") == "VALUE = 2"
    integrated = repo.integrate(stage_path)
    assert repo.integrate(stage_path) == integrated
    assert repo.is_ancestor(head, repo.integration_head())

    with monkeypatch.context() as context:
        context.setattr(GitRepository, "is_ancestor", lambda *_args: False)
        with pytest.raises(GitError, match="not reachable"):
            repo.integrate(stage_path)

    protected = repo.prepare_task_worktree("r02-t01-feedface", repo.integration_head())
    (protected / "PLAN.md").write_text("Changed plan\n")
    repo.commit_changes(protected, "change plan")
    with pytest.raises(GitError, match="PLAN.md"):
        repo.assert_plan_unchanged(protected, repo.integration_head())
    target_root = repo.state_root / "worktrees" / repo.target_id[:16]
    (target_root / "diagnostic.txt").write_text("keep")
    repo.cleanup_worktrees()
    assert not integration.exists()


def test_commit_and_merge_failures_are_normalized(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = GitRepository.inspect(repository, "PLAN.md", "HEAD", tmp_path / "state")
    integration = repo.prepare_integration_worktree()
    task = repo.prepare_task_worktree("task", repo.base_commit)
    (task / "app.py").write_text("VALUE = 2\n")

    original_run = subprocess.run

    def fail_commit(args, **kwargs):
        if "commit" in args:
            return subprocess.CompletedProcess(args, 1, "", "commit failed")
        return original_run(args, **kwargs)

    monkeypatch.setattr("agent_os.git.subprocess.run", fail_commit)
    with pytest.raises(GitError, match="commit failed"):
        repo.commit_changes(task, "fail")
    monkeypatch.setattr("agent_os.git.subprocess.run", original_run)
    git(task, "reset")
    repo.commit_changes(task, "succeed")

    original_git = __import__("agent_os.git", fromlist=["_git"])._git

    def fail_merge(path: Path, *args: str, check: bool = True):
        if args and args[0] == "merge":
            return subprocess.CompletedProcess([], 1, "", "merge failed")
        return original_git(path, *args, check=check)

    monkeypatch.setattr("agent_os.git._git", fail_merge)
    with pytest.raises(GitError, match="merge failed"):
        repo.prepare_staging_worktree("task", integration, task)
