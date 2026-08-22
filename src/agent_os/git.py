import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent_os.models import StageResult


class GitError(RuntimeError):
    pass


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GitError(detail)
    return result


def _git_bytes(
    path: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise GitError(detail)
    return result


def _safe_task_id(task_id: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", task_id):
        raise GitError(f"unsafe task id: {task_id}")
    return task_id


@dataclass(frozen=True, slots=True)
class GitRepository:
    root: Path
    plan_path: str
    base_commit: str
    plan_id: str
    target_id: str
    state_root: Path
    namespace: str

    @classmethod
    def inspect(
        cls,
        repository_path: Path,
        plan_path: str,
        base_ref: str,
        state_root: Path,
    ) -> GitRepository:
        candidate = repository_path.expanduser().resolve()
        try:
            root = Path(_git(candidate, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
        except GitError as exc:
            raise GitError(f"not a Git repository: {candidate}") from exc
        if _git(root, "status", "--porcelain", "--untracked-files=normal").stdout:
            raise GitError("repository worktree must be clean")
        base = _git(root, "rev-parse", "--verify", f"{base_ref}^{{commit}}").stdout.strip()
        resolved_state_root = state_root.expanduser().resolve()
        if resolved_state_root.is_relative_to(root):
            raise GitError("state directory must be outside the source repository")
        plan = _git_bytes(root, "show", f"{base}:{plan_path}", check=False)
        if plan.returncode:
            raise GitError(f"committed plan {plan_path!r} was not found at {base_ref!r}")
        plan_id = hashlib.sha256(plan.stdout).hexdigest()
        target_material = b"\0".join(
            [os.fsencode(root), plan_id.encode(), base.encode()]
        )
        target_id = hashlib.sha256(target_material).hexdigest()
        namespace = f"agent/{plan_id[:12]}-{base[:12]}"
        return cls(
            root=root,
            plan_path=plan_path,
            base_commit=base,
            plan_id=plan_id,
            target_id=target_id,
            state_root=resolved_state_root,
            namespace=namespace,
        )

    @property
    def integration_branch(self) -> str:
        return f"{self.namespace}/integration"

    def task_branch(self, task_id: str) -> str:
        return f"{self.namespace}/tasks/{_safe_task_id(task_id)}"

    def staging_branch(self, task_id: str) -> str:
        return f"{self.namespace}/staging/{_safe_task_id(task_id)}"

    def _worktree_path(self, leaf: str) -> Path:
        return self.state_root / "worktrees" / self.target_id[:16] / leaf

    def _branch_exists(self, branch: str) -> bool:
        return _git(
            self.root,
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
            check=False,
        ).returncode == 0

    def _prepare_worktree(self, path: Path, branch: str, start: str) -> Path:
        if path.exists():
            current = _git(path, "branch", "--show-current").stdout.strip()
            if current != branch:
                raise GitError(f"worktree {path} is on {current!r}, expected {branch!r}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._branch_exists(branch):
            _git(self.root, "worktree", "add", str(path), branch)
        else:
            _git(self.root, "worktree", "add", "-b", branch, str(path), start)
        return path

    def prepare_integration_worktree(self) -> Path:
        return self._prepare_worktree(
            self._worktree_path("integration"), self.integration_branch, self.base_commit
        )

    def prepare_task_worktree(self, task_id: str, start_commit: str) -> Path:
        safe = _safe_task_id(task_id)
        return self._prepare_worktree(
            self._worktree_path(f"task-{safe}"), self.task_branch(safe), start_commit
        )

    def commit_changes(self, worktree: Path, message: str) -> str | None:
        _git(worktree, "add", "-A")
        if _git(worktree, "diff", "--cached", "--quiet", check=False).returncode == 0:
            return None
        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_NAME": "agent-os",
                "GIT_AUTHOR_EMAIL": "agent-os@localhost",
                "GIT_COMMITTER_NAME": "agent-os",
                "GIT_COMMITTER_EMAIL": "agent-os@localhost",
                "GIT_EDITOR": "true",
            }
        )
        result = subprocess.run(
            ["git", "-C", str(worktree), "commit", "-m", message],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode:
            raise GitError(result.stderr.strip() or result.stdout.strip())
        return _git(worktree, "rev-parse", "HEAD").stdout.strip()

    def assert_plan_unchanged(self, worktree: Path, base_commit: str) -> None:
        changed = _git(
            worktree,
            "diff",
            "--name-only",
            f"{base_commit}..HEAD",
            "--",
            self.plan_path,
        ).stdout.strip()
        if changed:
            raise GitError(f"tasks may not modify {self.plan_path}")

    def prepare_staging_worktree(
        self, task_id: str, integration_worktree: Path, task_worktree: Path
    ) -> StageResult:
        safe = _safe_task_id(task_id)
        integration_head = _git(integration_worktree, "rev-parse", "HEAD").stdout.strip()
        stage_path = self._prepare_worktree(
            self._worktree_path(f"stage-{safe}"),
            self.staging_branch(safe),
            integration_head,
        )
        task_branch = _git(task_worktree, "branch", "--show-current").stdout.strip()
        merge = _git(stage_path, "merge", "--no-ff", "--no-edit", task_branch, check=False)
        conflicts = (
            _git(stage_path, "diff", "--name-only", "--diff-filter=U").stdout.splitlines()
            if merge.returncode
            else []
        )
        head = None if conflicts else _git(stage_path, "rev-parse", "HEAD").stdout.strip()
        if merge.returncode and not conflicts:
            raise GitError(merge.stderr.strip() or merge.stdout.strip())
        return StageResult(path=str(stage_path), head=head, conflicts=conflicts)

    def integrate(self, staging_worktree: Path) -> str:
        integration = self.prepare_integration_worktree()
        staging_head = _git(staging_worktree, "rev-parse", "HEAD").stdout.strip()
        current_head = self.integration_head()
        if self.is_ancestor(staging_head, current_head):
            return current_head
        _git(integration, "merge", "--ff-only", staging_head)
        integrated_head = self.integration_head()
        if not self.is_ancestor(staging_head, integrated_head):
            raise GitError(f"staging commit {staging_head} is not reachable after integration")
        return integrated_head

    def integration_head(self) -> str:
        integration = self.prepare_integration_worktree()
        return _git(integration, "rev-parse", "HEAD").stdout.strip()

    def head(self, worktree: Path) -> str:
        return _git(worktree, "rev-parse", "HEAD").stdout.strip()

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return _git(
            self.root,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
            check=False,
        ).returncode == 0

    def cleanup_worktrees(self) -> None:
        target_root = self.state_root / "worktrees" / self.target_id[:16]
        if not target_root.exists():
            return
        for worktree in sorted(target_root.iterdir(), reverse=True):
            if worktree.is_dir():
                _git(self.root, "worktree", "remove", "--force", str(worktree))
        _git(self.root, "worktree", "prune")
