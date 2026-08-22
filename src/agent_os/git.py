from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Concatenate

from agent_os.models import StageResult


class GitError(RuntimeError):
    pass


_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}
_LOCK_DEPTHS = threading.local()


def _reset_process_locks_after_fork() -> None:
    global _LOCKS_GUARD, _PROCESS_LOCKS, _LOCK_DEPTHS
    _LOCKS_GUARD = threading.Lock()
    _PROCESS_LOCKS = {}
    _LOCK_DEPTHS = threading.local()


os.register_at_fork(after_in_child=_reset_process_locks_after_fork)


def _target_lock_path(state_root: Path, target_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", target_id):
        raise GitError(f"unsafe target id: {target_id}")
    return state_root.expanduser().resolve() / "locks" / f"{target_id}.lock"


def _process_lock(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(path, threading.RLock())


def _lock_depths() -> dict[Path, int]:
    depths = getattr(_LOCK_DEPTHS, "values", None)
    if depths is None:
        depths = {}
        _LOCK_DEPTHS.values = depths
    return depths


def _open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    return os.open(path, flags, 0o600)


@contextmanager
def _target_operation_lock(state_root: Path, target_id: str) -> Iterator[None]:
    path = _target_lock_path(state_root, target_id)
    process_lock = _process_lock(path)
    with process_lock:
        depths = _lock_depths()
        depth = depths.get(path, 0)
        if depth:
            depths[path] = depth + 1
            try:
                yield
            finally:
                depths[path] -= 1
            return

        descriptor = _open_lock(path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            depths[path] = 1
            try:
                yield
            finally:
                depths.pop(path, None)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _target_operations_quiescent(state_root: Path, target_id: str) -> bool:
    with _try_target_operation_lock(state_root, target_id) as acquired:
        return acquired


@contextmanager
def _try_target_operation_lock(
    state_root: Path, target_id: str
) -> Iterator[bool]:
    path = _target_lock_path(state_root, target_id)
    process_lock = _process_lock(path)
    if not process_lock.acquire(blocking=False):
        yield False
        return
    descriptor: int | None = None
    file_lock_acquired = False
    try:
        if _lock_depths().get(path, 0):
            yield False
            return
        descriptor = _open_lock(path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        file_lock_acquired = True
        _lock_depths()[path] = 1
        try:
            yield True
        finally:
            _lock_depths().pop(path, None)
    finally:
        if descriptor is not None and file_lock_acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        if descriptor is not None:
            os.close(descriptor)
        process_lock.release()


def _locked_operation[**P, R](
    method: Callable[Concatenate[GitRepository, P], R],
) -> Callable[Concatenate[GitRepository, P], R]:
    @wraps(method)
    def locked(self: GitRepository, *args: P.args, **kwargs: P.kwargs) -> R:
        with self.operation_lock():
            return method(self, *args, **kwargs)

    return locked


def _git(
    path: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
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

    @staticmethod
    def target_operation_lock(
        state_root: Path, target_id: str
    ) -> AbstractContextManager[None]:
        return _target_operation_lock(state_root, target_id)

    @staticmethod
    def target_operations_quiescent(state_root: Path, target_id: str) -> bool:
        return _target_operations_quiescent(state_root, target_id)

    @staticmethod
    def try_target_operation_lock(
        state_root: Path, target_id: str
    ) -> AbstractContextManager[bool]:
        return _try_target_operation_lock(state_root, target_id)

    def operation_lock(self) -> AbstractContextManager[None]:
        return self.target_operation_lock(self.state_root, self.target_id)

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
            root = Path(
                _git(candidate, "rev-parse", "--show-toplevel").stdout.strip()
            ).resolve()
        except GitError as exc:
            raise GitError(f"not a Git repository: {candidate}") from exc
        if _git(root, "status", "--porcelain", "--untracked-files=normal").stdout:
            raise GitError("repository worktree must be clean")
        base = _git(
            root, "rev-parse", "--verify", f"{base_ref}^{{commit}}"
        ).stdout.strip()
        resolved_state_root = state_root.expanduser().resolve()
        if resolved_state_root.is_relative_to(root):
            raise GitError("state directory must be outside the source repository")
        plan = _git_bytes(root, "show", f"{base}:{plan_path}", check=False)
        if plan.returncode:
            raise GitError(
                f"committed plan {plan_path!r} was not found at {base_ref!r}"
            )
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
        return (
            _git(
                self.root,
                "show-ref",
                "--verify",
                "--quiet",
                f"refs/heads/{branch}",
                check=False,
            ).returncode
            == 0
        )

    @_locked_operation
    def _prepare_worktree(self, path: Path, branch: str, start: str) -> Path:
        if path.exists():
            current = _git(path, "branch", "--show-current").stdout.strip()
            if current != branch:
                raise GitError(
                    f"worktree {path} is on {current!r}, expected {branch!r}"
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._branch_exists(branch):
            _git(self.root, "worktree", "add", str(path), branch)
        else:
            _git(self.root, "worktree", "add", "-b", branch, str(path), start)
        return path

    def prepare_integration_worktree(self) -> Path:
        return self._prepare_worktree(
            self._worktree_path("integration"),
            self.integration_branch,
            self.base_commit,
        )

    def prepare_task_worktree(self, task_id: str, start_commit: str) -> Path:
        safe = _safe_task_id(task_id)
        return self._prepare_worktree(
            self._worktree_path(f"task-{safe}"), self.task_branch(safe), start_commit
        )

    @_locked_operation
    def commit_changes(self, worktree: Path, message: str) -> str | None:
        _git(worktree, "add", "-A")
        merge_in_progress = (
            _git(
                worktree, "rev-parse", "--quiet", "--verify", "MERGE_HEAD", check=False
            ).returncode
            == 0
        )
        if (
            not merge_in_progress
            and _git(worktree, "diff", "--cached", "--quiet", check=False).returncode
            == 0
        ):
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

    @_locked_operation
    def prepare_staging_worktree(
        self, task_id: str, integration_worktree: Path, task_worktree: Path
    ) -> StageResult:
        safe = _safe_task_id(task_id)
        integration_head = _git(
            integration_worktree, "rev-parse", "HEAD"
        ).stdout.strip()
        stage_path = self._prepare_worktree(
            self._worktree_path(f"stage-{safe}"),
            self.staging_branch(safe),
            integration_head,
        )
        task_branch = _git(task_worktree, "branch", "--show-current").stdout.strip()
        merge = _git(
            stage_path, "merge", "--no-ff", "--no-edit", task_branch, check=False
        )
        conflicts = (
            _git(
                stage_path, "diff", "--name-only", "--diff-filter=U"
            ).stdout.splitlines()
            if merge.returncode
            else []
        )
        head = (
            None if conflicts else _git(stage_path, "rev-parse", "HEAD").stdout.strip()
        )
        if merge.returncode and not conflicts:
            raise GitError(merge.stderr.strip() or merge.stdout.strip())
        return StageResult(path=str(stage_path), head=head, conflicts=conflicts)

    @_locked_operation
    def integrate(
        self, staging_worktree: Path, required_head: str | None = None
    ) -> str:
        integration = self.prepare_integration_worktree()
        staging_head = _git(staging_worktree, "rev-parse", "HEAD").stdout.strip()
        if required_head is not None and not self.is_ancestor(
            required_head, staging_head
        ):
            raise GitError(
                f"task commit {required_head} is not reachable from staging commit {staging_head}"
            )
        current_head = self.integration_head()
        if self.is_ancestor(staging_head, current_head):
            if required_head is not None and not self.is_ancestor(
                required_head, current_head
            ):
                raise GitError(
                    f"task commit {required_head} is not reachable after integration"
                )
            return current_head
        _git(integration, "merge", "--ff-only", staging_head)
        integrated_head = self.integration_head()
        if not self.is_ancestor(staging_head, integrated_head):
            raise GitError(
                f"staging commit {staging_head} is not reachable after integration"
            )
        if required_head is not None and not self.is_ancestor(
            required_head, integrated_head
        ):
            raise GitError(
                f"task commit {required_head} is not reachable after integration"
            )
        return integrated_head

    @_locked_operation
    def integration_head(self) -> str:
        integration = self.prepare_integration_worktree()
        return _git(integration, "rev-parse", "HEAD").stdout.strip()

    def head(self, worktree: Path) -> str:
        return _git(worktree, "rev-parse", "HEAD").stdout.strip()

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return (
            _git(
                self.root,
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
                check=False,
            ).returncode
            == 0
        )

    @_locked_operation
    def cleanup_worktrees(self) -> None:
        target_root = self.state_root / "worktrees" / self.target_id[:16]
        if not target_root.exists():
            return
        registered = {
            Path(line.removeprefix("worktree ")).resolve()
            for line in _git(
                self.root, "worktree", "list", "--porcelain"
            ).stdout.splitlines()
            if line.startswith("worktree ")
        }
        for worktree in sorted(target_root.iterdir(), reverse=True):
            managed = worktree.name == "integration" or worktree.name.startswith(
                ("task-", "stage-")
            )
            if not managed or not worktree.is_dir() or worktree.is_symlink():
                continue
            resolved = worktree.resolve()
            if resolved not in registered:
                shutil.rmtree(worktree)
                continue
            removal = _git(
                self.root,
                "worktree",
                "remove",
                "--force",
                str(worktree),
                check=False,
            )
            if removal.returncode and worktree.exists():
                raise GitError(removal.stderr.strip() or removal.stdout.strip())
        _git(self.root, "worktree", "prune")
