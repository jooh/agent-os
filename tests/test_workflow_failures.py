import asyncio
import inspect
from contextlib import nullcontext
from pathlib import Path

import pytest

from agent_os.models import (
    Candidate,
    DeveloperCommand,
    DeveloperSessionInput,
    ImplementationTask,
    PlanComparison,
    ReviewInput,
    ReviewIssue,
    ReviewResult,
    RunInput,
    RunStatus,
    StageResult,
)
from agent_os.store import StateStore
from agent_os.workflows import (
    _candidate,
    _developer_turn,
    developer_session,
    engineering_run,
    plan_comparison,
    technical_review,
)


def run_input(tmp_path: Path) -> RunInput:
    return RunInput(
        run_id="run-1",
        workflow_id="engineering-run:run-1",
        target_id="a" * 64,
        repository_path=str(tmp_path),
        plan_path="PLAN.md",
        plan_id="b" * 64,
        base_commit="c" * 40,
        integration_branch="agent/bbbbbbbbbbbb-cccccccccccc/integration",
        database_url=f"sqlite:///{tmp_path / 'state.db'}",
        state_dir=str(tmp_path / "state"),
        planner_model="test",
        developer_model="test",
        reviewer_model="test",
        max_rounds=10,
        max_review_cycles=3,
        max_developer_turns=6,
        model_request_limit=50,
        shell_timeout_seconds=30,
    )


def task() -> ImplementationTask:
    return ImplementationTask(
        id="task",
        description="Implement task",
        acceptance_criteria=["It works"],
    )


def session(tmp_path: Path) -> DeveloperSessionInput:
    value = run_input(tmp_path)
    return DeveloperSessionInput(
        run=value,
        task_id="task-1",
        task=task(),
        start_commit=value.base_commit,
        developer_workflow_id="developer-1",
    )


def test_fresh_agent_workflows_checkpoint_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path)
    store = StateStore(value.database_url)
    store.bootstrap()
    store.create_run(value, None, lambda _connection, _value: None)
    store.create_task(
        run_id=value.run_id,
        task_id="task-1",
        round_number=1,
        ordinal=1,
        description="Implement task",
        acceptance_criteria=["It works"],
    )
    monkeypatch.setattr("agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path))
    monkeypatch.setattr(
        "agent_os.workflows.new_execution_id_step", lambda role: f"{role}-failed"
    )

    async def fail(*_args, **_kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr("agent_os.workflows.run_planner_agent", fail)
    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(inspect.unwrap(plan_comparison)(value, 1))
    assert store.get_execution("planner-failed").status.value == "failed"

    monkeypatch.setattr("agent_os.workflows.run_developer_agent", fail)
    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(
            _developer_turn(
                session(tmp_path),
                turn=1,
                worktree=str(tmp_path),
                protected_base=value.base_commit,
                prompt="implement",
            )
        )
    assert store.get_execution("developer-failed").status.value == "failed"

    monkeypatch.setattr("agent_os.workflows.run_reviewer_agent", fail)
    review = ReviewInput(
        run=value,
        task_id="task-1",
        worktree=str(tmp_path),
        base_commit=value.base_commit,
        review_cycle=1,
        reviewer_workflow_id="reviewer-1",
    )
    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(inspect.unwrap(technical_review)(review))
    assert store.get_execution("reviewer-failed").status.value == "failed"


def test_developer_session_timeout_fix_and_turn_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = session(tmp_path)
    candidate = Candidate(turn=1, worktree=str(tmp_path), head="a" * 40)
    monkeypatch.setattr("agent_os.workflows.prepare_task_step", lambda *_args: str(tmp_path))
    monkeypatch.setattr("agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None)

    async def developer_turn(*_args, **_kwargs):
        return candidate

    async def set_event(*_args, **_kwargs):
        return None

    monkeypatch.setattr("agent_os.workflows._developer_turn", developer_turn)
    monkeypatch.setattr("agent_os.workflows.DBOS.set_event_async", set_event)

    async def timeout(*_args, **_kwargs):
        return None

    monkeypatch.setattr("agent_os.workflows.DBOS.recv_async", timeout)
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(inspect.unwrap(developer_session)(value))

    async def fix(*_args, **_kwargs):
        return {
            "action": "fix",
            "prompt": "fix it",
            "worktree": str(tmp_path),
            "protected_base": "a" * 40,
        }

    monkeypatch.setattr("agent_os.workflows.DBOS.recv_async", fix)
    with pytest.raises(RuntimeError, match="turn limit"):
        asyncio.run(inspect.unwrap(developer_session)(value))


def test_candidate_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(*_args, **_kwargs):
        return None

    monkeypatch.setattr("agent_os.workflows.DBOS.get_event_async", missing)
    with pytest.raises(TimeoutError, match="candidate"):
        asyncio.run(_candidate("developer-1", 1))


def test_engineering_run_handles_conflicts_and_round_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path).model_copy(update={"max_rounds": 1})
    comparison = PlanComparison(complete=False, tasks=[task()])
    updates: list[tuple[RunStatus, str | None]] = []
    sent: list[DeveloperCommand] = []

    class Handle:
        workflow_id = "developer-1"

        async def get_result(self):
            return comparison

    async def enqueue(*_args, **_kwargs):
        return Handle()

    candidates = iter(
        [
            Candidate(turn=1, worktree=str(tmp_path / "task"), head="a" * 40),
            Candidate(turn=2, worktree=str(tmp_path / "stage"), head="b" * 40),
            Candidate(turn=3, worktree=str(tmp_path / "stage"), head="c" * 40),
        ]
    )

    async def next_candidate(*_args, **_kwargs):
        return next(candidates)

    reviews = iter(
        [
            ReviewResult(
                approved=False,
                issues=[ReviewIssue(severity="P2", description="fix this")],
            ),
            ReviewResult(approved=True),
        ]
    )

    async def review(*_args, **_kwargs):
        return next(reviews)

    async def send(_workflow_id, command, _topic):
        sent.append(command)

    def update(_value, status, **kwargs):
        updates.append((status, kwargs.get("failure_reason")))

    monkeypatch.setattr("agent_os.workflows.SetWorkflowID", lambda _value: nullcontext())
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.DBOS.send_async", send)
    monkeypatch.setattr("agent_os.workflows._enqueue_developer", enqueue)
    monkeypatch.setattr("agent_os.workflows._candidate", next_candidate)
    monkeypatch.setattr("agent_os.workflows._review", review)
    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr("agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent_os.workflows.create_task_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path))
    monkeypatch.setattr("agent_os.workflows.integration_head_step", lambda _value: "a" * 40)
    monkeypatch.setattr(
        "agent_os.workflows.prepare_staging_step",
        lambda *_args: StageResult(
            path=str(tmp_path / "stage"), head=None, conflicts=["value.txt"]
        ),
    )
    monkeypatch.setattr("agent_os.workflows.integrate_step", lambda *_args: "b" * 40)

    with pytest.raises(RuntimeError, match="round limit"):
        asyncio.run(inspect.unwrap(engineering_run)(value))
    assert any(command.action == "fix" for command in sent)
    assert updates[-1] == (RunStatus.FAILED, "reconciliation round limit exceeded")


@pytest.mark.parametrize(
    ("review_cycles", "developer_turns", "reason"),
    [(1, 6, "review cycle limit"), (2, 1, "developer turn limit")],
)
def test_engineering_run_enforces_review_and_turn_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    review_cycles: int,
    developer_turns: int,
    reason: str,
) -> None:
    value = run_input(tmp_path).model_copy(
        update={
            "max_rounds": 1,
            "max_review_cycles": review_cycles,
            "max_developer_turns": developer_turns,
        }
    )
    comparison = PlanComparison(complete=False, tasks=[task()])

    class Handle:
        workflow_id = "developer-1"

        async def get_result(self):
            return comparison

    async def enqueue(*_args, **_kwargs):
        return Handle()

    async def candidate(*_args, **_kwargs):
        return Candidate(turn=1, worktree=str(tmp_path), head="a" * 40)

    async def rejected(*_args, **_kwargs):
        return ReviewResult(
            approved=False,
            issues=[ReviewIssue(severity="P1", description="broken")],
        )

    async def send(*_args, **_kwargs):
        return None

    monkeypatch.setattr("agent_os.workflows.SetWorkflowID", lambda _value: nullcontext())
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.DBOS.send_async", send)
    monkeypatch.setattr("agent_os.workflows._enqueue_developer", enqueue)
    monkeypatch.setattr("agent_os.workflows._candidate", candidate)
    monkeypatch.setattr("agent_os.workflows._review", rejected)
    monkeypatch.setattr("agent_os.workflows.update_run_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent_os.workflows.create_task_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path))
    monkeypatch.setattr("agent_os.workflows.integration_head_step", lambda _value: "a" * 40)
    monkeypatch.setattr(
        "agent_os.workflows.prepare_staging_step",
        lambda *_args: StageResult(path=str(tmp_path), head="a" * 40, conflicts=[]),
    )

    with pytest.raises(RuntimeError, match=reason):
        asyncio.run(inspect.unwrap(engineering_run)(value))
