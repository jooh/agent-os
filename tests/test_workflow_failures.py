import asyncio
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_os.git import GitRepository
from agent_os.models import (
    CancellationFinalizerInput,
    Candidate,
    DeveloperCommand,
    DeveloperSessionInput,
    DeveloperTurnResult,
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
    _finalize_success,
    _record_run_failure,
    cancel_workflow_tree_step,
    cancellation_finalizer,
    developer_session,
    engineering_run,
    finalize_cancellation_step,
    plan_comparison,
    run_status_step,
    target_operations_quiescent_step,
    technical_review,
    workflow_tree_quiescent_step,
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


def cancellation_input(tmp_path: Path) -> CancellationFinalizerInput:
    return CancellationFinalizerInput(
        run_id="run-1",
        root_workflow_id="engineering-run:run-1",
        target_id="a" * 64,
        database_url=f"sqlite:///{tmp_path / 'state.db'}",
        state_dir=str(tmp_path / "state"),
    )


def test_cancellation_finalizer_steps_and_durable_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = cancellation_input(tmp_path)
    calls: list[str] = []
    logical = iter([False, True])
    physical = iter([False, True])

    monkeypatch.setattr(
        "agent_os.workflows.cancel_workflow_tree_step",
        lambda workflow_id: calls.append(f"cancel:{workflow_id}"),
    )
    monkeypatch.setattr(
        "agent_os.workflows.workflow_tree_quiescent_step",
        lambda _workflow_id: next(logical),
    )
    monkeypatch.setattr(
        "agent_os.workflows.target_operations_quiescent_step",
        lambda _value: next(physical),
    )
    monkeypatch.setattr(
        "agent_os.workflows.finalize_cancellation_step",
        lambda _value: calls.append("finalize"),
    )

    async def sleep(_seconds):
        calls.append("sleep")

    monkeypatch.setattr("agent_os.workflows.DBOS.sleep_async", sleep)
    asyncio.run(inspect.unwrap(cancellation_finalizer)(value))

    assert calls == [f"cancel:{value.root_workflow_id}", "sleep", "finalize"]


def test_cancellation_finalizer_ignores_its_own_pending_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = cancellation_input(tmp_path)
    finalized: list[str] = []
    monkeypatch.setattr(
        "agent_os.workflows.cancel_workflow_tree_step", lambda _workflow_id: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.get_workflow_status",
        lambda _workflow_id: SimpleNamespace(status="CANCELLED"),
    )
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.list_workflows",
        lambda **_kwargs: [
            SimpleNamespace(workflow_id=value.finalizer_workflow_id, status="PENDING")
        ],
    )
    monkeypatch.setattr(
        "agent_os.workflows.target_operations_quiescent_step", lambda _value: True
    )
    monkeypatch.setattr(
        "agent_os.workflows.finalize_cancellation_step",
        lambda _value: finalized.append(value.run_id),
    )

    asyncio.run(inspect.unwrap(cancellation_finalizer)(value))
    assert finalized == [value.run_id]


def test_cancellation_finalizer_boundary_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = cancellation_input(tmp_path)
    cancelled: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.cancel_workflow",
        lambda workflow_id, *, cancel_children: cancelled.append(
            (workflow_id, cancel_children)
        ),
    )
    cancel_workflow_tree_step(value.root_workflow_id)
    assert cancelled == [(value.root_workflow_id, True)]

    monkeypatch.setattr(
        "agent_os.workflows.DBOS.get_workflow_status", lambda _workflow_id: None
    )
    assert not workflow_tree_quiescent_step(value.root_workflow_id)
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.get_workflow_status",
        lambda _workflow_id: SimpleNamespace(status="PENDING"),
    )
    assert not workflow_tree_quiescent_step(value.root_workflow_id)
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.get_workflow_status",
        lambda _workflow_id: SimpleNamespace(status="CANCELLED"),
    )
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.list_workflows",
        lambda **_kwargs: [SimpleNamespace(workflow_id="child", status="PENDING")],
    )
    assert not workflow_tree_quiescent_step(value.root_workflow_id)
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.list_workflows",
        lambda **_kwargs: [SimpleNamespace(workflow_id="child", status="CANCELLED")],
    )
    assert workflow_tree_quiescent_step(value.root_workflow_id)
    monkeypatch.setattr(
        "agent_os.workflows.DBOS.list_workflows",
        lambda **_kwargs: [
            SimpleNamespace(workflow_id=value.finalizer_workflow_id, status="PENDING")
        ],
    )
    assert workflow_tree_quiescent_step(value.root_workflow_id)

    monkeypatch.setattr(
        GitRepository,
        "target_operations_quiescent",
        lambda state_dir, target_id: (
            state_dir == Path(value.state_dir) and target_id == value.target_id
        ),
    )
    assert target_operations_quiescent_step(value)

    store = StateStore(value.database_url)
    store.bootstrap()
    store.create_run(run_input(tmp_path), None, lambda _connection, _value: None)
    store.request_cancellation(value.run_id)
    assert run_status_step(run_input(tmp_path)) is RunStatus.CANCELLING
    finalize_cancellation_step(value)
    assert store.get_run(value.run_id).status is RunStatus.CANCELLED
    store.engine.dispose()


def test_failure_recording_stops_at_authoritative_terminal_and_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path)
    attempts = 0

    def fail_update(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("agent_os.workflows.update_run_step", fail_update)
    monkeypatch.setattr(
        "agent_os.workflows.run_status_step", lambda _value: RunStatus.CANCELLED
    )
    _record_run_failure(value, "original failure")
    assert attempts == 1

    attempts = 0
    monkeypatch.setattr(
        "agent_os.workflows.run_status_step", lambda _value: RunStatus.RUNNING
    )
    with pytest.raises(RuntimeError, match="could not durably record"):
        _record_run_failure(value, "original failure")
    assert attempts == 3


def test_finalization_recovers_a_committed_transition_checkpoint_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path)
    status = RunStatus.FINALIZING
    cleanup_calls = 0

    def update(_value, requested, **_kwargs):
        nonlocal status
        if requested is RunStatus.FINALIZING:
            raise RuntimeError("checkpoint failed after commit")
        status = requested

    def cleanup(_value):
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr("agent_os.workflows.run_status_step", lambda _value: status)
    monkeypatch.setattr("agent_os.workflows.cleanup_step", cleanup)

    assert (
        _finalize_success(value, round_number=1, integration_head="a" * 40) == "a" * 40
    )
    assert cleanup_calls == 1
    assert status is RunStatus.COMPLETE


def test_finalization_transition_exhaustion_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path)
    status = RunStatus.RUNNING
    failure_reason: str | None = None

    def update(_value, requested, **kwargs):
        nonlocal status, failure_reason
        if requested is RunStatus.FINALIZING:
            raise RuntimeError("transition unavailable")
        status = requested
        failure_reason = kwargs.get("failure_reason")

    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr("agent_os.workflows.run_status_step", lambda _value: status)

    with pytest.raises(RuntimeError, match="could not enter finalization"):
        _finalize_success(value, round_number=1, integration_head="a" * 40)
    assert status is RunStatus.FAILED
    assert failure_reason is not None and "transition unavailable" in failure_reason


@pytest.mark.parametrize("observed_status", [RunStatus.COMPLETE, RunStatus.FAILED])
def test_finalization_respects_terminal_status_after_completion_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_status: RunStatus,
) -> None:
    value = run_input(tmp_path)
    status = RunStatus.RUNNING

    def update(_value, requested, **_kwargs):
        nonlocal status
        if requested is RunStatus.FINALIZING:
            status = requested
            return
        status = observed_status
        raise RuntimeError("checkpoint outcome ambiguous")

    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr("agent_os.workflows.run_status_step", lambda _value: status)
    monkeypatch.setattr("agent_os.workflows.cleanup_step", lambda _value: None)

    assert (
        _finalize_success(value, round_number=1, integration_head="a" * 40) == "a" * 40
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
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.new_execution_id_step", lambda role: f"{role}-failed"
    )

    async def fail(*_args, **_kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr("agent_os.workflows.run_planner_agent", fail)
    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(inspect.unwrap(plan_comparison)(value, 1))
    assert store.get_execution("planner-failed").status.value == "failed"
    assert [event.event_key for event in store.list_events("planner-failed")] == [
        "lifecycle.started",
        "final-output",
    ]

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
    store.engine.dispose()


def test_plan_comparison_enforces_configured_task_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path).model_copy(update={"planner_task_limit": 1})
    store = StateStore(value.database_url)
    store.bootstrap()
    store.create_run(value, None, lambda _connection, _value: None)
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.new_execution_id_step", lambda _role: "planner-over-limit"
    )
    prompts: list[str] = []

    async def over_limit(prompt, *_args):
        prompts.append(prompt)
        return PlanComparison(
            complete=False,
            tasks=[
                task(),
                ImplementationTask(
                    id="task-2",
                    description="Implement another task",
                    acceptance_criteria=["It also works"],
                ),
            ],
        )

    monkeypatch.setattr("agent_os.workflows.run_planner_agent", over_limit)

    with pytest.raises(RuntimeError, match="returned 2 tasks.*limit is 1"):
        asyncio.run(inspect.unwrap(plan_comparison)(value, 1))

    assert "at most 1 independent task" in prompts[0]
    execution = store.get_execution("planner-over-limit")
    assert execution.status.value == "failed"
    assert execution.final_output == {
        "error": "planner returned 2 tasks; configured limit is 1"
    }
    store.engine.dispose()


def test_developer_session_timeout_fix_and_turn_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = session(tmp_path)
    candidate = Candidate(turn=1, worktree=str(tmp_path), head="a" * 40)
    monkeypatch.setattr(
        "agent_os.workflows.prepare_task_step", lambda *_args: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None
    )

    async def developer_turn(*_args, **_kwargs):
        return (
            candidate,
            DeveloperTurnResult(summary="ready", ready_for_review=True),
        )

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


def test_developer_session_only_publishes_ready_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = session(tmp_path)
    candidates = [
        Candidate(turn=1, worktree=str(tmp_path), head="a" * 40),
        Candidate(turn=2, worktree=str(tmp_path), head="b" * 40),
    ]
    prompts: list[str] = []
    events: list[tuple[str, Candidate]] = []

    monkeypatch.setattr(
        "agent_os.workflows.prepare_task_step", lambda *_args: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None
    )

    async def developer_turn(*_args, **kwargs):
        prompts.append(kwargs["prompt"])
        candidate = candidates[len(prompts) - 1]
        return (
            candidate,
            DeveloperTurnResult(
                summary="turn completed",
                ready_for_review=len(prompts) == 2,
            ),
        )

    async def set_event(topic, candidate):
        events.append((topic, candidate))

    async def close(*_args, **_kwargs):
        return DeveloperCommand(action="close")

    monkeypatch.setattr("agent_os.workflows._developer_turn", developer_turn)
    monkeypatch.setattr("agent_os.workflows.DBOS.set_event_async", set_event)
    monkeypatch.setattr("agent_os.workflows.DBOS.recv_async", close)

    head = asyncio.run(inspect.unwrap(developer_session)(value))

    assert head == "b" * 40
    assert events == [("candidate:1", candidates[1])]
    assert len(prompts) == 2
    assert "not ready for review" in prompts[1]


def test_developer_session_publishes_failure_instead_of_leaving_parent_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = session(tmp_path)
    published: dict[str, object] = {}
    monkeypatch.setattr(
        "agent_os.workflows.prepare_task_step", lambda *_args: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None
    )

    async def fail(*_args, **_kwargs):
        raise RuntimeError("developer crashed")

    async def set_event(topic, event):
        published[topic] = event

    monkeypatch.setattr("agent_os.workflows._developer_turn", fail)
    monkeypatch.setattr("agent_os.workflows.DBOS.set_event_async", set_event)

    with pytest.raises(RuntimeError, match="developer crashed"):
        asyncio.run(inspect.unwrap(developer_session)(value))
    assert published["candidate:1"] == {"error": "developer crashed"}

    async def get_event(*_args, **_kwargs):
        return published["candidate:1"]

    monkeypatch.setattr("agent_os.workflows.DBOS.get_event_async", get_event)
    with pytest.raises(RuntimeError, match="developer crashed"):
        asyncio.run(_candidate("developer-1", 1))


def test_candidate_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(*_args, **_kwargs):
        return None

    monkeypatch.setattr("agent_os.workflows.DBOS.get_event_async", missing)
    with pytest.raises(TimeoutError, match="candidate"):
        asyncio.run(_candidate("developer-1", 1))


def test_completion_cleans_worktrees_before_releasing_active_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path)
    calls: list[str] = []

    class Handle:
        workflow_id = "planner-1"

        async def get_result(self):
            return PlanComparison(complete=True)

    async def enqueue(*_args, **_kwargs):
        return Handle()

    def update(_value, status, **_kwargs):
        calls.append(status.value)

    monkeypatch.setattr(
        "agent_os.workflows.SetWorkflowID", lambda _value: nullcontext()
    )
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.integration_head_step", lambda _value: "a" * 40
    )
    monkeypatch.setattr(
        "agent_os.workflows.cleanup_step", lambda _value: calls.append("cleanup")
    )

    assert asyncio.run(inspect.unwrap(engineering_run)(value)) == "a" * 40
    assert calls[-3:] == [
        RunStatus.FINALIZING.value,
        "cleanup",
        RunStatus.COMPLETE.value,
    ]


@pytest.mark.parametrize("failure_point", ["cleanup", "complete"])
def test_completion_finalization_retries_transient_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    value = run_input(tmp_path)
    calls: list[str] = []
    failures = 0
    status = RunStatus.RUNNING

    class Handle:
        workflow_id = "planner-1"

        async def get_result(self):
            return PlanComparison(complete=True)

    async def enqueue(*_args, **_kwargs):
        return Handle()

    def update(_value, status, **_kwargs):
        nonlocal failures
        calls.append(status.value)
        if (
            failure_point == "complete"
            and status is RunStatus.COMPLETE
            and failures < 2
        ):
            failures += 1
            raise RuntimeError("transient completion write failure")

    def cleanup(_value):
        nonlocal failures
        calls.append("cleanup")
        if failure_point == "cleanup" and failures < 2:
            failures += 1
            raise RuntimeError("transient cleanup failure")

    def current_status(_value):
        return status

    monkeypatch.setattr(
        "agent_os.workflows.SetWorkflowID", lambda _value: nullcontext()
    )
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr("agent_os.workflows.run_status_step", current_status)
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.integration_head_step", lambda _value: "a" * 40
    )
    monkeypatch.setattr("agent_os.workflows.cleanup_step", cleanup)

    assert asyncio.run(inspect.unwrap(engineering_run)(value)) == "a" * 40

    assert failures == 2
    assert calls[-1] == RunStatus.COMPLETE.value
    assert RunStatus.FAILED.value not in calls


@pytest.mark.parametrize("failure_point", ["cleanup", "complete"])
def test_completion_finalization_exhaustion_records_durable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    value = run_input(tmp_path)
    calls: list[tuple[RunStatus | str, str | None]] = []
    status = RunStatus.RUNNING

    class Handle:
        workflow_id = "planner-1"

        async def get_result(self):
            return PlanComparison(complete=True)

    async def enqueue(*_args, **_kwargs):
        return Handle()

    def update(_value, requested, **kwargs):
        nonlocal status
        calls.append((requested, kwargs.get("failure_reason")))
        if requested is RunStatus.FINALIZING:
            status = requested
        elif requested is RunStatus.COMPLETE and failure_point == "complete":
            raise RuntimeError("permanent completion write failure")
        else:
            status = requested

    def cleanup(_value):
        calls.append(("cleanup", None))
        if failure_point == "cleanup":
            raise RuntimeError("permanent cleanup failure")

    monkeypatch.setattr(
        "agent_os.workflows.SetWorkflowID", lambda _value: nullcontext()
    )
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr("agent_os.workflows.run_status_step", lambda _value: status)
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.integration_head_step", lambda _value: "a" * 40
    )
    monkeypatch.setattr("agent_os.workflows.cleanup_step", cleanup)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        asyncio.run(inspect.unwrap(engineering_run)(value))

    assert status is RunStatus.FAILED
    failure = calls[-1]
    assert failure[0] is RunStatus.FAILED
    assert failure[1] is not None
    expected_reason = "cleanup" if failure_point == "cleanup" else "completion"
    assert expected_reason in failure[1]


def test_cancellation_wins_before_finalization_and_preserves_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path)
    calls: list[RunStatus | str] = []

    class Handle:
        workflow_id = "planner-1"

        async def get_result(self):
            return PlanComparison(complete=True)

    async def enqueue(*_args, **_kwargs):
        return Handle()

    def update(_value, status, **_kwargs):
        calls.append(status)
        if status is RunStatus.FINALIZING:
            raise RuntimeError("cancellation won")

    monkeypatch.setattr(
        "agent_os.workflows.SetWorkflowID", lambda _value: nullcontext()
    )
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr(
        "agent_os.workflows.run_status_step", lambda _value: RunStatus.CANCELLING
    )
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.integration_head_step", lambda _value: "a" * 40
    )
    monkeypatch.setattr(
        "agent_os.workflows.cleanup_step", lambda _value: calls.append("cleanup")
    )

    assert asyncio.run(inspect.unwrap(engineering_run)(value)) == "a" * 40
    assert "cleanup" not in calls
    assert RunStatus.FAILED not in calls


def test_failure_attempts_every_developer_close_before_recording_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = run_input(tmp_path)
    comparison = PlanComparison(
        complete=False,
        tasks=[
            task(),
            ImplementationTask(
                id="task-2",
                description="Implement another task",
                acceptance_criteria=["It also works"],
            ),
        ],
    )
    closed: list[str] = []
    updates: list[tuple[RunStatus, str | None]] = []

    class PlannerHandle:
        workflow_id = "planner-1"

        async def get_result(self):
            return comparison

    class DeveloperHandle:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

    async def enqueue(*_args, **_kwargs):
        return PlannerHandle()

    async def enqueue_developer(developer_input):
        return DeveloperHandle(developer_input.developer_workflow_id)

    async def candidate(*_args, **_kwargs):
        raise RuntimeError("developer child failed")

    async def send(workflow_id, *_args, **_kwargs):
        closed.append(workflow_id)
        if len(closed) == 1:
            raise RuntimeError("close failed")

    def update(_value, status, **kwargs):
        updates.append((status, kwargs.get("failure_reason")))

    monkeypatch.setattr(
        "agent_os.workflows.SetWorkflowID", lambda _value: nullcontext()
    )
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.DBOS.send_async", send)
    monkeypatch.setattr("agent_os.workflows._enqueue_developer", enqueue_developer)
    monkeypatch.setattr("agent_os.workflows._candidate", candidate)
    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr(
        "agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.create_task_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.integration_head_step", lambda _value: "a" * 40
    )

    with pytest.raises(RuntimeError, match="developer child failed"):
        asyncio.run(inspect.unwrap(engineering_run)(value))

    assert len(closed) == 2
    assert updates[-1][0] is RunStatus.FAILED
    assert updates[-1][1] is not None
    assert "developer child failed" in updates[-1][1]
    assert "close failed" in updates[-1][1]


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

    monkeypatch.setattr(
        "agent_os.workflows.SetWorkflowID", lambda _value: nullcontext()
    )
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.DBOS.send_async", send)
    monkeypatch.setattr("agent_os.workflows._enqueue_developer", enqueue)
    monkeypatch.setattr("agent_os.workflows._candidate", next_candidate)
    monkeypatch.setattr("agent_os.workflows._review", review)
    monkeypatch.setattr("agent_os.workflows.update_run_step", update)
    monkeypatch.setattr(
        "agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.create_task_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.integration_head_step", lambda _value: "a" * 40
    )
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
    ("review_cycles", "developer_turns", "stage_kind", "reason"),
    [
        (1, 6, "ready", "review cycle limit"),
        (2, 1, "ready", "developer turn limit"),
        (2, 1, "conflict", "developer turn limit"),
        (2, 6, "missing-head", "staging produced no head"),
    ],
)
def test_engineering_run_enforces_review_and_turn_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    review_cycles: int,
    developer_turns: int,
    stage_kind: str,
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

    monkeypatch.setattr(
        "agent_os.workflows.SetWorkflowID", lambda _value: nullcontext()
    )
    monkeypatch.setattr("agent_os.workflows.DBOS.enqueue_workflow_async", enqueue)
    monkeypatch.setattr("agent_os.workflows.DBOS.send_async", send)
    monkeypatch.setattr("agent_os.workflows._enqueue_developer", enqueue)
    monkeypatch.setattr("agent_os.workflows._candidate", candidate)
    monkeypatch.setattr("agent_os.workflows._review", rejected)
    monkeypatch.setattr(
        "agent_os.workflows.update_run_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.update_task_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.create_task_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent_os.workflows.prepare_integration_step", lambda _value: str(tmp_path)
    )
    monkeypatch.setattr(
        "agent_os.workflows.integration_head_step", lambda _value: "a" * 40
    )
    monkeypatch.setattr(
        "agent_os.workflows.prepare_staging_step",
        lambda *_args: StageResult(
            path=str(tmp_path),
            head="a" * 40 if stage_kind == "ready" else None,
            conflicts=["conflict.txt"] if stage_kind == "conflict" else [],
        ),
    )

    with pytest.raises(RuntimeError, match=reason):
        asyncio.run(inspect.unwrap(engineering_run)(value))
