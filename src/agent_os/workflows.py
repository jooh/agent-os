import hashlib
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

from dbos import DBOS, SetWorkflowID
from pydantic import JsonValue

from agent_os.agents import (
    AgentDeps,
    deserialize_history,
    run_developer_agent,
    run_planner_agent,
    run_reviewer_agent,
    serialize_history,
)
from agent_os.git import GitRepository
from agent_os.models import (
    CancellationFinalizerInput,
    Candidate,
    DeveloperCommand,
    DeveloperSessionInput,
    DeveloperTurnResult,
    ExecutionStatus,
    ImplementationTask,
    PlanComparison,
    ReviewInput,
    ReviewResult,
    RunInput,
    RunStatus,
    StageResult,
    TaskStatus,
)
from agent_os.runtime import DEVELOPER_QUEUE, PLANNER_QUEUE, REVIEWER_QUEUE
from agent_os.store import StateStore

COMMAND_TOPIC = "developer-command"
EVENT_TIMEOUT_SECONDS = 86_400
FINALIZATION_ATTEMPTS = 3
CANCELLATION_POLL_SECONDS = 1.0
TERMINAL_DBOS_STATUSES = frozenset(
    {"SUCCESS", "ERROR", "MAX_RECOVERY_ATTEMPTS_EXCEEDED", "CANCELLED"}
)
ORDINARY_GIT_LEASE_STATUSES = frozenset({RunStatus.RUNNING})
FINALIZATION_GIT_LEASE_STATUSES = frozenset({RunStatus.FINALIZING})


class DeveloperWorkflowError(RuntimeError):
    pass


class RunFinalizationError(RuntimeError):
    pass


def _repository(value: RunInput) -> GitRepository:
    return GitRepository(
        root=Path(value.repository_path),
        plan_path=value.plan_path,
        base_commit=value.base_commit,
        plan_id=value.plan_id,
        target_id=value.target_id,
        state_root=Path(value.state_dir),
        namespace=value.integration_branch.removesuffix("/integration"),
    )


@contextmanager
def _leased_repository(
    value: RunInput, allowed_statuses: frozenset[RunStatus]
) -> Iterator[GitRepository]:
    repository = _repository(value)
    with repository.operation_lock():
        with _state_store(value) as store:
            store.assert_target_lease(
                value.run_id, value.target_id, allowed_statuses
            )
        yield repository


def task_identifier(
    round_number: int,
    ordinal: int,
    task: ImplementationTask,
    run_id: str | None = None,
) -> str:
    digest = hashlib.sha256(task.model_dump_json().encode()).hexdigest()[:8]
    prefix = f"{hashlib.sha256(run_id.encode()).hexdigest()[:16]}-" if run_id else ""
    return f"{prefix}r{round_number:02d}-t{ordinal:02d}-{digest}"


@contextmanager
def _state_store(value: RunInput) -> Iterator[StateStore]:
    store = StateStore(value.database_url)
    try:
        yield store
    finally:
        store.engine.dispose()


@DBOS.step(name="agent_os.git.prepare_integration")
def prepare_integration_step(value: RunInput) -> str:
    with _leased_repository(value, ORDINARY_GIT_LEASE_STATUSES) as repository:
        return str(repository.prepare_integration_worktree())


@DBOS.step(name="agent_os.git.prepare_task")
def prepare_task_step(value: RunInput, task_id: str, start_commit: str) -> str:
    with _leased_repository(value, ORDINARY_GIT_LEASE_STATUSES) as repository:
        return str(repository.prepare_task_worktree(task_id, start_commit))


@DBOS.step(name="agent_os.git.commit_candidate")
def commit_candidate_step(
    value: RunInput,
    worktree: str,
    message: str,
    protected_base: str,
    turn: int = 1,
) -> Candidate:
    with _leased_repository(value, ORDINARY_GIT_LEASE_STATUSES) as repository:
        path = Path(worktree)
        repository.commit_changes(path, message)
        repository.assert_plan_unchanged(path, protected_base)
        return Candidate(turn=turn, worktree=worktree, head=repository.head(path))


@DBOS.step(name="agent_os.git.prepare_staging")
def prepare_staging_step(
    value: RunInput,
    task_id: str,
    integration_worktree: str,
    task_worktree: str,
) -> StageResult:
    with _leased_repository(value, ORDINARY_GIT_LEASE_STATUSES) as repository:
        return repository.prepare_staging_worktree(
            task_id, Path(integration_worktree), Path(task_worktree)
        )


@DBOS.step(name="agent_os.git.integrate")
def integrate_step(
    value: RunInput, staging_worktree: str, task_head: str | None = None
) -> str:
    with _leased_repository(value, ORDINARY_GIT_LEASE_STATUSES) as repository:
        repository.assert_plan_unchanged(
            Path(staging_worktree), repository.integration_head()
        )
        return repository.integrate(Path(staging_worktree), required_head=task_head)


@DBOS.step(name="agent_os.git.integration_head")
def integration_head_step(value: RunInput) -> str:
    with _leased_repository(value, ORDINARY_GIT_LEASE_STATUSES) as repository:
        return repository.integration_head()


@DBOS.step(name="agent_os.git.cleanup")
def cleanup_step(value: RunInput) -> None:
    with _leased_repository(value, FINALIZATION_GIT_LEASE_STATUSES) as repository:
        repository.cleanup_worktrees()


@DBOS.step(name="agent_os.cancellation.cancel_tree")
def cancel_workflow_tree_step(root_workflow_id: str) -> None:
    DBOS.cancel_workflow(root_workflow_id, cancel_children=True)


@DBOS.step(name="agent_os.cancellation.logical_quiescence")
def workflow_tree_quiescent_step(root_workflow_id: str) -> bool:
    root = DBOS.get_workflow_status(root_workflow_id)
    if root is None or root.status not in TERMINAL_DBOS_STATUSES:
        return False
    children = DBOS.list_workflows(
        parent_workflow_id=root_workflow_id,
        load_input=False,
        load_output=False,
    )
    finalizer_prefix = f"{root_workflow_id}:cancellation-finalizer"
    return all(
        child.status in TERMINAL_DBOS_STATUSES
        for child in children
        if not child.workflow_id.startswith(finalizer_prefix)
    )


@DBOS.step(name="agent_os.cancellation.physical_quiescence")
def target_operations_quiescent_step(value: CancellationFinalizerInput) -> bool:
    return GitRepository.target_operations_quiescent(
        Path(value.state_dir), value.target_id
    )


@DBOS.step(name="agent_os.cancellation.finalize")
def finalize_cancellation_step(value: CancellationFinalizerInput) -> None:
    with GitRepository.target_operation_lock(Path(value.state_dir), value.target_id):
        store = StateStore(value.database_url)
        try:
            store.finalize_cancellation(value.run_id)
        finally:
            store.engine.dispose()


@DBOS.step(name="agent_os.ids.new_execution")
def new_execution_id_step(role: str) -> str:
    return f"{role}-{uuid.uuid7()}"


@DBOS.step(name="agent_os.state.update_run")
def update_run_step(
    value: RunInput,
    status: RunStatus,
    current_round: int | None = None,
    integration_head: str | None = None,
    failure_reason: str | None = None,
) -> None:
    with _state_store(value) as store:
        store.update_run(
            value.run_id,
            status=status,
            current_round=current_round,
            integration_head=integration_head,
            failure_reason=failure_reason,
        )


@DBOS.step(name="agent_os.state.run_status")
def run_status_step(value: RunInput) -> RunStatus:
    with _state_store(value) as store:
        return store.get_run(value.run_id).status


@DBOS.step(name="agent_os.state.create_task")
def create_task_step(
    value: RunInput,
    task_id: str,
    task: ImplementationTask,
    round_number: int,
    ordinal: int,
) -> None:
    with _state_store(value) as store:
        store.create_task(
            run_id=value.run_id,
            task_id=task_id,
            round_number=round_number,
            ordinal=ordinal,
            description=task.description,
            acceptance_criteria=task.acceptance_criteria,
        )


@DBOS.step(name="agent_os.state.update_task")
def update_task_step(
    value: RunInput,
    task_id: str,
    status: TaskStatus,
    developer_workflow_id: str | None = None,
) -> None:
    with _state_store(value) as store:
        store.update_task(task_id, status, developer_workflow_id=developer_workflow_id)


@DBOS.step(name="agent_os.state.start_execution")
def start_execution_step(
    value: RunInput,
    execution_id: str,
    role: Literal["planner", "developer", "reviewer"],
    workflow_id: str,
    session_id: str | None,
    task_id: str | None,
) -> None:
    with _state_store(value) as store:
        store.create_execution(
            execution_id=execution_id,
            run_id=value.run_id,
            role=role,
            workflow_id=workflow_id,
            session_id=session_id,
            task_id=task_id,
        )


@DBOS.step(name="agent_os.state.finish_execution")
def finish_execution_step(
    value: RunInput,
    execution_id: str,
    status: ExecutionStatus,
    output: JsonValue,
) -> None:
    with _state_store(value) as store:
        event, _execution = store.finish_execution_with_event(
            execution_id,
            status,
            output,
            event_key="final-output",
        )
        print(
            json.dumps(
                event.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
            ),
            flush=True,
        )


@DBOS.step(name="agent_os.state.lifecycle_event")
def lifecycle_event_step(
    value: RunInput, execution_id: str, event_type: str, payload: JsonValue
) -> None:
    with _state_store(value) as store:
        event = store.append_event(
            execution_id, event_type, payload, event_key=event_type
        )
        print(
            json.dumps(
                event.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
            ),
            flush=True,
        )


@DBOS.step(name="agent_os.state.load_history")
def load_history_step(value: RunInput, session_id: str) -> list[JsonValue]:
    with _state_store(value) as store:
        return store.load_history(session_id)


@DBOS.step(name="agent_os.state.save_history")
def save_history_step(
    value: RunInput, session_id: str, task_id: str, history: list[JsonValue]
) -> None:
    with _state_store(value) as store:
        store.save_history(session_id, task_id, history)


def _deps(
    value: RunInput,
    *,
    worktree: str,
    execution_id: str,
    workflow_id: str,
    session_id: str | None,
    task_id: str | None,
    allow_writes: bool,
    review_base: str | None = None,
) -> AgentDeps:
    return AgentDeps(
        database_url=value.database_url,
        worktree=worktree,
        plan_path=value.plan_path,
        execution_id=execution_id,
        run_id=value.run_id,
        workflow_id=workflow_id,
        session_id=session_id,
        task_id=task_id,
        shell_timeout_seconds=value.shell_timeout_seconds,
        allow_writes=allow_writes,
        review_base=review_base,
    )


def _model(value: str) -> str | None:
    return None if value == "test" else value


@DBOS.workflow(name="agent_os.plan_comparison")
async def plan_comparison(value: RunInput, round_number: int) -> PlanComparison:
    worktree = prepare_integration_step(value)
    execution_id = new_execution_id_step("planner")
    workflow_id = DBOS.workflow_id or f"planner:{value.run_id}:{round_number}"
    start_execution_step(value, execution_id, "planner", workflow_id, None, None)
    lifecycle_event_step(
        value, execution_id, "lifecycle.started", {"round": round_number}
    )
    deps = _deps(
        value,
        worktree=worktree,
        execution_id=execution_id,
        workflow_id=workflow_id,
        session_id=None,
        task_id=None,
        allow_writes=False,
    )
    prompt = (
        f"Round {round_number}: compare {value.plan_path} with the repository at {worktree}. "
        f"Return at most {value.planner_task_limit} independent "
        f"{'task' if value.planner_task_limit == 1 else 'tasks'} that are ready now."
    )
    try:
        output = await run_planner_agent(
            prompt, deps, _model(value.planner_model), value.model_request_limit
        )
        if len(output.tasks) > value.planner_task_limit:
            raise RuntimeError(
                f"planner returned {len(output.tasks)} tasks; configured limit is "
                f"{value.planner_task_limit}"
            )
    except Exception as exc:
        finish_execution_step(
            value, execution_id, ExecutionStatus.FAILED, {"error": str(exc)}
        )
        raise
    finish_execution_step(
        value,
        execution_id,
        ExecutionStatus.SUCCEEDED,
        cast(JsonValue, output.model_dump(mode="json")),
    )
    return output


async def _developer_turn(
    session: DeveloperSessionInput,
    *,
    turn: int,
    worktree: str,
    protected_base: str,
    prompt: str,
) -> tuple[Candidate, DeveloperTurnResult]:
    value = session.run
    session_id = f"developer-session:{session.task_id}"
    execution_id = new_execution_id_step("developer")
    workflow_id = DBOS.workflow_id or session.developer_workflow_id
    start_execution_step(
        value,
        execution_id,
        "developer",
        workflow_id,
        session_id,
        session.task_id,
    )
    lifecycle_event_step(
        value,
        execution_id,
        "lifecycle.started",
        {"task_id": session.task_id, "turn": turn},
    )
    history = deserialize_history(load_history_step(value, session_id))
    deps = _deps(
        value,
        worktree=worktree,
        execution_id=execution_id,
        workflow_id=workflow_id,
        session_id=session_id,
        task_id=session.task_id,
        allow_writes=True,
    )
    try:
        output, messages = await run_developer_agent(
            prompt,
            deps,
            _model(value.developer_model),
            value.model_request_limit,
            history,
        )
        save_history_step(
            value, session_id, session.task_id, serialize_history(messages)
        )
        finish_execution_step(
            value,
            execution_id,
            ExecutionStatus.SUCCEEDED,
            cast(JsonValue, output.model_dump(mode="json")),
        )
    except Exception as exc:
        finish_execution_step(
            value, execution_id, ExecutionStatus.FAILED, {"error": str(exc)}
        )
        raise
    return (
        commit_candidate_step(
            value,
            worktree,
            f"agent-os: {session.task_id} turn {turn}",
            protected_base,
            turn,
        ),
        output,
    )


@DBOS.workflow(name="agent_os.developer_session")
async def developer_session(session: DeveloperSessionInput) -> str:
    value = session.run
    next_candidate = 1
    turn = 1
    try:
        worktree = prepare_task_step(value, session.task_id, session.start_commit)
        update_task_step(value, session.task_id, TaskStatus.DEVELOPING)
        prompt = (
            f"Implement task {session.task.id}: {session.task.description}\n"
            f"Acceptance criteria: {json.dumps(session.task.acceptance_criteria)}"
        )
        protected_base = session.start_commit
        for turn in range(1, value.max_developer_turns + 1):
            candidate, result = await _developer_turn(
                session,
                turn=turn,
                worktree=worktree,
                protected_base=protected_base,
                prompt=prompt,
            )
            if not result.ready_for_review:
                prompt = (
                    "Continue the current task. Your previous turn reported that it was not ready "
                    f"for review. Finish the work and validation before reporting readiness. "
                    f"Previous result: {result.model_dump_json()}"
                )
                continue

            await DBOS.set_event_async(f"candidate:{next_candidate}", candidate)
            next_candidate += 1
            command_value = await DBOS.recv_async(
                COMMAND_TOPIC, timeout_seconds=EVENT_TIMEOUT_SECONDS
            )
            if command_value is None:
                raise TimeoutError(f"developer session {session.task_id} timed out")
            command = (
                command_value
                if isinstance(command_value, DeveloperCommand)
                else DeveloperCommand.model_validate(command_value)
            )
            if command.action == "close":
                return candidate.head
            update_task_step(value, session.task_id, TaskStatus.FIXING)
            assert command.prompt and command.worktree and command.protected_base
            prompt = command.prompt
            worktree = command.worktree
            protected_base = command.protected_base
        raise RuntimeError(f"developer turn limit exceeded for {session.task_id}")
    except Exception as exc:
        await DBOS.set_event_async(f"candidate:{next_candidate}", {"error": str(exc)})
        raise


@DBOS.workflow(name="agent_os.technical_review")
async def technical_review(review: ReviewInput) -> ReviewResult:
    value = review.run
    execution_id = new_execution_id_step("reviewer")
    workflow_id = DBOS.workflow_id or review.reviewer_workflow_id
    start_execution_step(
        value,
        execution_id,
        "reviewer",
        workflow_id,
        None,
        review.task_id,
    )
    lifecycle_event_step(
        value,
        execution_id,
        "lifecycle.started",
        {"task_id": review.task_id, "cycle": review.review_cycle},
    )
    deps = _deps(
        value,
        worktree=review.worktree,
        execution_id=execution_id,
        workflow_id=workflow_id,
        session_id=None,
        task_id=review.task_id,
        allow_writes=False,
        review_base=review.base_commit,
    )
    prompt = (
        f"Review task {review.task_id}. Inspect only changes in "
        f"{review.base_commit}..HEAD from {review.worktree}."
    )
    try:
        output = await run_reviewer_agent(
            prompt, deps, _model(value.reviewer_model), value.model_request_limit
        )
    except Exception as exc:
        finish_execution_step(
            value, execution_id, ExecutionStatus.FAILED, {"error": str(exc)}
        )
        raise
    finish_execution_step(
        value,
        execution_id,
        ExecutionStatus.SUCCEEDED,
        cast(JsonValue, output.model_dump(mode="json")),
    )
    return output


async def _enqueue_developer(session: DeveloperSessionInput):
    with SetWorkflowID(session.developer_workflow_id):
        return await DBOS.enqueue_workflow_async(
            DEVELOPER_QUEUE, developer_session, session
        )


async def _review(
    value: RunInput,
    task_id: str,
    worktree: str,
    base_commit: str,
    cycle: int,
) -> ReviewResult:
    workflow_id = f"{value.workflow_id}:reviewer:{task_id}:{cycle}"
    review = ReviewInput(
        run=value,
        task_id=task_id,
        worktree=worktree,
        base_commit=base_commit,
        review_cycle=cycle,
        reviewer_workflow_id=workflow_id,
    )
    with SetWorkflowID(workflow_id):
        handle = await DBOS.enqueue_workflow_async(
            REVIEWER_QUEUE, technical_review, review
        )
    return await handle.get_result()


async def _candidate(workflow_id: str, turn: int) -> Candidate:
    value = await DBOS.get_event_async(
        workflow_id, f"candidate:{turn}", timeout_seconds=EVENT_TIMEOUT_SECONDS
    )
    if value is None:
        raise TimeoutError(f"timed out waiting for {workflow_id} candidate {turn}")
    if isinstance(value, dict) and isinstance(value.get("error"), str):
        raise DeveloperWorkflowError(
            f"developer workflow {workflow_id} failed: {value['error']}"
        )
    return value if isinstance(value, Candidate) else Candidate.model_validate(value)


def _record_run_failure(value: RunInput, reason: str) -> None:
    last_error: Exception | None = None
    for _attempt in range(FINALIZATION_ATTEMPTS):
        try:
            update_run_step(value, RunStatus.FAILED, failure_reason=reason)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            status = run_status_step(value)
            if status in {
                RunStatus.COMPLETE,
                RunStatus.FAILED,
                RunStatus.CANCELLING,
                RunStatus.CANCELLED,
            }:
                return
    raise RunFinalizationError(
        f"could not durably record run failure after {FINALIZATION_ATTEMPTS} attempts: "
        f"{last_error}"
    ) from last_error


def _finalize_success(
    value: RunInput, *, round_number: int, integration_head: str
) -> str:
    transition_error: Exception | None = None
    for _attempt in range(FINALIZATION_ATTEMPTS):
        try:
            update_run_step(value, RunStatus.FINALIZING, current_round=round_number)
            break
        except Exception as exc:  # noqa: BLE001
            transition_error = exc
            status = run_status_step(value)
            if status is RunStatus.FINALIZING:
                break
            if status in {
                RunStatus.COMPLETE,
                RunStatus.FAILED,
                RunStatus.CANCELLING,
                RunStatus.CANCELLED,
            }:
                return integration_head
    else:
        reason = (
            f"could not enter finalization after {FINALIZATION_ATTEMPTS} attempts: "
            f"{transition_error}"
        )
        _record_run_failure(value, reason)
        raise RunFinalizationError(reason) from transition_error

    cleanup_error: Exception | None = None
    for _attempt in range(FINALIZATION_ATTEMPTS):
        try:
            cleanup_step(value)
            break
        except Exception as exc:  # noqa: BLE001
            cleanup_error = exc
    else:
        reason = (
            f"worktree cleanup failed after {FINALIZATION_ATTEMPTS} attempts: "
            f"{cleanup_error}"
        )
        _record_run_failure(value, reason)
        raise RunFinalizationError(reason) from cleanup_error

    completion_error: Exception | None = None
    for _attempt in range(FINALIZATION_ATTEMPTS):
        try:
            update_run_step(
                value,
                RunStatus.COMPLETE,
                current_round=round_number,
                integration_head=integration_head,
            )
            return integration_head
        except Exception as exc:  # noqa: BLE001
            completion_error = exc
            status = run_status_step(value)
            if status is RunStatus.COMPLETE:
                return integration_head
            if status in {
                RunStatus.FAILED,
                RunStatus.CANCELLING,
                RunStatus.CANCELLED,
            }:
                return integration_head
    reason = (
        f"completion status write failed after {FINALIZATION_ATTEMPTS} attempts: "
        f"{completion_error}"
    )
    _record_run_failure(value, reason)
    raise RunFinalizationError(reason) from completion_error


@DBOS.workflow(name="agent_os.cancellation_finalizer")
async def cancellation_finalizer(value: CancellationFinalizerInput) -> None:
    cancel_workflow_tree_step(value.root_workflow_id)
    while True:
        logical_quiescence = workflow_tree_quiescent_step(value.root_workflow_id)
        physical_quiescence = target_operations_quiescent_step(value)
        if logical_quiescence and physical_quiescence:
            finalize_cancellation_step(value)
            return
        await DBOS.sleep_async(CANCELLATION_POLL_SECONDS)


@DBOS.workflow(name="agent_os.engineering_run")
async def engineering_run(value: RunInput) -> str:
    active_developers: list[str] = []
    try:
        update_run_step(value, RunStatus.RUNNING)
        integration_worktree = prepare_integration_step(value)
        for round_number in range(1, value.max_rounds + 1):
            update_run_step(value, RunStatus.RUNNING, current_round=round_number)
            planner_id = f"{value.workflow_id}:planner:{round_number}"
            with SetWorkflowID(planner_id):
                planner_handle = await DBOS.enqueue_workflow_async(
                    PLANNER_QUEUE, plan_comparison, value, round_number
                )
            comparison = await planner_handle.get_result()
            if comparison.complete:
                head = integration_head_step(value)
                return _finalize_success(
                    value,
                    round_number=round_number,
                    integration_head=head,
                )

            start_commit = integration_head_step(value)
            sessions: list[tuple[str, DeveloperSessionInput]] = []
            for ordinal, task in enumerate(comparison.tasks, start=1):
                task_id = task_identifier(
                    round_number, ordinal, task, run_id=value.run_id
                )
                developer_id = f"{value.workflow_id}:developer:{task_id}"
                create_task_step(value, task_id, task, round_number, ordinal)
                session = DeveloperSessionInput(
                    run=value,
                    task_id=task_id,
                    task=task,
                    start_commit=start_commit,
                    developer_workflow_id=developer_id,
                )
                handle = await _enqueue_developer(session)
                active_developers.append(handle.workflow_id)
                update_task_step(
                    value,
                    task_id,
                    TaskStatus.DEVELOPING,
                    developer_workflow_id=handle.workflow_id,
                )
                sessions.append((handle.workflow_id, session))

            for developer_id, session in sessions:
                task_id = session.task_id
                candidate_number = 1
                candidate = await _candidate(developer_id, candidate_number)
                task_head = candidate.head
                update_task_step(value, task_id, TaskStatus.STAGING)
                review_base = integration_head_step(value)
                stage = prepare_staging_step(
                    value,
                    task_id,
                    integration_worktree,
                    candidate.worktree,
                )
                if stage.conflicts:
                    if candidate.turn >= value.max_developer_turns:
                        raise RuntimeError(
                            f"developer turn limit exceeded for {task_id}"
                        )
                    command = DeveloperCommand(
                        action="fix",
                        prompt=(
                            "Resolve the staging merge conflicts, validate the result, and leave it "
                            f"ready for review. Conflicts: {json.dumps(stage.conflicts)}"
                        ),
                        worktree=stage.path,
                        protected_base=review_base,
                    )
                    await DBOS.send_async(developer_id, command, COMMAND_TOPIC)
                    candidate_number += 1
                    candidate = await _candidate(developer_id, candidate_number)
                else:
                    if stage.head is None:
                        raise RuntimeError(f"staging produced no head for {task_id}")
                    candidate = Candidate(
                        turn=candidate.turn,
                        worktree=stage.path,
                        head=stage.head,
                    )

                for cycle in range(1, value.max_review_cycles + 1):  # pragma: no branch
                    update_task_step(value, task_id, TaskStatus.REVIEWING)
                    review_result = await _review(
                        value, task_id, candidate.worktree, review_base, cycle
                    )
                    if review_result.approved:
                        head = integrate_step(value, candidate.worktree, task_head)
                        update_task_step(value, task_id, TaskStatus.INTEGRATED)
                        await DBOS.send_async(
                            developer_id,
                            DeveloperCommand(action="close"),
                            COMMAND_TOPIC,
                        )
                        active_developers.remove(developer_id)
                        update_run_step(value, RunStatus.RUNNING, integration_head=head)
                        break
                    if cycle == value.max_review_cycles:
                        raise RuntimeError(f"review cycle limit exceeded for {task_id}")
                    if candidate.turn >= value.max_developer_turns:
                        raise RuntimeError(
                            f"developer turn limit exceeded for {task_id}"
                        )
                    update_task_step(value, task_id, TaskStatus.FIXING)
                    command = DeveloperCommand(
                        action="fix",
                        prompt=(
                            "Address every technical review finding, run relevant validation, and "
                            f"leave the staging worktree ready for review: "
                            f"{review_result.model_dump_json()}"
                        ),
                        worktree=stage.path,
                        protected_base=review_base,
                    )
                    await DBOS.send_async(developer_id, command, COMMAND_TOPIC)
                    candidate_number += 1
                    candidate = await _candidate(developer_id, candidate_number)
        raise RuntimeError("reconciliation round limit exceeded")
    except RunFinalizationError:
        raise
    except Exception as exc:
        close_failures: list[str] = []
        for developer_id in active_developers:
            try:
                await DBOS.send_async(
                    developer_id, DeveloperCommand(action="close"), COMMAND_TOPIC
                )
            except Exception as close_exc:  # noqa: BLE001
                close_failures.append(f"{developer_id}: {close_exc}")
        failure_reason = str(exc)
        if close_failures:
            failure_reason += f"; developer close failures: {'; '.join(close_failures)}"
        _record_run_failure(value, failure_reason)
        raise
