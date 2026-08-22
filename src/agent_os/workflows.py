import hashlib
import json
import uuid
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
    Candidate,
    DeveloperCommand,
    DeveloperSessionInput,
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


def task_identifier(
    round_number: int,
    ordinal: int,
    task: ImplementationTask,
    run_id: str | None = None,
) -> str:
    digest = hashlib.sha256(task.model_dump_json().encode()).hexdigest()[:8]
    prefix = f"{run_id[:8]}-" if run_id else ""
    return f"{prefix}r{round_number:02d}-t{ordinal:02d}-{digest}"


@DBOS.step(name="agent_os.git.prepare_integration")
def prepare_integration_step(value: RunInput) -> str:
    return str(_repository(value).prepare_integration_worktree())


@DBOS.step(name="agent_os.git.prepare_task")
def prepare_task_step(value: RunInput, task_id: str, start_commit: str) -> str:
    return str(_repository(value).prepare_task_worktree(task_id, start_commit))


@DBOS.step(name="agent_os.git.commit_candidate")
def commit_candidate_step(
    value: RunInput,
    worktree: str,
    message: str,
    protected_base: str,
    turn: int = 1,
) -> Candidate:
    repository = _repository(value)
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
    return _repository(value).prepare_staging_worktree(
        task_id, Path(integration_worktree), Path(task_worktree)
    )


@DBOS.step(name="agent_os.git.integrate")
def integrate_step(value: RunInput, staging_worktree: str) -> str:
    repository = _repository(value)
    repository.assert_plan_unchanged(Path(staging_worktree), repository.integration_head())
    return repository.integrate(Path(staging_worktree))


@DBOS.step(name="agent_os.git.integration_head")
def integration_head_step(value: RunInput) -> str:
    return _repository(value).integration_head()


@DBOS.step(name="agent_os.git.cleanup")
def cleanup_step(value: RunInput) -> None:
    _repository(value).cleanup_worktrees()


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
    StateStore(value.database_url).update_run(
        value.run_id,
        status=status,
        current_round=current_round,
        integration_head=integration_head,
        failure_reason=failure_reason,
    )


@DBOS.step(name="agent_os.state.create_task")
def create_task_step(
    value: RunInput,
    task_id: str,
    task: ImplementationTask,
    round_number: int,
    ordinal: int,
) -> None:
    StateStore(value.database_url).create_task(
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
    StateStore(value.database_url).update_task(
        task_id, status, developer_workflow_id=developer_workflow_id
    )


@DBOS.step(name="agent_os.state.start_execution")
def start_execution_step(
    value: RunInput,
    execution_id: str,
    role: Literal["planner", "developer", "reviewer"],
    workflow_id: str,
    session_id: str | None,
    task_id: str | None,
) -> None:
    StateStore(value.database_url).create_execution(
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
    store = StateStore(value.database_url)
    event = store.append_event(execution_id, "agent.final_output", output)
    print(
        json.dumps(event.model_dump(mode="json"), separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    store.finish_execution(execution_id, status, output)


@DBOS.step(name="agent_os.state.lifecycle_event")
def lifecycle_event_step(
    value: RunInput, execution_id: str, event_type: str, payload: JsonValue
) -> None:
    event = StateStore(value.database_url).append_event(execution_id, event_type, payload)
    print(
        json.dumps(event.model_dump(mode="json"), separators=(",", ":"), sort_keys=True),
        flush=True,
    )


@DBOS.step(name="agent_os.state.load_history")
def load_history_step(value: RunInput, session_id: str) -> list[JsonValue]:
    return StateStore(value.database_url).load_history(session_id)


@DBOS.step(name="agent_os.state.save_history")
def save_history_step(
    value: RunInput, session_id: str, task_id: str, history: list[JsonValue]
) -> None:
    StateStore(value.database_url).save_history(session_id, task_id, history)


def _deps(
    value: RunInput,
    *,
    worktree: str,
    execution_id: str,
    workflow_id: str,
    session_id: str | None,
    task_id: str | None,
    allow_writes: bool,
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
        "Return at most two independent tasks that are ready now."
    )
    try:
        output = await run_planner_agent(
            prompt, deps, _model(value.planner_model), value.model_request_limit
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
) -> Candidate:
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
        save_history_step(value, session_id, session.task_id, serialize_history(messages))
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
    return commit_candidate_step(
        value,
        worktree,
        f"agent-os: {session.task_id} turn {turn}",
        protected_base,
        turn,
    )


@DBOS.workflow(name="agent_os.developer_session")
async def developer_session(session: DeveloperSessionInput) -> str:
    value = session.run
    worktree = prepare_task_step(value, session.task_id, session.start_commit)
    update_task_step(value, session.task_id, TaskStatus.DEVELOPING)
    prompt = (
        f"Implement task {session.task.id}: {session.task.description}\n"
        f"Acceptance criteria: {json.dumps(session.task.acceptance_criteria)}"
    )
    protected_base = session.start_commit
    for turn in range(1, value.max_developer_turns + 1):
        candidate = await _developer_turn(
            session,
            turn=turn,
            worktree=worktree,
            protected_base=protected_base,
            prompt=prompt,
        )
        await DBOS.set_event_async(f"candidate:{turn}", candidate)
        command_value = await DBOS.recv_async(COMMAND_TOPIC, timeout_seconds=EVENT_TIMEOUT_SECONDS)
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
        handle = await DBOS.enqueue_workflow_async(REVIEWER_QUEUE, technical_review, review)
    return await handle.get_result()


async def _candidate(workflow_id: str, turn: int) -> Candidate:
    value = await DBOS.get_event_async(
        workflow_id, f"candidate:{turn}", timeout_seconds=EVENT_TIMEOUT_SECONDS
    )
    if value is None:
        raise TimeoutError(f"timed out waiting for {workflow_id} candidate {turn}")
    return value if isinstance(value, Candidate) else Candidate.model_validate(value)


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
                update_run_step(
                    value,
                    RunStatus.COMPLETE,
                    current_round=round_number,
                    integration_head=head,
                )
                cleanup_step(value)
                return head

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
                candidate = await _candidate(developer_id, 1)
                update_task_step(value, task_id, TaskStatus.STAGING)
                review_base = integration_head_step(value)
                stage = prepare_staging_step(
                    value,
                    task_id,
                    integration_worktree,
                    candidate.worktree,
                )
                turn = 1
                if stage.conflicts:
                    turn += 1
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
                    candidate = await _candidate(developer_id, turn)

                for cycle in range(1, value.max_review_cycles + 1):  # pragma: no branch
                    update_task_step(value, task_id, TaskStatus.REVIEWING)
                    review_result = await _review(
                        value, task_id, candidate.worktree, review_base, cycle
                    )
                    if review_result.approved:
                        head = integrate_step(value, candidate.worktree)
                        update_task_step(value, task_id, TaskStatus.INTEGRATED)
                        await DBOS.send_async(
                            developer_id, DeveloperCommand(action="close"), COMMAND_TOPIC
                        )
                        active_developers.remove(developer_id)
                        update_run_step(
                            value, RunStatus.RUNNING, integration_head=head
                        )
                        break
                    if cycle == value.max_review_cycles:
                        raise RuntimeError(f"review cycle limit exceeded for {task_id}")
                    turn += 1
                    if turn > value.max_developer_turns:
                        raise RuntimeError(f"developer turn limit exceeded for {task_id}")
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
                    candidate = await _candidate(developer_id, turn)
        raise RuntimeError("reconciliation round limit exceeded")
    except Exception as exc:
        for developer_id in active_developers:
            await DBOS.send_async(
                developer_id, DeveloperCommand(action="close"), COMMAND_TOPIC
            )
        update_run_step(value, RunStatus.FAILED, failure_reason=str(exc))
        raise
