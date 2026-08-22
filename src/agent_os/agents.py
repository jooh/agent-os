import json
import re
import shlex
import shutil
import subprocess
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dbos import DBOS
from pydantic import JsonValue, TypeAdapter
from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    RunContext,
)
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

from agent_os.models import DeveloperTurnResult, PlanComparison, ReviewResult
from agent_os.store import StateStore

_EVENT_ADAPTER = TypeAdapter(AgentStreamEvent)
_READ_ONLY_COMMANDS = frozenset({"git", "rg", "ls", "find", "sed", "pwd", "pytest", "uv", "make"})


@dataclass(frozen=True, slots=True)
class AgentDeps:
    database_url: str
    worktree: str
    plan_path: str
    execution_id: str
    run_id: str
    workflow_id: str
    session_id: str | None
    task_id: str | None
    shell_timeout_seconds: int
    allow_writes: bool


def _resolve(root: str, relative_path: str) -> Path:
    worktree = Path(root).resolve()
    candidate = (worktree / relative_path).resolve()
    if not candidate.is_relative_to(worktree):
        raise ValueError("path points outside the worktree")
    return candidate


@DBOS.step(name="agent_os.tools.read_file")
def read_file_step(root: str, relative_path: str) -> str:
    return _resolve(root, relative_path).read_text()


def _fallback_repository_files(root: str) -> list[str]:
    root_path = Path(root)
    return sorted(
        path.relative_to(root_path).as_posix()
        for path in root_path.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root_path).parts
    )


@DBOS.step(name="agent_os.tools.list_files")
def list_files_step(root: str) -> list[str]:
    if shutil.which("rg") is None:
        return _fallback_repository_files(root)
    result = subprocess.run(
        ["rg", "--files"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.splitlines()


@DBOS.step(name="agent_os.tools.search_repo")
def search_repo_step(root: str, pattern: str) -> str:
    if shutil.which("rg") is None:
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            raise RuntimeError(str(exc)) from exc
        matches: list[str] = []
        for relative_path in _fallback_repository_files(root):
            contents = (Path(root) / relative_path).read_text(errors="ignore")
            matches.extend(
                f"{relative_path}:{line_number}:{line}"
                for line_number, line in enumerate(contents.splitlines(), start=1)
                if expression.search(line)
            )
        return "".join(f"{match}\n" for match in matches)
    result = subprocess.run(
        ["rg", "-n", "--", pattern, "."],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


@DBOS.step(name="agent_os.tools.write_file")
def write_file_step(
    root: str,
    relative_path: str,
    content: str,
    plan_path: str,
    allow_writes: bool,
) -> str:
    if not allow_writes:
        raise PermissionError("this agent has read-only repository access")
    if relative_path == plan_path:
        raise ValueError(f"agents may not modify {plan_path}")
    path = _resolve(root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return relative_path


@DBOS.step(name="agent_os.tools.replace_text")
def replace_text_step(
    root: str,
    relative_path: str,
    old: str,
    new: str,
    plan_path: str,
    allow_writes: bool,
) -> str:
    current = read_file_step(root, relative_path)
    if old not in current:
        raise ValueError(f"text was not found in {relative_path}")
    if current.count(old) != 1:
        raise ValueError(f"text occurs more than once in {relative_path}")
    return write_file_step(
        root, relative_path, current.replace(old, new), plan_path, allow_writes
    )


@DBOS.step(name="agent_os.tools.run_command")
def run_command_step(
    root: str, command: str, timeout_seconds: int, allow_writes: bool
) -> dict[str, str | int]:
    try:
        executable = shlex.split(command)[0]
    except (ValueError, IndexError) as exc:
        raise ValueError("command must not be empty") from exc
    if not allow_writes and executable not in _READ_ONLY_COMMANDS:
        raise PermissionError(f"{executable!r} is not allowed for a read-only agent")
    try:
        result = subprocess.run(
            ["/bin/sh", "-lc", command],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"command exceeded {timeout_seconds} seconds") from exc
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def read_file(ctx: RunContext[AgentDeps], path: str) -> str:
    """Read a UTF-8 text file relative to the current worktree."""
    return read_file_step(ctx.deps.worktree, path)


async def list_files(ctx: RunContext[AgentDeps]) -> list[str]:
    """List repository files visible to ripgrep."""
    return list_files_step(ctx.deps.worktree)


async def search_repo(ctx: RunContext[AgentDeps], pattern: str) -> str:
    """Search repository text with a regular expression and return matching lines."""
    return search_repo_step(ctx.deps.worktree, pattern)


async def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """Write a complete UTF-8 text file inside the worktree."""
    return write_file_step(
        ctx.deps.worktree,
        path,
        content,
        ctx.deps.plan_path,
        ctx.deps.allow_writes,
    )


async def replace_text(
    ctx: RunContext[AgentDeps], path: str, old: str, new: str
) -> str:
    """Replace one exact text occurrence in a worktree file."""
    return replace_text_step(
        ctx.deps.worktree,
        path,
        old,
        new,
        ctx.deps.plan_path,
        ctx.deps.allow_writes,
    )


async def run_command(ctx: RunContext[AgentDeps], command: str) -> dict[str, str | int]:
    """Run a trusted local shell command from the worktree and capture its output."""
    return run_command_step(
        ctx.deps.worktree,
        command,
        ctx.deps.shell_timeout_seconds,
        ctx.deps.allow_writes,
    )


def serialize_history(messages: Sequence[ModelMessage]) -> list[JsonValue]:
    value = ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")
    return cast(list[JsonValue], value)


def deserialize_history(history: list[JsonValue]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(history)


@DBOS.step(name="agent_os.state.transcript_event")
def _record_event(deps: AgentDeps, event_type: str, payload: JsonValue) -> None:
    event = StateStore(deps.database_url).append_event(deps.execution_id, event_type, payload)
    print(json.dumps(event.model_dump(mode="json"), separators=(",", ":"), sort_keys=True), flush=True)


def normalize_event_type(event: AgentStreamEvent) -> str:
    name = event.__class__.__name__
    if "ToolCall" in name:
        return "tool.call"
    if "ToolResult" in name:
        return "tool.result"
    if "FinalResult" in name:
        return "agent.final_output"
    return "model.stream"


async def transcript_handler(
    ctx: RunContext[AgentDeps], stream: AsyncIterable[AgentStreamEvent]
) -> None:
    async for event in stream:
        payload = cast(JsonValue, _EVENT_ADAPTER.dump_python(event, mode="json"))
        _record_event(ctx.deps, normalize_event_type(event), payload)


def _durability(name: str) -> DBOSDurability[AgentDeps]:
    return DBOSDurability(
        name=name,
        event_stream_handler=transcript_handler,
        model_step_config={
            "retries_allowed": True,
            "max_attempts": 3,
            "interval_seconds": 1.0,
            "backoff_rate": 2.0,
        },
        parallel_execution_mode="sequential",
    )


_READ_TOOLS = [read_file, list_files, search_repo, run_command]
_WRITE_TOOLS = [*_READ_TOOLS, write_file, replace_text]

planner_agent = Agent(
    TestModel(call_tools=[], custom_output_args={"complete": True, "tasks": []}),
    name="agent_os_planner",
    deps_type=AgentDeps,
    output_type=PlanComparison,
    instructions=(
        "Compare the committed PLAN.md with the current implementation. Inspect the repository "
        "using tools. Return complete only when the plan is fully satisfied. Otherwise return at "
        "most two mutually independent, immediately actionable tasks. Do not implement changes."
    ),
    tools=_READ_TOOLS,
    capabilities=[_durability("agent_os_planner")],
    retries=2,
    defer_model_check=True,
)

developer_agent = Agent(
    TestModel(
        call_tools=[],
        custom_output_args={"summary": "test", "validation": [], "ready_for_review": True},
    ),
    name="agent_os_developer",
    deps_type=AgentDeps,
    output_type=DeveloperTurnResult,
    instructions=(
        "Implement the assigned task in the current worktree. Inspect existing conventions, make "
        "focused edits, run relevant validation, and never modify PLAN.md. Do not create Git commits; "
        "the orchestration layer owns commits. Return a concise structured summary."
    ),
    tools=_WRITE_TOOLS,
    capabilities=[_durability("agent_os_developer")],
    retries=2,
    defer_model_check=True,
)

reviewer_agent = Agent(
    TestModel(call_tools=[], custom_output_args={"approved": True, "issues": []}),
    name="agent_os_reviewer",
    deps_type=AgentDeps,
    output_type=ReviewResult,
    instructions=(
        "Review only the task-associated diff against the supplied base for technical correctness. "
        "Inspect context and run relevant checks, but do not edit files and do not judge overall plan "
        "completeness. Approve exactly when there are no actionable P0-P3 findings."
    ),
    tools=_READ_TOOLS,
    capabilities=[_durability("agent_os_reviewer")],
    retries=2,
    defer_model_check=True,
)


async def run_planner_agent(
    prompt: str,
    deps: AgentDeps,
    model: str | Model | None,
    request_limit: int,
) -> PlanComparison:
    result = await planner_agent.run(
        prompt,
        deps=deps,
        model=model,
        run_id=deps.execution_id,
        conversation_id=deps.execution_id,
        usage_limits=UsageLimits(request_limit=request_limit),
    )
    return result.output


async def run_developer_agent(
    prompt: str,
    deps: AgentDeps,
    model: str | Model | None,
    request_limit: int,
    history: list[ModelMessage],
) -> tuple[DeveloperTurnResult, list[ModelMessage]]:
    result = await developer_agent.run(
        prompt,
        deps=deps,
        model=model,
        run_id=deps.execution_id,
        conversation_id=deps.session_id,
        message_history=history or None,
        usage_limits=UsageLimits(request_limit=request_limit),
    )
    return result.output, result.all_messages()


async def run_reviewer_agent(
    prompt: str,
    deps: AgentDeps,
    model: str | Model | None,
    request_limit: int,
) -> ReviewResult:
    result = await reviewer_agent.run(
        prompt,
        deps=deps,
        model=model,
        run_id=deps.execution_id,
        conversation_id=deps.execution_id,
        usage_limits=UsageLimits(request_limit=request_limit),
    )
    return result.output
