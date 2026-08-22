import asyncio
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai import (
    AgentStreamEvent,
    ModelRequest,
    PartStartEvent,
    RunContext,
    TextPart,
)
from pydantic_ai.models.test import TestModel

from agent_os.agents import (
    AgentDeps,
    _record_event,
    _run_read_only_git,
    deserialize_history,
    list_files,
    list_files_step,
    normalize_event_type,
    planner_agent,
    read_file,
    read_file_step,
    replace_text,
    replace_text_step,
    review_git_diff,
    review_git_diff_step,
    run_command,
    run_command_step,
    run_developer_agent,
    run_planner_agent,
    run_reviewer_agent,
    search_repo,
    search_repo_step,
    serialize_history,
    transcript_handler,
    write_file,
    write_file_step,
)
from agent_os.models import PlanComparison, TranscriptEvent


def deps(
    tmp_path: Path,
    *,
    allow_writes: bool = True,
    review_base: str | None = None,
) -> AgentDeps:
    return AgentDeps(
        database_url=f"sqlite:///{tmp_path / 'state.db'}",
        worktree=str(tmp_path),
        plan_path="PLAN.md",
        execution_id="execution-1",
        run_id="run-1",
        workflow_id="workflow-1",
        session_id=None,
        task_id=None,
        shell_timeout_seconds=5,
        allow_writes=allow_writes,
        review_base=review_base,
    )


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_history_serialization_round_trip() -> None:
    messages = [ModelRequest.user_text_prompt("hello", instructions="be concise")]
    encoded = serialize_history(messages)
    decoded = deserialize_history(encoded)
    assert decoded == messages


def test_repository_steps_confine_paths_and_plan_writes(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan")
    (tmp_path / "file.txt").write_text("old")
    assert read_file_step(str(tmp_path), "file.txt") == "old"
    write_file_step(str(tmp_path), "file.txt", "new", "PLAN.md", True)
    assert (tmp_path / "file.txt").read_text() == "new"

    with pytest.raises(ValueError, match="outside"):
        read_file_step(str(tmp_path), "../outside.txt")
    with pytest.raises(ValueError, match="PLAN.md"):
        write_file_step(str(tmp_path), "PLAN.md", "changed", "PLAN.md", True)
    with pytest.raises(PermissionError):
        write_file_step(str(tmp_path), "file.txt", "changed", "PLAN.md", False)


def test_shell_step_enforces_read_only_mode(tmp_path: Path) -> None:
    output = run_command_step(str(tmp_path), "pwd", 5, True)
    assert output["returncode"] == 0
    for command in ("pwd", "git status; touch changed", "make --version && touch changed"):
        with pytest.raises(PermissionError, match="read-only"):
            run_command_step(str(tmp_path), command, 5, False)

    with pytest.raises(ValueError, match="empty"):
        run_command_step(str(tmp_path), "", 5, True)


def test_repository_tool_steps_and_async_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "file.txt").write_text("old value\n")
    context = cast(RunContext[AgentDeps], SimpleNamespace(deps=deps(tmp_path)))

    async def scenario() -> None:
        assert await read_file(context, "file.txt") == "old value\n"
        assert "file.txt" in await list_files(context)
        assert "old value" in await search_repo(context, "old value")
        assert await write_file(context, "new.txt", "new") == "new.txt"
        assert await replace_text(context, "file.txt", "old", "new") == "file.txt"
        assert (await run_command(context, "pwd"))["returncode"] == 0

    asyncio.run(scenario())
    assert list_files_step(str(tmp_path))
    assert "new value" in search_repo_step(str(tmp_path), "new value")

    with pytest.raises(ValueError, match="not found"):
        replace_text_step(str(tmp_path), "file.txt", "missing", "x", "PLAN.md", True)
    (tmp_path / "file.txt").write_text("twice twice")
    with pytest.raises(ValueError, match="more than once"):
        replace_text_step(str(tmp_path), "file.txt", "twice", "x", "PLAN.md", True)

    def successful_run(args, **_kwargs):
        stdout = "file.txt\n" if args[1] == "--files" else "file.txt:1:twice twice\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr("agent_os.agents.shutil.which", lambda _command: "/usr/bin/rg")
    monkeypatch.setattr("agent_os.agents.subprocess.run", successful_run)
    assert list_files_step(str(tmp_path)) == ["file.txt"]
    assert search_repo_step(str(tmp_path), "twice") == "file.txt:1:twice twice\n"

    def failed_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 2, "", "tool failed")

    monkeypatch.setattr("agent_os.agents.subprocess.run", failed_run)
    with pytest.raises(RuntimeError, match="tool failed"):
        list_files_step(str(tmp_path))
    with pytest.raises(RuntimeError, match="tool failed"):
        search_repo_step(str(tmp_path), "value")


def test_repository_steps_fall_back_when_ripgrep_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "file.txt").write_text("first\nneedle\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "internal").write_text("needle\n")
    monkeypatch.setattr("agent_os.agents.shutil.which", lambda _command: None)

    assert list_files_step(str(tmp_path)) == ["nested/file.txt"]
    assert search_repo_step(str(tmp_path), "needle") == "nested/file.txt:2:needle\n"
    with pytest.raises(RuntimeError, match="unterminated character set"):
        search_repo_step(str(tmp_path), "[")


def test_shell_timeout_is_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("command", 1)

    monkeypatch.setattr("agent_os.agents.subprocess.run", timeout)
    with pytest.raises(TimeoutError, match="exceeded"):
        run_command_step(str(tmp_path), "pwd", 1, True)


def test_review_git_diff_step_inspects_exact_protected_range(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "tracked.txt").write_text("old\n")
    git(tmp_path, "add", "tracked.txt")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "tracked.txt").write_text("new\n")
    (tmp_path / "added.txt").write_text("added\n")
    git(tmp_path, "add", "tracked.txt", "added.txt")
    git(tmp_path, "commit", "-qm", "change")
    head = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "untracked.txt").write_text("not reviewed\n")

    result = review_git_diff_step(str(tmp_path), base, None, 5)
    assert result["base"] == base
    assert result["head"] == head
    assert result["changed_files"] == ["added.txt", "tracked.txt"]
    assert "-old" in result["patch"]
    assert "+new" in result["patch"]
    assert "untracked.txt" not in result["patch"]

    one_file = review_git_diff_step(str(tmp_path), base, "tracked.txt", 5)
    assert one_file["changed_files"] == ["tracked.txt"]
    assert "added.txt" not in one_file["patch"]


def test_review_git_diff_rejects_unprotected_or_unsafe_ranges(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "tracked.txt").write_text("old\n")
    git(tmp_path, "add", "tracked.txt")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="protected review base"):
        review_git_diff_step(str(tmp_path), None, None, 5)
    with pytest.raises(ValueError, match="full commit ID"):
        review_git_diff_step(str(tmp_path), "HEAD; touch escaped", None, 5)
    with pytest.raises(ValueError, match="outside"):
        review_git_diff_step(str(tmp_path), base, "../outside.txt", 5)
    with pytest.raises(ValueError, match="must name"):
        review_git_diff_step(str(tmp_path), base, "", 5)

    nested = tmp_path / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="repository root"):
        review_git_diff_step(str(nested), base, None, 5)


def test_read_only_git_normalizes_timeout_and_command_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("git", 1)

    monkeypatch.setattr("agent_os.agents.subprocess.run", timeout)
    with pytest.raises(TimeoutError, match="Git inspection exceeded"):
        _run_read_only_git(tmp_path, ["status"], 1)

    for stdout, stderr, expected in (
        ("", "git failed", "git failed"),
        ("stdout failure", "", "stdout failure"),
        ("", "", "Git inspection failed"),
    ):
        monkeypatch.setattr(
            "agent_os.agents.subprocess.run",
            lambda *_args, _stdout=stdout, _stderr=stderr, **_kwargs: subprocess.CompletedProcess(
                [], 2, _stdout, _stderr
            ),
        )
        with pytest.raises(RuntimeError, match=expected):
            _run_read_only_git(tmp_path, ["status"], 1)


def test_review_git_diff_rejects_mismatched_and_non_ancestor_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = "a" * 40
    head = "b" * 40

    def result(stdout: str = "", returncode: int = 0, stderr: str = ""):
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    responses = iter([result(str(tmp_path)), result("c" * 40)])
    monkeypatch.setattr(
        "agent_os.agents._run_read_only_git", lambda *_args, **_kwargs: next(responses)
    )
    with pytest.raises(ValueError, match="resolve exactly"):
        review_git_diff_step(str(tmp_path), base, None, 5)

    for ancestry, expected in (
        (result(returncode=1), "not an ancestor"),
        (result(returncode=2, stderr="ancestry failed"), "ancestry failed"),
        (result(returncode=2), "could not validate"),
    ):
        responses = iter(
            [result(str(tmp_path)), result(base), result(head), ancestry]
        )
        monkeypatch.setattr(
            "agent_os.agents._run_read_only_git",
            lambda *_args, _responses=responses, **_kwargs: next(_responses),
        )
        with pytest.raises((ValueError, RuntimeError), match=expected):
            review_git_diff_step(str(tmp_path), base, None, 5)


def test_review_git_diff_async_tool_uses_orchestrator_base(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "tracked.txt").write_text("old\n")
    git(tmp_path, "add", "tracked.txt")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD")
    agent_deps = deps(tmp_path, allow_writes=False, review_base=base)
    context = cast(RunContext[AgentDeps], SimpleNamespace(deps=agent_deps))

    result = asyncio.run(review_git_diff(context))
    assert result["base"] == base
    assert result["changed_files"] == []


def test_record_event_writes_json_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event = TranscriptEvent(
        execution_id="execution-1",
        sequence=1,
        event_type="lifecycle.started",
        payload={"ok": True},
        created_at=datetime.now(UTC),
    )

    disposed = False

    class FakeEngine:
        def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    class FakeStore:
        def __init__(self, _database_url: str):
            self.engine = FakeEngine()

        def append_event(
            self, execution_id: str, event_type: str, payload, *, event_key: str
        ):
            assert (execution_id, event_type, payload, event_key) == (
                "execution-1",
                "lifecycle.started",
                {"ok": True},
                "event-key",
            )
            return event

    monkeypatch.setattr("agent_os.agents.StateStore", FakeStore)
    _record_event(deps(tmp_path), "lifecycle.started", {"ok": True}, "event-key")
    assert '"sequence":1' in capsys.readouterr().out
    assert disposed


def test_record_event_disposes_engine_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disposed = False

    class FakeEngine:
        def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    class FakeStore:
        def __init__(self, _database_url: str):
            self.engine = FakeEngine()

        def append_event(self, *_args, **_kwargs):
            raise RuntimeError("append failed")

    monkeypatch.setattr("agent_os.agents.StateStore", FakeStore)
    with pytest.raises(RuntimeError, match="append failed"):
        _record_event(deps(tmp_path), "model.stream", {}, "event-key")
    assert disposed


def test_transcript_event_types_are_normalized() -> None:
    event = lambda name: cast(AgentStreamEvent, cast(Any, type(name, (), {})()))
    assert normalize_event_type(event("FunctionToolCallEvent")) == "tool.call"
    assert normalize_event_type(event("FunctionToolResultEvent")) == "tool.result"
    assert normalize_event_type(event("FinalResultEvent")) == "model.final_result"
    assert normalize_event_type(event("PartDeltaEvent")) == "model.stream"


def test_transcript_retry_keys_preserve_differing_attempts_and_dedupe_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, tuple[str, object]] = {}
    calls: list[str] = []

    def record(_deps, event_type: str, payload, event_key: str) -> None:
        value = (event_type, payload)
        existing = recorded.get(event_key)
        if existing is not None and existing != value:
            raise RuntimeError(f"conflicting replay for {event_key}")
        recorded[event_key] = value
        calls.append(event_key)

    step_status = SimpleNamespace(current_attempt=0)
    monkeypatch.setattr("agent_os.agents._record_event", record)
    monkeypatch.setattr(
        "agent_os.agents.DBOS",
        SimpleNamespace(
            workflow_id="planner-workflow", step_id=7, step_status=step_status
        ),
    )
    context = cast(RunContext[AgentDeps], SimpleNamespace(deps=deps(tmp_path)))

    async def stream(content: str):
        yield PartStartEvent(index=0, part=TextPart(content=content))

    async def scenario() -> None:
        await transcript_handler(context, stream("partial response"))
        # A recovery can restart the same DBOS retry attempt. Identical events must replay
        # idempotently, while a provider that emits different partial content must be audited.
        await transcript_handler(context, stream("partial response"))
        await transcript_handler(context, stream("different partial response"))
        step_status.current_attempt = 1
        await transcript_handler(context, stream("partial response"))

    asyncio.run(scenario())

    assert calls[0] == calls[1]
    assert len(set(calls)) == 3
    assert len(recorded) == 3
    assert ":attempt:0:0:" in calls[0]
    assert ":attempt:1:0:" in calls[3]


def test_planner_agent_forces_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "agent_os.agents._record_event",
        lambda _deps, event_type, _payload, event_key: recorded.append(
            (event_type, event_key)
        ),
    )
    monkeypatch.setattr(
        "agent_os.agents.DBOS",
        SimpleNamespace(workflow_id="planner-workflow", step_id=7),
    )
    model = TestModel(
        call_tools=[],
        custom_output_args={
            "complete": False,
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Implement feature",
                    "acceptance_criteria": ["Tests pass"],
                }
            ],
        },
    )

    async def run() -> PlanComparison:
        with planner_agent.override(model=model):
            result = await planner_agent.run(
                "Compare the plan",
                deps=deps(tmp_path, allow_writes=False),
                run_id="execution-1",
                conversation_id="execution-1",
            )
        return result.output

    output = asyncio.run(run())
    assert isinstance(output, PlanComparison)
    assert output.tasks[0].id == "task-1"
    assert recorded
    assert all(
        event_key.startswith("dbos:planner-workflow:7:") for _, event_key in recorded
    )


def test_named_agent_helpers_use_structured_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_os.agents._record_event", lambda *_args: None)
    agent_deps = deps(tmp_path)

    planner_model = TestModel(
        call_tools=[], custom_output_args={"complete": True, "tasks": []}
    )
    developer_model = TestModel(
        call_tools=[],
        custom_output_args={
            "summary": "done",
            "validation": ["pytest"],
            "ready_for_review": True,
        },
    )
    reviewer_model = TestModel(
        call_tools=[], custom_output_args={"approved": True, "issues": []}
    )

    async def scenario() -> None:
        planned = await run_planner_agent(
            "compare",
            agent_deps,
            planner_model,
            5,
        )
        developed, history = await run_developer_agent(
            "implement",
            agent_deps,
            developer_model,
            5,
            [],
        )
        reviewed = await run_reviewer_agent(
            "review",
            agent_deps,
            reviewer_model,
            5,
        )
        assert planned.complete
        assert developed.summary == "done"
        assert history
        assert reviewed.approved

    asyncio.run(scenario())
    planner_parameters = planner_model.last_model_request_parameters
    reviewer_parameters = reviewer_model.last_model_request_parameters
    developer_parameters = developer_model.last_model_request_parameters
    assert planner_parameters is not None
    assert reviewer_parameters is not None
    assert developer_parameters is not None
    read_tool_names = {"list_files", "read_file", "search_repo"}
    assert {tool.name for tool in planner_parameters.function_tools} == read_tool_names
    assert {tool.name for tool in reviewer_parameters.function_tools} == read_tool_names | {
        "review_git_diff"
    }
    assert {tool.name for tool in developer_parameters.function_tools} == read_tool_names | {
        "replace_text",
        "run_command",
        "write_file",
    }
