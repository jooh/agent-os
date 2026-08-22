import asyncio
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai import AgentStreamEvent, ModelRequest, RunContext
from pydantic_ai.models.test import TestModel

from agent_os.agents import (
    AgentDeps,
    _record_event,
    deserialize_history,
    list_files,
    list_files_step,
    normalize_event_type,
    planner_agent,
    read_file,
    read_file_step,
    replace_text,
    replace_text_step,
    run_command,
    run_command_step,
    run_developer_agent,
    run_planner_agent,
    run_reviewer_agent,
    search_repo,
    search_repo_step,
    serialize_history,
    write_file,
    write_file_step,
)
from agent_os.models import PlanComparison, TranscriptEvent


def deps(tmp_path: Path, *, allow_writes: bool = True) -> AgentDeps:
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
    )


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
    with pytest.raises(PermissionError):
        run_command_step(str(tmp_path), "touch changed", 5, False)

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

    def failed_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 2, "", "tool failed")

    monkeypatch.setattr("agent_os.agents.subprocess.run", failed_run)
    with pytest.raises(RuntimeError, match="tool failed"):
        list_files_step(str(tmp_path))
    with pytest.raises(RuntimeError, match="tool failed"):
        search_repo_step(str(tmp_path), "value")


def test_shell_timeout_is_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("command", 1)

    monkeypatch.setattr("agent_os.agents.subprocess.run", timeout)
    with pytest.raises(TimeoutError, match="exceeded"):
        run_command_step(str(tmp_path), "pwd", 1, True)


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

    class FakeStore:
        def __init__(self, _database_url: str):
            pass

        def append_event(self, execution_id: str, event_type: str, payload):
            assert (execution_id, event_type, payload) == (
                "execution-1",
                "lifecycle.started",
                {"ok": True},
            )
            return event

    monkeypatch.setattr("agent_os.agents.StateStore", FakeStore)
    _record_event(deps(tmp_path), "lifecycle.started", {"ok": True})
    assert '"sequence":1' in capsys.readouterr().out


def test_transcript_event_types_are_normalized() -> None:
    event = lambda name: cast(AgentStreamEvent, cast(Any, type(name, (), {})()))
    assert normalize_event_type(event("FunctionToolCallEvent")) == "tool.call"
    assert normalize_event_type(event("FunctionToolResultEvent")) == "tool.result"
    assert normalize_event_type(event("FinalResultEvent")) == "agent.final_output"
    assert normalize_event_type(event("PartDeltaEvent")) == "model.stream"


def test_planner_agent_forces_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[str] = []
    monkeypatch.setattr(
        "agent_os.agents._record_event",
        lambda _deps, event_type, _payload: recorded.append(event_type),
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


def test_named_agent_helpers_use_structured_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_os.agents._record_event", lambda *_args: None)
    agent_deps = deps(tmp_path)

    async def scenario() -> None:
        planned = await run_planner_agent(
            "compare",
            agent_deps,
            TestModel(call_tools=[], custom_output_args={"complete": True, "tasks": []}),
            5,
        )
        developed, history = await run_developer_agent(
            "implement",
            agent_deps,
            TestModel(
                call_tools=[],
                custom_output_args={
                    "summary": "done",
                    "validation": ["pytest"],
                    "ready_for_review": True,
                },
            ),
            5,
            [],
        )
        reviewed = await run_reviewer_agent(
            "review",
            agent_deps,
            TestModel(call_tools=[], custom_output_args={"approved": True, "issues": []}),
            5,
        )
        assert planned.complete
        assert developed.summary == "done"
        assert history
        assert reviewed.approved

    asyncio.run(scenario())
