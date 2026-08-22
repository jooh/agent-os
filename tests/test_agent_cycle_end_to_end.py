import asyncio
import os
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pytest
from dbos import DBOS, DBOSConfig
from pydantic import JsonValue
from pydantic_ai import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.messages import ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from agent_os import agents as agent_module
from agent_os.agents import deserialize_history
from agent_os.config import Settings
from agent_os.git import GitRepository
from agent_os.models import ExecutionStatus, RunInput, RunStatus, TaskStatus
from agent_os.runtime import register_worker_queues
from agent_os.store import StateStore
from agent_os.workflows import engineering_run

Role = Literal["planner", "developer", "reviewer"]
Responder = Callable[[list[ModelMessage], AgentInfo], Awaitable[ModelResponse]]

PLAN = (
    "Implement parity.is_even(value) for every integer. It must return the correct boolean "
    "for positive, zero, and negative values.\n"
)
INITIAL_IMPLEMENTATION = """def is_even(value: int) -> bool:
    return False
"""
FLAWED_IMPLEMENTATION = """def is_even(value: int) -> bool:
    return value > 0 and value % 2 == 0
"""
CORRECT_IMPLEMENTATION = """def is_even(value: int) -> bool:
    return value % 2 == 0
"""


@dataclass(frozen=True, slots=True)
class RequestCapture:
    messages: list[ModelMessage]
    function_tools: frozenset[str]
    output_title: str

    @property
    def run_id(self) -> str:
        run_id = self.messages[-1].run_id
        assert run_id is not None
        return run_id


@dataclass(frozen=True, slots=True)
class ScriptedAgentModels:
    planner: FunctionModel
    developer: FunctionModel
    reviewer: FunctionModel
    requests: dict[Role, list[RequestCapture]]

    @property
    def registry(self) -> dict[str, Model]:
        return {
            "scripted:planner": self.planner,
            "scripted:developer": self.developer,
            "scripted:reviewer": self.reviewer,
        }


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "PLAN.md").write_text(PLAN)
    (root / "parity.py").write_text(INITIAL_IMPLEMENTATION)
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root


def make_input(
    repository: GitRepository, database_url: str, state_dir: Path, run_id: str
) -> RunInput:
    return RunInput(
        run_id=run_id,
        workflow_id=f"engineering-run:{run_id}",
        target_id=repository.target_id,
        repository_path=str(repository.root),
        plan_path=repository.plan_path,
        plan_id=repository.plan_id,
        base_commit=repository.base_commit,
        integration_branch=repository.integration_branch,
        database_url=database_url,
        state_dir=str(state_dir),
        planner_model="scripted:planner",
        developer_model="scripted:developer",
        reviewer_model="scripted:reviewer",
        max_rounds=3,
        max_review_cycles=2,
        max_developer_turns=2,
        model_request_limit=20,
        shell_timeout_seconds=30,
        planner_task_limit=1,
    )


def _latest_user_prompt(messages: list[ModelMessage]) -> tuple[int, str]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    assert isinstance(part.content, str)
                    return index, part.content
    raise AssertionError("model request did not contain a user prompt")


def _tool_calls(messages: list[ModelMessage]) -> list[ToolCallPart]:
    start, _prompt = _latest_user_prompt(messages)
    return [
        part
        for message in messages[start:]
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    start, _prompt = _latest_user_prompt(messages)
    return [
        part
        for message in messages[start:]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _response(tool_name: str, args: dict[str, object], tool_call_id: str) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name, args, tool_call_id=tool_call_id)]
    )


def _output(info: AgentInfo, args: dict[str, object], tool_call_id: str) -> ModelResponse:
    assert len(info.output_tools) == 1
    return _response(info.output_tools[0].name, args, tool_call_id)


def _capture(
    role: Role,
    messages: list[ModelMessage],
    info: AgentInfo,
    requests: dict[Role, list[RequestCapture]],
) -> str:
    assert len(info.output_tools) == 1
    title = info.output_tools[0].parameters_json_schema.get("title")
    assert isinstance(title, str)
    requests[role].append(
        RequestCapture(
            messages=deepcopy(messages),
            function_tools=frozenset(tool.name for tool in info.function_tools),
            output_title=title,
        )
    )
    return f"{role}-{len(requests[role])}"


def _function_model(role: Role, responder: Responder) -> FunctionModel:
    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        response = await responder(messages, info)
        assert len(response.parts) == 1
        part = response.parts[0]
        assert isinstance(part, ToolCallPart)
        yield {
            0: DeltaToolCall(
                name=part.tool_name,
                json_args=part.args_as_json_str(),
                tool_call_id=part.tool_call_id,
            )
        }

    return FunctionModel(responder, stream_function=stream, model_name=f"scripted-{role}")


@pytest.fixture()
def scripted_agent_models() -> ScriptedAgentModels:
    requests: dict[Role, list[RequestCapture]] = {
        "planner": [],
        "developer": [],
        "reviewer": [],
    }

    async def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call_id = _capture("planner", messages, info, requests)
        calls = _tool_calls(messages)
        if not calls:
            return _response("read_file", {"path": "PLAN.md"}, call_id)
        if len(calls) == 1:
            return _response("read_file", {"path": "parity.py"}, call_id)
        parity = _tool_returns(messages)[-1].content
        assert isinstance(parity, str)
        if parity == CORRECT_IMPLEMENTATION:
            return _output(info, {"complete": True, "tasks": []}, call_id)
        return _output(
            info,
            {
                "complete": False,
                "tasks": [
                    {
                        "id": "fix-is-even",
                        "description": "Implement is_even for all integers",
                        "acceptance_criteria": [
                            "Positive, zero, and negative integers return correct booleans"
                        ],
                    }
                ],
            },
            call_id,
        )

    async def developer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call_id = _capture("developer", messages, info, requests)
        calls = _tool_calls(messages)
        _start, prompt = _latest_user_prompt(messages)
        python = shlex.quote(sys.executable)
        if prompt.startswith("Implement task"):
            if not calls:
                return _response("read_file", {"path": "parity.py"}, call_id)
            if len(calls) == 1:
                return _response(
                    "write_file",
                    {"path": "parity.py", "content": FLAWED_IMPLEMENTATION},
                    call_id,
                )
            if len(calls) == 2:
                return _response(
                    "run_command",
                    {
                        "command": (
                            f"{python} -B -c 'from parity import is_even; "
                            "assert is_even(2); assert not is_even(3)'"
                        )
                    },
                    call_id,
                )
            return _output(
                info,
                {
                    "summary": "implemented positive parity handling",
                    "validation": ["checked positive even and odd inputs"],
                    "ready_for_review": True,
                },
                call_id,
            )
        if not calls:
            return _response(
                "replace_text",
                {
                    "path": "parity.py",
                    "old": "return value > 0 and value % 2 == 0",
                    "new": "return value % 2 == 0",
                },
                call_id,
            )
        if len(calls) == 1:
            return _response(
                "run_command",
                {
                    "command": (
                        f"{python} -B -c 'from parity import is_even; "
                        "assert is_even(2); assert not is_even(3); assert is_even(0); "
                        "assert is_even(-2); assert not is_even(-3)'"
                    )
                },
                call_id,
            )
        return _output(
            info,
            {
                "summary": "fixed parity handling for all integers",
                "validation": ["checked positive, zero, and negative inputs"],
                "ready_for_review": True,
            },
            call_id,
        )

    async def reviewer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call_id = _capture("reviewer", messages, info, requests)
        calls = _tool_calls(messages)
        if not calls:
            return _response("review_git_diff", {}, call_id)
        review = _tool_returns(messages)[-1].content
        assert isinstance(review, dict)
        patch = review["patch"]
        assert isinstance(patch, str)
        if "value > 0" in patch:
            return _output(
                info,
                {
                    "approved": False,
                    "issues": [
                        {
                            "severity": "P1",
                            "description": (
                                "Zero and negative even integers incorrectly return false"
                            ),
                            "file": "parity.py",
                            "line": 2,
                        }
                    ],
                },
                call_id,
            )
        return _output(info, {"approved": True, "issues": []}, call_id)

    return ScriptedAgentModels(
        planner=_function_model("planner", planner),
        developer=_function_model("developer", developer),
        reviewer=_function_model("reviewer", reviewer),
        requests=requests,
    )


def _captures_by_run(
    captures: list[RequestCapture],
) -> dict[str, list[RequestCapture]]:
    result: dict[str, list[RequestCapture]] = {}
    for capture in captures:
        result.setdefault(capture.run_id, []).append(capture)
    return result


def _captures_for_prompt(
    captures: list[RequestCapture], prefix: str
) -> list[RequestCapture]:
    return [
        capture
        for capture in captures
        if _latest_user_prompt(capture.messages)[1].startswith(prefix)
    ]


def _parts_before_latest_user_prompt(messages: list[ModelMessage]) -> list[object]:
    prompt_index, _prompt = _latest_user_prompt(messages)
    parts: list[object] = []
    for index, message in enumerate(messages):
        for part in message.parts:
            if index == prompt_index and isinstance(part, UserPromptPart):
                return parts
            parts.append(part)
    raise AssertionError("latest user prompt was not found")


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _head_refs(root: Path) -> dict[str, str]:
    lines = git(
        root,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/heads",
    ).splitlines()
    return dict(line.split(" ", 1) for line in lines)


def _worktree_paths(root: Path) -> set[Path]:
    return {
        Path(line.removeprefix("worktree ")).resolve()
        for line in git(root, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    }


def test_agent_cycle_uses_real_agents_tools_dbos_sqlite_and_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted_agent_models: ScriptedAgentModels,
) -> None:
    for name in tuple(os.environ):
        if name in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"} or name.startswith(
            ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    root = make_repository(tmp_path)
    state_dir = tmp_path / "state"
    repository = GitRepository.inspect(root, "PLAN.md", "HEAD", state_dir)
    database_url = f"sqlite:///{tmp_path / 'agent-os.sqlite3'}"
    first_input = make_input(repository, database_url, state_dir, "agent-cycle-1")
    monkeypatch.setattr(
        agent_module,
        "_MODEL_ID_OVERRIDES",
        scripted_agent_models.registry,
        raising=False,
    )
    config: DBOSConfig = {
        "name": "agent-os-test",
        "application_version": "0.1.0",
        "system_database_url": database_url,
        "run_admin_server": False,
    }
    DBOS.destroy()
    DBOS(config=config)
    with ExitStack() as cleanup:
        cleanup.callback(DBOS.destroy)
        DBOS.reset_system_database()
        DBOS.launch()
        settings = Settings.from_values_for_test(database_url, state_dir)
        register_worker_queues(settings)
        store = StateStore(database_url)
        cleanup.callback(store.engine.dispose)
        store.bootstrap()
        store.create_run(first_input, None, lambda _connection, _value: None)

        async def scenario() -> tuple[str, str]:
            first_head = await engineering_run(first_input)
            refs_after_first = _head_refs(root)
            commits_after_first = frozenset(git(root, "rev-list", "--all").splitlines())

            second_input = first_input.model_copy(
                update={
                    "run_id": "agent-cycle-2",
                    "workflow_id": "engineering-run:agent-cycle-2",
                }
            )
            store.create_run(second_input, None, lambda _connection, _value: None)
            second_head = await engineering_run(second_input)

            assert _head_refs(root) == refs_after_first
            assert frozenset(git(root, "rev-list", "--all").splitlines()) == (
                commits_after_first
            )
            return first_head, second_head

        first_head, second_head = asyncio.run(scenario())

        first_run = store.get_run("agent-cycle-1")
        assert first_run.status is RunStatus.COMPLETE
        assert first_run.integration_head == first_head == second_head
        assert len(first_run.tasks) == 1
        task = first_run.tasks[0]
        assert task.status is TaskStatus.INTEGRATED

        first_executions = store.list_executions("agent-cycle-1")
        assert len(first_executions) == 6
        assert all(
            execution.status is ExecutionStatus.SUCCEEDED
            for execution in first_executions
        )
        executions_by_role = {
            role: [item for item in first_executions if item.role == role]
            for role in cast(tuple[Role, ...], ("planner", "developer", "reviewer"))
        }
        assert {role: len(items) for role, items in executions_by_role.items()} == {
            "planner": 2,
            "developer": 2,
            "reviewer": 2,
        }
        assert [_object(item.final_output)["complete"] for item in executions_by_role["planner"]] == [
            False,
            True,
        ]
        assert [
            _object(item.final_output)["ready_for_review"]
            for item in executions_by_role["developer"]
        ] == [True, True]
        assert [_object(item.final_output)["approved"] for item in executions_by_role["reviewer"]] == [
            False,
            True,
        ]

        second_run = store.get_run("agent-cycle-2")
        assert second_run.status is RunStatus.COMPLETE
        assert second_run.tasks == []
        second_executions = store.list_executions("agent-cycle-2")
        assert len(second_executions) == 1
        assert second_executions[0].role == "planner"
        assert second_executions[0].status is ExecutionStatus.SUCCEEDED
        assert _object(second_executions[0].final_output)["complete"] is True

        expected_tools = {
            "planner": frozenset({"list_files", "read_file", "search_repo"}),
            "developer": frozenset(
                {
                    "list_files",
                    "read_file",
                    "replace_text",
                    "run_command",
                    "search_repo",
                    "write_file",
                }
            ),
            "reviewer": frozenset(
                {"list_files", "read_file", "review_git_diff", "search_repo"}
            ),
        }
        expected_outputs = {
            "planner": "PlanComparison",
            "developer": "DeveloperTurnResult",
            "reviewer": "ReviewResult",
        }
        for role, captures in scripted_agent_models.requests.items():
            assert captures
            assert all(item.function_tools == expected_tools[role] for item in captures)
            assert all(item.output_title == expected_outputs[role] for item in captures)

        planner_runs = _captures_by_run(scripted_agent_models.requests["planner"])
        reviewer_runs = _captures_by_run(scripted_agent_models.requests["reviewer"])
        for execution in [
            *executions_by_role["planner"],
            *executions_by_role["reviewer"],
            second_executions[0],
        ]:
            captures = (
                planner_runs[execution.id]
                if execution.role == "planner"
                else reviewer_runs[execution.id]
            )
            assert len(captures[0].messages) == 1
            assert isinstance(captures[0].messages[0], ModelRequest)

        developer_executions = executions_by_role["developer"]
        assert developer_executions[0].session_id is not None
        assert developer_executions[0].session_id == developer_executions[1].session_id
        session_id = developer_executions[0].session_id
        first_developer_captures = _captures_for_prompt(
            scripted_agent_models.requests["developer"], "Implement task"
        )
        second_developer_captures = _captures_for_prompt(
            scripted_agent_models.requests["developer"], "Address every"
        )
        assert len(first_developer_captures) == 4
        assert len(second_developer_captures) == 3
        first_developer_request = first_developer_captures[0]
        second_developer_request = second_developer_captures[0]
        assert len(first_developer_request.messages) == 1
        assert isinstance(first_developer_request.messages[0], ModelRequest)
        assert first_developer_request.messages[0].run_id == developer_executions[0].id
        assert any(
            message.run_id == developer_executions[0].id
            for message in second_developer_request.messages[:-1]
        )
        _feedback_index, feedback = _latest_user_prompt(second_developer_request.messages)
        assert "P1" in feedback
        assert "Zero and negative even integers incorrectly return false" in feedback
        stored_history = deserialize_history(store.load_history(session_id))
        assert {
            message.run_id for message in stored_history if message.run_id is not None
        } == {item.id for item in developer_executions}
        assert {message.conversation_id for message in stored_history} == {session_id}
        first_turn_parts = [
            part
            for message in stored_history
            if message.run_id == developer_executions[0].id
            for part in message.parts
        ]
        assert _parts_before_latest_user_prompt(second_developer_request.messages) == (
            first_turn_parts
        )

        task_head = git(root, "rev-parse", repository.task_branch(task.id))
        review_results: list[dict[str, object]] = []
        for execution in executions_by_role["reviewer"]:
            last_capture = reviewer_runs[execution.id][-1]
            result = _tool_returns(last_capture.messages)[-1].content
            assert isinstance(result, dict)
            review_results.append(result)
        assert [item["base"] for item in review_results] == [
            repository.base_commit,
            repository.base_commit,
        ]
        assert [item["changed_files"] for item in review_results] == [
            ["parity.py"],
            ["parity.py"],
        ]
        assert review_results[0]["head"] != review_results[1]["head"]
        assert review_results[1]["head"] == first_head
        first_review_head = review_results[0]["head"]
        assert isinstance(first_review_head, str)
        assert first_review_head == git(root, "rev-parse", f"{first_head}^")
        assert git(
            root, "rev-list", "--parents", "-n", "1", first_review_head
        ).split() == [first_review_head, repository.base_commit, task_head]

        events = [
            event
            for execution in [*first_executions, *second_executions]
            for event in store.list_events(execution.id)
        ]
        repository_tool_names = {
            "read_file",
            "write_file",
            "replace_text",
            "run_command",
            "review_git_diff",
        }
        calls: dict[str, dict[str, JsonValue]] = {}
        results: dict[str, dict[str, JsonValue]] = {}
        for event in events:
            if event.event_type not in {"tool.call", "tool.result"}:
                continue
            payload = _object(event.payload)
            part = payload.get("part")
            assert isinstance(part, dict)
            name = part.get("tool_name")
            call_id = part.get("tool_call_id")
            if name not in repository_tool_names:
                continue
            assert isinstance(call_id, str)
            target = calls if event.event_type == "tool.call" else results
            target[call_id] = part
        assert set(calls) == set(results)
        assert Counter(part["tool_name"] for part in calls.values()) == Counter(
            {
                "read_file": 7,
                "write_file": 1,
                "replace_text": 1,
                "run_command": 2,
                "review_git_diff": 2,
            }
        )
        assert all(part["outcome"] == "success" for part in results.values())
        command_results = [
            part for part in results.values() if part["tool_name"] == "run_command"
        ]
        assert all(_object(part["content"])["returncode"] == 0 for part in command_results)

        git(root, "merge-base", "--is-ancestor", task_head, first_head)
        assert git(root, "show", f"{repository.integration_branch}:parity.py") == (
            CORRECT_IMPLEMENTATION.strip()
        )
        assert git(root, "show", f"{repository.integration_branch}:PLAN.md") == PLAN.strip()
        assert git(root, "diff", f"{repository.base_commit}..{first_head}", "--", "PLAN.md") == ""
        assert git(root, "symbolic-ref", "--short", "HEAD") == "main"
        assert git(root, "rev-parse", "HEAD") == repository.base_commit
        assert git(root, "rev-parse", "main") == repository.base_commit
        assert (root / "parity.py").read_text() == INITIAL_IMPLEMENTATION
        assert (root / "PLAN.md").read_text() == PLAN
        assert git(root, "status", "--porcelain", "--untracked-files=normal") == ""
        assert _worktree_paths(root) == {root.resolve()}
