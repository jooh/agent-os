# Deterministic Agent-Cycle End-to-End Test

Status: implemented by `tests/test_agent_cycle_end_to_end.py`.

## Goal

Add one pytest test that completes a full reconciliation cycle through the real Pydantic AI agents,
repository tools, DBOS workflows, SQLite persistence, and Git integration without calling an
external model provider.

Implemented test: `tests/test_agent_cycle_end_to_end.py`.

## Mocking boundary

Mock only the LLM model by injecting role-specific Pydantic AI `FunctionModel` instances. Do not
mock `run_planner_agent`, `run_developer_agent`, `run_reviewer_agent`, repository tools, DBOS steps,
the state store, or Git operations.

Provide a pytest fixture such as `scripted_agent_models` that returns planner, developer, and
reviewer models plus captured model requests. Use a narrow model-resolution injection seam if the
current code cannot accept model objects in the in-process test.

Do not add `pydantic-ai-harness`; core Pydantic AI `FunctionModel` is sufficient and preserves the
POC's DBOS-checkpointed custom-tool boundary.

## Scenario

1. Create a temporary clean Git repository containing a committed `PLAN.md` and an incomplete file.
2. Start DBOS against a temporary SQLite database and run `engineering_run`.
3. The first planner execution returns one structured implementation task.
4. The developer model calls the real repository tools to edit the file and validate the change,
   then returns `DeveloperTurnResult(ready_for_review=True)`.
5. The first fresh reviewer returns one structured P1 issue.
6. Review feedback resumes the same developer conversation; the developer calls the real tools to
   fix the issue and validates again.
7. A fresh reviewer approves the corrected staging diff.
8. The integration branch advances, and a fresh planner reports the plan complete.
9. Run reconciliation again and verify that it creates no tasks, commits, or branch movement.

## Required assertions

- The actual planner, developer, and reviewer `Agent` objects and their structured output types ran.
- Real tool-call and tool-result transcript events were persisted.
- The two developer turns share one session/conversation and the second model request contains the
  checkpointed first-turn history plus review feedback.
- Planner and reviewer executions each start with fresh histories.
- Review covers the intended integration-to-staging range.
- The task commit is reachable from the final integration head.
- The source branch and caller worktree remain unchanged, and `PLAN.md` was not modified.
- The run finishes successfully with concrete execution, task, history, and transcript records.
- No provider credentials, network calls, Logfire, or proprietary infrastructure are used.

## Test separation

Keep this as an in-process deterministic agent test because Python `FunctionModel` objects are test
dependencies, not cross-process configuration. The current in-process workflow-cycle test mocks the
agent runner functions, so it does not satisfy this specification. Retain the existing five-process
SQLite smoke as the complementary process/API/queue-boundary check; it uses the built-in no-op test
model and does not execute a developer/reviewer correction cycle.

## Acceptance

The new test passes repeatedly on its own and under `make all-tests`, while preserving 100% configured
coverage, type checking, and Ruff compliance.
