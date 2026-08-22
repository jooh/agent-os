from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_os.models import (
    CancellationFinalizerInput,
    DeveloperCommand,
    DeveloperTurnResult,
    ImplementationTask,
    PlanComparison,
    ReviewIssue,
    ReviewResult,
    RunCreate,
    RunInput,
)


def test_plan_comparison_requires_tasks_exactly_when_incomplete() -> None:
    task = ImplementationTask(
        id="retry-handling",
        description="Implement retry handling",
        acceptance_criteria=["Transient failures are retried"],
    )
    assert PlanComparison(complete=False, tasks=[task]).tasks == [task]
    assert PlanComparison(complete=True).tasks == []

    with pytest.raises(ValidationError):
        PlanComparison(complete=True, tasks=[task])
    with pytest.raises(ValidationError):
        PlanComparison(complete=False)
    with pytest.raises(ValidationError):
        PlanComparison(complete=False, tasks=[task, task, task])


def test_review_result_keeps_approval_and_issues_consistent() -> None:
    issue = ReviewIssue(severity="P1", description="Unsafe retry")
    assert ReviewResult(approved=False, issues=[issue]).issues == [issue]
    assert ReviewResult(approved=True).issues == []

    with pytest.raises(ValidationError):
        ReviewResult(approved=True, issues=[issue])
    with pytest.raises(ValidationError):
        ReviewResult(approved=False)


def test_strict_boundary_models_reject_extra_or_coerced_values(tmp_path: Path) -> None:
    request = RunCreate(
        repository_path=str(tmp_path.resolve()), plan_path="docs/PLAN.md"
    )
    assert request.plan_path == "docs/PLAN.md"
    assert request.base_ref == "HEAD"
    assert DeveloperTurnResult(
        summary="done", validation=["pytest"], ready_for_review=True
    ).ready_for_review

    with pytest.raises(ValidationError):
        RunCreate(repository_path="relative")
    with pytest.raises(ValidationError):
        RunCreate(repository_path=str(tmp_path.resolve()), plan_path="../PLAN.md")
    with pytest.raises(ValidationError):
        DeveloperTurnResult(summary="done", ready_for_review=1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError, match="fix commands require"):
        DeveloperCommand(action="fix", prompt="missing paths")
    assert DeveloperCommand(action="close").action == "close"


def test_run_input_constrains_planner_task_limit(tmp_path: Path) -> None:
    values = {
        "run_id": "run-1",
        "workflow_id": "engineering-run:run-1",
        "target_id": "a" * 64,
        "repository_path": str(tmp_path),
        "plan_path": "PLAN.md",
        "plan_id": "b" * 64,
        "base_commit": "c" * 40,
        "integration_branch": "agent/bbbbbbbbbbbb-cccccccccccc/integration",
        "database_url": f"sqlite:///{tmp_path / 'state.db'}",
        "state_dir": str(tmp_path / "state"),
        "planner_model": "test:model",
        "developer_model": "test:model",
        "reviewer_model": "test:model",
        "max_rounds": 10,
        "max_review_cycles": 3,
        "max_developer_turns": 6,
        "model_request_limit": 50,
        "shell_timeout_seconds": 900,
    }

    assert RunInput.model_validate(values).planner_task_limit == 2
    assert RunInput.model_validate(values | {"planner_task_limit": 1}).planner_task_limit == 1
    for invalid_limit in (0, 3):
        with pytest.raises(ValidationError):
            RunInput.model_validate(values | {"planner_task_limit": invalid_limit})


def test_cancellation_finalizer_input_is_strict(tmp_path: Path) -> None:
    value = CancellationFinalizerInput(
        run_id="run-1",
        root_workflow_id="engineering-run:run-1",
        target_id="a" * 64,
        database_url=f"sqlite:///{tmp_path / 'state.db'}",
        state_dir=str(tmp_path / "state"),
    )

    assert value.finalizer_workflow_id == (
        "engineering-run:run-1:cancellation-finalizer"
    )
    with pytest.raises(ValidationError):
        CancellationFinalizerInput.model_validate(
            value.model_dump() | {"unexpected": True}
        )
