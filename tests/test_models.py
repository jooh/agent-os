from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_os.models import (
    DeveloperCommand,
    DeveloperTurnResult,
    ImplementationTask,
    PlanComparison,
    ReviewIssue,
    ReviewResult,
    RunCreate,
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
