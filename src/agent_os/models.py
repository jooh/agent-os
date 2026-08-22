import os
from datetime import datetime
from enum import StrEnum
from pathlib import PurePath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ImplementationTask(StrictModel):
    id: Annotated[str, Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")]
    description: NonEmptyString
    acceptance_criteria: Annotated[list[NonEmptyString], Field(min_length=1)]


class PlanComparison(StrictModel):
    complete: bool
    tasks: Annotated[list[ImplementationTask], Field(max_length=2)] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_completion(self) -> PlanComparison:
        if self.complete == bool(self.tasks):
            raise ValueError("complete must be true exactly when tasks is empty")
        return self


class DeveloperTurnResult(StrictModel):
    summary: NonEmptyString
    validation: list[NonEmptyString] = Field(default_factory=list)
    ready_for_review: bool


class ReviewIssue(StrictModel):
    severity: Literal["P0", "P1", "P2", "P3"]
    description: NonEmptyString
    file: str | None = None
    line: Annotated[int, Field(gt=0)] | None = None


class ReviewResult(StrictModel):
    approved: bool
    issues: list[ReviewIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_approval(self) -> ReviewResult:
        if self.approved == bool(self.issues):
            raise ValueError("approved must be true exactly when issues is empty")
        return self


class RunCreate(StrictModel):
    repository_path: NonEmptyString
    plan_path: NonEmptyString = "PLAN.md"
    base_ref: NonEmptyString = "HEAD"

    @field_validator("repository_path")
    @classmethod
    def repository_must_be_absolute(cls, value: str) -> str:
        if not os.path.isabs(value):
            raise ValueError("repository_path must be absolute")
        return value

    @field_validator("plan_path")
    @classmethod
    def plan_must_be_relative(cls, value: str) -> str:
        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("plan_path must stay within the repository")
        return value


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    DEVELOPING = "developing"
    STAGING = "staging"
    REVIEWING = "reviewing"
    FIXING = "fixing"
    INTEGRATED = "integrated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskView(StrictModel):
    id: str
    run_id: str
    round_number: int
    ordinal: int
    description: str
    acceptance_criteria: list[str]
    status: TaskStatus
    developer_workflow_id: str | None = None


class RunView(StrictModel):
    id: str
    target_id: str
    repository_path: str
    plan_path: str
    plan_id: str
    base_commit: str
    integration_branch: str
    integration_head: str | None
    status: RunStatus
    current_round: int
    failure_reason: str | None
    workflow_id: str
    created_at: datetime
    updated_at: datetime
    tasks: list[TaskView] = Field(default_factory=list)


class AgentExecutionView(StrictModel):
    id: str
    run_id: str
    role: Literal["planner", "developer", "reviewer"]
    workflow_id: str
    session_id: str | None
    task_id: str | None
    status: ExecutionStatus
    final_output: JsonValue = None
    started_at: datetime
    completed_at: datetime | None


class TranscriptEvent(StrictModel):
    execution_id: str
    sequence: Annotated[int, Field(gt=0)]
    event_key: NonEmptyString | None = None
    event_type: NonEmptyString
    payload: JsonValue
    created_at: datetime


class RunInput(StrictModel):
    run_id: str
    workflow_id: str
    target_id: str
    repository_path: str
    plan_path: str
    plan_id: str
    base_commit: str
    integration_branch: str
    database_url: str
    state_dir: str
    planner_model: str
    developer_model: str
    reviewer_model: str
    max_rounds: Annotated[int, Field(gt=0)]
    max_review_cycles: Annotated[int, Field(gt=0)]
    max_developer_turns: Annotated[int, Field(gt=0)]
    model_request_limit: Annotated[int, Field(gt=0)]
    shell_timeout_seconds: Annotated[int, Field(gt=0)]
    planner_task_limit: Annotated[int, Field(ge=1, le=2)] = 2


class CancellationFinalizerInput(StrictModel):
    run_id: NonEmptyString
    root_workflow_id: NonEmptyString
    target_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    database_url: NonEmptyString
    state_dir: NonEmptyString

    @property
    def finalizer_workflow_id(self) -> str:
        return f"{self.root_workflow_id}:cancellation-finalizer"


class StageResult(StrictModel):
    path: str
    head: str | None
    conflicts: list[str] = Field(default_factory=list)


class Candidate(StrictModel):
    turn: Annotated[int, Field(gt=0)]
    worktree: str
    head: str


class DeveloperSessionInput(StrictModel):
    run: RunInput
    task_id: str
    task: ImplementationTask
    start_commit: str
    developer_workflow_id: str


class DeveloperCommand(StrictModel):
    action: Literal["fix", "close"]
    prompt: str | None = None
    worktree: str | None = None
    protected_base: str | None = None

    @model_validator(mode="after")
    def validate_fix(self) -> DeveloperCommand:
        if self.action == "fix" and not all(
            (self.prompt, self.worktree, self.protected_base)
        ):
            raise ValueError("fix commands require prompt, worktree, and protected_base")
        return self


class ReviewInput(StrictModel):
    run: RunInput
    task_id: str
    worktree: str
    base_commit: str
    review_cycle: Annotated[int, Field(gt=0)]
    reviewer_workflow_id: str
