import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    model: str | None
    planner_model_override: str | None
    developer_model_override: str | None
    reviewer_model_override: str | None
    state_dir: Path
    application_version: str
    developer_parallelism: int
    planner_task_limit: int
    max_rounds: int
    max_review_cycles: int
    max_developer_turns: int
    model_request_limit: int
    shell_timeout_seconds: int
    api_host: str
    api_port: int

    @property
    def planner_model(self) -> str | None:
        return self.planner_model_override or self.model

    @property
    def developer_model(self) -> str | None:
        return self.developer_model_override or self.model

    @property
    def reviewer_model(self) -> str | None:
        return self.reviewer_model_override or self.model

    @classmethod
    def from_values_for_test(cls, database_url: str, state_dir: Path) -> Settings:
        return cls(
            database_url=database_url,
            model="test:model",
            planner_model_override=None,
            developer_model_override=None,
            reviewer_model_override=None,
            state_dir=state_dir,
            application_version="0.1.0",
            developer_parallelism=2,
            planner_task_limit=2,
            max_rounds=10,
            max_review_cycles=3,
            max_developer_turns=6,
            model_request_limit=50,
            shell_timeout_seconds=900,
            api_host="127.0.0.1",
            api_port=8000,
        )

    @classmethod
    def from_env(cls, *, require_model: bool = False) -> Settings:
        default_state = Path.home() / ".local" / "state" / "agent-os"
        state_dir = Path(
            os.environ.get("AGENT_OS_STATE_DIR", default_state)
        ).expanduser().resolve()
        database_url = os.environ.get("DBOS_SYSTEM_DATABASE_URL") or (
            f"sqlite:///{state_dir / 'agent-os.sqlite3'}"
        )
        model = os.environ.get("AGENT_OS_MODEL")
        overrides = (
            os.environ.get("AGENT_OS_PLANNER_MODEL"),
            os.environ.get("AGENT_OS_DEVELOPER_MODEL"),
            os.environ.get("AGENT_OS_REVIEWER_MODEL"),
        )
        if require_model and not model:
            raise ValueError("AGENT_OS_MODEL is required")
        api_host = os.environ.get("AGENT_OS_API_HOST", "127.0.0.1")
        if api_host != "127.0.0.1":
            raise ValueError("AGENT_OS_API_HOST must be 127.0.0.1")
        return cls(
            database_url=database_url,
            model=model,
            planner_model_override=overrides[0],
            developer_model_override=overrides[1],
            reviewer_model_override=overrides[2],
            state_dir=state_dir,
            application_version=os.environ.get("AGENT_OS_APPLICATION_VERSION", "0.1.0"),
            developer_parallelism=_positive_int("AGENT_OS_DEVELOPER_PARALLELISM", 2),
            planner_task_limit=_bounded_int(
                "AGENT_OS_PLANNER_TASK_LIMIT", 2, minimum=1, maximum=2
            ),
            max_rounds=_positive_int("AGENT_OS_MAX_ROUNDS", 10),
            max_review_cycles=_positive_int("AGENT_OS_MAX_REVIEW_CYCLES", 3),
            max_developer_turns=_positive_int("AGENT_OS_MAX_DEVELOPER_TURNS", 6),
            model_request_limit=_positive_int("AGENT_OS_MODEL_REQUEST_LIMIT", 50),
            shell_timeout_seconds=_positive_int("AGENT_OS_SHELL_TIMEOUT_SECONDS", 900),
            api_host=api_host,
            api_port=_positive_int("AGENT_OS_API_PORT", 8000),
        )
