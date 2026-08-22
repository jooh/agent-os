import pytest

from agent_os.config import Settings


def test_settings_load_models_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBOS_SYSTEM_DATABASE_URL", "postgresql+psycopg://localhost/agent_os")
    monkeypatch.setenv("AGENT_OS_MODEL", "openai:gpt-5.2")
    monkeypatch.setenv("AGENT_OS_PLANNER_MODEL", "anthropic:claude-sonnet-4-6")
    monkeypatch.setenv("AGENT_OS_MAX_ROUNDS", "4")

    settings = Settings.from_env(require_model=True)

    assert settings.planner_model == "anthropic:claude-sonnet-4-6"
    assert settings.developer_model == "openai:gpt-5.2"
    assert settings.max_rounds == 4
    assert settings.developer_parallelism == 2
    assert settings.api_host == "127.0.0.1"


def test_settings_default_to_sqlite_and_optionally_require_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)
    monkeypatch.delenv("AGENT_OS_MODEL", raising=False)
    monkeypatch.setenv("AGENT_OS_STATE_DIR", str(tmp_path))
    settings = Settings.from_env()
    assert settings.database_url == f"sqlite:///{tmp_path / 'agent-os.sqlite3'}"

    monkeypatch.setenv("DBOS_SYSTEM_DATABASE_URL", "sqlite:///state.db")
    assert Settings.from_env(require_model=False).model is None
    with pytest.raises(ValueError, match="AGENT_OS_MODEL"):
        Settings.from_env(require_model=True)

    monkeypatch.setenv("AGENT_OS_MAX_ROUNDS", "0")
    with pytest.raises(ValueError, match="positive"):
        Settings.from_env()

    monkeypatch.setenv("AGENT_OS_MAX_ROUNDS", "10")
    monkeypatch.setenv("AGENT_OS_API_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        Settings.from_env()
