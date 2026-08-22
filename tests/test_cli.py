import runpy
from pathlib import Path

from agent_os.cli import build_parser, main
from agent_os.config import Settings


def test_parser_accepts_api_and_role_workers() -> None:
    parser = build_parser()
    assert parser.parse_args(["api"]).command == "api"
    worker = parser.parse_args(["worker", "--role", "developer"])
    assert worker.command == "worker"
    assert worker.role == "developer"


def test_main_dispatches_worker(monkeypatch) -> None:
    settings = object()
    called: list[tuple[object, str]] = []
    monkeypatch.setattr(
        "agent_os.cli.Settings.from_env",
        lambda *, require_model: settings if require_model else None,
    )
    monkeypatch.setattr(
        "agent_os.cli.run_worker", lambda value, role: called.append((value, role))
    )
    assert main(["worker", "--role", "planner"]) == 0
    assert called == [(settings, "planner")]


def test_main_dispatches_api(monkeypatch) -> None:
    settings = Settings.from_values_for_test("sqlite:///state.db", Path("state"))
    called: list[Settings] = []
    monkeypatch.setattr("agent_os.cli.Settings.from_env", lambda *, require_model: settings)
    monkeypatch.setattr("agent_os.cli.run_api", called.append)
    assert main(["api"]) == 0
    assert called == [settings]


def test_parser_requires_a_command() -> None:
    parser = build_parser()
    try:
        parser.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("parser unexpectedly accepted no command")


def test_module_entrypoint_invokes_cli(monkeypatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("agent_os.cli.main", lambda: called.append(True))
    try:
        runpy.run_module("agent_os.__main__", run_name="__main__")
    except SystemExit as exc:
        assert exc.code is None
    assert called == [True]
