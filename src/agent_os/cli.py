import argparse
from collections.abc import Sequence
from typing import cast

import uvicorn
from dbos.cli.migration import run_dbos_database_migrations

from agent_os.api import RunCoordinator, create_app
from agent_os.config import Settings
from agent_os.runtime import DBOSEnqueuer, Role, run_worker
from agent_os.store import StateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-os")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("api", help="start the loopback HTTP API")
    worker = subparsers.add_parser("worker", help="start one DBOS queue worker")
    worker.add_argument(
        "--role",
        required=True,
        choices=("orchestrator", "planner", "developer", "reviewer"),
    )
    return parser


def run_api(settings: Settings) -> None:  # pragma: no cover - process boundary
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    run_dbos_database_migrations(settings.database_url)
    store = StateStore(settings.database_url)
    store.bootstrap()
    enqueuer = DBOSEnqueuer(settings.database_url)
    enqueuer.register_queues(settings.developer_parallelism)
    coordinator = RunCoordinator(settings, store, enqueuer)
    app = create_app(settings, store=store, coordinator=coordinator)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    settings = Settings.from_env(require_model=True)
    if arguments.command == "api":
        run_api(settings)
    else:
        run_worker(settings, cast(Role, arguments.role))
    return 0
