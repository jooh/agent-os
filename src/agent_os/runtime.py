import os
import socket
import threading
from typing import Literal

from dbos import DBOS, DBOSClient, DBOSConfig
from sqlalchemy.engine import Connection

from agent_os.config import Settings
from agent_os.models import RunInput

ORCHESTRATOR_QUEUE = "agent_os.orchestrator"
PLANNER_QUEUE = "agent_os.planner"
DEVELOPER_QUEUE = "agent_os.developer"
REVIEWER_QUEUE = "agent_os.reviewer"
QUEUES = (ORCHESTRATOR_QUEUE, PLANNER_QUEUE, DEVELOPER_QUEUE, REVIEWER_QUEUE)
Role = Literal["orchestrator", "planner", "developer", "reviewer"]


class DBOSEnqueuer:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def enqueue(self, connection: Connection, run_input: RunInput) -> None:
        client = DBOSClient(system_database_url=self.database_url)
        try:
            client.enqueue_in_transaction(
                connection,
                {
                    "workflow_name": "agent_os.engineering_run",
                    "queue_name": ORCHESTRATOR_QUEUE,
                    "workflow_id": run_input.workflow_id,
                },
                run_input,
            )
        finally:
            client.destroy()

    def cancel(self, workflow_id: str) -> None:
        client = DBOSClient(system_database_url=self.database_url)
        try:
            client.cancel_workflow(workflow_id, cancel_children=True)
        finally:
            client.destroy()

    def register_queues(self, developer_parallelism: int) -> None:
        client = DBOSClient(system_database_url=self.database_url)
        try:
            client.register_queue(ORCHESTRATOR_QUEUE, concurrency=1, application_name="agent-os")
            client.register_queue(PLANNER_QUEUE, worker_concurrency=2, application_name="agent-os")
            client.register_queue(
                DEVELOPER_QUEUE,
                concurrency=developer_parallelism,
                application_name="agent-os",
            )
            client.register_queue(REVIEWER_QUEUE, worker_concurrency=2, application_name="agent-os")
        finally:
            client.destroy()


def register_worker_queues(settings: Settings) -> None:
    DBOS.register_queue(ORCHESTRATOR_QUEUE, concurrency=1)
    DBOS.register_queue(PLANNER_QUEUE, worker_concurrency=2)
    DBOS.register_queue(DEVELOPER_QUEUE, concurrency=settings.developer_parallelism)
    DBOS.register_queue(REVIEWER_QUEUE, worker_concurrency=2)


def run_worker(settings: Settings, role: Role) -> None:  # pragma: no cover - process boundary
    from agent_os import workflows as _workflows  # noqa: F401

    settings.state_dir.mkdir(parents=True, exist_ok=True)
    queue_by_role = {
        "orchestrator": ORCHESTRATOR_QUEUE,
        "planner": PLANNER_QUEUE,
        "developer": DEVELOPER_QUEUE,
        "reviewer": REVIEWER_QUEUE,
    }
    config: DBOSConfig = {
        "name": "agent-os",
        "application_version": settings.application_version,
        "system_database_url": settings.database_url,
        "executor_id": f"{role}-{socket.gethostname()}-{os.getpid()}",
        "run_admin_server": False,
    }
    DBOS(config=config)
    DBOS.listen_queues([queue_by_role[role]])
    DBOS.launch()
    register_worker_queues(settings)
    try:
        threading.Event().wait()
    finally:
        DBOS.destroy()
