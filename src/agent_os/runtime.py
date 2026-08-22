import os
import socket
import threading
from typing import Literal

from dbos import DBOS, DBOSClient, DBOSConfig
from sqlalchemy.engine import Connection

from agent_os.config import Settings
from agent_os.models import CancellationFinalizerInput, RunInput

ORCHESTRATOR_QUEUE = "agent_os.orchestrator"
PLANNER_QUEUE = "agent_os.planner"
DEVELOPER_QUEUE = "agent_os.developer"
REVIEWER_QUEUE = "agent_os.reviewer"
QUEUES = (ORCHESTRATOR_QUEUE, PLANNER_QUEUE, DEVELOPER_QUEUE, REVIEWER_QUEUE)
Role = Literal["orchestrator", "planner", "developer", "reviewer"]
TERMINAL_DBOS_STATUSES = frozenset(
    {"SUCCESS", "ERROR", "CANCELLED", "MAX_RECOVERY_ATTEMPTS_EXCEEDED"}
)


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

    def enqueue_cancellation_finalizer(
        self,
        connection: Connection,
        finalizer_input: CancellationFinalizerInput,
    ) -> None:
        client = DBOSClient(system_database_url=self.database_url)
        try:
            client.enqueue_in_transaction(
                connection,
                {
                    "workflow_name": "agent_os.cancellation_finalizer",
                    "queue_name": ORCHESTRATOR_QUEUE,
                    "workflow_id": finalizer_input.finalizer_workflow_id,
                },
                finalizer_input,
            )
        finally:
            client.destroy()

    def cancel(self, workflow_id: str) -> None:
        client = DBOSClient(system_database_url=self.database_url)
        try:
            client.cancel_workflow(workflow_id, cancel_children=True)
        finally:
            client.destroy()

    def cancellation_confirmed(self, workflow_id: str) -> bool:
        """Return whether DBOS logically terminalized the run and its known children.

        DBOS may mark a cancelled workflow terminal while a non-preemptible step is
        still unwinding in its worker process. This confirms durable DBOS status,
        not physical process quiescence.
        """
        client = DBOSClient(system_database_url=self.database_url)
        try:
            offset = 0
            found = False
            finalizer_prefix = f"{workflow_id}:cancellation-finalizer"
            while True:
                page = client.list_workflows(
                    workflow_id_prefix=workflow_id,
                    load_input=False,
                    load_output=False,
                    limit=100,
                    offset=offset,
                )
                relevant = [
                    item
                    for item in page
                    if (
                        item.workflow_id == workflow_id
                        or item.workflow_id.startswith(f"{workflow_id}:")
                    )
                    and not item.workflow_id.startswith(finalizer_prefix)
                ]
                found = found or bool(relevant)
                if any(item.status not in TERMINAL_DBOS_STATUSES for item in relevant):
                    return False
                if len(page) < 100:
                    return found
                offset += len(page)
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
