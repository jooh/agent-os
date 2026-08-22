import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.engine import Connection

from agent_os.config import Settings
from agent_os.git import GitError, GitRepository
from agent_os.models import (
    AgentExecutionView,
    RunCreate,
    RunInput,
    RunStatus,
    RunView,
    TranscriptEvent,
)
from agent_os.store import ActiveRunError, StateStore


class Enqueuer(Protocol):
    def enqueue(self, connection: Connection, run_input: RunInput, /) -> None: ...

    def cancel(self, workflow_id: str) -> None: ...


class RunCoordinator:
    def __init__(self, settings: Settings, store: StateStore, enqueuer: Enqueuer):
        self.settings = settings
        self.store = store
        self.enqueuer = enqueuer

    def start_run(
        self,
        *,
        repository_path: str,
        plan_path: str,
        base_ref: str,
        idempotency_key: str | None,
    ) -> RunView:
        repository = GitRepository.inspect(
            Path(repository_path), plan_path, base_ref, self.settings.state_dir
        )
        planner_model = self.settings.planner_model
        developer_model = self.settings.developer_model
        reviewer_model = self.settings.reviewer_model
        if not planner_model or not developer_model or not reviewer_model:
            raise ValueError("models are required to start a run")
        run_id = str(uuid.uuid7())
        workflow_id = f"engineering-run:{run_id}"
        run_input = RunInput(
            run_id=run_id,
            workflow_id=workflow_id,
            target_id=repository.target_id,
            repository_path=str(repository.root),
            plan_path=repository.plan_path,
            plan_id=repository.plan_id,
            base_commit=repository.base_commit,
            integration_branch=repository.integration_branch,
            database_url=self.settings.database_url,
            state_dir=str(self.settings.state_dir),
            planner_model=planner_model,
            developer_model=developer_model,
            reviewer_model=reviewer_model,
            max_rounds=self.settings.max_rounds,
            max_review_cycles=self.settings.max_review_cycles,
            max_developer_turns=self.settings.max_developer_turns,
            model_request_limit=self.settings.model_request_limit,
            shell_timeout_seconds=self.settings.shell_timeout_seconds,
        )
        return self.store.create_run(
            run_input, idempotency_key, self.enqueuer.enqueue
        )[0]

    def cancel_run(self, run_id: str) -> RunView:
        run = self.store.get_run(run_id)
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
            self.enqueuer.cancel(run.workflow_id)
            return self.store.update_run(run_id, status=RunStatus.CANCELLED)
        return run


def _not_found(identifier: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"not found: {identifier}")


def create_app(
    settings: Settings,
    *,
    store: StateStore,
    coordinator: RunCoordinator,
) -> FastAPI:
    app = FastAPI(title="agent-os", version="0.1.0")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        store.health()
        return {"status": "ok"}

    @app.post("/v1/runs", response_model=RunView, status_code=status.HTTP_202_ACCEPTED)
    def start_run(
        request: RunCreate,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> RunView:
        try:
            return coordinator.start_run(
                repository_path=request.repository_path,
                plan_path=request.plan_path,
                base_ref=request.base_ref,
                idempotency_key=idempotency_key,
            )
        except ActiveRunError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except (GitError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
            ) from exc

    @app.get("/v1/runs/{run_id}", response_model=RunView)
    def get_run(run_id: str) -> RunView:
        try:
            return store.get_run(run_id)
        except KeyError as exc:
            raise _not_found(run_id) from exc

    @app.post("/v1/runs/{run_id}/cancel", response_model=RunView, status_code=202)
    def cancel_run(run_id: str) -> RunView:
        try:
            return coordinator.cancel_run(run_id)
        except KeyError as exc:
            raise _not_found(run_id) from exc

    @app.get("/v1/runs/{run_id}/executions", response_model=list[AgentExecutionView])
    def list_executions(run_id: str) -> list[AgentExecutionView]:
        try:
            store.get_run(run_id)
        except KeyError as exc:
            raise _not_found(run_id) from exc
        return store.list_executions(run_id)

    @app.get(
        "/v1/executions/{execution_id}/events", response_model=list[TranscriptEvent]
    )
    def list_events(
        execution_id: str,
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[TranscriptEvent]:
        try:
            store.get_execution(execution_id)
        except KeyError as exc:
            raise _not_found(execution_id) from exc
        return store.list_events(execution_id, after_sequence=after_sequence, limit=limit)

    @app.get("/v1/executions/{execution_id}/events/stream")
    def stream_events(
        execution_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            store.get_execution(execution_id)
        except KeyError as exc:
            raise _not_found(execution_id) from exc
        try:
            cursor = int(last_event_id or "0")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer") from exc

        async def generate() -> AsyncIterator[str]:
            nonlocal cursor
            while not await request.is_disconnected():
                page = store.list_events(execution_id, after_sequence=cursor, limit=100)
                for event in page:
                    cursor = event.sequence
                    data = json.dumps(event.payload, separators=(",", ":"), sort_keys=True)
                    yield f"id: {event.sequence}\nevent: {event.event_type}\ndata: {data}\n\n"
                if store.execution_terminal(execution_id) and not store.list_events(
                    execution_id, after_sequence=cursor, limit=1
                ):
                    return
                if not page:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app
