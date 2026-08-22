# Architecture

`agent-os` is a declarative reconciliation loop: the current integration branch is repeatedly
compared with a committed `PLAN.md`; tasks are implementation hints, while repository state and
the plan remain authoritative.

## Constraints

- The software and infrastructure stack must be open source. The POC does not use DBOS Conductor,
  Logfire, or another proprietary control plane. Model providers remain user-configurable.
- Development must remain laptop-friendly. SQLite is the default and only required durable store,
  so the POC has no database service or container dependency.
- Target repositories and plans are trusted. Worktree path checks prevent accidental traversal, but
  developer shell commands are not an operating-system security sandbox.

## Runtime

The loopback HTTP API uses `DBOSClient`; it never launches workflows in-process. Four worker roles
listen to independent database-backed queues: orchestrator, planner, developer, and reviewer. All
Git, filesystem, subprocess, database, and model I/O executed by workflows is isolated in DBOS
steps. One SQLite file stores both DBOS history and the application-owned run, session, execution,
and transcript tables.

Each target is identified by canonical repository path, the SHA-256 of the committed plan, and the
captured base commit. Results stop on `agent/<plan>-<base>/integration`; the caller's worktree and
source branch are never modified. Task and staging branches are sibling leaves so their Git refs can
coexist with the integration branch.

Developer workflows retain Pydantic AI message history as validated JSON and wait for
review/conflict messages. Planner and reviewer executions are fresh. Structured events are appended
to SQLite, printed as JSON Lines, and exposed through paginated JSON and resumable SSE.

## Recovery and deployment boundaries

DBOS application version `0.1.0` is the compatibility boundary for stored workflow history and may
be overridden only for an intentionally incompatible deployment. Workers register every queue but
listen only to their selected role, so a stopped process can be replaced without losing a developer
conversation. Successful turns checkpoint full Pydantic AI history before review; planner and
reviewer histories are deliberately discarded.

Failure and cancellation release the active-target database lock but retain diagnostic Git state.
Successful completion verifies and reports the integration head, then removes only reconstructable
worktrees. A later run against the same target performs a fresh comparison and can converge without
creating tasks, commits, or branch movements.

SQLite is a deliberate single-host POC boundary. The API and every role worker must resolve the same
absolute database path; SQLite serializes writes and DBOS polls instead of using PostgreSQL
`LISTEN/NOTIFY`. PostgreSQL remains the migration target when workers need to span hosts or the
write workload outgrows this low-concurrency design.
