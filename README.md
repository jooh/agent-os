# agent-os

[![CI](https://github.com/jooh/agent-os/actions/workflows/ci.yml/badge.svg)](https://github.com/jooh/agent-os/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- CHANGE ME: Uncomment after setting up codecov and add your token -->
<!-- [![codecov](https://codecov.io/gh/jooh/agent-os/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/jooh/agent-os) -->
<!-- CHANGE ME: Uncomment after first PyPI publish -->
<!-- [![PyPI](https://img.shields.io/pypi/v/agent-os.svg)](https://pypi.org/project/agent-os/) -->

Deterministic loop engineering with DBOS and Pydantic AI. A committed `PLAN.md` is compared with a
dedicated integration branch until the repository converges, with isolated developer worktrees,
fresh technical reviews, durable conversations, and inspectable transcripts.

## Installation

```bash
pip install agent-os
```

Or with uv:

```bash
uv add agent-os
```

## Local POC

The target repository and plan are trusted inputs. `agent-os` confines file tools to worktrees, but
developer shell commands are intentionally not an OS security sandbox.

```bash
cp .env.example .env
set -a; source .env; set +a
uv sync --dev
```

No database server or container is required. Unless `DBOS_SYSTEM_DATABASE_URL` is explicitly set,
the API and workers share `~/.local/state/agent-os/agent-os.sqlite3`.

Start these commands in separate tmux panes:

```bash
uv run agent-os api
uv run agent-os worker --role orchestrator
uv run agent-os worker --role planner
uv run agent-os worker --role developer
uv run agent-os worker --role reviewer
```

Start and inspect a run:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/runs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: my-first-run' \
  -d '{"repository_path":"/absolute/path/to/clean/repository"}'

curl -sS http://127.0.0.1:8000/v1/runs/RUN_ID
curl -sS http://127.0.0.1:8000/v1/runs/RUN_ID/executions
curl -sS 'http://127.0.0.1:8000/v1/executions/EXECUTION_ID/events?after_sequence=0&limit=100'
curl -N http://127.0.0.1:8000/v1/executions/EXECUTION_ID/events/stream
curl -N -H 'Last-Event-ID: 42' \
  http://127.0.0.1:8000/v1/executions/EXECUTION_ID/events/stream
curl -sS -X POST http://127.0.0.1:8000/v1/runs/RUN_ID/cancel
```

The repository must be clean and `PLAN.md` must exist in the captured base commit. Completion leaves
the result on the reported integration branch without changing the source branch.

Role-specific model variables may override `AGENT_OS_MODEL`. Provider credentials remain ordinary
environment variables and are never stored in application tables.

### Recovery and audit

DBOS resumes incomplete workflows from the shared SQLite file after an API or worker restart.
Restart each role on the same host, against the same database file, and with the same
`AGENT_OS_APPLICATION_VERSION`; set that variable explicitly when deploying an incompatible
workflow change. A failed or cancelled run retains its task/staging branches and worktrees for
diagnosis. A successful run retains its branches and database audit records but removes
reconstructable worktrees.

Inspect persisted transcripts directly when needed:

```bash
sqlite3 ~/.local/state/agent-os/agent-os.sqlite3 \
  'select execution_id, sequence, event_type, payload from transcript_events order by execution_id, sequence;'
```

The normal test suite uses deterministic Pydantic AI models and starts the API plus all four worker
processes against one SQLite file. For an optional provider smoke test, set `AGENT_OS_MODEL` and its
provider credential, start the five processes above, and target a disposable clean repository.

SQLite intentionally limits this POC to one host. PostgreSQL remains an optional migration path for
multi-host or higher-write-concurrency deployments: install `agent-os[postgres]` and set
`DBOS_SYSTEM_DATABASE_URL` to a PostgreSQL connection string.

## Library

```python
import agent_os
```

## Development

```bash
make test
make typecheck
make ruff
make all-tests
```

## Release process

Releases are automated on pushes to `main` by the CD workflow in `.github/workflows/cd.yml`.

1. The workflow determines the latest `v*` Git tag and runs `.github/workflows/cd_version.py` to
   resolve the next version. If the latest tag matches the current major/minor, it bumps the patch
   to one higher than the max of the current patch and the tag patch.
2. If `pyproject.toml` changes, the workflow commits the version bump back to `main`.
3. It tags the release as `v<version>`, builds the wheel with `uv build`, and creates a GitHub release.

To enable PyPI publishing, set `PUBLISH_TO_PYPI: true` in `.github/workflows/cd.yml` and configure
PyPI trusted publishing for this repository.
