# Agent guide for agent-os

## Repository overview
- `src/agent_os/` contains the library implementation.
- `tests/` holds pytest coverage for core behavior.
- `pyproject.toml` defines dependencies, entry points, and tooling.

## Development setup
- Requires Python 3.14+ (see `pyproject.toml`).
- Environment management is with uv.
- Run Python and related CLI tools via `uv run` so they use the uv virtualenv.

## Common commands
- Run tests: `make test`
- Run type checks: `make typecheck`
- Run ruff checks: `make ruff`
- Run all checks: `make all-tests`

## Style and conventions
- TDD for all code development - write test, then run to verify it fails, then develop, then verify the test passes.
- All tasks should end by running `make all-tests` and verifying it passes.
- Prefer updating or adding pytest tests in `tests/` for behavior changes.
- Target modern Python 3.14+ syntax, no need to be backwards compatible.

## Tips
- The main package is `agent_os`.
