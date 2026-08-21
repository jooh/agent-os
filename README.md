# agent-os

[![CI](https://github.com/jooh/agent-os/actions/workflows/ci.yml/badge.svg)](https://github.com/jooh/agent-os/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- CHANGE ME: Uncomment after setting up codecov and add your token -->
<!-- [![codecov](https://codecov.io/gh/jooh/agent-os/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/jooh/agent-os) -->
<!-- CHANGE ME: Uncomment after first PyPI publish -->
<!-- [![PyPI](https://img.shields.io/pypi/v/agent-os.svg)](https://pypi.org/project/agent-os/) -->

Deterministic loop engineering with DBOS

## Installation

```bash
pip install agent-os
```

Or with uv:

```bash
uv add agent-os
```

## Library

```python
import agent_os
```

## Development

```bash
make test
make typecheck
make ruff
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
