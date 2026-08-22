from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast


class CdVersionModule(Protocol):
    def main(self) -> None: ...


def load_cd_version() -> CdVersionModule:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / ".github" / "workflows" / "cd_version.py"
    spec = importlib.util.spec_from_file_location("cd_version", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(CdVersionModule, module)


def write_pyproject(path: Path, version: str) -> None:
    path.write_text(
        f"""
[project]
name = "agent-os"
version = "{version}"
"""
    )


def read_output(path: Path) -> str:
    return path.read_text().strip().split("=", 1)[1]


def test_bumps_patch_when_major_minor_match(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_pyproject(tmp_path / "pyproject.toml", "1.2.3")
    output_path = tmp_path / "out.txt"
    monkeypatch.setenv("LATEST_TAG", "v1.2.5")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    cd_version = load_cd_version()
    cd_version.main()

    assert read_output(output_path) == "1.2.6"
    assert 'version = "1.2.6"' in (tmp_path / "pyproject.toml").read_text()


def test_skips_bump_on_major_minor_change(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_pyproject(tmp_path / "pyproject.toml", "2.1.0")
    output_path = tmp_path / "out.txt"
    monkeypatch.setenv("LATEST_TAG", "v2.2.9")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    cd_version = load_cd_version()
    cd_version.main()

    assert read_output(output_path) == "2.1.0"
    assert 'version = "2.1.0"' in (tmp_path / "pyproject.toml").read_text()


def test_ignores_invalid_latest_tag(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_pyproject(tmp_path / "pyproject.toml", "0.4.1")
    output_path = tmp_path / "out.txt"
    monkeypatch.setenv("LATEST_TAG", "not-a-tag")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    cd_version = load_cd_version()
    cd_version.main()

    assert read_output(output_path) == "0.4.1"
    assert 'version = "0.4.1"' in (tmp_path / "pyproject.toml").read_text()


def test_bumps_patch_without_downgrade(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_pyproject(tmp_path / "pyproject.toml", "1.2.10")
    output_path = tmp_path / "out.txt"
    monkeypatch.setenv("LATEST_TAG", "v1.2.5")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    cd_version = load_cd_version()
    cd_version.main()

    assert read_output(output_path) == "1.2.11"
    assert 'version = "1.2.11"' in (tmp_path / "pyproject.toml").read_text()
