from __future__ import annotations

import os
import pathlib
import re
import tomllib
import tomli_w
from dataclasses import dataclass


# Strict semantic versioning pattern: MAJOR.MINOR.PATCH
# - MAJOR is either 0 (pre-1.0 semantics) or a non-zero integer without leading zeros.
# - MINOR and PATCH are non-negative integers.
# - Pre-release identifiers (e.g. "-alpha") and build metadata (e.g. "+build.1") are
#   intentionally not supported, because this script only needs to handle final release
#   versions when resolving and bumping versions.
VERSION_PATTERN = re.compile(r"^(?P<major>0|[1-9]\d*)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> "Version | None":
        match = VERSION_PATTERN.match(value)
        if not match:
            return None
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )

    def bump_patch(self) -> "Version":
        return Version(self.major, self.minor, self.patch + 1)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def read_current_version(pyproject_path: pathlib.Path) -> Version:
    data = tomllib.loads(pyproject_path.read_text())
    current = data["project"]["version"]
    parsed = Version.parse(current)
    if not parsed:
        raise ValueError(f"Invalid project.version in {pyproject_path}: {current}")
    return parsed


def write_version(pyproject_path: pathlib.Path, version: Version) -> None:
    data = tomllib.loads(pyproject_path.read_text())
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"Missing [project] table in {pyproject_path}")
    if "version" not in project:
        raise ValueError(f"Missing project.version in {pyproject_path}")
    project["version"] = str(version)
    pyproject_path.write_text(tomli_w.dumps(data))


def resolve_version(current: Version, latest_tag: str | None) -> Version:
    if not latest_tag:
        return current
    parsed = Version.parse(latest_tag.lstrip("v"))
    if not parsed:
        return current
    if current.major == parsed.major and current.minor == parsed.minor:
        new_patch = max(current.patch, parsed.patch) + 1
        return Version(current.major, current.minor, new_patch)
    return current


def main() -> None:
    pyproject_path = pathlib.Path("pyproject.toml")
    current = read_current_version(pyproject_path)
    new_version = resolve_version(current, os.environ.get("LATEST_TAG"))

    if new_version != current:
        write_version(pyproject_path, new_version)

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with pathlib.Path(output_path).open("a") as handle:
            handle.write(f"version={new_version}\n")
    else:
        raise RuntimeError("GITHUB_OUTPUT is not set")
    print(f"Resolved version: {new_version}")


if __name__ == "__main__":
    main()
