"""release_version: work out and write the next app version (`.github/workflows/release.yml`).

Every push to main whose CI passes becomes a release. The level of the bump comes from the commit
messages since the last app tag (`vMAJOR.MINOR.PATCH`, the form `omniscan update` accepts):

- `major`: a message with a `BREAKING CHANGE:` line or `[major]`
- `minor`: a subject starting `feat:` / `feat(...)` or a message with `[minor]`
- `patch`: anything else

With no app tag yet, or when pyproject.toml's version was raised by hand past the last tag, that
version is released as it is. The version lives
in pyproject.toml (`[project] version`) and `src/omniscan/__init__.py` (`__version__`); `write` sets
both, and the workflow then runs `uv lock` so uv.lock records it too.

    python scripts/release_version.py next [--level auto|patch|minor|major]   # prints the version
    python scripts/release_version.py write VERSION
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

Level = Literal["patch", "minor", "major"]

REPO_ROOT = Path(__file__).resolve().parents[1]
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_PYPROJECT_RE = re.compile(r'^(version = ")(\d+\.\d+\.\d+)(")$', re.MULTILINE)
_INIT_RE = re.compile(r'^(__version__ = ")(\d+\.\d+\.\d+)(")$', re.MULTILINE)
_FEAT_RE = re.compile(r"^feat(\([^)]*\))?!?:", re.IGNORECASE)
_BREAKING_RE = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)
_BANG_RE = re.compile(r"^\w+(\([^)]*\))?!:")


def bump_level(messages: Sequence[str]) -> Level:
    """The bump the commit messages ask for: major over minor over patch."""
    level: Level = "patch"
    for message in messages:
        if "[major]" in message or _BREAKING_RE.search(message) or _BANG_RE.match(message):
            return "major"
        if "[minor]" in message or _FEAT_RE.match(message):
            level = "minor"
    return level


def bump(version: str, level: Level) -> str:
    """`version` raised by one `level` ("1.2.3", "minor" -> "1.3.0")."""
    major, minor, patch = _key(version)
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _key(version: str) -> tuple[int, int, int]:
    """`version` as a comparable (major, minor, patch)."""
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def read_version(root: Path = REPO_ROOT) -> str:
    """The version in pyproject.toml's `[project]` table."""
    match = _PYPROJECT_RE.search((root / "pyproject.toml").read_text(encoding="utf-8"))
    if match is None:
        raise ValueError('pyproject.toml has no `version = "X.Y.Z"` line')
    return match.group(2)


def write_version(version: str, root: Path = REPO_ROOT) -> None:
    """Set `version` in pyproject.toml and `__version__` in src/omniscan/__init__.py."""
    if _VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    for path, pattern in (
        (root / "pyproject.toml", _PYPROJECT_RE),
        (root / "src" / "omniscan" / "__init__.py", _INIT_RE),
    ):
        text = path.read_text(encoding="utf-8")
        new, count = pattern.subn(rf"\g<1>{version}\g<3>", text, count=1)
        if count != 1:
            raise ValueError(f"no version line in {path.name}")
        path.write_text(new, encoding="utf-8", newline="\n")


def _git(*args: str, root: Path) -> str:
    """Run git in `root` and return its stdout."""
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def last_app_tag(root: Path = REPO_ROOT) -> str | None:
    """The newest `vX.Y.Z` tag reachable from HEAD, or None when there is none."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*.[0-9]*.[0-9]*"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    tag = result.stdout.strip()
    return tag if result.returncode == 0 and re.fullmatch(r"v\d+\.\d+\.\d+", tag) else None


def commit_messages(since: str, root: Path = REPO_ROOT) -> list[str]:
    """Full messages of the commits after `since` up to HEAD."""
    out = _git("log", "--format=%B%x00", f"{since}..HEAD", root=root)
    return [message.strip() for message in out.split("\x00") if message.strip()]


def next_version(level: str = "auto", root: Path = REPO_ROOT) -> str:
    """The version the next release gets."""
    current = read_version(root)
    tag = last_app_tag(root)
    if tag is None:
        return current
    released = tag.removeprefix("v")
    if _key(current) > _key(released):
        return current  # raised by hand since the last release: released as it is
    if level == "auto":
        return bump(released, bump_level(commit_messages(tag, root)))
    if level not in ("patch", "minor", "major"):
        raise ValueError(f"unknown level: {level!r}")
    return bump(released, cast(Level, level))


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    next_parser = commands.add_parser("next", help="print the next release version")
    next_parser.add_argument("--level", default="auto", choices=["auto", "patch", "minor", "major"])
    write_parser = commands.add_parser("write", help="write VERSION into pyproject.toml and __init__.py")
    write_parser.add_argument("version")
    args = parser.parse_args(argv)
    if args.command == "next":
        print(next_version(args.level))
    else:
        write_version(args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
