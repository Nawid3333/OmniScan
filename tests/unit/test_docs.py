"""Tests that README.md and docs/USER_GUIDE.md cannot drift from the real CLI."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import typer.main
from typer.core import TyperGroup

import omniscan.cli

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "USER_GUIDE.md")

_LANGUAGES = ("bash", "sh", "console")
_FENCE_RE = re.compile(r"^```([^\n]*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)
_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def _cli_tree() -> TyperGroup:
    """The real click command tree of the typer app — never a hand-written list."""
    command = typer.main.get_command(omniscan.cli.app)
    assert isinstance(command, TyperGroup)
    return command


_CLI = _cli_tree()


def extract_commands(text: str) -> list[list[str]]:
    """Tokens after `omniscan` in every fenced bash/sh/console line that invokes the CLI."""
    commands: list[list[str]] = []
    for info, body in _FENCE_RE.findall(text):
        words = info.strip().split()
        if not words or words[0] not in _LANGUAGES:
            continue
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("$ "):
                line = line[2:].strip()
            if line.startswith("uv run "):
                line = line[len("uv run ") :].strip()
            if not line.startswith("omniscan "):
                continue
            commands.append(shlex.split(line)[1:])
    return commands


def doc_command_errors(text: str) -> list[str]:
    """Human-readable problems with the `omniscan` invocations fenced in `text`."""
    errors: list[str] = []
    for tokens in extract_commands(text):
        bare = [t for t in tokens if not t.startswith("-")]
        if not bare:
            errors.append(f"omniscan invocation without a command name: {tokens!r}")
            continue
        pos = next(i for i, t in enumerate(tokens) if not t.startswith("-"))
        name = bare[0]
        command = _CLI.commands.get(name)
        if command is None:
            errors.append(f"unknown omniscan command {name!r}")
            continue
        _check_stub(command, name, errors)
        if isinstance(command, TyperGroup):
            sub_name = next((t for t in tokens[pos + 1 :] if not t.startswith("-")), None)
            if sub_name is None:
                errors.append(f"omniscan {name!r} is a group but no subcommand is given")
            elif sub_name not in command.commands:
                errors.append(f"unknown omniscan subcommand {sub_name!r} of {name!r}")
            else:
                _check_stub(command.commands[sub_name], f"{name} {sub_name}", errors)
    return errors


def _check_stub(command: object, label: str, errors: list[str]) -> None:
    """Flag commands whose help starts with 'Not implemented yet' — they must not be documented as working."""
    help_text = getattr(command, "help", None)
    if help_text and help_text.startswith("Not implemented yet"):
        errors.append(f"{label!r} is a stub; do not document it as working")


def doc_link_errors(doc_path: Path, text: str) -> list[str]:
    """Relative markdown link targets in `text` that do not resolve to an existing file or directory."""
    errors: list[str] = []
    for target in _LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_part = target.split("#", 1)[0]
        resolved = (doc_path.parent / path_part).resolve()
        if not (resolved.is_file() or resolved.is_dir()):
            errors.append(f"broken link {target!r} in {doc_path.name}")
    return errors


def test_every_documented_command_exists() -> None:
    for path in DOC_PATHS:
        errors = doc_command_errors(path.read_text(encoding="utf-8"))
        assert not errors, f"{path.name}:\n" + "\n".join(errors)


def test_no_stub_is_documented_as_working() -> None:
    for path in DOC_PATHS:
        errors = [e for e in doc_command_errors(path.read_text(encoding="utf-8")) if "stub" in e]
        assert not errors, f"{path.name}:\n" + "\n".join(errors)


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_relative_links_resolve(path: Path) -> None:
    errors = doc_link_errors(path, path.read_text(encoding="utf-8"))
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_doc_exists_and_is_non_empty(path: Path) -> None:
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip()


def test_readme_line_count() -> None:
    readme = REPO_ROOT / "README.md"
    assert len(readme.read_text(encoding="utf-8").splitlines()) <= 150


def test_extraction_helper_detects_unknown_command() -> None:
    """Prove the extraction/validation actually fires on a bad command name."""
    errors = doc_command_errors("```bash\nomniscan definitely-not-a-command\n```")
    assert len(errors) == 1
    assert "definitely-not-a-command" in errors[0]


def test_extraction_helper_flags_stub_as_documented() -> None:
    errors = doc_command_errors("```bash\nomniscan reference DemoSeries\n```")
    assert any("stub" in e for e in errors)
