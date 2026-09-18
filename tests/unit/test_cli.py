"""Tests for omniscan.cli."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.doctor import CheckResult

STUB_COMMANDS = (
    "acquire",
    "ingest",
    "slice",
    "filter",
    "detect",
    "ocr",
    "glossary",
    "judge",
    "inpaint",
    "typeset",
    "export",
    "run",
    "serve",
    "reference",
)

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("doctor", "version", *STUB_COMMANDS):
        assert name in result.output


def test_stub_exits_2() -> None:
    result = runner.invoke(app, ["ocr"])
    assert result.exit_code == 2
    assert "ocr: not implemented yet" in result.output


def test_doctor_json_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        CheckResult(name="python", status="OK", detail="3.14.1"),
        CheckResult(name="rocm", status="FAIL", detail="boom"),
    ]
    monkeypatch.setattr(omniscan.cli, "run_all_checks", lambda *a, **k: results)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    parsed = json.loads(result.output)
    assert isinstance(parsed, list) and len(parsed) == 2
    assert set(parsed[0]) == {"name", "status", "detail"}


def test_doctor_table_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [CheckResult(name="python", status="OK", detail="3.14.1")]
    monkeypatch.setattr(omniscan.cli, "run_all_checks", lambda *a, **k: results)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ok, 0 warn, 0 fail" in result.output
