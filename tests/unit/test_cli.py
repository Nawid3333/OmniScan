"""Tests for omniscan.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.manifest import load_manifest
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    InpaintArtifact,
    InpaintItem,
    Region,
    RegionsArtifact,
)
from omniscan.doctor import CheckResult
from tests.fixtures import images

STUB_COMMANDS = (
    "acquire",
    "ingest",
    "slice",
    "filter",
    "ocr",
    "glossary",
    "export",
    "run",
    "serve",
    "reference",
)

SERIES = "S"

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("doctor", "version", "typeset", *STUB_COMMANDS):
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


# ---------------------------------------------------------------- typeset


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Config with all paths under tmp_path."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


def write_typeset_inputs(cfg: Config, chapter: str) -> None:
    """Hand-write the three upstream artifacts the typeset stage reads."""
    work = cfg.paths.work_root / SERIES / chapter
    work.mkdir(parents=True, exist_ok=True)
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=150, y0=80, x1=250, y1=120),
                bubble_bbox=BBox(x0=0, y0=0, x1=400, y1=200),
                text="안녕",
            )
        ]
    ).save(work / "ocr.json")
    FinalArtifact(
        judge_model="judge-fake",
        lines=[FinalLine(region_id="r0001", text="Hello there.", decision="pick")],
    ).save(work / "final.json")
    InpaintArtifact(
        items=[
            InpaintItem(
                region_id="r0001",
                box=BBox(x0=0, y0=0, x1=400, y1=200),
                method="flat",
                fill=(255, 255, 255),
            )
        ]
    ).save(work / "inpaint.json")


def test_cli_typeset(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    for chapter in ("Chapter 1", "Chapter 2"):
        (cfg.paths.library_root / SERIES / chapter).mkdir(parents=True)
        write_typeset_inputs(cfg, chapter)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["typeset", SERIES])
    assert result.exit_code == 0
    for chapter in ("Chapter 1", "Chapter 2"):
        assert f"{SERIES}/{chapter} typeset: done" in result.output
        work = cfg.paths.work_root / SERIES / chapter
        assert (work / "layout.json").is_file()
        assert list(load_manifest(work / "manifest.json", SERIES, chapter).stages) == ["typeset"]

    result = runner.invoke(app, ["typeset", SERIES, "--chapter", "Chapter 1"])
    assert result.exit_code == 0
    assert "Chapter 1 typeset: skipped" in result.output

    result = runner.invoke(app, ["typeset", SERIES, "--force", "--chapter", "Chapter 1"])
    assert result.exit_code == 0
    assert "Chapter 1 typeset: done" in result.output

    result = runner.invoke(app, ["typeset", "NoSuchSeries"])
    assert result.exit_code == 2
    assert "no chapters found" in result.output


def test_cli_slice_unaffected(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = cfg.paths.library_root / SERIES / "Chapter 1"
    raw.mkdir(parents=True)
    for i in (1, 2, 3):
        images.plain_jpeg(raw / f"{i:03d}.jpg", size=(400, 300), color=(200, 60, 60))
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)

    result = runner.invoke(app, ["slice", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1 slice: done" in result.output
    assert (cfg.paths.work_root / SERIES / "Chapter 1" / "slices.json").is_file()
