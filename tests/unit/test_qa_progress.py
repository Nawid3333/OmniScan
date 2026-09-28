"""Tests for where each chapter of a series stands (qa/progress.py), `omniscan status` and GET …/progress."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cleanup.store import CLEANUP_FILE
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    Manifest,
    Region,
    RegionKind,
    RegionsArtifact,
    StageRecord,
)
from omniscan.edits import store
from omniscan.qa.progress import ChapterProgress, chapter_progress, series_progress
from omniscan.web.app import create_app

NOW = datetime.now(UTC)
PIPELINE = (
    "ingest",
    "slice",
    "detect",
    "ocr",
    "translate",
    "judge",
    "inpaint",
    "inpaint_lama",
    "typeset",
    "export",
)


def region(rid: str, y0: int, text: str, kind: RegionKind = "bubble_text") -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=BBox(x0=10, y0=y0, x1=110, y1=y0 + 50), text=text)


def record(stage: str, finished: datetime, status: str = "done") -> StageRecord:
    return StageRecord.model_validate(
        {
            "stage": stage,
            "version": 1,
            "input_hash": "i",
            "config_hash": "c",
            "outputs": [],
            "status": status,
            "started_at": finished - timedelta(seconds=5),
            "finished_at": finished,
        }
    )


def write_manifest(paths: ChapterPaths, records: dict[str, StageRecord]) -> None:
    Manifest(series=paths.series, chapter=paths.chapter, stages=records).save(paths.manifest)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    config = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: config)
    return config


@pytest.fixture
def series(cfg: Config) -> SeriesPaths:
    """Chapter 1: only raws. Chapter 2: read, judged and exported an hour from now, with one line blank, one
    edited by hand, one checked (with a typo), a sound effect and a watermark. Chapter 3: detection failed after
    slicing."""
    paths = SeriesPaths.from_config(cfg, "S")
    for name in ("Chapter 1", "Chapter 2", "Chapter 3"):
        (paths.library_dir / name).mkdir(parents=True)
    two = paths.chapter("Chapter 2")
    regions = [
        region("r0001", 10, "안녕"),
        region("r0002", 100, "반가워"),
        region("r0003", 200, "쾅", kind="sfx"),
        region("r0004", 300, "site.com", kind="watermark"),
        region("r0005", 400, "끝", kind="free_text"),
        region("r0006", 500, " "),
    ]
    RegionsArtifact(regions=regions).save(two.artifact("ocr.json"))
    lines = {"r0001": "Hi", "r0002": " ", "r0003": "BOOM", "r0005": "The endd."}  # r0002: a blank line
    FinalArtifact(
        judge_model="judge",
        lines=[FinalLine(region_id=rid, text=text, decision="pick") for rid, text in lines.items()],
    ).save(two.artifact("final.json"))
    store.set_translation(two, "r0001", "Hi there", direction="ltr")  # an edited line, not checked
    store.set_checked(two, ["r0005"])  # checked with its typo: not an open problem
    write_manifest(two, {name: record(name, NOW + timedelta(hours=1)) for name in PIPELINE})
    three = paths.chapter("Chapter 3")
    three.work_dir.mkdir(parents=True)
    write_manifest(
        three,
        {
            "ingest": record("ingest", NOW),
            "slice": record("slice", NOW),
            "detect": record("detect", NOW, status="failed"),
        },
    )
    return paths


def test_where_each_chapter_stands(series: SeriesPaths) -> None:
    first, second, third = series_progress(series)
    assert first == ChapterProgress("Chapter 1", 0, 0, 0, 0, None, None, False, False)
    assert second == ChapterProgress("Chapter 2", 3, 2, 1, 1, "export", None, True, False)
    assert third == ChapterProgress("Chapter 3", 0, 0, 0, 0, "slice", "detect", False, False)


def test_an_export_is_outdated_after_a_hand_edit_a_hand_clean_or_a_re_run(series: SeriesPaths) -> None:
    two = series.chapter("Chapter 2")
    stages = {name: record(name, NOW - timedelta(hours=1)) for name in PIPELINE}
    write_manifest(two, stages)  # exported an hour ago; the check (edits.json) was made after it
    assert chapter_progress(two).outdated
    later = NOW + timedelta(hours=1)
    os.utime(two.artifact(store.EDITS_FILE), (later.timestamp() - 9000, later.timestamp() - 9000))
    assert not chapter_progress(two).outdated
    two.artifact(CLEANUP_FILE).write_text("{}", encoding="utf-8")  # a hand clean after the export
    assert chapter_progress(two).outdated
    two.artifact(CLEANUP_FILE).unlink()
    write_manifest(two, {**stages, "judge": record("judge", NOW)})  # the judge ran again after the export
    assert chapter_progress(two).outdated
    write_manifest(two, {**stages, "qa": record("qa", NOW)})  # the finished-page re-read does not count
    assert not chapter_progress(two).outdated


def test_a_failed_export_is_not_exported(series: SeriesPaths) -> None:
    two = series.chapter("Chapter 2")
    write_manifest(two, {"judge": record("judge", NOW), "export": record("export", NOW, status="failed")})
    progress = chapter_progress(two)
    assert (progress.last_stage, progress.failed, progress.exported, progress.outdated) == (
        "judge",
        "export",
        False,
        False,
    )


def test_omniscan_status(series: SeriesPaths) -> None:
    runner = CliRunner()
    shown = runner.invoke(app, ["status", "S"])
    assert shown.exit_code == 0, shown.output
    rows = {
        re.split(r"[│┃|]", line)[1].strip(): line for line in shown.output.splitlines() if "Chapter " in line
    }
    assert "not started" in rows["Chapter 1"] and "0/0" in rows["Chapter 1"]
    assert "through export" in rows["Chapter 2"] and "2/3" in rows["Chapter 2"] and "1/3" in rows["Chapter 2"]
    assert "failed: detect" in rows["Chapter 3"]
    rows_json = json.loads(runner.invoke(app, ["status", "S", "--json"]).output)
    assert [row["chapter"] for row in rows_json] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert rows_json[1]["problems"] == 1 and rows_json[1]["exported"] is True
    assert re.split(r"[│┃|]", rows["Chapter 2"])[6].strip() == "yes"
    two = series.chapter("Chapter 2")
    write_manifest(two, {name: record(name, NOW - timedelta(hours=1)) for name in PIPELINE})
    again = runner.invoke(app, ["status", "S"]).output
    assert "outdated" in next(line for line in again.splitlines() if "Chapter 2" in line)
    missing = runner.invoke(app, ["status", "Ghost"])
    assert missing.exit_code == 2 and "no chapters found for series 'Ghost'" in missing.output


def test_progress_over_the_web_api(series: SeriesPaths, cfg: Config) -> None:
    client = TestClient(create_app(cfg))
    response = client.get("/api/series/S/progress")
    assert response.status_code == 200
    body = response.json()
    assert [row["chapter"] for row in body] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert body[1] == {
        "chapter": "Chapter 2",
        "lines": 3,
        "translated": 2,
        "checked": 1,
        "problems": 1,
        "last_stage": "export",
        "failed": None,
        "exported": True,
        "outdated": False,
    }
    assert client.get("/api/series/Ghost/progress").json() == []
    assert client.get("/api/series/..%2F..%2Fx/progress").status_code == 404
