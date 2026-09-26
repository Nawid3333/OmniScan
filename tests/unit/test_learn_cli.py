"""Tests for `omniscan learn show|enable|disable` — a hand-built series, no GPU, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import BBox, Region, RegionsArtifact, Slice, SlicesArtifact
from omniscan.edits import store

runner = CliRunner()


@pytest.fixture
def series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SeriesPaths:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "library", work_root=tmp_path / "work", output_root=tmp_path / "output"
        )
    )
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    series = SeriesPaths.from_config(cfg, "S")
    for number, text in ((1, "Jlnwoo, run!"), (2, "Jlnwoo?")):
        (series.library_dir / f"Chapter {number}").mkdir(parents=True)
        paths = series.chapter(f"Chapter {number}")
        SlicesArtifact(
            strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]
        ).save(paths.artifact("slices.json"))
        box = BBox(x0=10, y0=10, x1=110, y1=60)
        RegionsArtifact(
            regions=[Region(id="r0001", slice_index=0, kind="bubble_text", bbox=box, text=text)]
        ).save(paths.artifact("ocr.json"))
        store.update_region(paths, "r0001", direction="ltr", text=text.replace("Jlnwoo", "Jinwoo"))
    return series


def test_learn_show_lists_rules_and_switches_them(series: SeriesPaths) -> None:
    result = runner.invoke(app, ["learn", "show", "S", "--json"])
    assert result.exit_code == 0, result.output
    shown = json.loads(result.output)
    (rule,) = shown["rules"]
    assert (rule["kind"], rule["wrong"], rule["right"], rule["count"], rule["active"]) == (
        "ocr_fix",
        "Jlnwoo",
        "Jinwoo",
        2,
        True,
    )
    assert shown["translations"] == []

    table = runner.invoke(app, ["learn", "show", "S"])
    assert table.exit_code == 0 and "Jinwoo" in table.output and "active" in table.output

    assert (
        runner.invoke(app, ["learn", "disable", "S", rule["id"]]).output == f"learn: rule {rule['id']} off\n"
    )
    shown = json.loads(runner.invoke(app, ["learn", "show", "S", "--json", "--rebuild"]).output)
    assert (shown["rules"][0]["enabled"], shown["rules"][0]["active"]) == (False, False)
    assert runner.invoke(app, ["learn", "enable", "S", rule["id"]]).exit_code == 0
    missing = runner.invoke(app, ["learn", "enable", "S", "nope"])
    assert missing.exit_code == 2 and "no learned rule 'nope'" in missing.output
