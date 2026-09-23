"""Tests for `omniscan usage` (card B34): collect_usage/totals plus the CLI report; CPU-only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.schemas import CandidateRun, FinalArtifact
from omniscan.usage import UsageRow, collect_usage, totals

SERIES = "S"
runner = CliRunner()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
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


def write_two_chapter_tree(cfg: Config) -> None:
    """The card's example tree: chapter 1 = one run + a judge, chapter 2 = a legacy final.json only."""
    chapter1 = cfg.paths.work_root / SERIES / "Chapter 1"
    CandidateRun(
        run_id="prof",
        profile="prof",
        model="run-model",
        candidates=[],
        usage={"prompt_tokens": 100.0, "completion_tokens": 50.0, "requests": 2.0, "regions": 3.0},
    ).save(chapter1 / "translations" / "prof.json")
    FinalArtifact(
        judge_model="judge-model",
        lines=[],
        usage={"prompt_tokens": 10.0, "completion_tokens": 4.0, "requests": 1.0, "seconds": 0.5},
    ).save(chapter1 / "final.json")
    # a legacy artifact from before usage recording existed: the file has no usage key at all
    (cfg.paths.work_root / SERIES / "Chapter 2").mkdir(parents=True)
    (cfg.paths.work_root / SERIES / "Chapter 2" / "final.json").write_text(
        json.dumps({"judge_model": "judge-model", "lines": []}), encoding="utf-8"
    )


def test_collect_usage_rows_and_legacy_zero(cfg: Config) -> None:
    write_two_chapter_tree(cfg)
    rows = collect_usage(cfg, SERIES)
    assert [(row.series, row.chapter, row.source, row.model) for row in rows] == [
        (SERIES, "Chapter 1", "prof", "run-model"),
        (SERIES, "Chapter 1", "judge", "judge-model"),
        (SERIES, "Chapter 2", "judge", "judge-model"),
    ]
    assert rows[0].usage == {
        "prompt_tokens": 100.0,
        "completion_tokens": 50.0,
        "requests": 2.0,
        "regions": 3.0,
    }
    assert rows[1].usage == {
        "prompt_tokens": 10.0,
        "completion_tokens": 4.0,
        "requests": 1.0,
        "seconds": 0.5,
    }
    assert rows[2].usage == {}  # legacy artifact: zeros, never an error and never a skip


def test_collect_usage_all_series_and_corrupt(cfg: Config) -> None:
    work = cfg.paths.work_root
    for series, chapter in (("Beta", "Chapter 1"), ("Alpha", "Chapter 10"), ("Alpha", "Chapter 9")):
        CandidateRun(run_id="p", profile="p", model="m", candidates=[], usage={"requests": 1.0}).save(
            work / series / chapter / "translations" / "p.json"
        )
    CandidateRun(run_id="part", profile="part", model="m", candidates=[], usage={"requests": 9.0}).save(
        work / "Alpha" / "Chapter 9" / "translations" / ".part.json"
    )  # partial file, never a row
    (work / "Alpha" / "Chapter 2" / "translations").mkdir(parents=True)
    (work / "Alpha" / "Chapter 2" / "translations" / "x.json").write_bytes(b"{invalid json")
    (work / "Beta" / "Chapter 9").mkdir()  # a chapter with no artifacts contributes no rows
    (work / "_reference_en" / "Chapter 1").mkdir(parents=True)  # '_'-prefixed series are not series

    rows = collect_usage(cfg)  # series=None: every series, name-sorted, chapters in reading order
    assert [(row.series, row.chapter) for row in rows] == [
        ("Alpha", "Chapter 9"),
        ("Alpha", "Chapter 10"),
        ("Beta", "Chapter 1"),
    ]


def test_totals_generic() -> None:
    rows = [
        UsageRow(series="s", chapter="c1", source="p", model="m", usage={"regions": 3.0}),
        UsageRow(series="s", chapter="c2", source="judge", model="j", usage={"prompt_tokens": 5.0}),
        UsageRow(
            series="s", chapter="c3", source="p", model="m", usage={"regions": 2.0, "prompt_tokens": 1.5}
        ),
    ]
    assert totals(rows) == {"regions": 5.0, "prompt_tokens": 6.5}
    assert totals([]) == {}


# ---------------------------------------------------------------- the CLI


def test_usage_cli_table(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    write_two_chapter_tree(cfg)
    result = runner.invoke(app, ["usage", SERIES])
    assert result.exit_code == 0
    assert "prof" in result.output
    assert "judge" in result.output
    # the TOTAL line's numbers are the pinned sums of the tree above
    assert result.output.splitlines()[-1].split() == ["TOTAL", "110", "54", "3", "0.5"]


def test_usage_cli_json(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    write_two_chapter_tree(cfg)
    result = runner.invoke(app, ["usage", SERIES, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [(row["chapter"], row["source"], row["model"]) for row in payload["rows"]] == [
        ("Chapter 1", "prof", "run-model"),
        ("Chapter 1", "judge", "judge-model"),
        ("Chapter 2", "judge", "judge-model"),
    ]
    assert set(payload["rows"][0]) == {"series", "chapter", "source", "model", "usage"}
    assert payload["totals"]["prompt_tokens"] == 110.0
    assert payload["totals"]["completion_tokens"] == 54.0
    assert payload["totals"]["requests"] == 3.0
    assert payload["totals"]["seconds"] == 0.5


def test_usage_cli_unknown_series(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    result = runner.invoke(app, ["usage", "NoSuchSeries"])
    assert result.exit_code == 2
    assert "unknown series" in result.output


def test_usage_cli_empty(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert [line.split() for line in result.output.splitlines()] == [["TOTAL", "0", "0", "0", "0.0"]]
