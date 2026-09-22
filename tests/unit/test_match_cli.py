"""Tests for the `omniscan match chapters` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omniscan.cli import app
from omniscan.match.chapters import ChapterMapping
from tests.fixtures.chapter_sets import page_seeds, write_raw_series, write_translated_series

runner = CliRunner()

RAW = {f"Chapter {n:03d}": page_seeds(1000 * n + 1, 4) for n in range(1, 5)}
TRANSLATED = {f"ch_{n:02d}": RAW[f"Chapter {n:03d}"] for n in range(1, 5)}


def build_series(tmp_path: Path) -> tuple[Path, Path]:
    dir_a = write_raw_series(tmp_path / "raw", RAW)
    dir_b = write_translated_series(tmp_path / "translated", TRANSLATED)
    return dir_a, dir_b


def test_match_chapters_writes_and_summarises(tmp_path: Path) -> None:
    dir_a, dir_b = build_series(tmp_path)
    out = tmp_path / "mapping.json"
    result = runner.invoke(app, ["match", "chapters", str(dir_a), str(dir_b), "--out", str(out)])
    assert result.exit_code == 0
    assert "4 matched, 0 unmatched in A, 0 unmatched in B" in result.output
    assert str(out) in result.output
    mapping = ChapterMapping.load(out)
    assert [(m.a, m.b) for m in mapping.matched] == [
        ("Chapter 001", "ch_01"),
        ("Chapter 002", "ch_02"),
        ("Chapter 003", "ch_03"),
        ("Chapter 004", "ch_04"),
    ]
    assert "review:" not in result.output  # everything matched confidently


def test_match_chapters_default_out_is_cwd_and_json_flag_prints_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dir_a, dir_b = build_series(tmp_path)
    result = runner.invoke(app, ["match", "chapters", str(dir_a), str(dir_b), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [m["a"] for m in payload["matched"]] == [f"Chapter {n:03d}" for n in range(1, 5)]
    assert Path("chapter-match.json").is_file()  # the documented default destination


def test_match_chapters_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    dir_a, dir_b = build_series(tmp_path)
    out = tmp_path / "mapping.json"
    first = runner.invoke(app, ["match", "chapters", str(dir_a), str(dir_b), "--out", str(out)])
    assert first.exit_code == 0
    before = out.read_text(encoding="utf-8")
    second = runner.invoke(app, ["match", "chapters", str(dir_a), str(dir_b), "--out", str(out)])
    assert second.exit_code == 2
    assert "--force" in second.output
    assert out.read_text(encoding="utf-8") == before  # the hand-edited mapping was not touched
    third = runner.invoke(app, ["match", "chapters", str(dir_a), str(dir_b), "--out", str(out), "--force"])
    assert third.exit_code == 0


def test_match_chapters_lists_unmatched_and_review_matches(tmp_path: Path) -> None:
    dir_a = write_raw_series(tmp_path / "raw", RAW)
    dir_b = write_translated_series(
        tmp_path / "translated",
        {**TRANSLATED, "ad_chapter": page_seeds(9001, 3)},
    )
    out = tmp_path / "mapping.json"
    # a very strict review threshold flags every match, whatever its quality
    result = runner.invoke(
        app, ["match", "chapters", str(dir_a), str(dir_b), "--out", str(out), "--review-quality", "0.95"]
    )
    assert result.exit_code == 0
    assert "4 matched, 0 unmatched in A, 1 unmatched in B" in result.output
    assert "B only:" in result.output and "ad_chapter" in result.output
    assert "review: A 'Chapter 001' <-> B 'ch_01'" in result.output


def test_match_chapters_flags_low_confidence_matches(tmp_path: Path) -> None:
    dir_a, dir_b = build_series(tmp_path)
    out = tmp_path / "mapping.json"
    result = runner.invoke(
        app,
        ["match", "chapters", str(dir_a), str(dir_b), "--review-quality", "0.96", "--out", str(out)],
    )
    assert result.exit_code == 0
    mapping = ChapterMapping.load(out)
    assert all(m.review for m in mapping.matched)  # everything sits below the strict review bar
    assert result.output.count("review: A ") == 4
    assert "runner-up" not in result.output  # nothing else came close: no alternative partners at all


def test_match_chapters_missing_or_empty_dirs_fail_cleanly(tmp_path: Path) -> None:
    dir_a, _dir_b = build_series(tmp_path)
    missing = runner.invoke(app, ["match", "chapters", str(dir_a), str(tmp_path / "nope")])
    assert missing.exit_code == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    no_chapters = runner.invoke(app, ["match", "chapters", str(dir_a), str(empty)])
    assert no_chapters.exit_code == 2
    assert "no chapter folders" in no_chapters.output


def test_match_chapters_threshold_flags_reach_the_artifact(tmp_path: Path) -> None:
    dir_a, dir_b = build_series(tmp_path)
    out = tmp_path / "mapping.json"
    result = runner.invoke(
        app,
        [
            "match",
            "chapters",
            str(dir_a),
            str(dir_b),
            "--out",
            str(out),
            "--page-similarity",
            "0.99",
            "--chapter-gap",
            "0.7",
            "--min-quality",
            "0.2",
        ],
    )
    assert result.exit_code == 0
    mapping = ChapterMapping.load(out)
    assert mapping.thresholds.page_similarity == 0.99
    assert mapping.thresholds.chapter_gap == 0.7
    assert mapping.thresholds.min_quality == 0.2
