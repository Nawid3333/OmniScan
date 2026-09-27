"""Tests for the series consistency report (qa/consistency.py) through the CLI and the web API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    GlossaryEntry,
    Region,
    RegionKind,
    RegionsArtifact,
)
from omniscan.glossary.store import GlossaryStore
from omniscan.qa.consistency import (
    Divergence,
    Line,
    Rendering,
    TermMiss,
    divergences,
    glossary,
    series_lines,
    term_misses,
)
from omniscan.web.app import create_app


def line(chapter: str, rid: str, source: str, english: str, kind: RegionKind = "bubble_text") -> Line:
    return Line(chapter, rid, kind, "ko", source, english)


def test_divergences_group_repeated_lines_translated_differently() -> None:
    lines = [
        line("1", "r0001", "진우야", "Jinwoo!"),
        line("2", "r0004", "진우야 ", "jinwoo."),  # the same rendering: case, spacing and punctuation aside
        line("3", "r0002", "진우야", "Hey, Jinwoo!"),
        line("1", "r0002", "가자", "Let's go"),
        line("2", "r0001", "가자", "Let's go!"),  # one rendering only: consistent
        line("1", "r0003", "쾅쾅", "BOOM", kind="sfx"),
        line("2", "r0003", "쾅쾅", "BANG", kind="sfx"),  # sound effects may differ
        line("1", "r0005", "헉", "Gasp"),
        line("2", "r0005", "헉", "Huh"),  # one character: left out
        line("3", "r0006", "진우야", ""),  # untranslated: left out
    ]
    assert divergences(lines) == [
        Divergence(
            "진우야",
            [
                Rendering("Jinwoo!", [("1", "r0001"), ("2", "r0004")]),
                Rendering("Hey, Jinwoo!", [("3", "r0002")]),
            ],
        )
    ]


def test_term_misses_list_locked_terms_whose_english_is_missing() -> None:
    entries = [
        GlossaryEntry(id=1, source="헌터", target="Hunter", status="locked"),
        GlossaryEntry(id=2, source="게이트", target="Gate", status="proposed"),  # not agreed yet
    ]
    lines = [
        line("1", "r0001", "헌터가 왔다", "The Hunter is here"),
        line(
            "1", "r0002", "헌터는 헌터다", "A hunter is a hunter"
        ),  # case matters: "Hunter" is the agreed form
        line("2", "r0001", "게이트가 열렸다", "The portal opened"),
        line("2", "r0002", "헌터", ""),  # untranslated
    ]
    assert term_misses(lines, entries) == [TermMiss("1", "r0002", "헌터", "Hunter", "A hunter is a hunter")]
    assert term_misses(lines, []) == []


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def series(cfg: Config) -> SeriesPaths:
    series = SeriesPaths.from_config(cfg, "S")
    chapters = {
        "Chapter 1": [("r0001", "진우야", "Jinwoo!"), ("r0002", "헌터가 왔다", "The hunter came")],
        "Chapter 2": [("r0001", "진우야", "Hey, Jinwoo!"), ("r0002", "site.com", "")],
    }
    for name, rows in chapters.items():
        paths = series.chapter(name)
        paths.raw_dir.mkdir(parents=True)
        regions = [
            Region(
                id=rid,
                slice_index=0,
                kind="watermark" if source == "site.com" else "bubble_text",
                bbox=BBox(x0=0, y0=100 * i, x1=50, y1=100 * i + 40),
                text=source,
            )
            for i, (rid, source, _english) in enumerate(rows)
        ]
        RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
        FinalArtifact(
            judge_model="judge",
            lines=[
                FinalLine(region_id=rid, text=english, decision="pick")
                for rid, _s, english in rows
                if english
            ],
        ).save(paths.artifact("final.json"))
    return series


def test_series_lines_and_glossary_read_the_series(series: SeriesPaths) -> None:
    lines = series_lines(series)
    assert [(x.chapter, x.region_id, x.english) for x in lines] == [
        ("Chapter 1", "r0001", "Jinwoo!"),
        ("Chapter 1", "r0002", "The hunter came"),
        ("Chapter 2", "r0001", "Hey, Jinwoo!"),
    ]  # the watermark is left out
    assert [x.chapter for x in series_lines(series, ["Chapter 2"])] == ["Chapter 2"]
    assert glossary(series) == [] and not series.db.exists()  # no glossary yet: none created
    series.work_dir.mkdir(parents=True, exist_ok=True)
    with GlossaryStore(series.db) as db:
        db.add(GlossaryEntry(source="헌터", target="Hunter", status="locked"))
    assert [entry.target for entry in glossary(series)] == ["Hunter"]


def test_consistency_over_the_web_api(series: SeriesPaths, cfg: Config) -> None:
    series.work_dir.mkdir(parents=True, exist_ok=True)
    with GlossaryStore(series.db) as db:
        db.add(GlossaryEntry(source="헌터", target="Hunter", status="locked"))
    report = TestClient(create_app(cfg)).get("/api/series/S/consistency").json()
    assert report["divergences"] == [
        {
            "source": "진우야",
            "renderings": [
                {"english": "Jinwoo!", "places": [["Chapter 1", "r0001"]]},
                {"english": "Hey, Jinwoo!", "places": [["Chapter 2", "r0001"]]},
            ],
        }
    ]
    assert report["term_misses"] == [
        {
            "chapter": "Chapter 1",
            "region_id": "r0002",
            "term": "헌터",
            "target": "Hunter",
            "english": "The hunter came",
        }
    ]


def test_consistency_on_the_command_line(
    series: SeriesPaths, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import omniscan.cli

    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    runner = CliRunner()
    result = runner.invoke(omniscan.cli.app, ["consistency", "S"])
    assert result.output.splitlines() == [
        "consistency: '진우야' is translated 2 ways:",
        "  'Jinwoo!' ×1: Chapter 1 r0001",
        "  'Hey, Jinwoo!' ×1: Chapter 2 r0001",
        "consistency: 1 line(s) translated differently, 0 glossary term(s) missing",
    ]
    as_json = json.loads(runner.invoke(omniscan.cli.app, ["consistency", "S", "--json"]).output)
    assert len(as_json["divergences"]) == 1 and as_json["term_misses"] == []
    unknown = runner.invoke(omniscan.cli.app, ["consistency", "S", "-c", "Chapter 9"])
    assert unknown.exit_code == 2 and "no chapter 'Chapter 9' in S" in unknown.output
