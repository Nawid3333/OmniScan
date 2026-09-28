"""Tests for the typo check of the English lines (qa/typos.py) and where it shows: the desktop Studio's QA pass,
its session, and the web proofreading report."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
from omniscan.edits.session import StudioSession
from omniscan.glossary.store import GlossaryStore
from omniscan.qa.consistency import Line
from omniscan.qa.typos import (
    WORDS_FILE,
    SeriesTypo,
    TypoChecker,
    allow_word,
    allowed_words,
    checker_for,
    series_typos,
    series_words,
)
from omniscan.studio.qa import check_chapter
from omniscan.web.app import create_app


def words(text: str, checker: TypoChecker | None = None) -> list[str]:
    return [typo.word for typo in (checker or TypoChecker()).check(text)]


def test_an_unknown_word_is_a_typo_with_the_closest_words_first() -> None:
    typos = TypoChecker().check("Grab teh sword, now")
    assert [typo.word for typo in typos] == ["teh"]
    assert typos[0].suggestions[0] == "the"
    assert "teh" not in typos[0].suggestions and len(typos[0].suggestions) <= 3


def test_correct_english_has_no_typos() -> None:
    assert words("I don't know. Can't you see? We're fine, you'll be fine, it won't hurt.") == []
    assert words("") == [] and words("...!?") == []


def test_names_acronyms_interjections_and_drawn_out_words_are_left_alone() -> None:
    assert (
        words("Tell Dokja the NPC is here") == []
    )  # a capital inside a sentence: a name; all caps: an acronym
    assert words("Nooo! Ugh, hmph. Hyung-nim, wait!") == []
    assert words("Dokja, run!") == ["Dokja"]  # a capital opening a sentence says nothing: checked
    assert words("Dokja, run!", TypoChecker(["Kim Dokja"])) == []  # a name's words are known


def test_an_all_caps_line_is_checked_word_by_word() -> None:
    assert words("GRAB TEH SWORD") == ["TEH"]


def test_each_unknown_word_is_reported_once_per_line() -> None:
    assert words("teh sword and teh shield, Teh end") == ["teh"]


def test_series_typos_skip_sound_effects_watermarks_and_untranslated_lines() -> None:
    def line(rid: str, english: str, kind: RegionKind = "bubble_text") -> Line:
        return Line("Chapter 1", rid, kind, "ko", "원문", english)

    lines = [
        line("r0001", "Grab teh sword"),
        line("r0002", "KRZZHT", kind="sfx"),
        line("r0003", "wwww.teh.com", kind="watermark"),
        line("r0004", ""),
    ]
    typos = series_typos(lines, TypoChecker())
    assert [(t.chapter, t.region_id, t.word, t.english) for t in typos] == [
        ("Chapter 1", "r0001", "teh", "Grab teh sword")
    ]
    assert isinstance(typos[0], SeriesTypo)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def series(cfg: Config) -> SeriesPaths:
    """Series S, one chapter: a line with a real typo, a name, a known word list and an effect."""
    series = SeriesPaths.from_config(cfg, "S")
    paths = series.chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    rows = [
        ("r0001", "bubble_text", "Igris! Grab teh sword"),
        ("r0002", "bubble_text", "Hwaiting!"),
        ("r0003", "sfx", "KRZZHT"),
    ]
    RegionsArtifact(
        regions=[
            Region(
                id=rid,
                slice_index=0,
                kind=kind,
                bbox=BBox(x0=0, y0=100 * i, x1=50, y1=100 * i + 40),
                text="원문",
            )
            for i, (rid, kind, _english) in enumerate(rows)
        ]
    ).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[FinalLine(region_id=rid, text=english, decision="pick") for rid, _k, english in rows],
    ).save(paths.artifact("final.json"))
    return series


def test_the_series_glossary_characters_and_word_list_are_known(series: SeriesPaths) -> None:
    assert allowed_words(series) == set()  # nothing yet, and nothing created
    assert not series.db.exists()
    series.work_dir.mkdir(parents=True, exist_ok=True)
    with GlossaryStore(series.db) as db:
        db.add(GlossaryEntry(source="이그리트", target="Igris", status="locked"))
        db.add(GlossaryEntry(source="베르", target="Beru", status="rejected"))  # rejected: not agreed English
    (series.library_dir / "voices.toml").write_text(
        '[[character]]\nname = "Jinwoo"\naliases = ["Sung Jinwoo"]\n', encoding="utf-8"
    )
    (series.library_dir / WORDS_FILE).write_text("# not typos\nHwaiting\n\n", encoding="utf-8")
    assert allowed_words(series) == {"Igris", "Jinwoo", "Sung Jinwoo", "hwaiting"}
    checker = checker_for(series)
    assert words("Igris! Jinwoo! Sung! Hwaiting!", checker) == []
    assert words("Beru!", checker) == ["Beru"]


def test_a_broken_voices_file_does_not_stop_the_check(series: SeriesPaths) -> None:
    (series.library_dir / "voices.toml").write_text("[[character]]\nvoice = 3\n", encoding="utf-8")
    assert words("Grab teh sword", checker_for(series)) == ["teh"]


def test_allow_word_keeps_one_sorted_list_per_series(series: SeriesPaths) -> None:
    assert series_words(series) == []
    assert allow_word(series, "Hwaiting") == ["hwaiting"]
    assert allow_word(series, " igris ") == ["hwaiting", "igris"]
    assert allow_word(series, "HWAITING") == ["hwaiting", "igris"]
    assert (series.library_dir / WORDS_FILE).read_text(encoding="utf-8") == "hwaiting\nigris\n"
    for bad in ("two words", "", "123", "a-b"):
        with pytest.raises(ValueError):
            allow_word(series, bad)
    missing = SeriesPaths(
        "T", series.library_dir.parent / "T", series.work_dir.parent / "T", series.output_dir.parent / "T"
    )
    with pytest.raises(FileNotFoundError):
        allow_word(missing, "igris")  # never creates a series
    assert not missing.library_dir.exists()


def test_the_studio_check_reports_typos_only_when_given_a_checker() -> None:
    regions = [
        Region(id="r0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=0, y0=0, x1=9, y1=9), text="잡아"),
        Region(id="r0002", slice_index=0, kind="sfx", bbox=BBox(x0=0, y0=20, x1=9, y1=29), text="쾅"),
    ]
    english = {"r0001": "Grab teh sword", "r0002": "KRABOOM"}
    assert check_chapter(regions, english) == []
    issues = check_chapter(regions, english, typos=TypoChecker())
    assert [(issue.region_id, issue.kind, issue.word) for issue in issues] == [("r0001", "typo", "teh")]
    assert issues[0].message.startswith("'teh' is not in the dictionary (did you mean 'the'")


def test_the_studio_session_checks_typos_and_remembers_not_a_typo(series: SeriesPaths) -> None:
    session = StudioSession(series.chapter("Chapter 1"), direction="ltr")
    found = [(issue.region_id, issue.word) for issue in session.issues() if issue.kind == "typo"]
    assert found == [("r0001", "Igris"), ("r0001", "teh"), ("r0002", "Hwaiting")]
    assert session.allow_word("Igris") == ["igris"]
    session.set_translation("r0001", "Igris! Grab the sword")  # an unsaved fix counts too
    assert [issue.word for issue in session.issues() if issue.kind == "typo"] == ["Hwaiting"]
    assert series_words(series) == ["igris"]


def test_the_web_report_lists_typos_and_learns_not_a_typo(series: SeriesPaths, cfg: Config) -> None:
    client = TestClient(create_app(cfg))
    typos = client.get("/api/series/S/consistency").json()["typos"]
    assert [(t["chapter"], t["region_id"], t["word"], t["english"]) for t in typos] == [
        ("Chapter 1", "r0001", "Igris", "Igris! Grab teh sword"),
        ("Chapter 1", "r0001", "teh", "Igris! Grab teh sword"),
        ("Chapter 1", "r0002", "Hwaiting", "Hwaiting!"),
    ]
    assert typos[1]["suggestions"][0] == "the"
    url = "/api/series/S/typo-words"
    assert client.post(url, content='{"word": "Igris"}').status_code == 415  # not sent as JSON
    assert client.post(url, json={"word": "two words"}).status_code == 422
    assert client.post("/api/series/Nope/typo-words", json={"word": "Igris"}).status_code == 404
    assert not (cfg.paths.library_root / "Nope").exists()
    assert client.post(url, json={"word": "Igris"}).json() == {"words": ["igris"]}
    assert client.post(url, json={"word": "hwaiting"}).json() == {"words": ["hwaiting", "igris"]}
    typos = client.get("/api/series/S/consistency").json()["typos"]
    assert [t["word"] for t in typos] == ["teh"]
