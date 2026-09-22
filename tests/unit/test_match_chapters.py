"""End-to-end chapter-matching tests on synthetic chapter sets (the card's acceptance scenarios)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.match.chapters import ChapterMapping, PagePair, Thresholds, match_chapters
from tests.fixtures.chapter_sets import (
    TRANSLATED_QUALITY,
    TRANSLATED_SIZE,
    TRANSLATED_TEXT_OFFSET,
    page_seeds,
    write_raw_series,
    write_translated_series,
)

RAW = {  # folder names and art seeds of a clean 4-chapter raw set (c1..c4)
    "Chapter 001": page_seeds(1001, 6),
    "Chapter 002": page_seeds(2001, 6),
    "Chapter 003": page_seeds(3001, 6),
    "Chapter 004": page_seeds(4001, 6),
}
TRANSLATED = {
    "ch_01": RAW["Chapter 001"],
    "ch_02": RAW["Chapter 002"],
    "ch_03": RAW["Chapter 003"],
    "ch_04": RAW["Chapter 004"],
}


def match(tmp_path: Path, raw: dict[str, list[int]], translated: dict[str, list[int]]) -> ChapterMapping:
    """Match a synthetic raw set against a synthetic translated set (smaller pages, shifted lettering)."""
    dir_a = write_raw_series(tmp_path / "raw", raw)
    dir_b = write_translated_series(tmp_path / "translated", translated)
    return match_chapters(dir_a, dir_b, Thresholds())


def pairs(mapping: ChapterMapping) -> list[tuple[str, str]]:
    return [(m.a, m.b) for m in mapping.matched]


def test_clean_one_to_one_series(tmp_path: Path) -> None:
    mapping = match(tmp_path, RAW, TRANSLATED)
    assert pairs(mapping) == [
        ("Chapter 001", "ch_01"),
        ("Chapter 002", "ch_02"),
        ("Chapter 003", "ch_03"),
        ("Chapter 004", "ch_04"),
    ]
    assert mapping.unmatched_a == [] and mapping.unmatched_b == []
    assert all(m.quality > 0.8 for m in mapping.matched)
    assert all(not m.review for m in mapping.matched)
    one = mapping.matched[0]
    assert one.pages_a == 6 and one.pages_b == 6
    assert one.pages == [PagePair(page_a=i, page_b=i, similarity=one.pages[i].similarity) for i in range(6)]
    assert all(p.similarity > 0.8 for p in one.pages)
    assert mapping.b_of("Chapter 003") == "ch_03" and mapping.a_of("ch_03") == "Chapter 003"
    assert mapping.b_of("Chapter 009") is None


def test_chapter_only_in_a_is_flagged_unmatched(tmp_path: Path) -> None:
    raw = {"Chapter 000": page_seeds(501, 4), **RAW}  # a raw-only prologue in front
    mapping = match(tmp_path, raw, TRANSLATED)
    assert pairs(mapping) == [
        ("Chapter 001", "ch_01"),
        ("Chapter 002", "ch_02"),
        ("Chapter 003", "ch_03"),
        ("Chapter 004", "ch_04"),
    ]
    assert mapping.unmatched_a == ["Chapter 000"]
    assert mapping.unmatched_b == []


def test_chapter_only_in_b_is_flagged_unmatched(tmp_path: Path) -> None:
    translated = {
        "ch_01": RAW["Chapter 001"],
        "ch_02": RAW["Chapter 002"],
        "ad_page": page_seeds(9001, 3),
        "ch_03": RAW["Chapter 003"],
        "ch_04": RAW["Chapter 004"],
    }
    mapping = match(tmp_path, RAW, translated)  # an English-only ad chapter in the middle
    assert pairs(mapping) == [
        ("Chapter 001", "ch_01"),
        ("Chapter 002", "ch_02"),
        ("Chapter 003", "ch_03"),
        ("Chapter 004", "ch_04"),
    ]
    assert mapping.unmatched_a == []
    assert mapping.unmatched_b == ["ad_page"]


def test_chapter_numbering_offset_is_handled(tmp_path: Path) -> None:
    # B has one extra early chapter, so B's numbering runs one ahead of A's the whole way
    translated = {
        "ch_01": page_seeds(7001, 5),
        "ch_02": RAW["Chapter 001"],
        "ch_03": RAW["Chapter 002"],
        "ch_04": RAW["Chapter 003"],
        "ch_05": RAW["Chapter 004"],
    }
    mapping = match(tmp_path, RAW, translated)
    assert pairs(mapping) == [
        ("Chapter 001", "ch_02"),
        ("Chapter 002", "ch_03"),
        ("Chapter 003", "ch_04"),
        ("Chapter 004", "ch_05"),
    ]
    assert mapping.unmatched_a == []
    assert mapping.unmatched_b == ["ch_01"]
    assert all(m.quality > 0.8 for m in mapping.matched)


def test_ad_page_inside_a_chapter_is_absorbed(tmp_path: Path) -> None:
    # an extra ad page sits mid-chapter on the B side: the pair still matches, pages shift around it
    translated = {
        "ch_01": RAW["Chapter 001"],
        "ch_02": [2001, 2002, 8001, 2003, 2004, 2005, 2006],
        "ch_03": RAW["Chapter 003"],
        "ch_04": RAW["Chapter 004"],
    }
    mapping = match(tmp_path, RAW, translated)
    assert pairs(mapping) == [
        ("Chapter 001", "ch_01"),
        ("Chapter 002", "ch_02"),
        ("Chapter 003", "ch_03"),
        ("Chapter 004", "ch_04"),
    ]
    two = mapping.matched[1]
    assert (two.pages_a, two.pages_b) == (6, 7)
    assert [(p.page_a, p.page_b) for p in two.pages] == [(0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (5, 6)]
    assert two.quality > 0.75 and not two.review  # six perfect pages out of seven


def test_near_duplicate_chapters_are_not_confused(tmp_path: Path) -> None:
    # B's ch_05 reprints A chapter 3's last page as a recap; that must not pull chapter 3 onto ch_04
    translated = {
        "ch_01": RAW["Chapter 001"],
        "ch_02": RAW["Chapter 002"],
        "ch_03": RAW["Chapter 003"],
        "ch_04": [3006, *RAW["Chapter 004"]],
    }
    mapping = match(tmp_path, RAW, translated)
    assert pairs(mapping) == [
        ("Chapter 001", "ch_01"),
        ("Chapter 002", "ch_02"),
        ("Chapter 003", "ch_03"),
        ("Chapter 004", "ch_04"),
    ]
    assert mapping.matched[2].quality > 0.8  # chapter 3 kept its own partner
    assert mapping.matched[3].quality > 0.7  # chapter 4: six of its seven pages matched


def test_mapping_artifact_round_trips_and_is_hand_editable(tmp_path: Path) -> None:
    mapping = match(tmp_path, RAW, TRANSLATED)
    path = tmp_path / "mapping.json"
    mapping.save(path)
    loaded = ChapterMapping.load(path)
    assert loaded == mapping
    assert loaded.schema_version == 1
    assert loaded.thresholds.page_similarity == 0.75
    # the owner corrects a match by hand: drop one pair, add it as matched to a different chapter
    loaded.matched[-1].b = "ch_04-corrected"
    loaded.save(path)
    reloaded = ChapterMapping.load(path)
    assert reloaded.matched[-1].b == "ch_04-corrected"
    assert reloaded == loaded


def test_empty_and_degenerate_chapters_come_out_unmatched(tmp_path: Path) -> None:
    # one chapter folder on each side has no page images at all: they cannot match anything
    raw = {**RAW, "Chapter 00X": []}
    translated = {
        "ch_01": RAW["Chapter 001"],
        "ch_02": RAW["Chapter 002"],
        "ch_03": RAW["Chapter 003"],
        "ch_04": RAW["Chapter 004"],
        "ch_0X": [],
    }
    mapping = match(tmp_path, raw, translated)
    assert pairs(mapping) == [(f"Chapter 00{i}", f"ch_0{i}") for i in (1, 2, 3, 4)]
    assert mapping.unmatched_a == ["Chapter 00X"]
    assert mapping.unmatched_b == ["ch_0X"]


def test_all_empty_or_missing_chapter_sets_raise(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="no chapter folders"):
        match_chapters(empty_root, empty_root)


def test_chapter_sets_with_only_empty_folders_match_to_nothing(tmp_path: Path) -> None:
    # chapter folders exist but hold no page images: the run succeeds, everything is unmatched
    only_empty = write_raw_series(tmp_path / "blank", {"Chapter 001": []})
    mapping = match_chapters(only_empty, only_empty)
    assert mapping.matched == []
    assert mapping.unmatched_a == ["Chapter 001"] and mapping.unmatched_b == ["Chapter 001"]


def test_translated_side_defaults_differ_from_raw_side(tmp_path: Path) -> None:
    # the two sides are rendered differently on purpose: smaller pages, lower quality, shifted lettering
    dir_a = write_raw_series(tmp_path / "raw", {"Chapter 001": page_seeds(1001, 2)})
    dir_b = write_translated_series(tmp_path / "trn", {"ch_01": page_seeds(1001, 2)})
    first_raw = next((dir_a / "Chapter 001").iterdir())
    first_trn = next((dir_b / "ch_01").iterdir())
    assert first_trn.stat().st_size < first_raw.stat().st_size  # lower quality + smaller page
    assert TRANSLATED_TEXT_OFFSET != 0  # the lettering seed shift is what makes the two sides differ
    assert (TRANSLATED_SIZE, TRANSLATED_QUALITY) != ((96, 144), 92)
