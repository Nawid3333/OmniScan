"""Duplicate-detection tests on synthetic chapter sets (the card's acceptance scenarios)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.match.duplicates import DuplicatePair, DuplicateThresholds, find_duplicate_chapters
from tests.fixtures.chapter_sets import page_seeds, write_raw_series

EXACT_DUPLICATE = {  # "Chapter 001 copy" holds pixel-identical copies of "Chapter 001"'s pages
    "Chapter 001": page_seeds(1, 6),
    "Chapter 001 copy": page_seeds(1, 6),
    "Chapter 002": page_seeds(500, 6),
}


def build(tmp_path: Path, chapters: dict[str, list[int]]) -> Path:
    return write_raw_series(tmp_path / "set", chapters)


def test_identical_chapters_are_flagged_with_exact_quality(tmp_path: Path) -> None:
    root = build(tmp_path, EXACT_DUPLICATE)
    report = find_duplicate_chapters(root)
    assert report.duplicates == [DuplicatePair(a="Chapter 001", b="Chapter 001 copy", quality=1.0)]
    assert all("Chapter 002" not in (pair.a, pair.b) for pair in report.duplicates)
    assert report.root == str(root)
    assert report.thresholds == DuplicateThresholds()  # defaults reached the report unchanged


def test_well_separated_chapters_yield_no_duplicates(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "Chapter 001": page_seeds(1, 6),
            "Chapter 002": page_seeds(500, 6),
            "Chapter 003": page_seeds(9000, 6),
        },
    )
    report = find_duplicate_chapters(root)
    assert report.duplicates == []


def test_no_pair_is_a_chapter_paired_with_itself(tmp_path: Path) -> None:
    root = build(tmp_path, {**EXACT_DUPLICATE, "Chapter 001 again": page_seeds(1, 6)})
    report = find_duplicate_chapters(root)
    assert len(report.duplicates) >= 1  # the identical trio does pair up ...
    assert all(pair.a != pair.b for pair in report.duplicates)  # ... but never with itself
    assert {(pair.a, pair.b) for pair in report.duplicates} == {
        ("Chapter 001", "Chapter 001 again"),
        ("Chapter 001", "Chapter 001 copy"),
        ("Chapter 001 again", "Chapter 001 copy"),
    }  # a is always the natural-sort-earlier name of the two
    assert all(pair.quality == 1.0 for pair in report.duplicates)


def test_equal_qualities_tie_break_by_ascending_names(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {  # two unrelated exact-duplicate pairs: both pairs score exactly 1.0
            "Chapter 001": page_seeds(1, 6),
            "Chapter 001 copy": page_seeds(1, 6),
            "Chapter 005": page_seeds(100, 6),
            "Chapter 005 copy": page_seeds(100, 6),
        },
    )
    report = find_duplicate_chapters(root)
    assert [(pair.a, pair.b) for pair in report.duplicates] == [
        ("Chapter 001", "Chapter 001 copy"),
        ("Chapter 005", "Chapter 005 copy"),
    ]


def test_empty_chapter_folder_never_appears_in_a_pair(tmp_path: Path) -> None:
    root = build(tmp_path, {**EXACT_DUPLICATE, "Chapter 003 empty": []})
    report = find_duplicate_chapters(root)
    assert [DuplicatePair(a="Chapter 001", b="Chapter 001 copy", quality=1.0)] == report.duplicates
    assert all("Chapter 003 empty" not in (pair.a, pair.b) for pair in report.duplicates)


def test_only_empty_chapter_folders_yield_no_duplicates(tmp_path: Path) -> None:
    root = build(tmp_path, {"Chapter 001": [], "Chapter 002": []})
    assert find_duplicate_chapters(root).duplicates == []


def test_chapter_less_root_raises_with_match_chapters_wording(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError) as exc_info:  # the exact message is pinned below
        find_duplicate_chapters(empty_root)
    assert str(exc_info.value) == f"no chapter folders with page images under {empty_root}"


def test_custom_thresholds_reach_the_run_and_the_report(tmp_path: Path) -> None:
    root = build(tmp_path, EXACT_DUPLICATE)
    thresholds = DuplicateThresholds(page_similarity=0.99, page_gap=0.3, min_quality=1.1)
    report = find_duplicate_chapters(root, thresholds)
    assert report.thresholds is thresholds
    assert report.duplicates == []  # quality 1.0 sits below the impossible bar of 1.1
