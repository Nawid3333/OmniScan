"""Tests for omniscan.acquire.completeness."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from omniscan.acquire.completeness import (
    Finding,
    PageInfo,
    check_numbering,
    check_pages,
    inspect_chapter,
    verdict,
)


def make_jpeg(path: Path, width: int, height: int, fill: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    """Write a solid-colour JPEG of the given size and return the exact file bytes."""
    buffer = BytesIO()
    Image.new("RGB", (width, height), fill).save(buffer, format="JPEG")
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


def page(name: str, width: int, sha256: str | None = None) -> PageInfo:
    """A PageInfo stub with fixed height/size and a distinct hash unless one is given."""
    return PageInfo(name=name, width=width, height=1200, size_bytes=100_000, sha256=sha256 or f"sha-{name}")


def pages_of(count: int, width: int = 800) -> list[PageInfo]:
    """`count` distinct clean pages of one width."""
    return [page(f"p{i}", width) for i in range(count)]


# --- inspect_chapter ---


def test_inspect_chapter_pages_in_natural_order_plus_unreadable(tmp_path: Path) -> None:
    chapter = tmp_path / "Chapter 1"
    chapter.mkdir()
    bytes1 = make_jpeg(chapter / "1.jpg", 800, 1200, (10, 20, 30))
    bytes2 = make_jpeg(chapter / "2.jpg", 850, 1100, (40, 50, 60))
    bytes3 = make_jpeg(chapter / "10.jpg", 800, 1200, (70, 80, 90))
    (chapter / "3.jpg").write_text("not an image", encoding="utf-8")

    pages, findings = inspect_chapter(chapter)

    assert [page.name for page in pages] == ["1.jpg", "2.jpg", "10.jpg"]
    assert pages[0] == PageInfo("1.jpg", 800, 1200, len(bytes1), hashlib.sha256(bytes1).hexdigest())
    assert pages[1] == PageInfo("2.jpg", 850, 1100, len(bytes2), hashlib.sha256(bytes2).hexdigest())
    assert pages[2] == PageInfo("10.jpg", 800, 1200, len(bytes3), hashlib.sha256(bytes3).hexdigest())
    assert findings == [Finding("error", "unreadable", "3.jpg: cannot be decoded")]


def test_inspect_chapter_missing_and_empty_directory(tmp_path: Path) -> None:
    assert inspect_chapter(tmp_path / "nope") == ([], [])
    empty = tmp_path / "empty"
    empty.mkdir()
    assert inspect_chapter(empty) == ([], [])


# --- check_pages ---


def test_check_pages_no_pages() -> None:
    assert check_pages([]) == [Finding("error", "no_pages", "no pages found")]


def test_check_pages_few_by_median() -> None:
    findings = check_pages(pages_of(4), median_pages=10.0)
    assert findings == [Finding("warn", "few_pages", "4 pages; the series median is 10")]


def test_check_pages_few_by_previous() -> None:
    findings = check_pages(pages_of(4), previous_pages=10)
    assert findings == [Finding("warn", "few_pages", "4 pages; the previous chapter had 10")]


def test_check_pages_median_message_preferred_when_both_apply() -> None:
    findings = check_pages(pages_of(4), median_pages=10.0, previous_pages=10)
    assert [finding.message for finding in findings] == ["4 pages; the series median is 10"]


def test_check_pages_half_of_median_is_not_few() -> None:
    assert check_pages(pages_of(5), median_pages=10.0) == []


def test_check_pages_many_only_against_median() -> None:
    assert [finding.code for finding in check_pages(pages_of(21), median_pages=10.0)] == ["many_pages"]
    assert check_pages(pages_of(20), median_pages=10.0) == []


def test_check_pages_mixed_widths_over_five_percent() -> None:
    widths = [800] * 9 + [900]
    findings = check_pages([page(f"p{i}", width) for i, width in enumerate(widths)])
    assert findings == [Finding("warn", "mixed_widths", "1 page(s) differ in width from the usual 800 px")]


def test_check_pages_within_five_percent_is_not_mixed() -> None:
    assert check_pages([*pages_of(9), page("odd", 830)]) == []


def test_check_pages_width_tie_picks_the_smaller_usual() -> None:
    widths = [800] * 5 + [1000] * 5
    findings = check_pages([page(f"p{i}", width) for i, width in enumerate(widths)])
    assert findings == [Finding("warn", "mixed_widths", "5 page(s) differ in width from the usual 800 px")]


def test_check_pages_duplicates_count_extra_copies() -> None:
    twin = pages_of(5)
    twin[2] = PageInfo(twin[2].name, 800, 1200, 100_000, twin[0].sha256)
    assert check_pages(twin) == [
        Finding("warn", "duplicate_pages", "1 page(s) are exact duplicates of another page")
    ]

    triple = pages_of(5)
    triple[1] = PageInfo(triple[1].name, 800, 1200, 100_000, triple[0].sha256)
    triple[2] = PageInfo(triple[2].name, 800, 1200, 100_000, triple[0].sha256)
    assert check_pages(triple) == [
        Finding("warn", "duplicate_pages", "2 page(s) are exact duplicates of another page")
    ]


def test_check_pages_clean_chapter() -> None:
    assert check_pages(pages_of(10), median_pages=10.0) == []


def test_check_pages_finding_order() -> None:
    chapter_pages = [
        page("a", 800, "h1"),
        page("b", 800, "h2"),
        page("c", 900, "h3"),
        page("d", 800, "h2"),
    ]
    findings = check_pages(chapter_pages, median_pages=10.0)
    assert [finding.code for finding in findings] == ["few_pages", "mixed_widths", "duplicate_pages"]


# --- check_numbering ---


def test_check_numbering_multi_chapter_gap() -> None:
    assert check_numbering(["Chapter 1", "Chapter 2", "Chapter 5"]) == [
        Finding("warn", "numbering_gap", "chapters 3–4 are missing")
    ]


def test_check_numbering_single_chapter_gap() -> None:
    assert check_numbering(["Chapter 1", "Chapter 3"]) == [
        Finding("warn", "numbering_gap", "chapter 2 is missing")
    ]


def test_check_numbering_two_gaps_ascending_from_unsorted_input() -> None:
    findings = check_numbering(["Chapter 5", "Chapter 1", "Chapter 3", "Chapter 6"])
    assert [finding.message for finding in findings] == ["chapter 2 is missing", "chapter 4 is missing"]


def test_check_numbering_decimal_is_ignored() -> None:
    assert check_numbering(["Chapter 1", "Chapter 1.5", "Chapter 2"]) == []


def test_check_numbering_unnumbered_names_are_ignored() -> None:
    assert check_numbering(["Prologue", "Chapter 1", "Chapter 3"]) == [
        Finding("warn", "numbering_gap", "chapter 2 is missing")
    ]


def test_check_numbering_single_name_and_empty_list() -> None:
    assert check_numbering(["Chapter 1"]) == []
    assert check_numbering([]) == []


# --- verdict ---


def test_verdict_error_beats_warn_beats_ok() -> None:
    assert verdict([]) == "ok"
    assert verdict([Finding("warn", "few_pages", "m")]) == "review"
    assert verdict([Finding("error", "no_pages", "m")]) == "failed"
    assert verdict([Finding("warn", "w", "m"), Finding("error", "e", "m")]) == "failed"
