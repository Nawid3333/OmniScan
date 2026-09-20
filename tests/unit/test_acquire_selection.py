"""Tests for omniscan.acquire.selection."""

from __future__ import annotations

import re

import pytest

from omniscan.acquire.selection import read_url_list, select_chapters
from omniscan.acquire.sources import ChapterSource


def numbered(first: int, last: int) -> list[ChapterSource]:
    """Chapter N sources for N = first..last."""
    return [ChapterSource(f"Chapter {n}", f"https://x.test/{n}") for n in range(first, last + 1)]


def names(sources: list[ChapterSource]) -> list[str]:
    return [source.name for source in sources]


# --- read_url_list ---


def test_read_url_list_mixed_lines_numbering_over_non_ignored() -> None:
    text = (
        "# series: Test\n\nhttps://x.test/10\n\nChapter 2 | https://x.test/b\n# comment\nhttps://x.test/c\n"
    )
    assert read_url_list(text, first_number=10) == [
        ChapterSource("Chapter 10", "https://x.test/10"),
        ChapterSource("Chapter 2", "https://x.test/b"),
        ChapterSource("Chapter 12", "https://x.test/c"),
    ]


def test_read_url_list_named_line_extra_spaces() -> None:
    assert read_url_list("  Ch 1   |   https://x.test/1  ") == [ChapterSource("Ch 1", "https://x.test/1")]


def test_read_url_list_splits_on_first_separator_only() -> None:
    assert read_url_list("A | https://x.test/a | b") == [ChapterSource("A", "https://x.test/a | b")]


def test_read_url_list_windows_line_endings() -> None:
    assert read_url_list("https://x.test/1\r\nhttps://x.test/2\r\n") == [
        ChapterSource("Chapter 1", "https://x.test/1"),
        ChapterSource("Chapter 2", "https://x.test/2"),
    ]


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Chapter 1", "chapter #1"),
        ("https://x.test/1\nftp://x/2", "chapter #2"),
        ("https://x.test/1\nChapter 2.5", "chapter #2"),
    ],
)
def test_read_url_list_bad_line_gets_chapter_label(text: str, label: str) -> None:
    with pytest.raises(ValueError, match=re.escape(label)):
        read_url_list(text)


def test_read_url_list_only_comments_is_empty() -> None:
    with pytest.raises(ValueError, match="no chapter links found"):
        read_url_list("# a\n\n   \n# b\n")


# --- select_chapters ---


@pytest.fixture
def sources() -> list[ChapterSource]:
    """Chapter 10 .. Chapter 20."""
    return numbered(10, 20)


def test_select_all(sources: list[ChapterSource]) -> None:
    for spec in (None, "", "   ", "all", "ALL"):
        assert select_chapters(sources, spec) == sources


def test_select_single_number(sources: list[ChapterSource]) -> None:
    assert names(select_chapters(sources, "12")) == ["Chapter 12"]


def test_select_closed_range(sources: list[ChapterSource]) -> None:
    assert names(select_chapters(sources, "12-14")) == ["Chapter 12", "Chapter 13", "Chapter 14"]


def test_select_open_ranges(sources: list[ChapterSource]) -> None:
    assert names(select_chapters(sources, "18-")) == ["Chapter 18", "Chapter 19", "Chapter 20"]
    assert names(select_chapters(sources, "-11")) == ["Chapter 10", "Chapter 11"]


def test_select_tokens_allow_spaces(sources: list[ChapterSource]) -> None:
    assert names(select_chapters(sources, "11, 13-14 ,20")) == [
        "Chapter 11",
        "Chapter 13",
        "Chapter 14",
        "Chapter 20",
    ]


def test_select_overlapping_tokens_no_duplicates(sources: list[ChapterSource]) -> None:
    assert names(select_chapters(sources, "12-14,13")) == ["Chapter 12", "Chapter 13", "Chapter 14"]


def test_select_by_position(sources: list[ChapterSource]) -> None:
    assert names(select_chapters(sources, "1-3", by="position")) == [
        "Chapter 10",
        "Chapter 11",
        "Chapter 12",
    ]


def test_select_auto_falls_back_to_position() -> None:
    chapter_sources = [
        ChapterSource("Prologue", "https://x.test/p"),
        ChapterSource("Chapter 10", "https://x.test/10"),
    ]
    assert names(select_chapters(chapter_sources, "1-2")) == ["Prologue", "Chapter 10"]


def test_select_range_partly_outside_list_is_fine(sources: list[ChapterSource]) -> None:
    assert select_chapters(sources, "10-99") == sources


def test_select_decimal_chapter_matched_only_by_range() -> None:
    chapter_sources = [
        ChapterSource("Chapter 5", "https://x.test/5"),
        ChapterSource("Chapter 5.5", "https://x.test/5-5"),
        ChapterSource("Chapter 6", "https://x.test/6"),
    ]
    assert names(select_chapters(chapter_sources, "5")) == ["Chapter 5"]
    assert names(select_chapters(chapter_sources, "5-6")) == ["Chapter 5", "Chapter 5.5", "Chapter 6"]


@pytest.mark.parametrize("spec", ["5-2", "1,,3", "abc", "1-x", "--3"])
def test_select_invalid_tokens(sources: list[ChapterSource], spec: str) -> None:
    with pytest.raises(ValueError, match="selection:"):
        select_chapters(sources, spec)


def test_select_token_matching_nothing(sources: list[ChapterSource]) -> None:
    with pytest.raises(ValueError, match=re.escape("selection: nothing matches '99'")):
        select_chapters(sources, "99")
