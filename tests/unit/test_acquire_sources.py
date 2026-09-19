"""Tests for omniscan.acquire.sources."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from omniscan.acquire.sources import ChapterSource, load_sources, sources_path


def write(tmp_path: Path, text: str) -> Path:
    """Write a sources.toml into tmp_path and return its path."""
    path = tmp_path / "sources.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_sources_path() -> None:
    assert sources_path(Path("lib"), "Series") == Path("lib") / "Series" / "sources.toml"


def test_explicit_then_template_order(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [[chapter]]
        name = "Chapter 1"
        url = "https://example.org/a/1"

        [[chapter]]
        name = "Chapter 1.5"
        url = "https://example.org/a/1-5"

        [template]
        url = "https://example.org/a/{n}"
        first = 2
        last = 4
        """,
    )
    assert load_sources(path) == [
        ChapterSource("Chapter 1", "https://example.org/a/1"),
        ChapterSource("Chapter 1.5", "https://example.org/a/1-5"),
        ChapterSource("Chapter 2", "https://example.org/a/2"),
        ChapterSource("Chapter 3", "https://example.org/a/3"),
        ChapterSource("Chapter 4", "https://example.org/a/4"),
    ]


def test_template_zero_padded_and_custom_name(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [template]
        url = "https://example.org/a/{n:03d}"
        first = 2
        last = 3
        name = "Ep {n}"
        """,
    )
    assert load_sources(path) == [
        ChapterSource("Ep 2", "https://example.org/a/002"),
        ChapterSource("Ep 3", "https://example.org/a/003"),
    ]


def test_template_default_name(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [template]
        url = "https://example.org/a/{n}"
        first = 7
        last = 8
        """,
    )
    assert load_sources(path) == [
        ChapterSource("Chapter 7", "https://example.org/a/7"),
        ChapterSource("Chapter 8", "https://example.org/a/8"),
    ]


def test_empty_file_gives_no_chapters(tmp_path: Path) -> None:
    assert load_sources(write(tmp_path, "")) == []


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sources(tmp_path / "sources.toml")


def test_unknown_keys_and_tables_ignored(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [site]
        name = "Example"

        [[chapter]]
        name = "Chapter 1"
        url = "https://example.org/a/1"
        note = "hi"

        [template]
        url = "https://example.org/a/{n}"
        first = 2
        last = 3
        step = 2
        """,
    )
    assert [source.name for source in load_sources(path)] == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_invalid_toml_propagates(tmp_path: Path) -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        load_sources(write(tmp_path, "[[chapter"))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[[chapter]]\nurl = "https://example.org/a"\n', "chapter #1: missing 'name'"),
        ('[[chapter]]\nname = "Chapter 1"\n', "chapter #1: missing 'url'"),
        ("[template]\nfirst = 1\nlast = 2\n", "template: missing 'url'"),
        ('[template]\nurl = "https://example.org/a/{n}"\nlast = 2\n', "template: missing 'first'"),
        ('[template]\nurl = "https://example.org/a/{n}"\nfirst = 1\n', "template: missing 'last'"),
        (
            '[template]\nurl = "https://example.org/a/{n}"\nfirst = 5\nlast = 3\n',
            "template: first (5) is greater than last (3)",
        ),
        (
            '[template]\nurl = "https://example.org/a/{n}"\nfirst = 1\nlast = 1001\n',
            "template: more than 1000 chapters",
        ),
        (
            '[[chapter]]\nname = "Chapter 1"\nurl = "https://example.org/a/1"\n\n[[chapter]]\nname = "Chapter 2"\nurl = "ftp://x/2"\n',
            "chapter #2: url must be http(s)",
        ),
        ('[template]\nurl = "ftp://x/{n}"\nfirst = 1\nlast = 2\n', "template: url must be http(s)"),
        ('[[chapter]]\nname = "  "\nurl = "https://example.org/a"\n', "unsafe chapter name '  '"),
        ('[[chapter]]\nname = "a/b"\nurl = "https://example.org/a"\n', "unsafe chapter name 'a/b'"),
        ('[[chapter]]\nname = "a\\\\b"\nurl = "https://example.org/a"\n', "unsafe chapter name 'a\\\\b'"),
        ('[[chapter]]\nname = "."\nurl = "https://example.org/a"\n', "unsafe chapter name '.'"),
        ('[[chapter]]\nname = ".."\nurl = "https://example.org/a"\n', "unsafe chapter name '..'"),
        (
            '[template]\nurl = "https://example.org/a/{n}"\nfirst = 1\nlast = 2\nname = "x/{n}"\n',
            "unsafe chapter name 'x/1'",
        ),
        (
            '[[chapter]]\nname = "Chapter 1"\nurl = "https://example.org/a"\n'
            '[template]\nurl = "https://example.org/a/{n}"\nfirst = 1\nlast = 2\n',
            "duplicate chapter name 'Chapter 1'",
        ),
    ],
)
def test_invalid_sources_raise(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        load_sources(write(tmp_path, text))


def test_thousand_template_chapters_ok(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [template]
        url = "https://example.org/a/{n}"
        first = 1
        last = 1000
        """,
    )
    sources = load_sources(path)
    assert len(sources) == 1000
    assert sources[-1] == ChapterSource("Chapter 1000", "https://example.org/a/1000")
