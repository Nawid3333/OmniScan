"""Unit tests for omniscan.typeset.fonts (CPU only)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from omniscan.typeset.fonts import DEFAULT_FONT_FILES, default_font_path, fonts_dir, load_font


def test_default_font_files_table() -> None:
    assert DEFAULT_FONT_FILES == {
        "dialogue": "ComicNeue-Bold.ttf",
        "thought": "PatrickHand-Regular.ttf",
        "shout": "Bangers-Regular.ttf",
        "narration": "ComicNeue-Regular.ttf",
        "free": "ComicNeue-Bold.ttf",
        "sfx": "Bangers-Regular.ttf",
    }


def test_default_font_path_dialogue_exists() -> None:
    path = default_font_path("dialogue")
    assert path == fonts_dir() / "ComicNeue-Bold.ttf"
    assert path.is_file()


def test_default_font_path_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMNISCAN_FONTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="font not found"):
        default_font_path("dialogue")


def test_default_font_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shutil.copyfile(fonts_dir() / "ComicNeue-Bold.ttf", tmp_path / "ComicNeue-Bold.ttf")
    monkeypatch.setenv("OMNISCAN_FONTS_DIR", str(tmp_path))
    assert default_font_path("dialogue") == tmp_path / "ComicNeue-Bold.ttf"


def test_load_font_measures_and_caches() -> None:
    path = default_font_path("dialogue")
    font = load_font(path, 20)
    assert font.getlength("Hello") > 0
    assert load_font(path, 20) is font
    assert load_font(path, 24) is not font


def test_load_font_size_validation() -> None:
    with pytest.raises(ValueError, match="size_px"):
        load_font(default_font_path("dialogue"), 0)


def test_load_font_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="font not found"):
        load_font(tmp_path / "missing.ttf", 20)
