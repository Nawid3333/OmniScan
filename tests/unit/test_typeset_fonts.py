"""Unit tests for omniscan.typeset.fonts (CPU only)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from omniscan.typeset.fonts import (
    DEFAULT_FONT_FILES,
    SFX_FONTS,
    STYLE_FONTS,
    STYLE_UPPERCASE,
    default_font_path,
    font_file,
    fonts_dir,
    layout_font_name,
    load_font,
)


def test_default_font_files_table() -> None:
    assert (
        DEFAULT_FONT_FILES
        == STYLE_FONTS["webtoon"]
        == {
            "dialogue": "Mali-SemiBold.ttf",
            "thought": "Mali-MediumItalic.ttf",
            "shout": "Mali-Bold.ttf",
            "narration": "Mali-SemiBold.ttf",
            "free": "Mali-Bold.ttf",
            "sfx": "Knewave-Regular.ttf",
        }
    )


def test_every_preset_font_is_shipped_with_its_licence() -> None:
    names = {name for table in STYLE_FONTS.values() for name in table.values()} | set(SFX_FONTS.values())
    for name in names:
        assert (fonts_dir() / name).is_file(), name
        family = name.split("-")[0]
        assert any(fonts_dir().glob(f"*-{family}.txt")), f"no licence file for {family}"


def test_manga_preset_letters_in_capitals() -> None:
    assert STYLE_UPPERCASE == {"webtoon": False, "manga": True}
    assert default_font_path("dialogue", "manga") == fonts_dir() / "Kalam-Bold.ttf"


def test_font_file_accepts_names_and_absolute_paths(tmp_path: Path) -> None:
    assert font_file("Kalam-Bold.ttf") == fonts_dir() / "Kalam-Bold.ttf"
    own = tmp_path / "MyLettering.ttf"
    shutil.copyfile(fonts_dir() / "Kalam-Bold.ttf", own)
    assert font_file(str(own)) == own
    assert layout_font_name(own) == str(own)  # the user's own font keeps its full path
    assert layout_font_name(fonts_dir() / "Kalam-Bold.ttf") == "Kalam-Bold.ttf"
    with pytest.raises(FileNotFoundError, match="font not found"):
        font_file(str(tmp_path / "missing.ttf"))


def test_default_font_path_dialogue_exists() -> None:
    path = default_font_path("dialogue")
    assert path == fonts_dir() / "Mali-SemiBold.ttf"
    assert path.is_file()


def test_default_font_path_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMNISCAN_FONTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="font not found"):
        default_font_path("dialogue")


def test_default_font_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shutil.copyfile(fonts_dir() / "Mali-SemiBold.ttf", tmp_path / "Mali-SemiBold.ttf")
    monkeypatch.setenv("OMNISCAN_FONTS_DIR", str(tmp_path))
    assert default_font_path("dialogue") == tmp_path / "Mali-SemiBold.ttf"


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
