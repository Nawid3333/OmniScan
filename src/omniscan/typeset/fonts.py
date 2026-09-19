"""Font file discovery and caching for the typesetter."""

from __future__ import annotations

import functools
import os
from pathlib import Path

from PIL import ImageFont

from omniscan.core.schemas import FontRole

# Open-licensed lettering defaults per font role (see fonts/README.md).
DEFAULT_FONT_FILES: dict[FontRole, str] = {
    "dialogue": "ComicNeue-Bold.ttf",
    "thought": "PatrickHand-Regular.ttf",
    "shout": "Bangers-Regular.ttf",
    "narration": "ComicNeue-Regular.ttf",
    "free": "ComicNeue-Bold.ttf",
    "sfx": "Bangers-Regular.ttf",
}


def fonts_dir() -> Path:
    """Directory holding the lettering fonts: `$OMNISCAN_FONTS_DIR` or, as a development default that
    packaging will replace later, `<repo root>/fonts`."""
    env_dir = os.environ.get("OMNISCAN_FONTS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[3] / "fonts"


def default_font_path(role: FontRole) -> Path:
    """Path of the default font file for a font role; `FileNotFoundError` when it is missing."""
    path = fonts_dir() / DEFAULT_FONT_FILES[role]
    if not path.is_file():
        raise FileNotFoundError(f"font not found: {path}")
    return path


@functools.lru_cache(maxsize=64)
def load_font(path: Path, size_px: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at `size_px`, cached per (path, size)."""
    if size_px < 1:
        raise ValueError(f"size_px must be >= 1, got {size_px}")
    if not path.is_file():
        raise FileNotFoundError(f"font not found: {path}")
    return ImageFont.truetype(str(path), size_px)
