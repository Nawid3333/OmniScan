"""Font file discovery, lettering style presets and caching for the typesetter."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Literal

from PIL import ImageFont

from omniscan.core.schemas import FontRole

LetteringStyle = Literal["webtoon", "manga"]
SfxWeight = Literal["heavy", "bold", "light"]

# Open-licensed lettering presets per style and font role (see fonts/README.md). "webtoon": the clean,
# rounded mixed-case look of official English webtoon releases; "manga": the slanted hand-lettered
# capitals of official English manga (the role CC Wild Words plays there). Users with licensed
# lettering fonts override single roles in [typeset] (font_dialogue, ...).
STYLE_FONTS: dict[LetteringStyle, dict[FontRole, str]] = {
    "webtoon": {
        "dialogue": "Mali-SemiBold.ttf",
        "thought": "Mali-MediumItalic.ttf",
        "shout": "Mali-Bold.ttf",
        "narration": "Mali-SemiBold.ttf",
        "free": "Mali-Bold.ttf",
        "sfx": "Knewave-Regular.ttf",
    },
    "manga": {
        "dialogue": "Kalam-Bold.ttf",
        "thought": "Kalam-Regular.ttf",
        "shout": "Bangers-Regular.ttf",
        "narration": "Kalam-Regular.ttf",
        "free": "Kalam-Bold.ttf",
        "sfx": "Knewave-Regular.ttf",
    },
}
STYLE_UPPERCASE: dict[LetteringStyle, bool] = {"webtoon": False, "manga": True}
DEFAULT_FONT_FILES: dict[FontRole, str] = STYLE_FONTS["webtoon"]

# Sound-effect lettering by the stroke weight of the original SFX (heavy block letters, bold brush
# strokes, thin marker lines), so the English effect keeps the original's visual weight.
SFX_FONTS: dict[SfxWeight, str] = {
    "heavy": "LuckiestGuy-Regular.ttf",
    "bold": "Knewave-Regular.ttf",
    "light": "PermanentMarker-Regular.ttf",
}


def fonts_dir() -> Path:
    """Directory holding the lettering fonts: `$OMNISCAN_FONTS_DIR` or, as a development default that
    packaging will replace later, `<repo root>/fonts`."""
    env_dir = os.environ.get("OMNISCAN_FONTS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[3] / "fonts"


def font_file(name: str) -> Path:
    """A font given by file name (looked up in the fonts folder) or by absolute path; `FileNotFoundError`
    when it does not exist."""
    path = Path(name)
    if not path.is_absolute():
        path = fonts_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"font not found: {path}")
    return path


def default_font_path(role: FontRole, style: LetteringStyle = "webtoon") -> Path:
    """Path of the preset font file for a font role; `FileNotFoundError` when it is missing."""
    return font_file(STYLE_FONTS[style][role])


def layout_font_name(path: Path) -> str:
    """How a LayoutItem names its font: the bare file name for a font in the fonts folder (portable
    between machines), else the absolute path of the user's own font file."""
    return path.name if path.parent == fonts_dir() else str(path)


@functools.lru_cache(maxsize=64)
def load_font(path: Path, size_px: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at `size_px`, cached per (path, size)."""
    if size_px < 1:
        raise ValueError(f"size_px must be >= 1, got {size_px}")
    if not path.is_file():
        raise FileNotFoundError(f"font not found: {path}")
    return ImageFont.truetype(str(path), size_px)
