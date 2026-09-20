"""Real manga-ocr on the GPU: rendered Japanese crops -> text (card O1b, test 9).

Skipped unless `ocr-rec-manga-ocr-2025` is installed in the models dir and a font in `fonts/`
actually has Japanese glyphs — the test never downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from omniscan.core.config import OcrConfig
from omniscan.gpu.device import resolve_device
from omniscan.models.resolve import local_model_source
from omniscan.ocr.crop_readers import MangaOcrReader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO = "jzhang533/manga-ocr-base-2025"

_TEXTS = (
    "あれしまったまどをあけたまま",
    "つよいかぜのおとがするよ",
)  # kana only: comic fonts rarely carry kanji


def _glyph(font: ImageFont.FreeTypeFont, char: str) -> bytes:
    """The pixels the font draws for `char` (identical for every character the font lacks)."""
    image = Image.new("L", (80, 80), 0)
    ImageDraw.Draw(image).text((10, 10), char, font=font, fill=255)
    return image.tobytes()


def _font_with_japanese() -> ImageFont.FreeTypeFont:
    """The first font in `fonts/` with a glyph for every character of the test texts."""
    chars = {c for text in _TEXTS for c in text}
    for path in sorted((_REPO_ROOT / "fonts").glob("*.tt[fc]")):
        font = ImageFont.truetype(str(path), 48)
        missing = _glyph(font, "􏿾")  # what the font draws for a character it does not have
        if all(_glyph(font, c) != missing for c in chars):
            return font
    raise FileNotFoundError("no font in fonts/ can render the kana of the test texts")


def _crop(text: str, font: ImageFont.FreeTypeFont) -> torch.Tensor:
    """A tight white uint8 [3, h, w] crop of `text` drawn in black, wrapped to 4 characters per line.

    manga-ocr squashes every crop to 224x224, so like a real speech bubble the crop must be compact:
    a 15:1 single line is unreadable to it.
    """
    lines = [text[i : i + 4] for i in range(0, len(text), 4)]
    image = Image.new("RGB", (400, 60 * len(lines) + 40), "white")
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(lines):
        draw.text((20, 20 + 60 * row), line, font=font, fill="black")
    x0, y0, x1, y1 = ImageOps.invert(image.convert("L")).getbbox() or (
        0,
        0,
        image.width - 1,
        image.height - 1,
    )
    tight = image.crop((max(0, x0 - 8), max(0, y0 - 8), min(image.width, x1 + 8), min(image.height, y1 + 8)))
    return torch.from_numpy(np.asarray(tight, dtype=np.uint8)).permute(2, 0, 1).contiguous()


@pytest.mark.gpu
def test_manga_ocr_reads_rendered_japanese_crops() -> None:
    models_dir = _REPO_ROOT / "models"
    if local_model_source(_REPO, models_dir) is None:
        pytest.skip(
            f"{_REPO} is not installed in {models_dir} - run 'omniscan models download ocr-rec-manga-ocr-2025'"
        )
    try:
        font = _font_with_japanese()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    device = resolve_device()
    reader = MangaOcrReader.load(OcrConfig(engine="manga_ocr"), device, models_dir=models_dir)
    crops = [_crop(text, font).to(device) for text in _TEXTS]  # on the GPU: the processor round trips

    readings = reader.read(crops)

    for truth, (text, score) in zip(_TEXTS, readings, strict=True):
        assert 0.0 < score <= 1.0
        wanted = [c for c in truth if not c.isspace()]
        found = sum(1 for c in wanted if c in text)
        assert found >= len(wanted) / 2, f"{text!r} vs {truth!r}: {found}/{len(wanted)} characters"
    torch.cuda.empty_cache()
