"""Real manga-ocr on the GPU: rendered Japanese crops -> text (card O1b, test 9).

Skipped unless `ocr-rec-manga-ocr-2025` is installed in the models dir and a font in `fonts/`
actually has Japanese glyphs — the test never downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.config import OcrConfig
from omniscan.gpu.device import resolve_device
from omniscan.models.resolve import local_model_source
from omniscan.ocr.crop_readers import MangaOcrReader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO = "jzhang533/manga-ocr-base-2025"

_TEXTS = ("窓を開けたまま寝ちゃった", "強い風の音がする")


def _font_with_japanese() -> ImageFont.FreeTypeFont:
    """The first font in `fonts/` whose mask for あ is non-empty."""
    for path in sorted((_REPO_ROOT / "fonts").glob("*.tt[fc]")):
        font = ImageFont.truetype(str(path), 48)
        if font.getmask("あ").getbbox() is not None:
            return font
    raise FileNotFoundError("no font in fonts/ can render Japanese kana")


def _crop(text: str, font: ImageFont.FreeTypeFont) -> torch.Tensor:
    """A tight white uint8 [3, h, w] crop of `text` drawn in black."""
    image = Image.new("RGB", (900, 120), "white")
    ImageDraw.Draw(image).text((20, 20), text, font=font, fill="black")
    x0, y0, x1, y1 = image.getbbox() or (0, 0, 899, 119)
    tight = image.crop((max(0, x0 - 8), max(0, y0 - 8), min(image.width, x1 + 8), min(image.height, y1 + 8)))
    return torch.from_numpy(np.asarray(tight, dtype=np.uint8)).permute(2, 0, 1).contiguous()


@pytest.mark.gpu
def test_manga_ocr_reads_rendered_japanese_crops() -> None:
    models_dir = _REPO_ROOT / "models"
    if local_model_source(_REPO, models_dir) is None:
        pytest.skip(
            f"{_REPO} is not installed in {models_dir} - run 'omniscan models download ocr-rec-manga-ocr-2025'"
        )
    font = _font_with_japanese()

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
