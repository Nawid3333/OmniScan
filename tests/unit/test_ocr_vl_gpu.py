"""Real PaddleOCR-VL on the GPU: rendered English crops -> text (card O1d, test 6).

Skipped unless `ocr-vl-1.6` is installed in the models dir — the test never downloads.
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
from omniscan.ocr.crop_readers import PaddleOcrVlReader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO = "PaddlePaddle/PaddleOCR-VL-1.6"

_TEXTS = (
    "I even brought my clothes, hat, and potions!",
    "Wow, the potion really worked this time!",
)


def _wrap(text: str, width: int = 18) -> list[str]:
    """Lines of at most `width` characters, broken between words (like a speech bubble)."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _crop(text: str, font: ImageFont.FreeTypeFont) -> torch.Tensor:
    """A tight white uint8 [3, h, w] crop of `text` wrapped to lines of <= 18 characters."""
    lines = _wrap(text)
    image = Image.new("RGB", (420, 64 * len(lines) + 40), "white")
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(lines):
        draw.text((20, 20 + 64 * row), line, font=font, fill="black")
    x0, y0, x1, y1 = ImageOps.invert(image.convert("L")).getbbox() or (
        0,
        0,
        image.width - 1,
        image.height - 1,
    )
    tight = image.crop((max(0, x0 - 8), max(0, y0 - 8), min(image.width, x1 + 8), min(image.height, y1 + 8)))
    return torch.from_numpy(np.asarray(tight, dtype=np.uint8)).permute(2, 0, 1).contiguous()


@pytest.mark.gpu
def test_paddleocr_vl_reads_rendered_english_crops() -> None:
    models_dir = _REPO_ROOT / "models"
    if local_model_source(_REPO, models_dir) is None:
        pytest.skip(f"{_REPO} is not installed in {models_dir} - run 'omniscan models download ocr-vl-1.6'")
    font = ImageFont.truetype(str(_REPO_ROOT / "fonts" / "ComicNeue-Regular.ttf"), 36)

    device = resolve_device()
    reader = PaddleOcrVlReader.load(OcrConfig(engine="paddleocr_vl"), device, models_dir=models_dir)
    crops = [_crop(text, font).to(device) for text in _TEXTS]  # on the GPU: prepare_vl_image round trips

    readings = reader.read(crops)

    for truth, (text, score) in zip(_TEXTS, readings, strict=True):
        assert 0.0 < score <= 1.0
        wanted = [c.lower() for c in truth if c.isalnum()]
        found = sum(1 for c in wanted if c in text.lower())
        assert found >= 0.6 * len(wanted), f"{text!r} vs {truth!r}: {found}/{len(wanted)} characters"
    torch.cuda.empty_cache()
