"""Assemble per-line readings back into regions (pure; inputs are never mutated)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from omniscan.core.schemas import BBox, Lang, OcrLine, Region
from omniscan.detect.postprocess import Box
from omniscan.ocr.lines import LineBox


def build_ocr_regions(
    regions: Sequence[Region],
    by_region: Mapping[str, Sequence[LineBox]],
    readings: Mapping[tuple[str, int], tuple[str, float]],
    *,
    engine: str,
    lang: Lang,
    strip_width: int,
    strip_height: int,
) -> list[Region]:
    """New regions in input order with `lines`/`text`/`confidence` filled from per-line readings.

    A line whose text is empty after `.strip()` is dropped; `confidence` is the minimum surviving
    line score (0.0 without lines).
    """
    out: list[Region] = []
    for region in regions:
        lines: list[OcrLine] = []
        for index, line in enumerate(by_region.get(region.id, ())):
            text, score = readings[(region.id, index)]
            stripped = text.strip()
            if not stripped:
                continue
            lines.append(
                OcrLine(
                    bbox=_line_bbox(line.box, strip_width, strip_height),
                    text=stripped,
                    score=score,
                    engine=engine,
                )
            )
        out.append(
            region.model_copy(
                update={
                    "lines": lines,
                    "text": "\n".join(line.text for line in lines),
                    "confidence": round(min(line.score for line in lines), 4) if lines else 0.0,
                    "lang": lang,
                    "orientation": "h",
                    "ocr_alt": None,
                }
            )
        )
    return out


def _line_bbox(box: Box, strip_width: int, strip_height: int) -> BBox:
    """The line box rounded outwards (floor x0/y0, ceil x1/y1) and clamped to the strip."""
    x0 = max(0, math.floor(box[0]))
    y0 = max(0, math.floor(box[1]))
    x1 = min(strip_width, math.ceil(box[2]))
    y1 = min(strip_height, math.ceil(box[3]))
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
