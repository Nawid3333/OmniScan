"""Assemble per-line readings back into regions (pure except for the text-colour sample; inputs
are never mutated)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from omniscan.core.schemas import RGB, BBox, Lang, OcrLine, Region
from omniscan.detect.postprocess import Box
from omniscan.ocr.lines import LineBox

_MIN_INK_PX = 16  # a smaller minority cluster is more likely noise/anti-aliasing than real ink


def build_ocr_regions(
    regions: Sequence[Region],
    by_region: Mapping[str, Sequence[LineBox]],
    readings: Mapping[tuple[str, int], tuple[str, float]],
    strip: torch.Tensor,
    *,
    engine: str,
    lang: Lang,
    strip_width: int,
    strip_height: int,
) -> list[Region]:
    """New regions in input order with `lines`/`text`/`confidence` filled from per-line readings.

    A line whose text is empty after `.strip()` is dropped; `confidence` is the minimum surviving
    line score (0.0 without lines). `text_color` is sampled from `strip` at the region's own line
    boxes when the region does not already have one (a pre-set value is always kept as-is); a region
    with no lines, too small a crop, or a degenerate colour split keeps whatever `text_color` it
    already had (usually None, left to the typeset stage's own kind-based default).
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
        text_color = region.text_color
        if text_color is None:
            boxes = [line.box for line in by_region.get(region.id, ())]
            text_color = _sample_text_color(strip, boxes)
        out.append(
            region.model_copy(
                update={
                    "lines": lines,
                    "text": "\n".join(line.text for line in lines),
                    "confidence": round(min(line.score for line in lines), 4) if lines else 0.0,
                    "lang": lang,
                    "orientation": "h",
                    "ocr_alt": None,
                    "text_color": text_color,
                }
            )
        )
    return out


def _sample_text_color(strip: torch.Tensor, boxes: Sequence[Box]) -> RGB | None:
    """The real colour of a region's own text ink, sampled from its OCR line boxes on `strip`.

    Splits the union of `boxes`' pixels into two clusters by an Otsu threshold on luminance and
    returns the smaller cluster's median colour: real ink almost always covers less area within a
    line's own tight box than the background peeking through the gaps between letters, regardless of
    whether the ink is dark-on-light or light-on-dark. Validated on three real Solo Leveling regions
    this way: a white-on-navy caption (ink was the ~15% bright cluster), a dark-red-on-white title
    (ink was the ~17% dark-red cluster) and ordinary black-on-white dialogue (ink was the ~15% dark
    cluster) — the minority cluster was the correct ink colour every time (`docs/reports/` has none
    of this yet; it was gathered interactively against `data/raws/SoloLeveling`, not from a card).
    None when there are no boxes, the sample is smaller than `_MIN_INK_PX`, or the split is
    degenerate (every pixel landed on one side — nothing to separate).
    """
    crops = [
        strip[
            :,
            max(0, math.floor(y0)) : min(strip.shape[-2], math.ceil(y1)),
            max(0, math.floor(x0)) : min(strip.shape[-1], math.ceil(x1)),
        ]
        for x0, y0, x1, y1 in boxes
    ]
    crops = [crop for crop in crops if crop.numel() > 0]
    if not crops:
        return None
    pixels = torch.cat([crop.reshape(3, -1) for crop in crops], dim=1).float()
    if pixels.shape[1] < _MIN_INK_PX:
        return None
    luminance = 0.299 * pixels[0] + 0.587 * pixels[1] + 0.114 * pixels[2]
    threshold = _otsu_threshold(luminance)
    low = luminance < threshold
    low_count = int(low.sum())
    high_count = pixels.shape[1] - low_count
    if low_count == 0 or high_count == 0:
        return None
    minority = pixels[:, low] if low_count < high_count else pixels[:, ~low]
    if minority.shape[1] < _MIN_INK_PX:
        return None
    values = minority.median(dim=1).values.round().tolist()
    return (int(values[0]), int(values[1]), int(values[2]))


def _otsu_threshold(luminance: torch.Tensor) -> float:
    """The luminance cutoff maximising between-class variance of a two-way split (Otsu's method).

    `luminance < cutoff` is the "low" side, `luminance >= cutoff` the "high" side: `torch.histc`'s
    256 bins each span exactly one luminance unit ([i, i+1) for bin i), so the bin index maximising
    variance must be offset by +1 to become a correct exclusive cutoff — a value sitting exactly on
    an integer luminance (e.g. 40.0, in bin 40) must land on the same side as the rest of its bin.
    """
    hist = torch.histc(luminance, bins=256, min=0, max=256)
    weights = torch.arange(256, dtype=torch.float32, device=hist.device)
    cum_weight = torch.cumsum(hist, dim=0)
    cum_moment = torch.cumsum(hist * weights, dim=0)
    total_weight = cum_weight[-1]
    total_moment = cum_moment[-1]
    weight_lo = cum_weight.clamp(min=1e-6)
    weight_hi = (total_weight - cum_weight).clamp(min=1e-6)
    mean_lo = cum_moment / weight_lo
    mean_hi = (total_moment - cum_moment) / weight_hi
    variance = cum_weight * (total_weight - cum_weight) * (mean_lo - mean_hi) ** 2
    return float(torch.argmax(variance)) + 1.0


def _line_bbox(box: Box, strip_width: int, strip_height: int) -> BBox:
    """The line box rounded outwards (floor x0/y0, ceil x1/y1) and clamped to the strip."""
    x0 = max(0, math.floor(box[0]))
    y0 = max(0, math.floor(box[1]))
    x1 = min(strip_width, math.ceil(box[2]))
    y1 = min(strip_height, math.ceil(box[3]))
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
