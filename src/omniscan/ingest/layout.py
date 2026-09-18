"""Strip layout: dominant width and the y-ranges each image occupies in the stitched strip."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def dominant_width(widths: Sequence[int]) -> int:
    """The most frequent width; ties broken by the larger width."""
    if not widths:
        raise ValueError("widths is empty")
    counts = Counter(widths)
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def stack_layout(sizes: Sequence[tuple[int, int]], strip_width: int) -> list[tuple[float, int, int]]:
    """Scale each (width, height) to `strip_width` and stack vertically; return [(scale, y0, y1), ...]."""
    layout: list[tuple[float, int, int]] = []
    y = 0
    for width, height in sizes:
        scale = strip_width / width
        scaled_height = max(1, round(height * scale))
        layout.append((scale, y, y + scaled_height))
        y += scaled_height
    return layout
