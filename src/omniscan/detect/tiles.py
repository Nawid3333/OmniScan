"""Tile planning: cover the chapter strip with overlapping tiles the detector can digest.

The detector was trained on images resized to 640x640 (tall webtoons split vertically), so the strip is cut into
tiles that are as wide as the strip (capped at `tile_px`), sliding down (and across, for wide pages) with `overlap`.
With overlap 0.5 every object no larger than half a tile lies wholly inside at least one tile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tile:
    """A window of the strip in strip pixels; the flags mark sides that cut through the strip interior."""

    x0: int
    y0: int
    x1: int
    y1: int
    left: bool
    top: bool
    right: bool
    bottom: bool

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _starts(length: int, size: int, overlap: float) -> list[int]:
    """Window start offsets covering [0, length) with windows of `size`; the last one is flush with the end."""
    if length <= size:
        return [0]
    step = max(1, int(size * (1.0 - overlap)))
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return starts


def plan_tiles(width: int, height: int, tile_px: int, overlap: float) -> list[Tile]:
    """Row-major tiles of side min(tile_px, width) (height capped at the strip height) covering the strip."""
    if width <= 0 or height <= 0:
        raise ValueError(f"empty strip {width}x{height}")
    if tile_px < 64:
        raise ValueError(f"tile_px must be >= 64, got {tile_px}")
    if not 0.0 <= overlap < 0.9:
        raise ValueError(f"overlap must be in [0, 0.9), got {overlap}")
    side = min(tile_px, width)
    tile_h = min(side, height)
    tiles: list[Tile] = []
    for y in _starts(height, tile_h, overlap):
        for x in _starts(width, side, overlap):
            tiles.append(
                Tile(
                    x0=x,
                    y0=y,
                    x1=x + side,
                    y1=y + tile_h,
                    left=x > 0,
                    top=y > 0,
                    right=x + side < width,
                    bottom=y + tile_h < height,
                )
            )
    return tiles


def keep_tiles(tiles: Sequence[Tile], active: Sequence[tuple[int, int]]) -> list[Tile]:
    """Tiles that overlap at least one active (y0, y1) range; used to skip blank and filtered slices."""
    return [t for t in tiles if any(t.y0 < y1 and y0 < t.y1 for y0, y1 in active)]
