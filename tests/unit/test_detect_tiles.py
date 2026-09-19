"""Golden values and coverage properties for detect tile planning (card C3)."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np
import pytest

from omniscan.detect.tiles import Tile, keep_tiles, plan_tiles


def flags(tile: Tile) -> tuple[bool, bool, bool, bool]:
    return tile.left, tile.top, tile.right, tile.bottom


def test_plan_tall_strip_golden() -> None:
    tiles = plan_tiles(1000, 3000, 1280, 0.5)
    assert [(t.x0, t.y0, t.x1, t.y1, *flags(t)) for t in tiles] == [
        (0, 0, 1000, 1000, False, False, False, True),
        (0, 500, 1000, 1500, False, True, False, True),
        (0, 1000, 1000, 2000, False, True, False, True),
        (0, 1500, 1000, 2500, False, True, False, True),
        (0, 2000, 1000, 3000, False, True, False, False),
    ]


def test_plan_two_rows_three_columns_golden() -> None:
    tiles = plan_tiles(2000, 1500, 1280, 0.5)
    assert len(tiles) == 6
    assert [(t.x0, t.y0, t.x1, t.y1, *flags(t)) for t in tiles] == [
        (0, 0, 1280, 1280, False, False, True, True),
        (640, 0, 1920, 1280, True, False, True, True),
        (720, 0, 2000, 1280, True, False, False, True),
        (0, 220, 1280, 1500, False, True, True, False),
        (640, 220, 1920, 1500, True, True, True, False),
        (720, 220, 2000, 1500, True, True, False, False),
    ]


def test_plan_single_tiles() -> None:
    assert [(t.x0, t.y0, t.x1, t.y1, *flags(t)) for t in plan_tiles(800, 800, 1280, 0.5)] == [
        (0, 0, 800, 800, False, False, False, False)
    ]
    assert [(t.x0, t.y0, t.x1, t.y1, *flags(t)) for t in plan_tiles(800, 300, 1280, 0.5)] == [
        (0, 0, 800, 300, False, False, False, False)
    ]


@pytest.mark.parametrize(
    ("width", "height", "tile_px", "overlap"),
    [(0, 100, 1280, 0.5), (100, 0, 1280, 0.5), (-5, 100, 1280, 0.5), (100, -5, 1280, 0.5)],
)
def test_plan_rejects_empty_strip(width: int, height: int, tile_px: int, overlap: float) -> None:
    with pytest.raises(ValueError, match="strip"):
        plan_tiles(width, height, tile_px, overlap)


def test_plan_rejects_bad_tile_px_and_overlap() -> None:
    with pytest.raises(ValueError, match="tile_px"):
        plan_tiles(800, 800, 63, 0.5)
    with pytest.raises(ValueError, match="overlap"):
        plan_tiles(800, 800, 1280, 0.9)
    with pytest.raises(ValueError, match="overlap"):
        plan_tiles(800, 800, 1280, -0.1)


def _assert_covers(tiles: list[Tile], width: int, height: int, tile_px: int) -> None:
    rows: dict[int, list[Tile]] = defaultdict(list)
    for t in tiles:
        assert 0 <= t.x0 < t.x1 <= width and 0 <= t.y0 < t.y1 <= height
        assert t.x1 - t.x0 <= tile_px and t.y1 - t.y0 <= tile_px
        rows[t.y0].append(t)
    ys = sorted(rows)
    prev_y1 = 0
    for y in ys:
        row = sorted(rows[y], key=lambda t: t.x0)
        assert row[0].x0 == 0 and row[-1].x1 == width
        assert y <= prev_y1  # no vertical gap between tile rows
        prev_y1 = row[0].y1
        for a, b in pairwise(row):
            assert a.x1 >= b.x0  # no horizontal gap within the row
    assert prev_y1 == height


@pytest.mark.parametrize("seed", range(20))
def test_plan_covers_strip(seed: int) -> None:
    rng = np.random.default_rng(seed)
    width, height = (int(v) for v in rng.integers(32, 3000, size=2))
    tile_px = int(rng.integers(64, 1300))
    overlap = float(rng.uniform(0.0, 0.89))
    _assert_covers(plan_tiles(width, height, tile_px, overlap), width, height, tile_px)


def test_keep_tiles_by_active_ranges() -> None:
    tiles = plan_tiles(1000, 3000, 1280, 0.5)
    kept = keep_tiles(tiles, [(0, 400), (2600, 3000)])
    assert [t.y0 for t in kept] == [0, 2000]
    assert keep_tiles(tiles, []) == []
    # (1000, 1100) touches the first tile's end edge exactly: that tile is dropped, the rest stay
    assert [t.y0 for t in keep_tiles(tiles, [(1000, 1100)])] == [500, 1000]
