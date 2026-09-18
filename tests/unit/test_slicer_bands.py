"""Unit tests for row_stats and find_uniform_bands."""

import torch

from omniscan.slicer.bands import find_uniform_bands, row_stats
from tests.fixtures.strips import art, gradient, noisy_solid, solid, stack

W = 720


def _bands(strip: torch.Tensor) -> list:
    return find_uniform_bands(row_stats(strip, 10), min_band=50, tol=10, max_drift=2.0)


def test_no_band_below_min_px_and_one_at_min() -> None:
    short = stack(art(300, W, 0), solid(49, W, (255, 255, 255)), art(300, W, 1))
    assert _bands(short) == []

    exact = stack(art(300, W, 0), solid(50, W, (255, 255, 255)), art(300, W, 1))
    bands = _bands(exact)
    assert len(bands) == 1
    assert (bands[0].y0, bands[0].y1) == (300, 350)
    assert bands[0].color == (255, 255, 255)


def test_solid_band_not_gradient() -> None:
    strip = stack(art(300, W, 0), solid(200, W, (255, 255, 255)), art(300, W, 1))
    bands = _bands(strip)
    assert len(bands) == 1
    assert bands[0].y1 - bands[0].y0 == 200
    assert bands[0].is_gradient is False


def test_noisy_solid_is_uniform() -> None:
    strip = stack(art(300, W, 0), noisy_solid(150, W, (255, 255, 255), amp=6, seed=3), art(300, W, 1))
    bands = _bands(strip)
    assert len(bands) == 1
    assert bands[0].y1 - bands[0].y0 == 150


def test_gradient_band_is_gradient() -> None:
    strip = stack(art(300, W, 0), gradient(300, W, (0, 0, 0), (255, 255, 255)), art(300, W, 1))
    bands = _bands(strip)
    assert len(bands) == 1
    assert bands[0].y1 - bands[0].y0 == 300
    assert bands[0].is_gradient is True


def test_hard_colour_jump_splits_bands() -> None:
    strip = stack(
        art(300, W, 0),
        solid(100, W, (200, 200, 200)),
        solid(100, W, (40, 40, 40)),
        art(300, W, 1),
    )
    bands = _bands(strip)
    assert [(b.y0, b.y1) for b in bands] == [(300, 400), (400, 500)]
    assert [b.color for b in bands] == [(200, 200, 200), (40, 40, 40)]


def test_row_stats_chunking_is_transparent() -> None:
    strip = stack(art(200, W, 0), gradient(100, W, (10, 10, 10), (250, 250, 250)), solid(200, W, (5, 5, 5)))
    a = row_stats(strip, 10, chunk_rows=7)
    b = row_stats(strip, 10, chunk_rows=8192)
    assert torch.equal(a.median, b.median)
    assert torch.equal(a.uniform, b.uniform)
    assert torch.equal(a.detail, b.detail)
