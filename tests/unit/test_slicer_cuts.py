"""Unit tests for plan_cuts."""

import itertools

import pytest
import torch

from omniscan.core.config import SlicerConfig
from omniscan.gpu.device import resolve_device
from omniscan.slicer.bands import find_uniform_bands, row_stats
from omniscan.slicer.cuts import plan_cuts
from tests.fixtures.strips import art, gradient, solid, stack

W = 720


def _plan(strip: torch.Tensor, cfg: SlicerConfig) -> list:
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    return plan_cuts(strip.shape[1], bands, stats.detail, cfg)


def _heights(strip: torch.Tensor, cuts: list) -> list[int]:
    boundaries = [0, *(c.y for c in cuts), strip.shape[1]]
    return [b - a for a, b in itertools.pairwise(boundaries)]


def test_cuts_near_target_height() -> None:
    cfg = SlicerConfig()
    blocks = []
    for k in range(27):
        blocks.append(art(1000, W, seed=k))
        blocks.append(solid(100, W, (255, 255, 255)))
    blocks.append(art(300, W, seed=1000))
    strip = stack(*blocks)
    assert strip.shape[1] == 30000

    cuts = _plan(strip, cfg)
    heights = _heights(strip, cuts)
    assert heights
    assert all(cfg.min_height <= h <= cfg.max_height for h in heights)
    assert abs(sum(heights) / len(heights) - cfg.target_height) <= 0.2 * cfg.target_height

    band_ranges = [
        (b.y0, b.y1)
        for b in find_uniform_bands(
            row_stats(strip, cfg.uniform_tol), cfg.band_min_px, cfg.uniform_tol, cfg.max_drift
        )
    ]
    for cut in cuts:
        if not cut.forced:
            assert any(y0 < cut.y < y1 for y0, y1 in band_ranges)


def test_forced_cuts_when_no_bands() -> None:
    cfg = SlicerConfig()
    strip = art(40000, W, seed=7)
    stats = row_stats(strip, cfg.uniform_tol)
    bands = find_uniform_bands(stats, cfg.band_min_px, cfg.uniform_tol, cfg.max_drift)
    assert bands == []
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, cfg)
    assert cuts
    assert all(c.forced for c in cuts)
    assert all(h <= cfg.hard_max_height for h in _heights(strip, cuts))


def test_long_uniform_block_becomes_own_slice() -> None:
    cfg = SlicerConfig()
    strip = stack(art(2000, W, 1), solid(4000, W, (0, 0, 0)), art(2000, W, 2))
    cuts = _plan(strip, cfg)
    assert [c.y for c in cuts] == [2025, 5975]
    assert all(not c.forced for c in cuts)
    assert bool((strip[:, 2025:5975, :] == 0).all())


def test_deterministic() -> None:
    cfg = SlicerConfig()
    strip = stack(
        art(1000, W, 11),
        solid(120, W, (255, 255, 255)),
        art(2500, W, 12),
        gradient(500, W, (0, 0, 0), (255, 255, 255)),
        art(1000, W, 13),
    )
    assert _plan(strip, cfg) == _plan(strip, cfg)


@pytest.mark.gpu
def test_cuts_on_gpu_match_cpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    cfg = SlicerConfig()
    strip = stack(
        art(3000, W, 21),
        solid(200, W, (255, 255, 255)),
        art(6000, W, 22),
        solid(200, W, (0, 0, 0)),
        art(3000, W, 23),
    )
    cpu = _plan(strip, cfg)
    gpu = _plan(strip.to(resolve_device()), cfg)
    assert [(c.y, c.forced) for c in cpu] == [(c.y, c.forced) for c in gpu]
