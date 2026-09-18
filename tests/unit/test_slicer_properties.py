"""Property tests for the slicer over seeded random strips (card B27). Bug hunt: src/ is read-only here."""

from __future__ import annotations

import functools
import itertools
import math

import pytest
import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import Band, SlicesArtifact
from omniscan.slicer.bands import RowStats, find_uniform_bands, row_stats
from omniscan.slicer.cuts import Cut, plan_cuts
from omniscan.slicer.slice import slice_strip
from tests.fixtures.strip_layouts import Segment, art_strip, random_strip
from tests.fixtures.strips import art, gradient, solid, stack

SMALL = SlicerConfig(
    band_min_px=50,
    target_height=500,
    min_height=250,
    max_height=1000,
    hard_max_height=2500,
    uniform_tol=10,
    max_drift=2.0,
)
DEFAULT = SlicerConfig()
SEEDS = range(150)


@functools.cache
def run_case(
    seed: int,
) -> tuple[torch.Tensor, list[Segment], RowStats, list[Band], list[Cut], SlicesArtifact]:
    """Build the seeded random strip and run the whole slicer pipeline with SMALL (cached per seed)."""
    strip, segments = random_strip(seed)
    stats = row_stats(strip, SMALL.uniform_tol)
    bands = find_uniform_bands(stats, SMALL.band_min_px, SMALL.uniform_tol, SMALL.max_drift)
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, SMALL)
    artifact = slice_strip(strip, SMALL)
    return strip, segments, stats, bands, cuts, artifact


def centres_and_edges(bands: list[Band], cfg: SlicerConfig, height: int) -> tuple[set[int], set[int]]:
    """Recompute plan_cuts' centre and edge candidate positions from `bands` (mirrors the P8 formula)."""
    centres = {(b.y0 + b.y1) // 2 for b in bands if b.y1 - b.y0 < cfg.min_height + cfg.band_min_px}
    edges: set[int] = set()
    for b in bands:
        if b.y1 - b.y0 >= cfg.min_height + cfg.band_min_px:
            edges.add(b.y0 + cfg.band_min_px // 2)
            edges.add(b.y1 - cfg.band_min_px // 2)
    centres = {p for p in centres if 0 < p < height}
    edges = {p for p in edges if 0 < p < height}
    return centres - edges, edges


def assert_tiling(artifact: SlicesArtifact, height: int) -> None:
    """Slices tile [0, height) exactly, are non-empty, and carry their position as index."""
    slices = artifact.slices
    assert slices, "no slices produced"
    assert slices[0].y0 == 0
    assert slices[-1].y1 == height
    for i, s in enumerate(slices):
        assert s.y0 < s.y1
        assert s.index == i
    for a, b in itertools.pairwise(slices):
        assert a.y1 == b.y0


# ---------------------------------------------------------------- P1


@pytest.mark.parametrize("seed", SEEDS)
def test_p1_band_recall_precision(seed: int) -> None:
    """Bands are exactly the solid/gradient segments at least band_min_px tall, edges included."""
    _, segments, _, bands, _, _ = run_case(seed)
    expected = {(s.y0, s.y1) for s in segments if s.kind != "art" and s.y1 - s.y0 >= 50}
    got = {(b.y0, b.y1) for b in bands}
    assert got == expected


# ---------------------------------------------------------------- P2


@pytest.mark.parametrize("seed", SEEDS)
def test_p2_band_metadata(seed: int) -> None:
    """Solid bands are not gradients, all bands are >= 50 tall, sorted and non-overlapping."""
    _, segments, _, bands, _, _ = run_case(seed)
    solid_spans = {(s.y0, s.y1) for s in segments if s.kind == "solid"}
    for b in bands:
        assert b.y1 - b.y0 >= 50
        if (b.y0, b.y1) in solid_spans:
            assert b.is_gradient is False
    assert [b.y0 for b in bands] == sorted(b.y0 for b in bands)
    for a, b in itertools.pairwise(bands):
        assert a.y1 <= b.y0


# ---------------------------------------------------------------- P3


@pytest.mark.parametrize("seed", SEEDS)
def test_p3_cuts_well_formed(seed: int) -> None:
    """Cut y values are strictly increasing and strictly inside the strip."""
    strip, _, _, _, cuts, _ = run_case(seed)
    ys = [c.y for c in cuts]
    assert ys == sorted(ys)
    assert len(ys) == len(set(ys))
    height = strip.shape[1]
    assert all(0 < y < height for y in ys)


# ---------------------------------------------------------------- P4


@pytest.mark.parametrize("seed", SEEDS)
def test_p4_non_forced_cuts_inside_band(seed: int) -> None:
    """Every non-forced cut lies inside a band and its row is uniform."""
    _, _, stats, bands, cuts, _ = run_case(seed)
    for c in cuts:
        if c.forced:
            continue
        assert any(b.y0 <= c.y < b.y1 for b in bands)
        assert bool(stats.uniform[c.y].item())


# ---------------------------------------------------------------- P5


@pytest.mark.parametrize("seed", SEEDS)
def test_p5_slices_tile_strip(seed: int) -> None:
    """Slices tile the strip exactly and the artifact records the strip dimensions."""
    strip, _, _, _, _, artifact = run_case(seed)
    assert_tiling(artifact, strip.shape[1])
    assert artifact.strip_height == strip.shape[1]
    assert artifact.strip_width == strip.shape[2]


# ---------------------------------------------------------------- P6


@pytest.mark.parametrize("seed", SEEDS)
def test_p6_flags(seed: int) -> None:
    """blank marks fully-uniform slices; forced_cut mirrors the cut list and is False for the last slice."""
    _, _, stats, _, cuts, artifact = run_case(seed)
    for i, s in enumerate(artifact.slices):
        assert s.blank == bool(stats.uniform[s.y0 : s.y1].all().item())
        if i < len(cuts):
            assert s.forced_cut == cuts[i].forced
        else:
            assert s.forced_cut is False


# ---------------------------------------------------------------- P7


@pytest.mark.parametrize("seed", SEEDS)
def test_p7_size_ceiling(seed: int) -> None:
    """No slice is taller than hard_max_height."""
    _, _, _, _, _, artifact = run_case(seed)
    for s in artifact.slices:
        assert s.y1 - s.y0 <= SMALL.hard_max_height


# ---------------------------------------------------------------- P8


@pytest.mark.parametrize("seed", SEEDS)
def test_p8_dp_optimality_consequence(seed: int) -> None:
    """A slice taller than max_height never leaves a splittable centre unused (DP optimality)."""
    strip, _, _, bands, _, artifact = run_case(seed)
    height = strip.shape[1]
    centres, _ = centres_and_edges(bands, SMALL, height)
    for s in artifact.slices:
        if s.y1 - s.y0 <= SMALL.max_height:
            continue
        bad = [
            p
            for p in centres
            if s.y0 < p < s.y1 and p - s.y0 <= SMALL.max_height and s.y1 - p <= SMALL.max_height
        ]
        assert not bad, f"oversized slice [{s.y0}, {s.y1}) leaves usable centres {bad}"


# ---------------------------------------------------------------- P9


@pytest.mark.parametrize("seed", SEEDS)
def test_p9_forced_cuts_only_when_needed(seed: int) -> None:
    """Forced cuts imply some candidate-free gap exceeds hard_max_height."""
    strip, _, _, bands, cuts, _ = run_case(seed)
    if not any(c.forced for c in cuts):
        return
    height = strip.shape[1]
    centres, edges = centres_and_edges(bands, SMALL, height)
    positions = sorted({0, height} | centres | edges)
    assert any(b - a > SMALL.hard_max_height for a, b in itertools.pairwise(positions))


# ---------------------------------------------------------------- P10


@pytest.mark.parametrize("seed", SEEDS)
def test_p10_determinism(seed: int) -> None:
    """slice_strip is deterministic: two runs on the same strip give equal artifacts."""
    strip, _, _, _, _, artifact = run_case(seed)
    assert slice_strip(strip, SMALL).model_dump() == artifact.model_dump()


# ---------------------------------------------------------------- P11


@pytest.mark.parametrize("seed", range(40))
def test_p11_chunk_independence(seed: int) -> None:
    """row_stats results do not depend on chunk_rows (chunk-boundary bug catcher)."""
    strip, _ = random_strip(seed, max_rows=1200)
    ref = row_stats(strip, 10, chunk_rows=8192)
    for k in (1, 7, 97, 8192):
        stats = row_stats(strip, 10, chunk_rows=k)
        assert torch.equal(stats.median, ref.median)
        assert torch.equal(stats.uniform, ref.uniform)
        assert torch.equal(stats.detail, ref.detail)


# ---------------------------------------------------------------- P12


def test_p12a_tall_pure_art_forced_cuts() -> None:
    """A 16000-row pure-art strip under DEFAULT yields exactly 2 forced cuts and no blank slices."""
    strip = art_strip(16000, width=8, seed=1)
    stats = row_stats(strip, DEFAULT.uniform_tol)
    bands = find_uniform_bands(stats, DEFAULT.band_min_px, DEFAULT.uniform_tol, DEFAULT.max_drift)
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, DEFAULT)
    expected = math.ceil(16000 / DEFAULT.max_height) - 1
    assert len(cuts) == expected == 2
    assert all(c.forced for c in cuts)
    artifact = slice_strip(strip, DEFAULT)
    assert_tiling(artifact, 16000)
    assert len(artifact.slices) == expected + 1
    for s in artifact.slices:
        assert s.y1 - s.y0 <= DEFAULT.hard_max_height
        assert s.blank is False
    assert [s.forced_cut for s in artifact.slices] == [True] * expected + [False]


def test_p12b_chunk_boundary_gutters() -> None:
    """An 18000-row strip with two solid gutters: centres are candidates, cuts stay in gutters."""
    gutters = [(4000, 4200), (11200, 11260)]
    strip = stack(
        art(4000, 8, 101),
        solid(200, 8, (7, 7, 7)),
        art(7000, 8, 102),
        solid(60, 8, (250, 250, 250)),
        art(6740, 8, 103),
    )
    stats = row_stats(strip, DEFAULT.uniform_tol)
    bands = find_uniform_bands(stats, DEFAULT.band_min_px, DEFAULT.uniform_tol, DEFAULT.max_drift)
    assert [(b.y0, b.y1) for b in bands] == gutters
    centres, _ = centres_and_edges(bands, DEFAULT, strip.shape[1])
    for g0, g1 in gutters:
        assert (g0 + g1) // 2 in centres
    cuts = plan_cuts(strip.shape[1], bands, stats.detail, DEFAULT)
    for c in cuts:
        if not c.forced:
            assert any(g0 <= c.y < g1 for g0, g1 in gutters)
    artifact = slice_strip(strip, DEFAULT)
    assert_tiling(artifact, strip.shape[1])
    for s in artifact.slices:
        assert s.y1 - s.y0 <= DEFAULT.hard_max_height


# ---------------------------------------------------------------- P13 (added in review: pins max_drift)


def test_p13_hard_colour_edge_splits_bands() -> None:
    """Two adjacent uniform blocks of different colour are two bands, not one (drift 255 > max_drift)."""
    strip = stack(art(300, 8, 1), solid(100, 8, (255, 255, 255)), solid(100, 8, (0, 0, 0)), art(300, 8, 2))
    stats = row_stats(strip, SMALL.uniform_tol)
    bands = find_uniform_bands(stats, SMALL.band_min_px, SMALL.uniform_tol, SMALL.max_drift)
    assert [(b.y0, b.y1) for b in bands] == [(300, 400), (400, 500)]


def test_p13_drift_boundary_is_inclusive() -> None:
    """A gradient stepping exactly max_drift per row (2.0) stays one band; stepping 3 per row splits into nothing."""
    slow = stack(art(100, 8, 3), gradient(100, 8, (0, 0, 0), (198, 198, 198)), art(100, 8, 4))
    fast = stack(art(100, 8, 5), gradient(80, 8, (0, 0, 0), (237, 237, 237)), art(100, 8, 6))
    args = (SMALL.band_min_px, SMALL.uniform_tol, SMALL.max_drift)
    slow_bands = find_uniform_bands(row_stats(slow, SMALL.uniform_tol), *args)
    fast_bands = find_uniform_bands(row_stats(fast, SMALL.uniform_tol), *args)
    assert [(b.y0, b.y1, b.is_gradient) for b in slow_bands] == [(100, 200, True)]
    assert fast_bands == []
