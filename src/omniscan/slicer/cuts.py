"""Cut planning: choose slice boundaries inside uniform bands, with forced cuts as fallback."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

from omniscan.core.config import SlicerConfig
from omniscan.core.schemas import Band


@dataclass(frozen=True, slots=True)
class Cut:
    """A slice boundary at row `y`; `forced` marks cuts placed outside a uniform band."""

    y: int
    forced: bool


def _cost(length: int, target: int, min_height: int, max_height: int) -> float:
    """Cost of a single slice segment of the given length."""
    c = ((length - target) / target) ** 2
    if length < min_height:
        c += 4.0
    if length > max_height:
        c += 100.0
    return c


def _least_detail(detail: torch.Tensor, t: int, a: int, b: int) -> int:
    """Row with minimum detail in [t-256, t+256] clipped to (a, b); ties -> closest to t, then smaller y."""
    lo = max(t - 256, a + 1)
    hi = min(t + 256, b - 1)
    if lo > hi:
        return t
    window = detail[lo : hi + 1]
    m = int(window.min().item())
    idx = torch.nonzero(window == m).squeeze(1)
    dist = (idx + lo - t).abs()
    return lo + int(idx[dist.argmin()].item())


def _dp_cuts(
    a: int,
    b: int,
    cands: list[int],
    target: int,
    min_height: int,
    max_height: int,
    hard_max: int,
) -> list[int]:
    """Choose a subset of centre candidates in (a, b) minimising total segment cost."""
    if not cands:
        return []
    pos = [a, *cands, b]
    m = len(cands)
    dp: list[tuple[float, int, tuple[int, ...]] | None] = [None] * (m + 2)
    dp[0] = (0.0, 0, ())
    for j in range(1, m + 2):
        best: tuple[float, int, tuple[int, ...]] | None = None
        for i in range(j):
            prev = dp[i]
            if prev is None:
                continue
            seg = pos[j] - pos[i]
            if seg > hard_max:
                continue
            is_cut = 1 <= i <= m
            cand = (
                prev[0] + _cost(seg, target, min_height, max_height),
                prev[1] + (1 if is_cut else 0),
                prev[2] + ((pos[i],) if is_cut else ()),
            )
            if best is None or cand < best:
                best = cand
        dp[j] = best
    best = dp[m + 1]
    assert best is not None
    return list(best[2])


def plan_cuts(height: int, bands: list[Band], detail: torch.Tensor, cfg: SlicerConfig) -> list[Cut]:
    """Plan slice boundaries from uniform bands and forced cuts (see B4 spec)."""
    min_band = cfg.band_min_px
    centres: set[int] = set()
    edges: set[int] = set()
    for band in bands:
        length = band.y1 - band.y0
        if length >= cfg.min_height + min_band:
            edges.add(band.y0 + min_band // 2)
            edges.add(band.y1 - min_band // 2)
        else:
            centres.add((band.y0 + band.y1) // 2)
    centres = {p for p in centres if 0 < p < height}
    edges = {p for p in edges if 0 < p < height}
    centres -= edges

    positions = [0, *sorted(centres | edges), height]
    forced: list[int] = []
    for a, b in itertools.pairwise(positions):
        gap = b - a
        if gap > cfg.hard_max_height:
            k = math.ceil(gap / cfg.max_height) - 1
            for i in range(1, k + 1):
                t = a + round(i * gap / (k + 1))
                forced.append(_least_detail(detail, t, a, b))

    mandatory = sorted({0, height} | edges | set(forced))
    chosen: list[int] = []
    for a, b in itertools.pairwise(mandatory):
        cands = [p for p in sorted(centres) if a < p < b]
        chosen.extend(
            _dp_cuts(a, b, cands, cfg.target_height, cfg.min_height, cfg.max_height, cfg.hard_max_height)
        )

    cuts: dict[int, bool] = {}
    for y in edges:
        cuts[y] = False
    for y in forced:
        cuts[y] = True
    for y in chosen:
        cuts.setdefault(y, False)
    return [Cut(y=y, forced=cuts[y]) for y in sorted(cuts)]
