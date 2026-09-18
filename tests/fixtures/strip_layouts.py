"""Seeded random strip layouts for slicer property tests (deterministic per seed, uint8 [3, H, W] on CPU)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch

Kind = Literal["art", "solid", "gradient"]


@dataclass(frozen=True)
class Segment:
    kind: Kind
    y0: int
    y1: int


def art_strip(rows: int, width: int, seed: int) -> torch.Tensor:
    """Pure full-range noise strip, uint8 [3, rows, width] on CPU."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, rows, width), dtype=torch.int32, generator=gen).to(torch.uint8)


def _art(h: int, width: int, gen: torch.Generator) -> torch.Tensor:
    """Full-range noise block; no art row can be uniform."""
    return torch.randint(0, 256, (3, h, width), dtype=torch.int32, generator=gen).to(torch.uint8)


def _solid(h: int, width: int, rng: random.Random, gen: torch.Generator) -> torch.Tensor:
    """One random RGB colour plus per-pixel noise in {-amp..+amp}, amp in {0, 2}, clamped."""
    c = torch.randint(0, 256, (3,), dtype=torch.int32, generator=gen)
    amp = rng.choice((0, 2))
    noise = torch.randint(-amp, amp + 1, (3, h, width), dtype=torch.int32, generator=gen)
    return (c.view(3, 1, 1) + noise).clamp(0, 255).to(torch.uint8)


def _gradient(h: int, width: int, gen: torch.Generator) -> torch.Tensor:
    """Rows step from c0 to c1 = clamp(c0 + delta); consecutive rows differ by at most 1 per channel."""
    c0 = torch.randint(0, 256, (3,), dtype=torch.int32, generator=gen)
    delta = torch.randint(-(h // 2), h // 2 + 1, (3,), dtype=torch.int32, generator=gen)
    c1 = (c0 + delta).clamp(0, 255)
    t = torch.arange(h, dtype=torch.float32) / (h - 1)
    rows = (c0.to(torch.float32)[:, None] + (c1 - c0).to(torch.float32)[:, None] * t[None, :]).round()
    return rows.to(torch.uint8)[:, :, None].expand(3, h, width).contiguous()


def random_strip(seed: int, width: int = 48, max_rows: int = 6000) -> tuple[torch.Tensor, list[Segment]]:
    """Deterministic for a given seed. Returns (uint8 [3, H, width] strip, segments in top-to-bottom order)."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    segments: list[Segment] = []
    blocks: list[torch.Tensor] = []
    total = 0
    next_is_art = rng.random() < 0.5
    while total < max_rows:
        if next_is_art:
            kind: Kind = "art"
            h = rng.randint(20, 900)
        else:
            kind = "solid" if rng.random() < 0.65 else "gradient"
            h = rng.randint(10, 400) if kind == "solid" else rng.randint(60, 400)
        remaining = max_rows - total
        if h > remaining:
            if remaining < 10:
                break
            h = remaining
        if kind == "art":
            blocks.append(_art(h, width, gen))
        elif kind == "solid":
            blocks.append(_solid(h, width, rng, gen))
        else:
            blocks.append(_gradient(h, width, gen))
        segments.append(Segment(kind=kind, y0=total, y1=total + h))
        total += h
        next_is_art = not next_is_art
    return torch.cat(blocks, dim=1), segments
