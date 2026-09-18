"""Synthetic strip generators for slicer tests (deterministic, uint8 [3, H, W] on CPU)."""

from __future__ import annotations

import torch


def art(h: int, w: int, seed: int) -> torch.Tensor:
    """Random uint8 noise (every row non-uniform)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, h, w), dtype=torch.int32, generator=gen).to(torch.uint8)


def solid(h: int, w: int, rgb: tuple[int, int, int]) -> torch.Tensor:
    """Solid colour block."""
    return torch.tensor(rgb, dtype=torch.uint8).view(3, 1, 1).expand(3, h, w).contiguous()


def noisy_solid(h: int, w: int, rgb: tuple[int, int, int], amp: int, seed: int) -> torch.Tensor:
    """Solid colour plus uniform integer noise in [-amp, amp], clamped to 0..255."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randint(-amp, amp + 1, (3, h, w), dtype=torch.int32, generator=gen)
    base = torch.tensor(rgb, dtype=torch.int32).view(3, 1, 1)
    return (base + noise).clamp(0, 255).to(torch.uint8)


def gradient(h: int, w: int, rgb_top: tuple[int, int, int], rgb_bottom: tuple[int, int, int]) -> torch.Tensor:
    """Each row a solid colour linearly interpolated between the two endpoints (rounded)."""
    top = torch.tensor(rgb_top, dtype=torch.float32)
    bot = torch.tensor(rgb_bottom, dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, h, dtype=torch.float32)  # [h]
    rows = (
        (top[:, None] * (1.0 - t)[None, :] + bot[:, None] * t[None, :]).round().clamp(0, 255).to(torch.uint8)
    )
    return rows[:, :, None].expand(3, h, w).contiguous()


def bubble(h: int, w: int, seed: int) -> torch.Tensor:
    """White block with a black ellipse outline and text bars; no uniform band >= 50 inside."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.full((3, h, w), 255, dtype=torch.uint8)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rx = 0.4 * w
    ry = h / 2.0 - 6.0
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    f_outer = ((xx[None, :] - cx) / rx) ** 2 + ((yy[:, None] - cy) / ry) ** 2
    f_inner = ((xx[None, :] - cx) / (rx - 2.0)) ** 2 + ((yy[:, None] - cy) / (ry - 2.0)) ** 2
    ring = (f_outer <= 1.0) & (f_inner > 1.0)
    out[:, ring] = 0
    widths = torch.randint(int(0.20 * w), int(0.30 * w), (3,), generator=gen)
    offsets = torch.randint(-int(0.30 * ry), int(0.30 * ry), (3,), generator=gen)
    bar_half_h = 4
    for width, off in zip(widths.tolist(), offsets.tolist(), strict=True):
        half = width // 2
        yc = int(cy) + off
        x0 = int(cx) - half
        x1 = int(cx) + half
        out[:, max(0, yc - bar_half_h) : yc + bar_half_h, x0:x1] = 0
    return out


def stack(*blocks: torch.Tensor) -> torch.Tensor:
    """Concatenate blocks along H (dim 1)."""
    return torch.cat(blocks, dim=1)
