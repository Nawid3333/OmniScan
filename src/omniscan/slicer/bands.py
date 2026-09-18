"""Per-row uniformity statistics and uniform-band extraction for the slicer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from omniscan.core.schemas import Band


@dataclass(frozen=True, slots=True)
class RowStats:
    """Per-row statistics computed from a strip; tensors live on the strip's device."""

    median: torch.Tensor  # uint8 [3, H] per-channel row median (lower median, like torch.median)
    uniform: torch.Tensor  # bool [H]
    detail: torch.Tensor  # uint8 [H] = max over channels and pixels of |pixel - row median|


def row_stats(strip: torch.Tensor, tol: int, chunk_rows: int = 8192) -> RowStats:
    """Compute per-row median, uniformity and detail for a uint8 [3, H, W] strip."""
    device = strip.device
    channels, height = strip.shape[0], strip.shape[1]
    median = torch.empty((channels, height), dtype=torch.uint8, device=device)
    uniform = torch.empty(height, dtype=torch.bool, device=device)
    detail = torch.empty(height, dtype=torch.uint8, device=device)
    for start in range(0, height, chunk_rows):
        end = min(start + chunk_rows, height)
        chunk = strip[:, start:end, :]  # [3, h, W]
        med = torch.median(chunk, dim=2).values  # uint8 [3, h]
        med_i16 = med.to(torch.int16)
        lo = (med_i16 - tol).clamp(0, 255).to(torch.uint8)
        hi = (med_i16 + tol).clamp(0, 255).to(torch.uint8)
        diff = chunk.to(torch.int16) - med_i16.unsqueeze(2)  # [3, h, W]
        det = diff.abs().amax(dim=(0, 2))  # int16 [h]
        unif = ((chunk >= lo.unsqueeze(2)) & (chunk <= hi.unsqueeze(2))).all(dim=0).all(dim=1)
        median[:, start:end] = med
        detail[start:end] = det.to(torch.uint8)
        uniform[start:end] = unif
    return RowStats(median=median, uniform=uniform, detail=detail)


def find_uniform_bands(stats: RowStats, min_band: int, tol: int, max_drift: float) -> list[Band]:
    """Extract maximal runs of mutually-linked uniform rows at least `min_band` tall."""
    uniform = stats.uniform
    height = uniform.shape[0]
    if height == 0:
        return []
    device = uniform.device
    med = stats.median.to(torch.float32)  # [3, H]
    start = uniform.clone()
    if height >= 2:
        diff = (med[:, 1:] - med[:, :-1]).abs().mean(dim=0)  # [H-1]
        link = uniform[1:] & uniform[:-1] & (diff <= max_drift)
        start[1:] = uniform[1:] & ~link
        end = torch.zeros(height, dtype=torch.bool, device=device)
        end[1:] = uniform[:-1] & ~link
    else:
        end = torch.zeros(height, dtype=torch.bool, device=device)
    start_idx = torch.nonzero(start).squeeze(1).tolist()
    end_idx = torch.nonzero(end).squeeze(1).tolist()
    if bool(uniform[-1].item()):
        end_idx.append(height)
    bands: list[Band] = []
    for y0, y1 in zip(start_idx, end_idx, strict=True):
        if y1 - y0 < min_band:
            continue
        band_med = stats.median[:, y0:y1]  # uint8 [3, n]
        r, g, b = torch.median(band_med, dim=1).values.tolist()
        color = (int(r), int(g), int(b))
        is_gradient = bool((med[:, y1 - 1] - med[:, y0]).abs().max().item() > tol)
        bands.append(Band(y0=y0, y1=y1, color=color, is_gradient=is_gradient))
    bands.sort(key=lambda b: b.y0)
    return bands
