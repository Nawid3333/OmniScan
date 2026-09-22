"""Perceptual hashing for the promo filter (dHash over Pillow / torch, no extra dependency)."""

from __future__ import annotations

import torch
from PIL import Image

_LUMA = (0.299, 0.587, 0.114)  # ITU-R 601 luma weights (PIL's 'L' conversion)
_MAX_ANTIALIAS_INPUT = 512  # larger inputs are coarse-averaged first: ROCm's antialiased
# interpolate cannot do one big downscale factor ("Too much shared memory required: 90268 vs
# 65536"); at 512 the final resize needs at most 57x64 box taps, far below the buffer limit.


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash. Convert to grayscale ('L'), resize to (hash_size+1, hash_size) with
    Image.Resampling.LANCZOS, then for each of the hash_size rows compare each of hash_size adjacent
    column pairs (col[x] > col[x+1]) left to right, top to bottom, packing bits MSB-first into an
    int of hash_size*hash_size bits. Deterministic for the same input image."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = gray.tobytes()
    value = 0
    for y in range(hash_size):
        row = y * (hash_size + 1)
        for x in range(hash_size):
            value = (value << 1) | (1 if pixels[row + x] > pixels[row + x + 1] else 0)
    return value


def dhash_tensor(image: torch.Tensor, hash_size: int = 8) -> int:
    """dhash of a uint8 [3, h, w] tensor on any device: grayscale (ITU-R 601 luma 0.299/0.587/0.114),
    an antialiased resize to (hash_size, hash_size + 1) rows x cols (full-page crops are coarse-
    averaged first, see `_antialiased_resize`), then the same MSB-first left>right comparison as
    `dhash`. One host transfer of hash_size*(hash_size+1) values; everything else stays on the device."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a uint8 [3, h, w] image tensor, got shape {tuple(image.shape)}")
    weights = torch.tensor(_LUMA, device=image.device).view(3, 1, 1)
    gray = image.to(torch.float32).mul(weights).sum(0)  # [h, w]
    resized = _antialiased_resize(gray[None, None], hash_size)
    pixels = resized.round().clamp(0, 255).to(torch.uint8).flatten().cpu().tolist()
    value = 0
    for y in range(hash_size):
        row = y * (hash_size + 1)
        for x in range(hash_size):
            value = (value << 1) | (1 if pixels[row + x] > pixels[row + x + 1] else 0)
    return value


def _antialiased_resize(gray: torch.Tensor, hash_size: int) -> torch.Tensor:
    """[1, 1, h, w] float -> [1, 1, hash_size, hash_size + 1], antialiased.

    One `interpolate` call for moderate sizes; a coarse average-pool stage first for larger ones
    (ROCm's antialiased interpolate overflows its box-filter buffer on big downscale factors)."""
    height, width = int(gray.shape[-2]), int(gray.shape[-1])
    if max(height, width) <= _MAX_ANTIALIAS_INPUT:
        return torch.nn.functional.interpolate(
            gray,
            size=(hash_size, hash_size + 1),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
    # integer factors keep the intermediate within 512 px per axis, so the final resize is small
    coarse = torch.nn.functional.avg_pool2d(
        gray,
        kernel_size=(-(-height // _MAX_ANTIALIAS_INPUT), -(-width // _MAX_ANTIALIAS_INPUT)),
        ceil_mode=True,
    )
    return torch.nn.functional.interpolate(
        coarse,
        size=(hash_size, hash_size + 1),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    )


def hamming(a: int, b: int) -> int:
    """Popcount of a ^ b (int.bit_count())."""
    return (a ^ b).bit_count()


def similarity(a: int, b: int, bits: int = 64) -> float:
    """1.0 - hamming(a, b) / bits, clamped to [0.0, 1.0]."""
    return max(0.0, 1.0 - hamming(a, b) / bits)
