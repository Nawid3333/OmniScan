"""Perceptual hashing for the promo filter (dHash over Pillow, no extra dependency)."""

from __future__ import annotations

from PIL import Image


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


def hamming(a: int, b: int) -> int:
    """Popcount of a ^ b (int.bit_count())."""
    return (a ^ b).bit_count()


def similarity(a: int, b: int, bits: int = 64) -> float:
    """1.0 - hamming(a, b) / bits, clamped to [0.0, 1.0]."""
    return max(0.0, 1.0 - hamming(a, b) / bits)
