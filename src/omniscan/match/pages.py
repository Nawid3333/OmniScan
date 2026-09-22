"""Page fingerprints for chapter matching: dHash of the page files (CPU/PIL, no torch).

Reuses `omniscan.filter.hashing.dhash` / `similarity` — the same promo-page hashing, read as a
language-independent "is this the same underlying art" signal. `np.bitwise_count` vectorises the
hamming distances of every page pair of every chapter pair in a handful of numpy ops."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from omniscan.core.paths import list_images
from omniscan.filter.hashing import dhash

BITS = 64  # dhash hash_size 8 -> 64-bit fingerprints
BLOCK_ELEMENTS = 16_000_000  # XOR intermediates are cut so one block stays <=128 MB


def chapter_hashes(chapter_dir: Path) -> list[int]:
    """dhash of every page image of a chapter folder, in reading order (natural sort)."""
    hashes = []
    for path in list_images(chapter_dir):
        with Image.open(path) as img:
            hashes.append(dhash(img))
    return hashes


def similarity_matrix(hashes_a: Sequence[int], hashes_b: Sequence[int]) -> np.ndarray:
    """[len_a, len_b] dhash similarity (0..1) of every page pair of two chapters."""
    a = np.asarray(hashes_a, dtype=np.uint64).reshape(-1, 1)
    b = np.asarray(hashes_b, dtype=np.uint64).reshape(1, -1)
    return 1.0 - np.bitwise_count(a ^ b) / BITS


def similarity_blocks(
    hashes_a: Sequence[Sequence[int]], hashes_b: Sequence[Sequence[int]]
) -> Iterator[tuple[range, np.ndarray]]:
    """Yield (a_indices, [len_block, len_b, p_max, q_max] similarity tensor) blocks of the full matrix.

    One block of A chapters against all of B at a time keeps the uint64 XOR intermediate bounded no
    matter how many chapters or pages a series has."""
    len_a, len_b = len(hashes_a), len(hashes_b)
    if len_a == 0 or len_b == 0:
        return
    p_max = max(len(chapter) for chapter in hashes_a)
    q_max = max(len(chapter) for chapter in hashes_b)
    packed_b = _pack(hashes_b, q_max)  # [len_b, q_max]
    block = max(1, BLOCK_ELEMENTS // max(1, len_b * max(1, p_max) * max(1, q_max) * 8))
    for start in range(0, len_a, block):
        stop = min(start + block, len_a)
        packed = _pack(hashes_a[start:stop], p_max)  # [len_block, p_max]
        xor = packed[:, None, :, None] ^ packed_b[None, :, None, :]
        yield range(start, stop), 1.0 - np.bitwise_count(xor) / BITS


def _pack(hashes: Sequence[Sequence[int]], width: int) -> np.ndarray:
    """Chapters' page hashes as a zero-padded [n, width] uint64 array."""
    packed = np.zeros((len(hashes), width), dtype=np.uint64)
    for index, chapter in enumerate(hashes):
        packed[index, : len(chapter)] = chapter
    return packed
