"""Synthetic chapter-set fixtures for the matcher (small images generated in code, no real manga).

A page's art is a seeded function of one integer: six random sinusoid waves + seeded panel blocks +
a focal disk + optional 'lettering' boxes. The same seed rendered at a different size, JPEG quality
and lettering placement stands in for "the same page, raw scan vs official translation" (dhash-close
— measured: hamming <= 14 of 64); different seeds stand in for different pages (dhash-far — median
hamming 32, 1st percentile 16)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PAGE_SIZE = (96, 144)  # the raw side's page size; the translated side renders the same art smaller
TRANSLATED_SIZE = (72, 108)
TEXT_SEED_BASE = 10_000  # lettering seeds live in a different range than art seeds


def page_image(
    seed: int,
    size: tuple[int, int] = PAGE_SIZE,
    *,
    text_seed: int | None = None,
) -> Image.Image:
    """One synthetic comic page; `text_seed` places small dark 'lettering' boxes (None draws none).

    A raw page and its translation carry their lettering at different spots, so the two renderings of
    one page use different text seeds while the art seed — and therefore the art — stays the same."""
    rng = np.random.Generator(np.random.PCG64(seed))
    w, h = size
    xs = np.arange(w)
    ys = np.arange(h)
    field = np.full((h, w), 128.0)
    for _ in range(3):  # horizontal waves
        freq = float(rng.integers(1, 5))
        field += rng.uniform(20, 50) * np.sin(2 * np.pi * freq * xs / w + rng.uniform(0, 2 * np.pi))[None, :]
    for _ in range(3):  # vertical waves
        freq = float(rng.integers(1, 5))
        field += rng.uniform(20, 50) * np.sin(2 * np.pi * freq * ys / h + rng.uniform(0, 2 * np.pi))[:, None]
    img = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(4, 8))):  # panel blocks: large, so they move many dhash cells
        x0, y0 = int(rng.integers(0, w - 8)), int(rng.integers(0, h - 8))
        x1, y1 = x0 + int(rng.integers(6, w // 2)), y0 + int(rng.integers(6, h // 2))
        draw.rectangle((x0, y0, min(x1, w - 1), min(y1, h - 1)), fill=(int(rng.integers(20, 230)),) * 3)
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h  # one focal disk per page
    radius = rng.uniform(0.08, 0.2) * min(w, h)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(int(rng.integers(20, 230)),) * 3)
    if text_seed is not None:
        text_rng = np.random.Generator(np.random.PCG64(text_seed))
        for _ in range(int(text_rng.integers(1, 4))):  # lettering: small, moves only a few dhash bits
            x0 = int(text_rng.uniform(0.05, 0.75) * w)
            y0 = int(text_rng.uniform(0.05, 0.85) * h)
            box_w = max(4, int(text_rng.uniform(0.05, 0.12) * w))
            box_h = max(3, int(text_rng.uniform(0.03, 0.07) * h))
            draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(0, 0, 0))
    return img


def page_seeds(first: int, count: int) -> list[int]:
    """`count` distinct page seeds starting at `first` (a chapter's worth of page art)."""
    return list(range(first, first + count))


def write_page(
    path: Path,
    seed: int,
    *,
    size: tuple[int, int] = PAGE_SIZE,
    text_seed: int | None = None,
    quality: int = 92,
) -> Path:
    """Write one synthetic page as a JPEG; returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    page_image(seed, size, text_seed=text_seed).save(path, format="JPEG", quality=quality)
    return path


def write_chapter(
    chapter_dir: Path,
    seeds: list[int],
    *,
    size: tuple[int, int] = PAGE_SIZE,
    text_seed_offset: int = 0,
    quality: int = 92,
) -> Path:
    """Write one synthetic chapter folder (001.jpg, 002.jpg, ...); returns the chapter dir.

    `text_seed_offset` shifts every page's lettering seed: raw and translated sides use different
    offsets, so their lettering sits at different spots while the art stays the same."""
    chapter_dir.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(seeds, start=1):
        write_page(
            chapter_dir / f"{index:03d}.jpg",
            seed,
            size=size,
            text_seed=TEXT_SEED_BASE + seed + text_seed_offset,
            quality=quality,
        )
    return chapter_dir


def write_series(
    root: Path,
    chapters: dict[str, list[int]],
    *,
    size: tuple[int, int] = PAGE_SIZE,
    text_seed_offset: int = 0,
    quality: int = 92,
) -> Path:
    """Write a chapter-set directory: one subfolder per chapter (folder names = dict keys)."""
    for name, seeds in chapters.items():
        write_chapter(root / name, seeds, size=size, text_seed_offset=text_seed_offset, quality=quality)
    return root


RAW_TEXT_OFFSET = 0  # the raw side's lettering seed offset
TRANSLATED_TEXT_OFFSET = 5_000  # the translated side's: same art, lettering at different spots
TRANSLATED_QUALITY = 85


def write_raw_series(root: Path, chapters: dict[str, list[int]]) -> Path:
    """A raw-language chapter set: full-size pages, quality 92."""
    return write_series(root, chapters, text_seed_offset=RAW_TEXT_OFFSET, quality=92)


def write_translated_series(root: Path, chapters: dict[str, list[int]]) -> Path:
    """An independently-sourced translated chapter set: smaller pages, lower quality, shifted lettering."""
    return write_series(
        root,
        chapters,
        size=TRANSLATED_SIZE,
        text_seed_offset=TRANSLATED_TEXT_OFFSET,
        quality=TRANSLATED_QUALITY,
    )
