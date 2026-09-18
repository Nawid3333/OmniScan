"""Synthetic image fixtures for ingest tests (small, generated in code — no real manga files)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

_ORIENT = 0x0112  # EXIF Orientation tag


def plain_jpeg(
    path: Path, size: tuple[int, int] = (400, 300), color: tuple[int, int, int] = (200, 60, 60)
) -> Path:
    """Write a baseline RGB JPEG with no EXIF."""
    Image.new("RGB", size, color).save(path, format="JPEG", quality=95)
    return path


def rotated_jpeg(
    path: Path,
    size: tuple[int, int] = (300, 400),
    color: tuple[int, int, int] = (60, 200, 60),
    orientation: int = 6,
) -> Path:
    """Write an RGB JPEG with raw pixels stored at swapped dimensions plus an EXIF Orientation tag.

    After `exif_transpose` the logical size becomes `size` (orientation 6 = rotate 90° CW on display).
    """
    stored = Image.new("RGB", (size[1], size[0]), color)
    exif = Image.Exif()
    exif[_ORIENT] = orientation
    stored.save(path, format="JPEG", quality=95, exif=exif)
    return path


def png_with_alpha(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    """Write an RGBA PNG with a semi-transparent gradient alpha (left edge fully transparent)."""
    w, h = size
    img = Image.new("RGBA", size, (40, 80, 160, 255))
    # alpha ramps up left→right; the first 8 columns are fully transparent so a whole JPEG
    # block near the corner stays exactly white after compositing
    alpha = Image.new("L", size, 255)
    alpha.putdata(
        [0 if x < 8 else min(255, (x - 8) * 255 // max(1, w - 8)) for y in range(h) for x in range(w)]
    )
    img.putalpha(alpha)
    img.save(path, format="PNG")
    return path


def cmyk_jpeg(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    """Write a CMYK-mode JPEG."""
    Image.new("RGB", size, (90, 90, 200)).convert("CMYK").save(path, format="JPEG", quality=95)
    return path


def grayscale_jpeg(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    """Write a mode-'L' JPEG."""
    Image.new("L", size, 128).save(path, format="JPEG", quality=95)
    return path


def webp_image(path: Path, size: tuple[int, int] = (250, 180)) -> Path:
    """Write a plain RGB WebP."""
    Image.new("RGB", size, (120, 40, 200)).save(path, format="WEBP")
    return path


def gradient_jpeg(path: Path, size: tuple[int, int] = (400, 300), invert: bool = False) -> Path:
    """Write a JPEG with a left-to-right luminance ramp (falling when inverted); non-uniform content."""
    w, h = size
    img = Image.new("L", size)
    img.putdata(
        [(w - 1 - x) * 255 // (w - 1) if invert else x * 255 // (w - 1) for y in range(h) for x in range(w)]
    )
    img.convert("RGB").save(path, format="JPEG", quality=95)
    return path
