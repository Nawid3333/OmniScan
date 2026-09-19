"""Tests for the scanner-style page degradation (card X2)."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageFilter

from tests.fixtures.korean_pages import make_korean_page
from tests.fixtures.scan_artifacts import degrade

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_korean_chapter.py"


def scan(image: Image.Image, seed: int = 0, **options: Any) -> Image.Image:
    """`degrade` with every artefact off unless the test names it."""
    return degrade(image, seed, **{"blur_px": 0, "noise_sigma": 0, "jpeg_quality": 100, **options})


def mean_abs_difference(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute per-channel pixel difference between two same-size RGB images."""
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).mean())


def dark_centroid(image: Image.Image) -> tuple[float, float]:
    """Centroid (x, y) of the dark (< 128) pixels of `image`."""
    ys, xs = np.nonzero(np.asarray(image.convert("L"), dtype=np.float64) < 128.0)
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------- 1 size, determinism, jpeg-only


@pytest.mark.parametrize("size", [(200, 120), (100, 100), (37, 53)])
def test_same_size_and_rgb(size: tuple[int, int]) -> None:
    source = Image.new("RGB", size, (128, 128, 128))
    out = degrade(source, 3)
    assert out.size == source.size
    assert out.mode == "RGB"


def test_determinism() -> None:
    source = Image.new("RGB", (80, 60), (120, 130, 140))
    assert degrade(source, 5).tobytes() == degrade(source, 5).tobytes()


def test_seed_changes_noise() -> None:
    source = Image.new("RGB", (80, 60), (128, 128, 128))
    assert degrade(source, 1).tobytes() != degrade(source, 2).tobytes()


def test_no_artefacts_is_jpeg_only() -> None:
    rows = 255.0 * np.arange(100, dtype=np.float64) / 99.0
    arr = np.repeat(rows[:, None, None], 100, axis=1)
    source = Image.fromarray(np.repeat(arr, 3, axis=2).astype(np.uint8), "RGB")
    assert mean_abs_difference(scan(source), source) <= 2.0


# ---------------------------------------------------------------- 2 noise


def test_noise_level_matches_sigma() -> None:
    base = Image.new("RGB", (200, 200), (128, 128, 128))
    values = np.asarray(degrade(base, 5, blur_px=0, noise_sigma=8.0, jpeg_quality=100), dtype=np.float64)
    flat = values.reshape(-1)
    assert 5.5 <= flat.std() <= 9.5
    assert 126.0 <= flat.mean() <= 130.0
    quiet = np.asarray(degrade(base, 5, blur_px=0, noise_sigma=0, jpeg_quality=100), dtype=np.float64)
    assert quiet.reshape(-1).std() <= 1.0


# ---------------------------------------------------------------- 3 jpeg quality


def _checkerboard_gradient() -> Image.Image:
    """Grey 4-px checkerboard over a horizontal gradient: high-frequency JPEG bait."""
    grad = 100.0 + 55.0 * np.arange(200, dtype=np.float64) / 199.0
    checker = ((np.arange(200)[:, None] // 4 + np.arange(200)[None, :] // 4) % 2) * 2.0 - 1.0
    arr = np.clip(grad[None, :, None] + 60.0 * checker[:, :, None], 0.0, 255.0)
    return Image.fromarray(np.repeat(arr, 3, axis=2).astype(np.uint8), "RGB")


def test_jpeg_quality_orders_damage() -> None:
    source = _checkerboard_gradient()
    coarse = scan(source, jpeg_quality=20)
    fine = scan(source, jpeg_quality=95)
    assert mean_abs_difference(coarse, source) > mean_abs_difference(fine, source)


# ---------------------------------------------------------------- 4 blur


def test_blur_softens_a_hard_line() -> None:
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[:, 50] = 0
    source = Image.fromarray(arr, "RGB")
    blurred = np.asarray(scan(source, blur_px=2.0), dtype=np.float64)[:, :, 0]
    assert blurred[:, 50].min() > 60.0  # the line got lighter
    assert blurred[:, 49].max() < 250.0  # its neighbours got darker
    assert blurred[:, 51].max() < 250.0
    sharp = np.asarray(scan(source, blur_px=0), dtype=np.float64)[:, :, 0]
    assert sharp[:, 50].min() < 20.0


# ---------------------------------------------------------------- 5 halftone


def test_halftone_dot_screen() -> None:
    base = Image.new("RGB", (200, 200), (200, 200, 200))
    on = np.asarray(scan(base, halftone=True), dtype=np.float64).reshape(-1)
    assert on.std() >= 3.0
    assert 184.0 <= on.mean() <= 196.0  # the dot screen darkens the page on average
    off = np.asarray(scan(base, halftone=False), dtype=np.float64).reshape(-1)
    assert off.std() <= 1.0


# ---------------------------------------------------------------- 6 rotation


def _black_square_page() -> Image.Image:
    arr = np.full((200, 200, 3), 255, dtype=np.uint8)
    arr[70:130, 70:130] = 0
    return Image.fromarray(arr, "RGB")


def test_rotation_zero_keeps_the_square() -> None:
    source = _black_square_page()
    (x_in, y_in), (x_out, y_out) = dark_centroid(source), dark_centroid(scan(source, rotate_deg=0.0))
    assert abs(x_out - x_in) <= 0.5
    assert abs(y_out - y_in) <= 0.5


def test_small_rotation_moves_pixels_not_centroid() -> None:
    source = _black_square_page()
    out = scan(source, rotate_deg=5.0)
    (x_in, y_in), (x_out, y_out) = dark_centroid(source), dark_centroid(out)
    assert abs(x_out - x_in) <= 3.0  # rotation is about the centre, so the square stays put
    assert abs(y_out - y_in) <= 3.0
    assert mean_abs_difference(out, scan(source, rotate_deg=0.0)) > 1.0


def test_rotation_fills_corners_white() -> None:
    arr = np.full((200, 200, 3), 255, dtype=np.uint8)
    arr[:12, :] = 0
    arr[-12:, :] = 0
    arr[:, :12] = 0
    arr[:, -12:] = 0
    source = Image.fromarray(arr, "RGB")
    assert np.asarray(source, dtype=np.float64)[0, 0].min() < 20.0  # the original corner was black
    out = np.asarray(scan(source, rotate_deg=10.0), dtype=np.float64)
    for x, y in [(0, 0), (199, 0), (0, 199), (199, 199)]:
        assert out[y, x].min() >= 250.0


# ---------------------------------------------------------------- 7 validation


@pytest.mark.parametrize("quality", [0, 101])
def test_jpeg_quality_out_of_bounds(quality: int) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        degrade(Image.new("RGB", (10, 10), (128, 128, 128)), 0, jpeg_quality=quality)


def test_negative_blur_and_noise() -> None:
    source = Image.new("RGB", (10, 10), (128, 128, 128))
    with pytest.raises(ValueError, match="blur_px"):
        degrade(source, 0, blur_px=-1.0)
    with pytest.raises(ValueError, match="noise_sigma"):
        degrade(source, 0, noise_sigma=-1.0)


# ---------------------------------------------------------------- 8 generator


def run_generator(library: Path, *options: str) -> bytes:
    """Run make_korean_chapter.py for one seeded page and return the page file's bytes."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--library-root",
            str(library),
            "--pages",
            "1",
            "--seed",
            "41",
            *options,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return (library / "KoreanDemo" / "Chapter 1" / "001.jpg").read_bytes()


def test_generator_scan_artifacts(tmp_path: Path) -> None:
    plain = run_generator(tmp_path / "plain")
    expected = io.BytesIO()
    make_korean_page(42, n_bubbles=4, n_free=1, n_sfx=1).image.save(expected, format="JPEG", quality=92)
    assert plain == expected.getvalue()  # without the flag: the documented recipe, byte for byte

    degraded = run_generator(tmp_path / "scan", "--scan-artifacts")
    assert degraded != plain
    decoded = Image.open(io.BytesIO(degraded))
    assert decoded.format == "JPEG"
    assert decoded.size == (800, 1400)

    again = run_generator(tmp_path / "again", "--scan-artifacts")
    assert again == degraded  # deterministic across runs

    harder = run_generator(
        tmp_path / "harder", "--scan-artifacts", "--halftone", "--rotate", "1.0", "--jpeg-quality", "30"
    )
    assert harder != degraded  # the extra degradation options reach the pages


# ---------------------------------------------------------------- gaps found by the director's mutation run


def test_halftone_screen_is_periodic_shallow_and_darkest_at_the_dots() -> None:
    out = np.asarray(scan(Image.new("RGB", (60, 60), (200, 200, 200)), halftone=True), dtype=np.float64)
    assert out.min() >= 182 and out.max() <= 202  # multiplier 1 - 0.08 * d with d in [0, 1]
    assert np.abs(out[:, 6:54] - out[:, 12:60]).mean() < 1.5  # a 6 px dot screen repeats every 6 columns
    assert np.abs(out[6:54, :] - out[12:60, :]).mean() < 1.5
    assert np.abs(out[:, :50] - out[:, 5:55]).mean() > 2.5  # ... and not every 5
    assert out[1, 1, 0] < out[4, 1, 0] - 5  # d(1, 1) = 0.875 is a dark dot, d(1, 4) = 0.125 a light gap


def test_default_blur_radius_is_the_gaussian_radius() -> None:
    edge = Image.new("RGB", (80, 80), (255, 255, 255))
    edge.paste((0, 0, 0), (0, 0, 40, 80))
    reference = scan(edge.filter(ImageFilter.GaussianBlur(radius=0.6)))
    assert mean_abs_difference(scan(edge, blur_px=0.6), reference) < 0.5
    assert mean_abs_difference(scan(edge, blur_px=0.6), scan(edge)) > 0.5  # sub-pixel blur still blurs


def test_noise_is_rounded_not_truncated_and_never_wraps() -> None:
    flat = np.asarray(scan(Image.new("RGB", (200, 200), (128, 128, 128)), noise_sigma=6.0), dtype=np.float64)
    assert abs(flat.mean() - 128.0) < 0.25  # truncation would shift the mean by half a level
    dark = np.asarray(scan(Image.new("RGB", (100, 100), (0, 0, 0)), noise_sigma=8.0))
    assert dark.max() < 60  # negative noise is clipped at 0, it must not wrap around to 255
