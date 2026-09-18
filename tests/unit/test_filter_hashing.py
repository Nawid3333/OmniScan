"""Tests for omniscan.filter.hashing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from omniscan.filter.hashing import dhash, hamming, similarity
from tests.fixtures import images


def test_dhash_deterministic(tmp_path: Path) -> None:
    images.gradient_jpeg(tmp_path / "a.jpg")
    with Image.open(tmp_path / "a.jpg") as img:
        assert dhash(img) == dhash(img)


def test_very_different_images_hash_far_apart(tmp_path: Path) -> None:
    # dHash compares adjacent pixels, so two uniform solid-color images (e.g. solid red vs solid blue)
    # both hash to all-zero bits regardless of color — structured content is needed to separate them.
    images.gradient_jpeg(tmp_path / "a.jpg")
    images.gradient_jpeg(tmp_path / "b.jpg", invert=True)
    with Image.open(tmp_path / "a.jpg") as a, Image.open(tmp_path / "b.jpg") as b:
        assert similarity(dhash(a), dhash(b)) < 0.5


def test_solid_color_images_hash_to_zero(tmp_path: Path) -> None:
    # Pins the uniform-field invariance: a dHash of any solid image is all-zero bits.
    images.plain_jpeg(tmp_path / "red.jpg", color=(200, 60, 60))
    images.plain_jpeg(tmp_path / "blue.jpg", color=(60, 60, 200))
    with Image.open(tmp_path / "red.jpg") as red, Image.open(tmp_path / "blue.jpg") as blue:
        assert dhash(red) == 0
        assert dhash(blue) == 0


def test_similarity_self_is_one() -> None:
    for value in (0, 1, 0xFFFF, 2**64 - 1):
        assert similarity(value, value) == 1.0


def test_hamming_and_similarity_bits() -> None:
    assert hamming(0, 0) == 0
    assert hamming(0, 2**64 - 1) == 64
    assert similarity(0, 2**64 - 1) == 0.0  # clamped at the low end
    assert similarity(0b1, 0) == 1.0 - 1 / 64


def test_jpeg_reencode_stays_similar(tmp_path: Path) -> None:
    images.plain_jpeg(tmp_path / "original.jpg", size=(400, 300))
    with Image.open(tmp_path / "original.jpg") as original:
        pixels = original.convert("RGB")
        original_hash = dhash(pixels)
        for quality in (95, 60):
            dest = tmp_path / f"q{quality}.jpg"
            pixels.save(dest, format="JPEG", quality=quality)
            with Image.open(dest) as reencoded:
                assert similarity(original_hash, dhash(reencoded)) >= 0.95
