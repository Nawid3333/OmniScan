"""Tests for omniscan.filter.hashing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from omniscan.filter.hashing import dhash, dhash_tensor, hamming, similarity
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


# ---------------------------------------------------------------- dhash_tensor


def tensor_of(pil_image: Image.Image) -> torch.Tensor:
    """The image as a uint8 [3, h, w] tensor (the form the slice stage hands to dhash_tensor)."""
    return torch.from_numpy(np.asarray(pil_image.convert("RGB")).copy()).permute(2, 0, 1).contiguous()


def synthetic_images(seed: int = 7) -> list[Image.Image]:
    """Seeded synthetic pages: noise, gradients, text-like stripes and solid banners.

    The noise pages are upscaled from a low-resolution grid: real pages are smooth, and per-pixel
    noise makes the two resize filters (PIL Lanczos vs antialiased bilinear) diverge beyond the
    6-bit tolerance the two dhash implementations guarantee."""
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    for i in range(18):
        low = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
        size = (360 + 4 * i, 260 + 3 * i)
        smooth = Image.fromarray(low).resize(size, Image.Resampling.BILINEAR)
        images.append(smooth)
    steps = np.repeat(np.arange(0, 256, 36).astype(np.uint8), 50)  # 400 px: 7 brightness plateaus
    for ramp in (steps, steps[::-1].copy()):
        pixels = np.repeat(ramp[None, :], 300, axis=0)
        images.append(Image.fromarray(np.stack([pixels] * 3, axis=-1)))
    stripes = np.zeros((300, 400, 3), dtype=np.uint8)  # text-like horizontal bars
    stripes[40:48, 30:200] = 255
    stripes[120:130, 100:380] = 255
    images.append(Image.fromarray(stripes))
    for color in ((200, 60, 60), (60, 60, 200), (30, 160, 90)):
        images.append(Image.fromarray(np.full((300, 400, 3), color, dtype=np.uint8)))
    return images


def test_dhash_tensor_matches_dhash_on_synthetic_images() -> None:
    """Noise, gradients, text-like stripes and solid banners stay within 6 bits of the PIL dhash."""
    probes = synthetic_images()
    assert len(probes) >= 20
    for image in probes:
        expected = dhash(image)
        actual = dhash_tensor(tensor_of(image))
        assert hamming(expected, actual) <= 6, f"{image.size}: {expected} vs {actual}"


def test_dhash_tensor_golden_gradient() -> None:
    """A pure horizontal ramp hashes to all-zero (rising) / all-ones (falling) bits in both functions."""
    w, h = 400, 300
    rising = np.tile((np.arange(w) * 255 // (w - 1)).astype(np.uint8), (h, 1))
    falling = np.tile(((w - 1 - np.arange(w)) * 255 // (w - 1)).astype(np.uint8), (h, 1))
    for pixels, expected in ((rising, 0), (falling, 2**64 - 1)):
        image = Image.fromarray(np.stack([pixels] * 3, axis=-1))
        assert dhash(image) == expected
        assert dhash_tensor(tensor_of(image)) == expected


def test_dhash_tensor_deterministic_per_tensor() -> None:
    image = torch.randint(0, 256, (3, 200, 400), dtype=torch.uint8)
    assert dhash_tensor(image) == dhash_tensor(image)


def test_dhash_tensor_device_independent() -> None:
    image = torch.randint(0, 256, (3, 200, 400), dtype=torch.uint8)
    assert dhash_tensor(image.to("cpu")) == dhash_tensor(image.to("cpu").clone())


@pytest.mark.gpu
def test_dhash_tensor_cuda_matches_cpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    image = torch.randint(0, 256, (3, 200, 400), dtype=torch.uint8)
    assert dhash_tensor(image) == dhash_tensor(image.to("cuda"))


def test_dhash_tensor_one_row_or_column() -> None:
    assert dhash_tensor(torch.randint(0, 256, (3, 1, 50), dtype=torch.uint8)) >= 0
    assert dhash_tensor(torch.randint(0, 256, (3, 50, 1), dtype=torch.uint8)) >= 0


def test_dhash_tensor_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="3, h, w"):
        dhash_tensor(torch.zeros((200, 400), dtype=torch.uint8))
