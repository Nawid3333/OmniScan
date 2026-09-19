"""Unit tests for omniscan.export.composite (CPU tensors; GPU parity in the gpu-marked test)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omniscan.core.schemas import BBox, InpaintItem
from omniscan.export.composite import apply_patch, apply_patches, blend_rgba
from omniscan.typeset.render import GlyphPatch

H, W = 10, 10


def solid(value: int) -> torch.Tensor:
    """A uint8 [3, H, W] strip filled with one value on every channel."""
    return torch.full((3, H, W), value, dtype=torch.uint8)


def flat_patch(pixel: tuple[int, int, int, int], height: int, width: int, x: int, y: int) -> GlyphPatch:
    """A GlyphPatch of one constant RGBA pixel."""
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:] = pixel
    return GlyphPatch(x=x, y=y, rgba=rgba)


def test_blend_rgba_alpha_levels() -> None:
    for alpha, expected in ((128, 150), (255, 200), (0, 100)):
        strip = solid(100)
        blend_rgba(strip, flat_patch((200, 200, 200, alpha), H, W, 0, 0))
        assert torch.all(strip == expected)
        assert strip.dtype == torch.uint8


def test_blend_rgba_mixed_channels() -> None:
    strip = solid(100)
    strip[0], strip[1], strip[2] = 250, 100, 10
    blend_rgba(strip, flat_patch((10, 100, 250, 51), H, W, 0, 0))  # alpha 51/255 = 0.2
    assert strip[0, 0, 0].item() == 202 and strip[1, 0, 0].item() == 100 and strip[2, 0, 0].item() == 58


def test_blend_rgba_clips_overhanging_patch() -> None:
    for x, y, region in ((-2, -2, (slice(0, 2), slice(0, 2))), (8, 8, (slice(8, 10), slice(8, 10)))):
        strip = solid(100)
        blend_rgba(strip, flat_patch((200, 200, 200, 255), 4, 4, x, y))
        expected = solid(100)
        expected[:, region[0], region[1]] = 200
        assert torch.equal(strip, expected), (x, y)


def test_blend_rgba_patch_outside_the_strip_changes_nothing() -> None:
    strip = solid(100)
    for x, y in ((-20, 2), (2, -20), (20, 2), (2, 20)):
        blend_rgba(strip, flat_patch((200, 200, 200, 255), 4, 4, x, y))
    assert torch.all(strip == 100)


def test_apply_patch_replaces_masked_pixels_only() -> None:
    strip = solid(100)
    pixels = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    apply_patch(strip, BBox(x0=3, y0=3, x1=7, y1=7), pixels, mask)
    expected = solid(100)
    expected[:, 4:6, 4:6] = 200
    assert torch.equal(strip, expected)


def test_apply_patch_clips_to_the_strip() -> None:
    strip = solid(100)
    pixels = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    apply_patch(strip, BBox(x0=-2, y0=-2, x1=2, y1=2), pixels, mask)
    expected = solid(100)
    expected[:, :2, :2] = 200
    assert torch.equal(strip, expected)
    apply_patch(strip, BBox(x0=8, y0=8, x1=12, y1=12), pixels, mask)
    expected[:, 8:10, 8:10] = 200
    assert torch.equal(strip, expected)
    before = strip.clone()
    apply_patch(strip, BBox(x0=20, y0=20, x1=24, y1=24), pixels, mask)
    assert torch.equal(strip, before)  # fully outside: ignored


def test_apply_patch_handles_mismatched_crops() -> None:
    strip = solid(100)
    apply_patch(
        strip,
        BBox(x0=2, y0=2, x1=8, y1=8),  # 6x6 box, but the arrays are 3x3
        np.full((3, 3, 3), 200, dtype=np.uint8),
        np.ones((3, 3), dtype=bool),
    )
    expected = solid(100)
    expected[:, 2:5, 2:5] = 200
    assert torch.equal(strip, expected)


def test_apply_patches_counts_skips_and_applies_in_item_order() -> None:
    strip = solid(100)
    items = [
        InpaintItem(region_id="r1", box=BBox(x0=0, y0=0, x1=2, y1=2), method="flat"),
        InpaintItem(region_id="rx", box=BBox(x0=4, y0=4, x1=6, y1=6), method="flat"),
        InpaintItem(region_id="r2", box=BBox(x0=0, y0=0, x1=2, y1=2), method="flat"),
    ]
    patches = {
        "r1": (np.full((2, 2, 3), 200, dtype=np.uint8), np.ones((2, 2), dtype=bool)),
        "r2": (np.full((2, 2, 3), 50, dtype=np.uint8), np.ones((2, 2), dtype=bool)),
    }
    assert apply_patches(strip, items, patches) == 2
    assert torch.all(strip[:, :2, :2] == 50)  # the later item for the same box overwrote the first
    assert torch.all(strip[:, 4:6, 4:6] == 100)  # the item without a patch entry was skipped


@pytest.mark.gpu
def test_blend_and_patch_are_device_independent() -> None:
    from omniscan.gpu.device import resolve_device

    rng = np.random.default_rng(0)
    cpu = torch.from_numpy(rng.integers(0, 256, (3, H, W), dtype=np.uint8))
    gpu = cpu.to(resolve_device())
    rgba = np.zeros((4, 6, 4), dtype=np.uint8)
    rgba[..., :3] = (10, 100, 250)
    rgba[..., 3] = (np.arange(24).reshape(4, 6) * 11).astype(np.uint8)
    pixels = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    mask = rng.random((4, 4)) < 0.5
    blend_rgba(cpu, GlyphPatch(x=-1, y=2, rgba=rgba))
    apply_patch(cpu, BBox(x0=4, y0=4, x1=8, y1=8), pixels, mask)
    blend_rgba(gpu, GlyphPatch(x=-1, y=2, rgba=rgba))
    apply_patch(gpu, BBox(x0=4, y0=4, x1=8, y1=8), pixels, mask)
    assert gpu.is_cuda  # the strip stays on the GPU device until the codec's encode
    assert torch.equal(cpu, gpu.cpu())  # bit-identical to the same operations on a CPU copy
