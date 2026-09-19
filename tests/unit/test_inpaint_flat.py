"""Tests for the text mask and flat fill primitives (card C6a, part 1)."""

from __future__ import annotations

import torch

from omniscan.inpaint.flat import flat_fill, line_mask

DEVICE = torch.device("cpu")


def test_line_mask_grows_boxes() -> None:
    mask = line_mask(20, 30, [(5, 5, 10, 8)], dilate_px=2, device=DEVICE)
    assert mask.shape == (20, 30)
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == 63
    rows = mask.any(dim=1).nonzero().flatten().tolist()
    cols = mask.any(dim=0).nonzero().flatten().tolist()
    assert rows == list(range(3, 10))  # y 3 <= y < 10
    assert cols == list(range(3, 12))  # x 3 <= x < 12


def test_line_mask_clamps_at_edges() -> None:
    mask = line_mask(20, 30, [(0, 0, 5, 5)], dilate_px=2, device=DEVICE)
    assert mask.shape == (20, 30)
    assert int(mask.sum()) == 49
    assert mask[:7, :7].all()
    assert int(mask[7:, :].sum()) == 0
    assert int(mask[:, 7:].sum()) == 0


def test_line_mask_unions_boxes() -> None:
    mask = line_mask(20, 30, [(2, 2, 6, 6), (10, 4, 14, 8)], dilate_px=0, device=DEVICE)
    assert int(mask.sum()) == 32  # two disjoint 4x4 boxes
    assert mask[2:6, 2:6].all()
    assert mask[4:8, 10:14].all()


def test_line_mask_no_boxes_is_empty() -> None:
    mask = line_mask(10, 10, [], dilate_px=3, device=DEVICE)
    assert int(mask.sum()) == 0


def test_line_mask_zero_dilate_equals_plain_boxes() -> None:
    mask = line_mask(10, 10, [(1, 1, 4, 3)], dilate_px=0, device=DEVICE)
    plain = torch.zeros((10, 10), dtype=torch.bool)
    plain[1:3, 1:4] = True
    assert torch.equal(mask, plain)


def test_flat_fill_white_bubble() -> None:
    crop = torch.full((3, 60, 200), 250, dtype=torch.uint8)
    crop[:, 20:40, 80:120] = 0  # the black text rectangle
    mask = torch.zeros((60, 200), dtype=torch.bool)
    mask[18:42, 78:122] = True  # the rectangle plus 2 px on every side
    before = crop.clone()
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=48)
    assert result.ok
    assert result.fill == (250, 250, 250)
    assert torch.equal(result.pixels, torch.full((3, 60, 200), 250, dtype=torch.uint8))
    assert torch.equal(crop, before)  # the input tensor is not modified


def test_flat_fill_dark_bubble() -> None:
    crop = torch.zeros((3, 60, 200), dtype=torch.uint8)
    crop[0], crop[1], crop[2] = 20, 20, 32
    crop[:, 20:40, 80:120] = 250  # white text on the dark interior
    mask = torch.zeros((60, 200), dtype=torch.bool)
    mask[18:42, 78:122] = True
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=48)
    assert result.ok
    assert result.fill == (20, 20, 32)
    expected = torch.zeros((3, 60, 200), dtype=torch.uint8)
    expected[0], expected[1], expected[2] = 20, 20, 32
    assert torch.equal(result.pixels, expected)


def test_flat_fill_rejects_noise_ring() -> None:
    generator = torch.Generator().manual_seed(7)
    crop = torch.randint(0, 256, (3, 60, 200), generator=generator, dtype=torch.uint8)
    mask = torch.zeros((60, 200), dtype=torch.bool)
    mask[20:40, 80:120] = True
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=48)
    assert not result.ok
    assert result.fill is None
    assert torch.equal(result.pixels, crop)


def test_flat_fill_tolerates_five_percent_outline() -> None:
    crop = torch.full((3, 60, 200), 240, dtype=torch.uint8)
    mask = torch.zeros((60, 200), dtype=torch.bool)
    mask[20:40, 80:120] = True  # 800 masked -> ring of 11200
    crop[:, 0:20, 0:28] = 0  # 560 black ring pixels = 5 % (a bubble outline crossing)
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=48)
    assert result.ok
    assert result.fill == (240, 240, 240)


def test_flat_fill_rejects_twenty_percent_outline() -> None:
    crop = torch.full((3, 60, 200), 240, dtype=torch.uint8)
    mask = torch.zeros((60, 200), dtype=torch.bool)
    mask[20:40, 80:120] = True
    crop[:, 0:40, 0:56] = 0  # 2240 black ring pixels = 20 %
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=48)
    assert not result.ok
    assert result.fill is None
    assert torch.equal(result.pixels, crop)


def test_flat_fill_threshold_is_inclusive() -> None:
    # ring of exactly 11 pixels, 9 uniform and 2 off by 8: the 90th percentile is exactly 8.0
    crop = torch.full((3, 4, 4), 240, dtype=torch.uint8)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[0, 0], mask[0, 1], mask[0, 2], mask[1, 0], mask[1, 1] = True, True, True, True, True
    crop[:, 2, 2] = 248
    crop[:, 3, 3] = 248
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=1)
    assert result.ok
    assert result.fill == (240, 240, 240)


def test_flat_fill_rejects_tiny_ring() -> None:
    crop = torch.full((3, 10, 10), 200, dtype=torch.uint8)
    mask = torch.zeros((10, 10), dtype=torch.bool)
    mask[:6, :] = True  # ring of 40 < min_ring_px
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=48)
    assert not result.ok
    assert result.fill is None
    assert torch.equal(result.pixels, crop)


def test_flat_fill_empty_mask() -> None:
    crop = torch.full((3, 10, 10), 200, dtype=torch.uint8)
    mask = torch.zeros((10, 10), dtype=torch.bool)
    result = flat_fill(crop, mask, flat_tol=8.0, min_ring_px=48)
    assert not result.ok
    assert result.fill is None
    assert torch.equal(result.pixels, crop)
