"""Tests for patches.npz writing and reading (card C6a, part 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from omniscan.inpaint.patches import load_patches, save_patches


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "patches.npz"
    p1 = torch.randint(0, 256, (3, 8, 10), dtype=torch.uint8)
    m1 = torch.zeros((8, 10), dtype=torch.bool)
    m1[2, 3] = True
    p2 = torch.randint(0, 256, (3, 5, 7), dtype=torch.uint8)
    m2 = torch.ones((5, 7), dtype=torch.bool)
    save_patches(path, {"r0001": (p1, m1), "r0002": (p2, m2)})

    loaded = load_patches(path)
    assert set(loaded) == {"r0001", "r0002"}
    pixels, mask = loaded["r0001"]
    assert pixels.dtype == np.uint8
    assert pixels.shape == (8, 10, 3)  # HWC
    assert np.array_equal(pixels, p1.permute(1, 2, 0).numpy())
    assert mask.dtype == np.bool_
    assert mask.shape == (8, 10)  # HW
    assert np.array_equal(mask, m1.numpy())
    pixels2, mask2 = loaded["r0002"]
    assert pixels2.shape == (5, 7, 3)
    assert np.array_equal(pixels2, p2.permute(1, 2, 0).numpy())
    assert mask2.all()


def test_empty_mapping_writes_valid_empty_npz(tmp_path: Path) -> None:
    path = tmp_path / "patches.npz"
    save_patches(path, {})
    assert path.is_file()
    assert load_patches(path) == {}
    with np.load(path) as data:
        assert data.files == []


def test_atomic_replace_and_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "patches.npz"
    first = torch.full((3, 4, 4), 1, dtype=torch.uint8)
    save_patches(path, {"r0001": (first, torch.ones((4, 4), dtype=torch.bool))})
    assert not list(tmp_path.glob("*.tmp"))
    second = torch.full((3, 4, 4), 9, dtype=torch.uint8)
    save_patches(path, {"r0002": (second, torch.zeros((4, 4), dtype=torch.bool))})
    assert not list(tmp_path.glob("*.tmp"))
    loaded = load_patches(path)
    assert set(loaded) == {"r0002"}
    assert np.array_equal(loaded["r0002"][0], second.permute(1, 2, 0).numpy())


def test_orphan_key_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "patches.npz"
    np.savez_compressed(path, **{"r0001.pixels": np.zeros((2, 2, 3), dtype=np.uint8)})  # type: ignore[arg-type]
    assert load_patches(path) == {}
