"""patches.npz IO: cleaned crops and text masks per region, written atomically."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_PIXELS_SUFFIX = ".pixels"
_MASK_SUFFIX = ".mask"


def save_patches(path: Path, patches: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Write `id -> (pixels [3,h,w] uint8, mask [h,w] bool)` as `<id>.pixels`/`<id>.mask` (HWC/HW), atomically."""
    arrays: dict[str, np.ndarray] = {}
    for region_id, (pixels, mask) in patches.items():
        arrays[region_id + _PIXELS_SUFFIX] = pixels.cpu().permute(1, 2, 0).numpy()
        arrays[region_id + _MASK_SUFFIX] = mask.cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)  # type: ignore[arg-type]  # plain arrays, never pickled
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_patches(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read patches.npz back: `id -> (pixels uint8 [h,w,3], mask bool [h,w])`; only ids with both arrays."""
    pixels: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for key in data.files:
            region_id, sep, field = key.rpartition(".")
            if not sep:
                continue
            if field == "pixels":
                pixels[region_id] = data[key]
            elif field == "mask":
                masks[region_id] = data[key]
    return {region_id: (pixels[region_id], masks[region_id]) for region_id in pixels if region_id in masks}
