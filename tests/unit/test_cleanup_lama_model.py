"""LaMa on a brush selection through the real window logic (inpaint/lama_pipeline.py) with a stand-in model;
needs torch (CPU is enough), so it runs in CI."""

from __future__ import annotations

import numpy as np
import torch

from omniscan.cleanup.lama_now import LamaRebuild
from omniscan.core.config import InpaintConfig


class Model:
    """Stands in for LamaInpainter: paints the masked pixels of each window 200 and records the windows."""

    def __init__(self) -> None:
        self.windows: list[tuple[tuple[int, ...], int]] = []

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.windows.append((tuple(image.shape), int(mask.sum())))
        return torch.where(mask, torch.full_like(image, 200), image)


def rebuild(context: np.ndarray, mask: np.ndarray, **cfg: int) -> tuple[np.ndarray, Model]:
    model = Model()
    lama = LamaRebuild(model, InpaintConfig(**cfg), torch.device("cpu"))
    return lama(context, mask), model


def test_a_stroke_is_rebuilt_inside_one_window_and_only_there() -> None:
    context = np.full((128, 128, 3), 50, dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:70, 40:80] = True
    out, model = rebuild(context, mask, lama_window=64, lama_dilate_px=0, lama_context_px=8)
    assert model.windows == [((3, 64, 64), 400)]
    assert (out[mask] == 200).all() and (out[~mask] == 50).all()
    assert LamaRebuild(model, InpaintConfig(lama_window=64), torch.device("cpu")).context_px == 32


def test_a_big_stroke_is_tiled_and_a_small_page_padded() -> None:
    context = np.full((40, 300, 3), 50, dtype=np.uint8)  # smaller than the window on one axis: padded
    mask = np.zeros((40, 300), dtype=bool)
    mask[10:30, 10:290] = True  # wider than window - 2 * context: tiled
    out, model = rebuild(context, mask, lama_window=128, lama_dilate_px=0, lama_context_px=16)
    assert len(model.windows) == 4 and all(
        shape == (3, 128, 128) for shape, _ in model.windows
    )  # 0, 64, 128, 184
    assert (out[mask] == 200).all() and (out[~mask] == 50).all()


def test_an_empty_mask_changes_nothing() -> None:
    context = np.full((32, 32, 3), 7, dtype=np.uint8)
    out, model = rebuild(context, np.zeros((32, 32), dtype=bool))
    assert model.windows == [] and (out == 7).all()
