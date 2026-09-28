"""LaMa on a brush selection, right away: the Studio's "lama" clean method (issue #35).

A stroke is rebuilt by the same model and the same window logic as the `inpaint_lama` stage
(inpaint/lama_pipeline.py): the page as it looks now around the stroke — one `lama_window` of context —
goes through `lama_regions` as one region whose mask is the stroke, so a big stroke is tiled and every window
keeps its context, exactly like an automatic clean. The result is an ordinary hand patch (cleanup/store.py),
exported, listed and removed like the others. The model comes from the inpaint group of a VRAM manager, with
exclusive GPU access, only while the stroke is cleaned. Torch is imported only then, so the web app and the
CLI stay importable without it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from omniscan.cleanup.store import add_patch, load_ingest, page_to_strip
from omniscan.core.config import Config, InpaintConfig
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, CleanupPatch, InpaintArtifact, InpaintItem
from omniscan.pipeline.on_demand import group_models

_STROKE = "stroke"  # the one region lama_regions sees


@dataclass(frozen=True, slots=True)
class LamaRebuild:
    """A loaded LaMa model as a `cleanup.store.Rebuild`: crop + mask in, crop with the mask rebuilt out."""

    inpainter: Any  # inpaint.lama.LamaInpainter (or a stand-in with the same `inpaint`)
    cfg: InpaintConfig
    device: Any  # the torch.device the model runs on

    @property
    def context_px(self) -> int:
        """Half a window on every side, so the window around a stroke is all page (no padding)."""
        return self.cfg.lama_window // 2

    def __call__(self, context: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """`context` (uint8 [h, w, 3]) with the pixels under `mask` (bool [h, w]) rebuilt by LaMa."""
        import torch  # deferred: only cleaning needs it

        from omniscan.inpaint.lama_pipeline import lama_regions

        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return context.copy()
        box = BBox(x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()) + 1, y1=int(ys.max()) + 1)
        strip = torch.from_numpy(np.ascontiguousarray(context.transpose(2, 0, 1))).to(self.device)
        stroke = mask[box.y0 : box.y1, box.x0 : box.x1]
        item = InpaintItem(
            region_id=_STROKE, box=box, method="flat", needs_lama=True, mask_px=int(stroke.sum())
        )
        _, patches, _ = lama_regions(
            strip,
            InpaintArtifact(items=[item]),
            {_STROKE: (context[box.y0 : box.y1, box.x0 : box.x1], stroke)},
            self.inpainter,
            self.cfg,
        )
        pixels, _ = patches[_STROKE]
        rebuilt = context.copy()
        rebuilt[box.y0 : box.y1, box.x0 : box.x1] = pixels.permute(1, 2, 0).cpu().numpy()  # once per stroke
        return rebuilt


@contextmanager
def lama_model(cfg: Config) -> Iterator[LamaRebuild]:
    """The LaMa model for one on-demand clean (the inpaint group, pipeline/on_demand.py)."""
    from omniscan.gpu.device import resolve_device
    from omniscan.gpu.groups import INPAINT_GROUP

    with group_models(cfg, INPAINT_GROUP) as models:
        yield LamaRebuild(models["lama"], cfg.inpaint, resolve_device(cfg.gpu.device))


def clean_with_lama(
    paths: ChapterPaths, cfg: Config, *, page: int, box: BBox, mask: np.ndarray
) -> CleanupPatch:
    """Clean a brush stroke painted on page `page` (page pixels) with LaMa and append it to the chapter's
    cleanup (the web API's and the desktop Studio's "lama" method); the model is loaded just for it, after the
    stroke checked out (ValueError / FileNotFoundError as `add_patch` raises them, before any model load)."""
    page_to_strip(load_ingest(paths), page, box, mask)
    with lama_model(cfg) as lama:
        return add_patch(paths, page=page, box=box, mask=mask, method="lama", lama=lama)
