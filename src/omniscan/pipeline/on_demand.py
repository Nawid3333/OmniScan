"""A model group loaded for one on-demand Studio action, the way a stage gets it.

Reading a region again (ocr/on_demand.py), finding missed text on a page (detect/on_demand.py) and LaMa on a
brush selection (cleanup/lama_now.py) each load one model group of a VRAM manager for a single action, holding
the GPU lock a pipeline run holds too (so they wait for a running chapter instead of fighting over the GPU), and
release everything afterwards. The GPU modules import torch, so they are imported only when a group is loaded.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from omniscan.core.config import Config


@contextmanager
def group_models(cfg: Config, group: str) -> Iterator[Mapping[str, Any]]:
    """The models of `group` (omniscan.gpu.groups: VISION_GROUP, INPAINT_GROUP) for one on-demand action, with
    exclusive GPU access (no lock on a CPU-only config); everything is released after."""
    from omniscan.gpu.groups import build_vram_manager
    from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock

    lock = acquire_gpu_lock() if cfg.gpu.device != "cpu" else None
    try:
        manager = build_vram_manager(cfg)
        try:
            yield manager.acquire(group)
        finally:
            manager.release()
    finally:
        if lock is not None:
            release_gpu_lock(lock)
