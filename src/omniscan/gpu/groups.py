"""VRAM model groups for the pipeline stages: one manager per command, model groups registered here.

Card C3 registers the vision group (detector); C4 will add the OCR models to the same group. Loaders import
their model lazily so importing this module — and `omniscan --help` — stays free of heavy model libraries.
"""

from __future__ import annotations

from typing import Any

import torch

from omniscan.core.config import Config
from omniscan.gpu.vram import VramManager

VISION_GROUP = "vision"


def build_vram_manager(cfg: Config) -> VramManager:
    """A VramManager for cfg.gpu with the pipeline's model groups registered."""

    def load_vision(device: torch.device) -> dict[str, Any]:
        from omniscan.detect.model import (
            Detector,
        )  # deferred: importing this module must not pull transformers

        return {"detector": Detector.load(cfg.detect, device)}

    manager = VramManager(cfg.gpu.device, cfg.gpu.vram_budget_gib, ollama_url=cfg.ollama.local_url)
    manager.register(VISION_GROUP, load_vision, est_gib=2.0)
    return manager
