"""VRAM model groups for the pipeline stages: one manager per command, model groups registered here.

The vision group holds the comic detector and the two OCR models (line detector + recognizer); they are
always resident together. Loaders import their models lazily so importing this module — and
`omniscan --help` — stays free of heavy model libraries.
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
        from omniscan.ocr.model import LineDetector, LineRecognizer  # deferred

        return {
            "detector": Detector.load(cfg.detect, device),
            "line_detector": LineDetector.load(cfg.ocr, device),
            "recognizer": LineRecognizer.load(cfg.ocr, device),
        }

    manager = VramManager(cfg.gpu.device, cfg.gpu.vram_budget_gib, ollama_url=cfg.ollama.local_url)
    manager.register(VISION_GROUP, load_vision, est_gib=3.0)
    return manager
