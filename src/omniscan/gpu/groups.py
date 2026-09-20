"""VRAM model groups for the pipeline stages: one manager per command, model groups registered here.

The vision group holds the comic detector plus the OCR models the configured engine needs (ppocr:
line detector + recognizer; manga_ocr: the crop reader); they are always resident together. Loaders
import their models lazily so importing this module — and `omniscan --help` — stays free of heavy
model libraries.
"""

from __future__ import annotations

from typing import Any

import torch

from omniscan.core.config import Config
from omniscan.gpu.vram import VramManager

VISION_GROUP = "vision"
INPAINT_GROUP = "inpaint"


def build_vram_manager(cfg: Config) -> VramManager:
    """A VramManager for cfg.gpu with the pipeline's model groups registered."""

    def load_vision(device: torch.device) -> dict[str, Any]:
        from omniscan.detect.model import (
            Detector,
        )  # deferred: importing this module must not pull transformers
        from omniscan.ocr.model import LineDetector, LineRecognizer  # deferred

        engine = cfg.ocr.engine
        if engine == "ppocr":
            ocr: dict[str, Any] = {
                "line_detector": LineDetector.load(cfg.ocr, device, models_dir=cfg.paths.models_dir),
                "recognizer": LineRecognizer.load(cfg.ocr, device, models_dir=cfg.paths.models_dir),
            }
        elif engine == "manga_ocr":
            from omniscan.ocr.crop_readers import MangaOcrReader  # deferred

            ocr = {"reader": MangaOcrReader.load(cfg.ocr, device, models_dir=cfg.paths.models_dir)}
        else:
            raise ValueError(f"OCR engine {engine!r} is not available yet")
        return {"detector": Detector.load(cfg.detect, device, models_dir=cfg.paths.models_dir), **ocr}

    def load_inpaint(device: torch.device) -> dict[str, Any]:
        from omniscan.inpaint.lama import (
            LamaInpainter,
        )  # deferred: importing this module must not download weights

        return {"lama": LamaInpainter.load(cfg.inpaint, cfg.paths.models_dir, device)}

    manager = VramManager(cfg.gpu.device, cfg.gpu.vram_budget_gib, ollama_url=cfg.ollama.local_url)
    manager.register(VISION_GROUP, load_vision, est_gib=3.0)
    manager.register(INPAINT_GROUP, load_inpaint, est_gib=2.0)
    return manager
