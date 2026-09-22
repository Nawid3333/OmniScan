"""VRAM model groups for the pipeline stages: one manager per command, model groups registered here.

The vision group holds the comic detector plus the OCR models the configured engine needs (ppocr:
line detector + recognizer; manga_ocr and paddleocr_vl: the crop reader); they are always resident
together. Loaders
import their models lazily so importing this module — and `omniscan --help` — stays free of heavy
model libraries.
"""

from __future__ import annotations

from typing import Any

import torch

from omniscan.core.config import Config
from omniscan.gpu.device import resolve_device
from omniscan.gpu.vram import VramManager
from omniscan.gpu.warmup import GpuWarmup, start_warmup

VISION_GROUP = "vision"
INPAINT_GROUP = "inpaint"


class WarmupVramManager(VramManager):
    """A VramManager carrying the GPU warm-up handle its build started (None when disabled)."""

    warmup: GpuWarmup | None


def build_vram_manager(cfg: Config) -> WarmupVramManager:
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
        elif engine == "paddleocr_vl":
            from omniscan.ocr.crop_readers import PaddleOcrVlReader  # deferred

            ocr = {"reader": PaddleOcrVlReader.load(cfg.ocr, device, models_dir=cfg.paths.models_dir)}
        else:
            raise ValueError(f"OCR engine {engine!r} is not available yet")
        return {"detector": Detector.load(cfg.detect, device, models_dir=cfg.paths.models_dir), **ocr}

    def load_inpaint(device: torch.device) -> dict[str, Any]:
        from omniscan.inpaint.lama import (
            LamaInpainter,
        )  # deferred: importing this module must not download weights

        return {"lama": LamaInpainter.load(cfg.inpaint, cfg.paths.models_dir, device)}

    device = resolve_device(cfg.gpu.device)
    manager = WarmupVramManager(device, cfg.gpu.vram_budget_gib, ollama_url=cfg.ollama.local_url)
    # 3.0 GiB comic detector + line OCR; paddleocr_vl's 0.9 B VLM (3.4 GiB fp32) on top makes 6.6
    est_gib = 6.6 if cfg.ocr.engine == "paddleocr_vl" else 3.0
    manager.register(VISION_GROUP, load_vision, est_gib=est_gib)
    manager.register(INPAINT_GROUP, load_inpaint, est_gib=2.0)
    # The warm-up runs on its own daemon thread and must not block the caller; resolve_device is the
    # same call VramManager makes, so both always land on the same device.
    manager.warmup = start_warmup(device) if cfg.gpu.warmup and device.type == "cuda" else None
    return manager
