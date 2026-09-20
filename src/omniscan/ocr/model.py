"""OCR model wrappers: PP-OCRv5 line detector and the source-language line recognizer over GPU tensors.

The detector turns strip tiles (uint8 `[3, h, w]`, any device) into `LineBox` lists in tile pixels; the
recognizer turns line crops (uint8 `[3, h, w]`) into (text, score) pairs, batched by width. Both run in
fp32 (MIOpen's fp16/bf16 kernels fail on this stack) and never pull the strip or crops to the host.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from omniscan.core.config import OcrConfig
from omniscan.models.resolve import local_model_source
from omniscan.ocr.engines import load_kwargs, model_entry, model_source
from omniscan.ocr.lines import LineBox, polygon_box

log = logging.getLogger(__name__)


def _to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """`model.to(device)` via a Module-typed parameter: transformers' @wraps-wrapped `.to` rejects a device in pyright."""
    return model.to(device)


def _check_model_role(model_id: str, expected_role: str) -> None:
    """A `det_model`/`rec_model` catalog id must name a model of the role its loader expects."""
    role = model_entry(model_id).role
    if role != expected_role:
        raise ValueError(f"OCR model {model_id!r} has role {role!r}, expected {expected_role!r}")


class LineDetector:
    """PP-OCRv5 server text-line detector (`cfg.det_repo`): tiles -> line boxes per tile (tile pixels)."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device,
        *,
        det_threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        min_size: int,
    ) -> None:
        self.model = model
        self.processor = processor
        self._device = device
        self.det_threshold = det_threshold
        self.box_threshold = box_threshold
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device, models_dir: Path | None = None) -> LineDetector:
        """Download (first use) and load the line detector, fp32 everywhere (no `.half()` on this stack)."""
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if device.type == "cuda":
            # MIOpen launches kernels against the current device (the iGPU here), not the tensors' device
            torch.cuda.set_device(device)
        if cfg.det_model is not None:
            _check_model_role(cfg.det_model, "text_line_detector")
            repo, extra = load_kwargs(model_source(cfg.det_model, models_dir), models_dir=models_dir)
        else:
            source = local_model_source(cfg.det_repo, models_dir) if models_dir is not None else None
            if source is not None:
                log.info("loading %s from %s", cfg.det_repo, source)
                extra: dict[str, Any] = {"local_files_only": True}
                repo = source
            else:
                if models_dir is not None:
                    log.warning(
                        "model %s is not installed in %s; using the Hugging Face hub/cache. "
                        'Run "omniscan models download --required" to install it.',
                        cfg.det_repo,
                        models_dir,
                    )
                extra = {"revision": cfg.det_revision} if cfg.det_revision else {}
                repo = cfg.det_repo
        model = AutoModelForObjectDetection.from_pretrained(repo, **extra)
        model = _to_device(model, device).eval()
        processor = AutoImageProcessor.from_pretrained(repo, **extra)
        return cls(
            model,
            processor,
            device,
            det_threshold=cfg.det_threshold,
            box_threshold=cfg.box_threshold,
            unclip_ratio=cfg.unclip_ratio,
            min_size=cfg.min_size,
        )

    def detect(self, tiles: Sequence[torch.Tensor]) -> list[list[LineBox]]:
        """Line boxes per tile (clipped to the tile, too-small boxes dropped); an empty batch runs nothing."""
        if not tiles:
            return []
        per_tile: list[list[LineBox]] = []
        for tile in tiles:
            height, width = int(tile.shape[-2]), int(tile.shape[-1])
            inputs = self.processor(images=tile.to(self._device), return_tensors="pt")
            with torch.inference_mode():
                outputs = self.model(pixel_values=inputs["pixel_values"])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=self.det_threshold,
                box_threshold=self.box_threshold,
                unclip_ratio=self.unclip_ratio,
                min_size=self.min_size,
                target_sizes=torch.tensor([[height, width]]),
            )
            result = results[0]
            scores = result["scores"].tolist()  # one host transfer per tile, never per detection
            polygons = result["boxes"].tolist()
            lines: list[LineBox] = []
            for score, points in zip(scores, polygons, strict=True):
                x0, y0, x1, y1 = polygon_box(points)
                clipped = (
                    min(max(x0, 0.0), float(width)),
                    min(max(y0, 0.0), float(height)),
                    min(max(x1, 0.0), float(width)),
                    min(max(y1, 0.0), float(height)),
                )
                if clipped[2] - clipped[0] < self.min_size or clipped[3] - clipped[1] < self.min_size:
                    continue
                lines.append(LineBox(box=clipped, score=float(score)))
            per_tile.append(lines)
        return per_tile


class LineRecognizer:
    """Source-language recognizer (`cfg.rec_repo`): line crops (uint8 `[3, h, w]`) -> (text, score)."""

    def __init__(self, model: Any, processor: Any, device: torch.device, *, batch_size: int) -> None:
        self.model = model
        self.processor = processor
        self._device = device
        self.batch_size = batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device, models_dir: Path | None = None) -> LineRecognizer:
        """Download (first use) and load the recognition model, fp32 everywhere (no `.half()` on this stack)."""
        from transformers import AutoImageProcessor, AutoModelForTextRecognition

        if device.type == "cuda":
            # MIOpen launches kernels against the current device (the iGPU here), not the tensors' device
            torch.cuda.set_device(device)
        if cfg.rec_model is not None:
            _check_model_role(cfg.rec_model, "recognizer")
            repo, extra = load_kwargs(model_source(cfg.rec_model, models_dir), models_dir=models_dir)
        else:
            source = local_model_source(cfg.rec_repo, models_dir) if models_dir is not None else None
            if source is not None:
                log.info("loading %s from %s", cfg.rec_repo, source)
                extra: dict[str, Any] = {"local_files_only": True}
                repo = source
            else:
                if models_dir is not None:
                    log.warning(
                        "model %s is not installed in %s; using the Hugging Face hub/cache. "
                        'Run "omniscan models download --required" to install it.',
                        cfg.rec_repo,
                        models_dir,
                    )
                extra = {"revision": cfg.rec_revision} if cfg.rec_revision else {}
                repo = cfg.rec_repo
        model = AutoModelForTextRecognition.from_pretrained(repo, **extra)
        model = _to_device(model, device).eval()
        processor = AutoImageProcessor.from_pretrained(repo, **extra)
        return cls(model, processor, device, batch_size=cfg.rec_batch_size)

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]:
        """(text, score) per crop in the original order; crops are batched by ascending width."""
        if not crops:
            return []
        readings: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: int(crops[i].shape[-1]))
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            inputs = self.processor(images=[crops[i].to(self._device) for i in chunk], return_tensors="pt")
            with torch.inference_mode():
                outputs = self.model(**inputs)
            results = self.processor.post_process_text_recognition(outputs)
            for index, result in zip(chunk, results, strict=True):
                readings[index] = (result["text"], float(result["score"]))
        return readings
