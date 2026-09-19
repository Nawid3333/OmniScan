"""Detector wrapper: the ogkalu/comic-text-and-bubble-detector RT-DETR-v2 model over chapter-strip tiles.

Tiles in (`uint8 [3, h, w]`, any device), one `RawDet` list per tile out, in tile pixels. The image
processor is used only to scale boxes back to tile pixels; resizing and rescaling are done here so the
strip never leaves the GPU and no copy per detection is made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias

from omniscan.core.config import DetectConfig
from omniscan.detect.postprocess import Box, DetClass

_CLASSES: tuple[DetClass, ...] = ("bubble", "text_bubble", "text_free")


def _to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """`model.to(device)` via a Module-typed parameter: transformers' @wraps-wrapped `.to` rejects a device in pyright."""
    return model.to(device)


@dataclass(frozen=True, slots=True)
class RawDet:
    """One detection in the pixels of the tile it was found in."""

    cls: DetClass
    score: float
    box: Box  # (x0, y0, x1, y1), tile pixels


class Detector:
    """RT-DETR-v2 (`cfg.repo`) run on GPU in batches: tiles -> boxes/scores per tile."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device,
        *,
        threshold: float,
        input_size: int = 640,
    ) -> None:
        self.model = model
        self.processor = processor
        self._device = device
        self.threshold = threshold
        self.input_size = input_size

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def load(cls, cfg: DetectConfig, device: torch.device) -> Detector:
        """Download (first use) and load the model, fp32 everywhere (MIOpen fp16 conv fails at small batch sizes)."""
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        if device.type == "cuda":
            # MIOpen launches kernels against the current device (the iGPU here), not the tensors' device
            torch.cuda.set_device(device)
        extra: dict[str, Any] = {"revision": cfg.revision} if cfg.revision else {}
        model = RTDetrV2ForObjectDetection.from_pretrained(cfg.repo, **extra)
        model = _to_device(model, device).eval()
        processor = AutoImageProcessor.from_pretrained(cfg.repo, **extra)
        return cls(model, processor, device, threshold=cfg.threshold)

    def detect(self, tiles: Sequence[torch.Tensor]) -> list[list[RawDet]]:
        """Detections per tile (tile pixels, highest score first); an empty batch returns without running the model."""
        if not tiles:
            return []
        scaled = [
            F.interpolate(
                tile.unsqueeze(0).float(),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )[0]
            / 255.0
            for tile in tiles
        ]
        batch = torch.stack(scaled).to(next(self.model.parameters()).dtype)
        with torch.inference_mode():
            outputs = self.model(pixel_values=batch)
        target_sizes = [(int(tile.shape[-2]), int(tile.shape[-1])) for tile in tiles]
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )
        id2label = self.model.config.id2label
        per_tile: list[list[RawDet]] = []
        for result, tile in zip(results, tiles, strict=True):
            height, width = int(tile.shape[-2]), int(tile.shape[-1])
            scores = result["scores"].tolist()  # one host transfer per tensor, never per detection
            labels = result["labels"].tolist()
            boxes = result["boxes"].tolist()
            dets: list[RawDet] = []
            for score, label, box in zip(scores, labels, boxes, strict=True):
                name = id2label[int(label)]
                if name not in _CLASSES:
                    continue
                clipped = (
                    min(max(box[0], 0.0), float(width)),
                    min(max(box[1], 0.0), float(height)),
                    min(max(box[2], 0.0), float(width)),
                    min(max(box[3], 0.0), float(height)),
                )
                dets.append(RawDet(cls=cast("DetClass", name), score=score, box=clipped))
            dets.sort(key=lambda det: -det.score)
            per_tile.append(dets)
        return per_tile
