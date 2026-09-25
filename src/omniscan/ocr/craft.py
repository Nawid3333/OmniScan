"""CRAFT scene-text detector: finds lettering anywhere on the page, including stylised sound effects.

The comic detector misses most sound effects drawn into the art (M10). CRAFT (Baek et al. 2019,
clovaai/CRAFT-pytorch, MIT; the `craft_mlt_25k` weights EasyOCR ships) scores every pixel for "centre of
a character" (region) and "between two characters of a word" (affinity), which also holds for big brush
lettering, outlines and tilted words. `CraftDetector.detect` turns tiles (uint8 `[3, h, w]`, any device)
into word boxes in tile pixels: the network runs on the GPU in fp32 (MIOpen's half kernels fail here),
and only the thresholded half-resolution maps — one uint8 transfer per tile — go to the host, where
OpenCV labels their connected components.
"""

from __future__ import annotations

import io
import logging
import math
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias
from torch import nn

from omniscan.detect.postprocess import Box
from omniscan.models.catalog import ModelEntry, load_catalog
from omniscan.models.download import download_model
from omniscan.models.store import install_path, model_status

log = logging.getLogger(__name__)

CRAFT_MODEL_ID = "sfx-detector-craft"  # catalog entry of the weights (config/models.toml)
CRAFT_WEIGHTS = "craft_mlt_25k.pth"  # the state dict inside the downloaded zip

_MEAN = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)  # ImageNet statistics the VGG backbone expects (RGB)
_STD = (0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0)
_STRIDE = 32  # the backbone downsamples 16x (4 pools) and pools once more at stride 1: pad to 32
_TEXT_THRESHOLD = 0.7  # a word needs one pixel this sure to be a character centre (EasyOCR's defaults)
_LINK_THRESHOLD = 0.4  # affinity above this joins characters into a word
_LOW_TEXT = 0.4  # region score above this belongs to a character
_MIN_AREA = 10  # smaller components (half-resolution pixels) are noise
# VGG16-BN's feature layers up to conv5_3's batch norm: channels per 3x3 conv, "M" a 2x2 max pool
_VGG16 = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512)
_SLICES = ((0, 12), (12, 19), (19, 29), (29, 39))  # conv2_2, conv3_3, conv4_3, conv5_3 outputs


def _vgg_layers() -> list[nn.Module]:
    """VGG16-BN's feature layers in torchvision's order (their indices name the checkpoint's keys)."""
    layers: list[nn.Module] = []
    channels = 3
    for spec in _VGG16:
        if spec == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers += [
            nn.Conv2d(channels, int(spec), 3, padding=1),
            nn.BatchNorm2d(int(spec)),
            nn.ReLU(inplace=True),
        ]
        channels = int(spec)
    return layers


class _Backbone(nn.Module):
    """VGG16-BN in four slices plus the dilated fc6/fc7 convolutions."""

    def __init__(self) -> None:
        super().__init__()
        layers = _vgg_layers()
        self.slice1, self.slice2, self.slice3, self.slice4 = (
            nn.Sequential(OrderedDict((str(i), layers[i]) for i in range(start, stop)))
            for start, stop in _SLICES
        )
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Feature maps from the deepest (fc7) to the shallowest (conv2_2)."""
        conv2 = self.slice1(x)
        conv3 = self.slice2(conv2)
        conv4 = self.slice3(conv3)
        conv5 = self.slice4(conv4)
        return self.slice5(conv5), conv5, conv4, conv3, conv2


class _UpConv(nn.Module):
    """U-net decoder step: 1x1 then 3x3 convolution over the upsampled map joined with the skip."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Craft(nn.Module):
    """The CRAFT network: image `[n, 3, h, w]` (h, w multiples of 32) -> region and affinity logits
    `[n, 2, h / 2, w / 2]`."""

    def __init__(self) -> None:
        super().__init__()
        self.basenet = _Backbone()
        self.upconv1 = _UpConv(1024, 512, 256)
        self.upconv2 = _UpConv(512, 256, 128)
        self.upconv3 = _UpConv(256, 128, 64)
        self.upconv4 = _UpConv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc7, conv5, conv4, conv3, conv2 = self.basenet(x)
        y = self.upconv1(torch.cat([fc7, conv5], dim=1))
        for up, skip in ((self.upconv2, conv4), (self.upconv3, conv3), (self.upconv4, conv2)):
            y = F.interpolate(y, size=skip.shape[2:], mode="bilinear", align_corners=False)
            y = up(torch.cat([y, skip], dim=1))
        return self.conv_cls(y)


def word_boxes(region: np.ndarray, affinity: np.ndarray) -> list[tuple[Box, float]]:
    """Word boxes in score-map pixels from region/affinity scores (float [h, w], 0..1): the connected
    components of "character or link" holding at least one confident character pixel, each grown by
    about its stroke (EasyOCR's rule: characters score high only at their centres), with the highest
    region score inside."""
    joined = ((region > _LOW_TEXT) | (affinity > _LINK_THRESHOLD)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=4)
    height, width = region.shape
    boxes: list[tuple[Box, float]] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < _MIN_AREA:
            continue
        inside = labels[y : y + h, x : x + w] == label
        peak = float(region[y : y + h, x : x + w][inside].max())
        if peak < _TEXT_THRESHOLD:
            continue
        grow = int(math.sqrt(area * min(w, h) / (w * h)) * 2)
        boxes.append(
            (
                (
                    float(max(0, x - grow)),
                    float(max(0, y - grow)),
                    float(min(width, x + w + grow)),
                    float(min(height, y + h + grow)),
                ),
                peak,
            )
        )
    return boxes


def _weights(models_dir: Path, catalog: Sequence[ModelEntry] | None) -> dict[str, torch.Tensor]:
    """The CRAFT state dict: the catalog zip under `models_dir`, downloaded (hash-checked) on first use."""
    entries = load_catalog() if catalog is None else catalog
    entry = next((e for e in entries if e.id == CRAFT_MODEL_ID), None)
    if entry is None:
        raise ValueError(f"model catalog has no {CRAFT_MODEL_ID!r} entry")
    path = install_path(entry, models_dir)
    if path is None:
        raise ValueError(f"{CRAFT_MODEL_ID} must be a file entry with an install_path")
    if model_status(entry, models_dir, ollama_names=None) != "installed":
        log.info("downloading %s into %s", CRAFT_MODEL_ID, models_dir)
        download_model(entry, models_dir)
    with zipfile.ZipFile(path) as archive:
        member = next((name for name in archive.namelist() if Path(name).name == CRAFT_WEIGHTS), None)
        if member is None:
            raise ValueError(f"{path} holds no {CRAFT_WEIGHTS}")
        state = torch.load(io.BytesIO(archive.read(member)), map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


class CraftDetector:
    """CRAFT on GPU tiles: `detect(tiles)` -> word boxes (tile pixels) with their peak character score."""

    def __init__(self, model: Craft, device: torch.device) -> None:
        self.model = model
        self._device = device
        self._mean = torch.tensor(_MEAN, device=device)[:, None, None]
        self._std = torch.tensor(_STD, device=device)[:, None, None]

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def load(
        cls, device: torch.device, models_dir: Path, catalog: Sequence[ModelEntry] | None = None
    ) -> CraftDetector:
        """Download (first use) and load the weights, fp32 (no `.half()` on this stack)."""
        if device.type == "cuda":
            # MIOpen launches kernels against the current device (the iGPU here), not the tensors' device
            torch.cuda.set_device(device)
        model = Craft()
        model.load_state_dict(_weights(Path(models_dir), catalog))
        return cls(model.to(device).eval(), device)

    def detect(self, tiles: Sequence[torch.Tensor]) -> list[list[tuple[Box, float]]]:
        """Word boxes per tile in tile pixels, each with its peak character score; runs one tile per forward."""
        per_tile: list[list[tuple[Box, float]]] = []
        for tile in tiles:
            height, width = int(tile.shape[-2]), int(tile.shape[-1])
            x = (tile.to(self._device).float() - self._mean) / self._std
            x = F.pad(x, (0, (-width) % _STRIDE, 0, (-height) % _STRIDE))
            with torch.inference_mode():
                scores = self.model(x[None])[0].clamp(0.0, 1.0)
            half_h, half_w = (height + 1) // 2, (width + 1) // 2
            maps = (scores[:, :half_h, :half_w] * 255.0).round().to(torch.uint8).cpu().numpy()  # one per tile
            boxes = word_boxes(maps[0].astype(np.float32) / 255.0, maps[1].astype(np.float32) / 255.0)
            per_tile.append(
                [
                    (
                        (
                            min(2.0 * b[0], float(width)),
                            min(2.0 * b[1], float(height)),
                            min(2.0 * b[2], float(width)),
                            min(2.0 * b[3], float(height)),
                        ),
                        score,
                    )
                    for b, score in boxes
                ]
            )
        return per_tile
