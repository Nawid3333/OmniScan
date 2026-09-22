"""LaMa inpainting model: TorchScript `big-lama.pt`, fp32, one fixed window shape warmed up per process."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from omniscan.core.config import InpaintConfig
from omniscan.gpu.timeline import mark
from omniscan.inpaint.lama_weights import ensure_lama_weights

log = logging.getLogger(__name__)


class LamaInpainter:
    """`model(image, mask)` over one fixed window: every new input shape costs a 10-25 s warm-up."""

    def __init__(self, model: Any, device: torch.device, *, window: int) -> None:
        self.model = model
        self._device = device
        self._window = window

    @property
    def window(self) -> int:
        """The one window side this inpainter was warmed up for."""
        return self._window

    @classmethod
    def load(cls, cfg: InpaintConfig, models_dir: Path, device: torch.device) -> LamaInpainter:
        """Download (first use) + load the fp32 weights and warm up the fixed window shape."""
        path = ensure_lama_weights(models_dir, cfg)
        mark("lama weights verified")
        if device.type == "cuda":
            # MIOpen launches kernels against the current device (the iGPU here), not the tensors' device
            torch.cuda.set_device(device)
        model = torch.jit.load(
            str(path), map_location=device
        ).eval()  # fp32 only: fp16 fails in the interpreter
        mark("lama jit loaded")
        window = cfg.lama_window
        image = torch.zeros((1, 3, window, window), device=device)
        mask = torch.zeros((1, 1, window, window), device=device)
        log.info("warming up LaMa (one-time, ~10-25 s)")
        # The forwards' MIOpen find is the pipeline's critical path (~15-28 s): it starts as early as
        # possible (the prefetch worker runs this loader once the first stage acquired) and overlaps the
        # vision pass. No lock vs the warm-up thread: concurrent find phases stretch each other but end
        # at nearly the same time as serialized ones, so a lock only delays the start (G3 measurement).
        with torch.inference_mode():
            model(image, mask)
            model(image, mask)
        mark("lama warmup forwards done")
        return cls(model, device, window=window)

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inpaint the bool [S, S] mask of a uint8 [3, S, S] window (S == window); result on the device."""
        window = self._window
        if image.shape != (3, window, window) or mask.shape != (window, window):
            raise ValueError(
                f"LaMa inpaint expects a uint8 [3, {window}, {window}] image and a bool "
                f"[{window}, {window}] mask, got {tuple(image.shape)} and {tuple(mask.shape)}"
            )
        x = image.to(self._device).float().div(255)[None]
        m = mask.to(self._device).float()[None, None]
        with torch.inference_mode():
            y = self.model(x, m)
        out = (y[0].clamp(0, 1) * 255).round().to(torch.uint8)
        # outside the mask the result is bit-identical to the input, exactly like the probe measured
        return torch.where(mask.to(self._device)[None], out, image.to(self._device))
