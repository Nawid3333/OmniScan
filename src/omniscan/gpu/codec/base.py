"""JPEG codec contract. Backends: rocjpeg (VCN hardware), hybrid (CPU entropy + GPU IDCT), turbo (CPU baseline).

All decoded images are uint8 RGB tensors shaped [3, H, W] (CHW) on the codec's device. `decode_into` writes
several images of identical width straight into rows of a preallocated strip, so stitching costs nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch

Subsampling = Literal["444", "422", "420"]


@dataclass(frozen=True, slots=True)
class JpegInfo:
    width: int
    height: int
    components: int  # 1 = grayscale, 3 = YCbCr
    subsampling: str  # e.g. "420", "444", "400"
    progressive: bool


@runtime_checkable
class JpegCodec(Protocol):
    name: str
    device: torch.device

    def available(self) -> bool:
        """True if this backend works on this machine right now."""
        ...

    def info(self, data: bytes) -> JpegInfo:
        """Parse the JPEG header only (no decode)."""
        ...

    def decode(self, data: bytes) -> torch.Tensor:
        """Decode one JPEG to a uint8 [3, H, W] tensor on `device` (grayscale is expanded to 3 channels)."""
        ...

    def decode_into(self, datas: Sequence[bytes], out: torch.Tensor, y_offsets: Sequence[int]) -> None:
        """Decode images of width == out.shape[2] into out[:, y:y+h, :] for each y offset (batched, async-safe).

        Raises ValueError if any image width differs from the strip width (the caller resizes those separately).
        """
        ...

    def encode(self, image: torch.Tensor, quality: int = 92, subsampling: Subsampling = "420") -> bytes:
        """Encode a uint8 [3, H, W] tensor (any device) to baseline JPEG bytes."""
        ...


class CodecUnavailableError(RuntimeError):
    """Raised when a requested backend cannot run on this machine."""
