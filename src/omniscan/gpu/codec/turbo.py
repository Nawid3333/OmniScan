"""Turbo CPU JPEG codec (Pillow/libjpeg-turbo baseline). Decodes on CPU threads, uploads to GPU.

Speed baseline for card C2's `hybrid` backend; correctness over speed. The thread pool is lazily
recreated after `close()`, so a closed instance still works (see `close`).
"""

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageOps, JpegImagePlugin

from omniscan.gpu.codec.base import JpegInfo, Subsampling
from omniscan.gpu.device import resolve_device
from omniscan.gpu.timeline import mark

# PIL hands out read-only arrays and the codec only ever reads them (copy into a tensor), so torch's
# "array is not writable" warning is noise that would otherwise appear once in every CLI run.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)

_EXIF_ORIENTATION = 0x0112  # EXIF Orientation tag id
_GET_SAMPLING_CODES = {0: "444", 1: "422", 2: "420"}  # newer Pillow: get_sampling -> code
_LAYER_CODES = {(1, 1): "444", (2, 1): "422", (2, 2): "420"}  # (luma h, v) -> subsampling
_ENCODE_SAMPLING_CODES: dict[str, int] = {"444": 0, "422": 1, "420": 2}  # Pillow save codes


class TurboCodec:
    """CPU JPEG codec; decodes via Pillow on worker threads and uploads tensors to `device`."""

    name = "turbo"

    def __init__(self, device: str | torch.device = "auto", max_workers: int | None = None) -> None:
        self.device = resolve_device(device)
        self._max_workers = max_workers if max_workers is not None else os.cpu_count()
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(self._max_workers)

    def _ensure_pool(self) -> ThreadPoolExecutor:
        """Return the worker pool, lazily recreating it after `close()`."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self._max_workers)
        return self._pool

    def available(self) -> bool:
        """Always True: Pillow is a hard dependency of this project."""
        return True

    def close(self) -> None:
        """Shut down the thread pool (idempotent); the pool is lazily recreated on the next decode."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def info(self, data: bytes) -> JpegInfo:
        """Parse the JPEG header only (no decode)."""
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            components = len(img.getbands())
            subsampling = "400" if components == 1 else _subsampling(img)
            # Pillow writes either key depending on version; both hold truthy values only when progressive
            progressive = bool(img.info.get("progressive")) or bool(img.info.get("progression"))
        return JpegInfo(width, height, components, subsampling, progressive)

    def decode(self, data: bytes) -> torch.Tensor:
        """Decode one JPEG to a uint8 [3, H, W] tensor on `device`."""
        arr = self._decode_cpu(data)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.device.type != "cuda":
            return tensor.to(self.device)
        pinned = torch.empty(tensor.shape, dtype=torch.uint8, pin_memory=True)
        pinned.copy_(tensor)
        return pinned.to(self.device, non_blocking=True)

    def decode_into(self, datas: Sequence[bytes], out: torch.Tensor, y_offsets: Sequence[int]) -> None:
        """Decode images of width == out.shape[2] into out[:, y:y+h, :] for each y offset (batched).

        Raises ValueError if any image width differs from the strip width.
        """
        pool = self._ensure_pool()
        arrs = list(pool.map(self._decode_cpu, datas))
        mark(f"turbo decoded {len(arrs)} pages")
        width = out.shape[2]
        for i, (arr, _y) in enumerate(zip(arrs, y_offsets, strict=True)):
            if arr.shape[1] != width:
                msg = f"image {i} width {arr.shape[1]} != strip width {width}"
                raise ValueError(msg)
        if self.device.type != "cuda":
            for arr, y in zip(arrs, y_offsets, strict=True):
                out[:, y : y + arr.shape[0], :].copy_(torch.from_numpy(arr).permute(2, 0, 1))
            mark("turbo copies done")
            return
        # One pinned mirror of the whole strip + a single H2D copy: during another thread's library
        # init (MIOpen), per-page blocking H2D copies each stall behind it, while CPU memcpys do not.
        staging = torch.empty(out.shape, dtype=torch.uint8, pin_memory=True)
        for arr, y in zip(arrs, y_offsets, strict=True):
            staging[:, y : y + arr.shape[0], :].copy_(torch.from_numpy(arr).permute(2, 0, 1))
        mark("staging filled")
        out.copy_(staging)
        mark("turbo copies done")

    def _decode_cpu(self, data: bytes) -> np.ndarray:
        """Decode JPEG bytes to an (H, W, 3) uint8 RGB array on the calling thread (no GPU work)."""
        with Image.open(io.BytesIO(data)) as img:
            if img.getexif().get(_EXIF_ORIENTATION, 1) != 1:
                transposed = ImageOps.exif_transpose(img)
                if transposed is not None:
                    img = transposed
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)

    def encode(self, image: torch.Tensor, quality: int = 92, subsampling: Subsampling = "420") -> bytes:
        """Encode a uint8 [3, H, W] tensor (any device) to baseline JPEG bytes."""
        cpu = image.cpu() if image.device.type != "cpu" else image
        buf = io.BytesIO()
        Image.fromarray(cpu.permute(1, 2, 0).numpy(), "RGB").save(
            buf,
            format="JPEG",
            quality=quality,
            subsampling=_ENCODE_SAMPLING_CODES[subsampling],
            optimize=False,
        )
        return buf.getvalue()


def _subsampling(img: Image.Image) -> str:
    """Map a Pillow JPEG image's chroma sampling to "444"/"422"/"420" ("420" when unknown)."""
    get_sampling = getattr(JpegImagePlugin, "get_sampling", None)
    if get_sampling is not None:
        code = get_sampling(img)
        if code in _GET_SAMPLING_CODES:
            return _GET_SAMPLING_CODES[code]
    layer = getattr(img, "layer", None)
    if layer:
        h, v = layer[0][1], layer[0][2]
        if (h, v) in _LAYER_CODES:
            return _LAYER_CODES[(h, v)]
    return "420"
