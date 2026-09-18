"""Codec factory: the JPEG backend requested by the config (card C2 adds rocjpeg/hybrid selection)."""

from __future__ import annotations

from omniscan.core.config import Config
from omniscan.gpu.codec.base import CodecUnavailableError, JpegCodec
from omniscan.gpu.codec.turbo import TurboCodec


def get_codec(cfg: Config) -> JpegCodec:
    """Return the codec for cfg.gpu.codec ('auto' currently means turbo; other backends are card C2)."""
    if cfg.gpu.codec in ("auto", "turbo"):
        return TurboCodec(cfg.gpu.device)
    raise CodecUnavailableError(f"codec backend {cfg.gpu.codec!r} is not implemented yet")
