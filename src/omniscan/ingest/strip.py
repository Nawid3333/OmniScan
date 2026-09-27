"""Strip building: decode a chapter's JPEGs straight into one preallocated strip tensor."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from omniscan.core.paths import jpeg_paths
from omniscan.core.schemas import IngestArtifact
from omniscan.core.stage import ChapterContext
from omniscan.gpu.codec.base import JpegCodec
from omniscan.gpu.codec.select import get_codec
from omniscan.gpu.timeline import mark


def build_strip(ingest: IngestArtifact, paths: Sequence[Path], codec: JpegCodec) -> torch.Tensor:
    """Fill the chapter's strip tensor on the codec's device; off-width pages are resized to the strip width."""
    if len(paths) != len(ingest.files):
        raise ValueError(f"{len(paths)} paths given for {len(ingest.files)} ingest files")

    strip = torch.empty((3, ingest.strip_height, ingest.strip_width), dtype=torch.uint8, device=codec.device)
    mark("strip allocated")
    _decode_full_width(ingest, paths, strip, codec)
    mark("strip full-width decoded")
    _resize_off_width(ingest, paths, strip, codec)
    mark("strip resize done")
    return strip


def _decode_full_width(
    ingest: IngestArtifact, paths: Sequence[Path], strip: torch.Tensor, codec: JpegCodec
) -> None:
    """Batch-decode every page already at the strip width straight into its strip rows (one call)."""
    entries = [
        (file, path)
        for file, path in zip(ingest.files, paths, strict=True)
        if file.width == ingest.strip_width
    ]
    datas = [path.read_bytes() for _, path in entries]
    mark(f"strip bytes read ({sum(len(d) for d in datas) // 1024} KiB)")
    codec.decode_into(datas, strip, [file.y0 for file, _ in entries])


def _resize_off_width(
    ingest: IngestArtifact, paths: Sequence[Path], strip: torch.Tensor, codec: JpegCodec
) -> None:
    """Decode each off-width page and bilinear-resize it into its strip rows."""
    for file, path in zip(ingest.files, paths, strict=True):
        if file.width == ingest.strip_width:
            continue
        image = codec.decode(path.read_bytes())
        resized = _resize(image, file.y1 - file.y0, ingest.strip_width)
        strip[:, file.y0 : file.y1, :] = resized


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear-resize a [3, h, w] tensor to [3, height, width] uint8 (antialiased)."""
    return (
        torch.nn.functional.interpolate(
            image.unsqueeze(0).float(),
            size=(height, width),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .squeeze(0)
    )


def load_strip(ctx: ChapterContext, ingest: IngestArtifact) -> torch.Tensor:
    """The chapter strip (uint8 [3, H, W] on the configured device), decoded once per chapter pass and shared."""

    def build() -> torch.Tensor:
        codec = get_codec(ctx.cfg)
        try:
            return build_strip(
                ingest, jpeg_paths(ingest, ctx.paths.raw_dir, ctx.paths.work_dir / "converted"), codec
            )
        finally:
            close = getattr(codec, "close", None)
            if close is not None:
                close()

    return ctx.lazy("strip", build)
