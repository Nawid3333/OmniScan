"""Export stage wrapper: patches + layout + slices -> the released English slices in the output folder."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import (
    ExportArtifact,
    ExportFile,
    IngestArtifact,
    InpaintArtifact,
    LayoutArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.core.stage import ChapterContext
from omniscan.export.composite import apply_patches, blend_rgba
from omniscan.gpu.codec.base import JpegCodec
from omniscan.gpu.codec.select import get_codec
from omniscan.ingest.strip import load_strip
from omniscan.inpaint.patches import load_patches
from omniscan.typeset.render import render_item

if TYPE_CHECKING:
    import torch

_STALE_FILE = re.compile(r"\d{4}\.jpg")

_REQUIRED = (
    ("ingest.json", "ingest"),
    ("slices.json", "slice"),
    ("inpaint.json", "inpaint"),
    ("patches.npz", "inpaint"),
    ("layout.json", "typeset"),
)


class ExportStage:
    """Cut the finished English strip into the output folder's JPEG slices (satisfies core.stage.Stage)."""

    name: ClassVar[str] = "export"
    version: ClassVar[int] = (
        2  # 2: strips decoded before the CUDA staging-buffer fix (2026-09-19) held duplicated pages
    )
    gpu_group: ClassVar[str | None] = None

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and upstream artifacts)."""
        inputs = [
            ctx.paths.artifact("ingest.json"),
            ctx.paths.artifact("slices.json"),
            ctx.paths.artifact("inpaint.json"),
            ctx.paths.artifact("patches.npz"),
            ctx.paths.artifact("layout.json"),
        ]
        for name in ("inpaint_lama.json", "patches_lama.npz"):
            path = ctx.paths.artifact(name)
            if path.is_file():
                inputs.append(path)
        return [*inputs, *list_images(ctx.paths.raw_dir)]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["export.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return cfg.export.model_dump()

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        for name, stage in _REQUIRED:
            if not ctx.paths.artifact(name).is_file():
                raise FileNotFoundError(f"{name} missing — run the {stage} stage first")
        ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
        strip = load_strip(ctx, ingest)
        patches = float(
            apply_patches(
                strip,
                InpaintArtifact.load(ctx.paths.artifact("inpaint.json")).items,
                load_patches(ctx.paths.artifact("patches.npz")),
            )
        )
        lama_json, lama_npz = ctx.paths.artifact("inpaint_lama.json"), ctx.paths.artifact("patches_lama.npz")
        if lama_json.is_file() and lama_npz.is_file():  # LaMa overrides the flat-fill placeholder
            patches += apply_patches(strip, InpaintArtifact.load(lama_json).items, load_patches(lama_npz))
        glyph_items = overflow_items = 0
        for item in LayoutArtifact.load(ctx.paths.artifact("layout.json")).items:
            overflow_items += int(item.overflow)
            glyph = render_item(item)
            if glyph is None:
                continue
            blend_rgba(strip, glyph)
            glyph_items += 1
        slices = SlicesArtifact.load(ctx.paths.artifact("slices.json")).slices
        codec = get_codec(ctx.cfg)
        try:
            files = _write_slices(ctx, strip, slices, codec)
        finally:
            close = getattr(codec, "close", None)
            if close is not None:
                close()
        ExportArtifact(
            quality=ctx.cfg.export.jpeg_quality,
            subsampling=ctx.cfg.export.subsampling,
            files=files,
        ).save(ctx.paths.artifact("export.json"))
        return {
            "slices": float(len(files)),
            "bytes": float(sum(f.bytes for f in files)),
            "patches": patches,
            "glyph_items": float(glyph_items),
            "overflow_items": float(overflow_items),
        }


def _write_slices(
    ctx: ChapterContext, strip: torch.Tensor, slices: list[Slice], codec: JpegCodec
) -> list[ExportFile]:
    """Delete stale NNNN.jpg files from the output folder, then encode and write the kept slices in order."""
    output = ctx.paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file() and _STALE_FILE.fullmatch(path.name):
            path.unlink()
    files: list[ExportFile] = []
    for number, s in enumerate((s for s in slices if not s.filtered), start=1):
        data = codec.encode(
            strip[:, s.y0 : s.y1, :],
            quality=ctx.cfg.export.jpeg_quality,
            subsampling=ctx.cfg.export.subsampling,
        )
        name = f"{number:04d}.jpg"
        (output / name).write_bytes(data)
        files.append(
            ExportFile(
                name=name,
                slice_index=s.index,
                width=int(strip.shape[2]),
                height=s.y1 - s.y0,
                bytes=len(data),
            )
        )
    return files
