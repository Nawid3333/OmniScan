"""The `qa` stage: the exported pages re-read with the chapter's OCR engine; regions whose original text is still
readable are listed in qa.json (qa/leftover.py decides). Not part of `omniscan run`: `omniscan qa` runs it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch

from omniscan.core.config import Config
from omniscan.core.schemas import ExportArtifact, IngestArtifact, QaArtifact, RegionsArtifact, SlicesArtifact
from omniscan.core.stage import ChapterContext
from omniscan.edits.store import EDITS_FILE, load_edits
from omniscan.export.segments import output_segments
from omniscan.gpu.codec.select import get_codec
from omniscan.gpu.groups import VISION_GROUP
from omniscan.ocr.engines import engine_rec_model
from omniscan.ocr.pipeline import read_region_crops, read_regions
from omniscan.qa.leftover import QA_FILE, cleaned_kinds, leftover_issues, regions_to_check, segment_rows


def lettered_strip(
    ctx: ChapterContext, ingest: IngestArtifact, rows: Sequence[tuple[str, int, int]]
) -> torch.Tensor:
    """The finished strip (uint8 [3, H, W] on the codec's device), decoded from the exported images; the rows no
    image covers (filtered slices) are black."""
    codec = get_codec(ctx.cfg)
    try:
        strip = torch.empty(
            (3, ingest.strip_height, ingest.strip_width), dtype=torch.uint8, device=codec.device
        )
        datas = [(ctx.paths.output_dir / name).read_bytes() for name, _, _ in rows]
        codec.decode_into(datas, strip, [y0 for _, y0, _ in rows])
    finally:
        close = getattr(codec, "close", None)
        if close is not None:
            close()
    covered = 0
    for _, y0, y1 in sorted(rows, key=lambda row: row[1]):
        strip[:, covered:y0, :] = 0  # decode_into leaves rows it has no image for undefined
        covered = max(covered, y1)
    strip[:, covered:, :] = 0
    return strip


class QaStage:
    """export.json + the exported images + ocr.json -> qa.json (the vision group's OCR models)."""

    name: ClassVar[str] = "qa"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = VISION_GROUP

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output."""
        inputs = [
            ctx.paths.artifact(name) for name in ("ingest.json", "slices.json", "ocr.json", "export.json")
        ]
        edits = ctx.paths.artifact(EDITS_FILE)
        if edits.is_file():
            inputs.append(edits)
        export = ctx.paths.artifact("export.json")
        if export.is_file():
            inputs.extend(ctx.paths.output_dir / f.name for f in ExportArtifact.load(export).files)
        return inputs

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return [QA_FILE]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {
            **cfg.ocr.model_dump(),
            "reading_direction": cfg.detect.reading_direction,
            "cleaned": sorted(cleaned_kinds(cfg)),
        }

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Re-read the finished pages, write qa.json, return metrics."""
        export_path = ctx.paths.artifact("export.json")
        if not export_path.is_file():
            raise FileNotFoundError("export.json missing — run the export stage first")
        ingest = IngestArtifact.load(ctx.paths.artifact("ingest.json"))
        slices = SlicesArtifact.load(ctx.paths.artifact("slices.json")).slices
        regions = RegionsArtifact.load(ctx.paths.artifact("ocr.json")).regions
        segments = output_segments(slices, ingest.strip_height, load_edits(ctx.paths).cuts)
        rows = segment_rows(ExportArtifact.load(export_path), segments)
        to_read = regions_to_check(regions, cleaned_kinds(ctx.cfg), segments)
        readings: dict[str, tuple[str, float]] = {}
        if to_read:
            strip = lettered_strip(ctx, ingest, rows)
            cfg = ctx.cfg
            if cfg.ocr.engine == "ppocr":
                engine = cfg.ocr.rec_model or cfg.ocr.rec_repo.split("/")[-1]
                read, _ = read_regions(
                    strip,
                    to_read,
                    models["line_detector"],
                    models["recognizer"],
                    cfg.ocr,
                    direction=cfg.detect.reading_direction,
                    engine=engine,
                )
            else:
                rec_model = engine_rec_model(cfg.ocr)
                if rec_model is None:
                    raise ValueError(f"OCR engine '{cfg.ocr.engine}' needs ocr.rec_model")
                read, _ = read_region_crops(strip, to_read, models["reader"], cfg.ocr, engine=rec_model)
            readings = {r.id: (r.text, r.confidence) for r in read}
        issues = leftover_issues(to_read, readings)
        QaArtifact(checked=len(to_read), issues=issues).save(ctx.paths.artifact(QA_FILE))
        return {
            "checked": float(len(to_read)),
            "issues": float(len(issues)),
            "source_left": float(sum(1 for i in issues if i.kind == "source_left")),
            "watermark_left": float(sum(1 for i in issues if i.kind == "watermark_left")),
        }
