"""Detect stage wrapper: ingest.json + slices.json + chapter JPEGs -> regions.json (text empty).

The strip is cut into overlapping tiles (skipping blank/filtered slices), the detector runs in batches,
detections are merged across tiles and become regions via the pure functions in detect/postprocess.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, RegionsArtifact, SlicesArtifact
from omniscan.core.stage import ChapterContext
from omniscan.detect.postprocess import Det, build_regions, merge_detections, tile_det_to_strip
from omniscan.detect.tiles import keep_tiles, plan_tiles
from omniscan.detect.watermark_position import (
    add_watermark_zone_regions,
    reclassify_watermark_position_regions,
)
from omniscan.gpu.groups import VISION_GROUP
from omniscan.ingest.strip import load_strip
from omniscan.watermark.resolve import resolve_watermark_regions
from omniscan.watermark.store import WatermarkStore


class DetectStage:
    """Find bubbles and text regions in chapter strips (satisfies core.stage.Stage)."""

    name: ClassVar[str] = "detect"
    version: ClassVar[int] = (
        4  # 2: strips decoded before the CUDA staging-buffer fix (2026-09-19) held duplicated pages
        # 3: fixed-position watermark reclassification (F2c)
        # 4: a watermark region for every stored fixed-position box (erased by inpaint)
    )
    gpu_group: ClassVar[str | None] = VISION_GROUP

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        return [
            ctx.paths.artifact("ingest.json"),
            ctx.paths.artifact("slices.json"),
            ctx.series.work_dir / "watermarks.json",
            *list_images(ctx.paths.raw_dir),
        ]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["regions.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return cfg.detect.model_dump()

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        ingest_path = ctx.paths.artifact("ingest.json")
        if not ingest_path.is_file():
            raise FileNotFoundError("ingest.json missing — run the slice stage first")
        slices_path = ctx.paths.artifact("slices.json")
        if not slices_path.is_file():
            raise FileNotFoundError("slices.json missing — run the slice stage first")
        detector = models["detector"]
        cfg = ctx.cfg.detect

        ingest = IngestArtifact.load(ingest_path)
        slices = SlicesArtifact.load(slices_path)
        strip = load_strip(ctx, ingest)

        active = [(s.y0, s.y1) for s in slices.slices if not s.blank and not s.filtered]
        all_tiles = plan_tiles(ingest.strip_width, ingest.strip_height, cfg.tile_px, cfg.overlap)
        tiles = keep_tiles(all_tiles, active)

        dets: list[Det] = []
        raw_count = 0
        for start in range(0, len(tiles), cfg.batch_size):
            batch = tiles[start : start + cfg.batch_size]
            crops = [strip[:, t.y0 : t.y1, t.x0 : t.x1] for t in batch]  # views, never copies
            for tile, raw in zip(batch, detector.detect(crops), strict=True):
                dets.extend(tile_det_to_strip(tile, det.cls, det.score, det.box) for det in raw)
                raw_count += len(raw)
        merged = merge_detections(
            dets, nms_iou=cfg.nms_iou, contain_thr=cfg.contain_thr, edge_penalty=cfg.edge_penalty
        )
        regions = build_regions(
            merged,
            slices.slices,
            strip_width=ingest.strip_width,
            strip_height=ingest.strip_height,
            merge_bubble_text=cfg.merge_bubble_text,
            direction=cfg.reading_direction,
        )
        # a fixed-position watermark stored for this series (card F2c) reclassifies overlapping regions
        watermark_boxes = resolve_watermark_regions(WatermarkStore(ctx.series.work_dir).list(), ingest)
        regions = reclassify_watermark_position_regions(regions, watermark_boxes)
        regions = add_watermark_zone_regions(regions, watermark_boxes, slices.slices)
        RegionsArtifact(regions=regions).save(ctx.paths.artifact("regions.json"))
        return {
            "tiles": float(len(tiles)),
            "tiles_skipped": float(len(all_tiles) - len(tiles)),
            "raw_detections": float(raw_count),
            "merged_detections": float(len(merged)),
            "regions": float(len(regions)),
            "watermarked": float(sum(1 for r in regions if r.kind == "watermark")),
        }
