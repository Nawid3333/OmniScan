"""Find text the detector missed on one page: the Studio's *Find missed text* and `omniscan edit find` (#35).

The `detect` stage's own functions run again on just that page — optionally at a lower score threshold, for
text the series setting passes over, or with detection settings changed since the chapter ran — and every box
no current region covers is read by the configured OCR (ocr/on_demand.py). Nothing is written here: the boxes
are suggestions the editor adds as hand-drawn regions (edits.store.add_region, undoable) or ignores, so the
pipeline's regions, the hand edits and the stage manifest stay as they are. The models come from the vision
group of a VRAM manager with exclusive GPU access, like the stages; torch is imported only when a page is
searched, so the web app and the CLI stay importable without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from omniscan.cleanup.strip import strip_crop
from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import BBox, IngestArtifact, Region, RegionKind, SlicesArtifact
from omniscan.detect.postprocess import Det, build_regions, ioa, merge_detections, tile_det_to_strip
from omniscan.detect.tiles import plan_tiles
from omniscan.edits import store
from omniscan.ocr.on_demand import in_crop, on_device, read_in, vision_models

COVERED_IOA = 0.5  # a found box this much inside a current region, or holding it this much, is that region's


@dataclass(frozen=True, slots=True)
class Found:
    """A text area the detector finds on a page that no current region covers, with what the OCR reads there."""

    kind: RegionKind
    bbox: BBox  # strip space
    bubble_bbox: BBox | None
    score: float  # the detector's
    text: str
    confidence: float  # the OCR's


def _box(bbox: BBox) -> tuple[float, float, float, float]:
    return (float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1))


def covered(box: BBox, regions: Sequence[Region]) -> bool:
    """Whether a current region already has the text at `box`: `box` mostly inside it (the same box, or a line of
    it), or the region mostly inside `box` (a bubble found around it)."""
    found = _box(box)
    return any(
        ioa(found, _box(region.bbox)) >= COVERED_IOA or ioa(_box(region.bbox), found) >= COVERED_IOA
        for region in regions
    )


def page_rows(ingest: IngestArtifact, page: int) -> BBox:
    """The strip rows of raw page `page` (SourceFile.index) across the strip's width; ValueError for no such page."""
    source = next((file for file in ingest.files if file.index == page and not file.filtered), None)
    if source is None:
        raise ValueError(f"the chapter has no page with index {page}")
    return BBox(x0=0, y0=source.y0, x1=ingest.strip_width, y1=source.y1)


def detect_page(
    tile: Any, rows: BBox, ingest: IngestArtifact, slices: SlicesArtifact, cfg: Config, detector: Any
) -> list[Region]:
    """The regions (text empty, strip space) the detector finds in `tile`, the pixels of `rows`."""
    det = cfg.detect
    tiles = plan_tiles(rows.width, rows.height, det.tile_px, det.overlap)
    dets: list[Det] = []
    for start in range(0, len(tiles), det.batch_size):
        batch = tiles[start : start + det.batch_size]
        crops = [tile[:, t.y0 : t.y1, t.x0 : t.x1] for t in batch]  # views, never copies
        for piece, raw in zip(batch, detector.detect(crops), strict=True):
            placed = replace(piece, y0=piece.y0 + rows.y0, y1=piece.y1 + rows.y0)  # into strip rows
            dets.extend(tile_det_to_strip(placed, found.cls, found.score, found.box) for found in raw)
    merged = merge_detections(
        dets, nms_iou=det.nms_iou, contain_thr=det.contain_thr, edge_penalty=det.edge_penalty
    )
    return build_regions(
        merged,
        slices.slices,
        strip_width=ingest.strip_width,
        strip_height=ingest.strip_height,
        merge_bubble_text=det.merge_bubble_text,
        direction=det.reading_direction,
    )


def find_on_page(
    paths: ChapterPaths, page: int, cfg: Config, models: Mapping[str, Any], *, threshold: float | None = None
) -> list[Found]:
    """The text areas on raw page `page` that no current region covers, read by the OCR, in reading order;
    `threshold` replaces the detector's score threshold for this search. FileNotFoundError without ingest.json,
    slices.json or a page file; ValueError for no such page."""
    for name in ("ingest.json", "slices.json"):
        if not paths.artifact(name).is_file():
            raise FileNotFoundError(f"{name} missing — run the slice stage first")
    ingest = IngestArtifact.load(paths.artifact("ingest.json"))
    slices = SlicesArtifact.load(paths.artifact("slices.json"))
    rows = page_rows(ingest, page)
    tile = on_device(strip_crop(paths, ingest, rows), cfg)
    detector = models["detector"]
    configured = detector.threshold
    detector.threshold = configured if threshold is None else threshold
    try:
        found = detect_page(tile, rows, ingest, slices, cfg, detector)
    finally:
        detector.threshold = configured
    current = store.current_regions(paths)
    new = [region for region in found if not covered(region.bbox, current)]
    if not new:
        return []
    read, _engine = read_in(tile, [in_crop(region, rows) for region in new], cfg, models)
    reading = {region.id: region for region in read}
    return [
        Found(
            kind=region.kind,
            bbox=region.bbox,
            bubble_bbox=region.bubble_bbox,
            score=region.confidence,
            text=reading[region.id].text,
            confidence=reading[region.id].confidence,
        )
        for region in new
    ]


def find_on_page_now(
    paths: ChapterPaths, page: int, cfg: Config, threshold: float | None = None
) -> list[Found]:
    """`find_on_page` with the vision models loaded just for this search (the web API's and the CLI's default)."""
    with vision_models(cfg) as models:
        return find_on_page(paths, page, cfg, models, threshold=threshold)
