"""Scoring for `omniscan eval`: detection recall/precision, OCR CER and translation chrF of a chapter."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

from omniscan.core.schemas import BBox, FinalArtifact, IngestArtifact, Region, RegionsArtifact
from omniscan.eval.metrics import chrf, levenshtein, normalize
from omniscan.eval.truth import TruthBox, TruthStats
from omniscan.translate.prompts import translatable

_MISS_LIMIT = 10
_WORST_LIMIT = 10


@dataclass(frozen=True, slots=True)
class BoxResult:
    """One truth box and, when detected, what the OCR read from it."""

    page: int
    bbox: BBox
    text: str
    detected: bool
    read: str = ""
    cer: float | None = None  # only when detected


@dataclass(frozen=True, slots=True)
class EvalReport:
    """All numbers of one evaluated chapter (written as eval.json in the chapter work dir)."""

    chapter: str
    series: str
    truth_boxes: int
    ignored_boxes: int
    dropped_boxes: int
    pages: int
    pages_without_truth: int
    regions: int
    assigned_regions: int
    detected_boxes: int
    recall: float | None
    precision: float | None
    cer_macro: float | None
    cer_micro: float | None
    cer_boxes: int
    chrf_mean: float | None
    chrf_pages: int
    missed: list[BoxResult]
    worst_cer: list[BoxResult]

    def to_json(self) -> str:
        """The report as pretty JSON (bboxes as [x0, y0, x1, y1])."""
        return json.dumps(_plain(self), indent=2, ensure_ascii=False)


def _plain(obj: object) -> object:
    """A JSON-ready tree: dataclasses as dicts, BBox as [x0, y0, x1, y1]."""
    if isinstance(obj, BBox):
        return [obj.x0, obj.y0, obj.x1, obj.y1]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: _plain(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_plain(item) for item in obj]
    return obj


def _contains(box: BBox, cx: float, cy: float) -> bool:
    """Inclusive containment test of a point in a box."""
    return box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1


def _area(box: BBox) -> int:
    return (box.x1 - box.x0) * (box.y1 - box.y0)


def _page_number(name: str) -> int | None:
    """The integer page number of a raw file name ('01.jpg' -> 1); None when the stem is not numeric."""
    try:
        return int(Path(name).stem)
    except ValueError:
        return None


def score_chapter(
    series: str,
    chapter: str,
    ingest: IngestArtifact,
    regions: RegionsArtifact,
    final: FinalArtifact | None,
    truth: Sequence[TruthBox],
    english_pages: dict[int, str],
    stats: TruthStats,
) -> EvalReport:
    """Score a chapter's ocr.json (and optional final.json) against its ground-truth boxes."""
    usable = [box for box in truth if normalize(box.text)]
    ignored = len(truth) - len(usable)
    predicted = [region for region in regions.regions if region.kind != "watermark"]

    assigned: list[list[Region]] = [[] for _ in usable]
    assigned_count = 0
    for region in predicted:
        cx = (region.bbox.x0 + region.bbox.x1) / 2
        cy = (region.bbox.y0 + region.bbox.y1) / 2
        best = -1
        best_area = 0
        for index, box in enumerate(usable):
            if _contains(box.bbox, cx, cy):
                area = _area(box.bbox)
                if best < 0 or area < best_area:
                    best, best_area = index, area
        if best >= 0:
            assigned[best].append(region)
            assigned_count += 1

    detected_boxes = 0
    distances = 0
    truth_chars = 0
    cers: list[float] = []
    detected: list[BoxResult] = []
    missed: list[BoxResult] = []
    for box, box_regions in zip(usable, assigned, strict=True):
        if not box_regions:
            missed.append(BoxResult(page=box.page, bbox=box.bbox, text=box.text, detected=False))
            continue
        detected_boxes += 1
        read = " ".join(
            region.text
            for region in sorted(box_regions, key=lambda r: (r.bbox.y0, r.bbox.x0))
        )
        distance = levenshtein(normalize(box.text), normalize(read))
        score = distance / len(normalize(box.text))
        distances += distance
        truth_chars += len(normalize(box.text))
        cers.append(score)
        detected.append(
            BoxResult(page=box.page, bbox=box.bbox, text=box.text, detected=True, read=read, cer=score)
        )
    missed.sort(key=lambda r: _area(r.bbox), reverse=True)

    chrf_scores = _page_chrfs(ingest, regions, final, english_pages) if final is not None else []

    return EvalReport(
        chapter=chapter,
        series=series,
        truth_boxes=len(usable),
        ignored_boxes=ignored,
        dropped_boxes=stats.dropped,
        pages=stats.pages,
        pages_without_truth=stats.pages_without_truth,
        regions=len(predicted),
        assigned_regions=assigned_count,
        detected_boxes=detected_boxes,
        recall=detected_boxes / len(usable) if usable else None,
        precision=assigned_count / len(predicted) if predicted else None,
        cer_macro=sum(cers) / len(cers) if cers else None,
        cer_micro=distances / truth_chars if truth_chars else None,
        cer_boxes=detected_boxes,
        chrf_mean=sum(chrf_scores) / len(chrf_scores) if chrf_scores else None,
        chrf_pages=len(chrf_scores),
        missed=missed[:_MISS_LIMIT],
        worst_cer=sorted(detected, key=lambda r: r.cer or 0.0, reverse=True)[:_WORST_LIMIT],
    )


def _page_chrfs(
    ingest: IngestArtifact,
    regions: RegionsArtifact,
    final: FinalArtifact,
    english_pages: dict[int, str],
) -> list[float]:
    """chrF of final.json against the English truth, one score per scorable page."""
    by_region = {line.region_id: line.text for line in final.lines}
    ordered = translatable(regions.regions)
    rows = {
        page: (source.y0, source.y1)
        for source in ingest.files
        if not source.filtered and (page := _page_number(source.name)) is not None
    }
    scores: list[float] = []
    for page, reference in english_pages.items():
        if page not in rows:
            continue
        y0, y1 = rows[page]
        hypothesis = " ".join(
            by_region[region.id]
            for region in ordered
            if region.id in by_region and y0 <= (region.bbox.y0 + region.bbox.y1) / 2 < y1
        )
        if not reference:
            continue  # nothing to score against on this page
        scores.append(chrf(hypothesis, reference) if hypothesis else 0.0)
    return scores


def to_json_lines(reports: Sequence[EvalReport]) -> str:
    """The reports as compact JSON, one object per line (for `eval --json`)."""
    return "\n".join(json.dumps(_plain(report), ensure_ascii=False) for report in reports)