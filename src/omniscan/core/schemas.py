"""Artifact contracts exchanged between pipeline stages (JSON files in the chapter work dir).

Coordinate system: every pixel coordinate is in **strip space** — the chapter's stitched strip after width
normalisation (x right, y down, integers, half-open [x0, x1) / [y0, y1)). A slice is only a y-range of the strip.
Every top-level artifact carries `schema_version`; bump it on any breaking change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

RGB = tuple[int, int, int]
Lang = Literal["ko", "zh", "ja", "en"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Model(BaseModel):
    """Base for all artifacts: strict field set, immutable-by-convention."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Artifact(Model):
    """Top-level JSON artifact; `save`/`load` are atomic and versioned."""

    schema_version: int = 1

    def save(self, path: Path) -> None:
        """Atomically write as pretty JSON (tmp file + rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class BBox(Model):
    x0: int
    y0: int
    x1: int
    y1: int

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"degenerate bbox {self}")
        return self

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def iou(self, other: BBox) -> float:
        ix = max(0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union else 0.0


# ---------------------------------------------------------------- ingest / slicer


class SourceFile(Model):
    """One raw image of the chapter and where it lands in the strip."""

    index: int
    name: str  # file name inside the raw chapter dir
    sha256: str
    width: int  # original pixel size
    height: int
    y0: int  # rows in strip space after scaling to the strip width
    y1: int
    scale: float = 1.0  # strip_width / width
    converted_from: str | None = None  # original suffix if ingest converted it to JPEG
    filtered: bool = False  # removed by the file-level promo pre-check (not part of the strip)


class IngestArtifact(Artifact):
    series: str
    chapter: str
    strip_width: int
    strip_height: int
    files: list[
        SourceFile
    ]  # the files that make up the strip (promo files dropped by the file-level check are not listed)
    filtered_files: list[str] = Field(
        default_factory=list
    )  # names of raw files the file-level promo check left out


class Band(Model):
    """A run of uniform rows (safe cut zone)."""

    y0: int
    y1: int
    color: RGB
    is_gradient: bool = False


class Slice(Model):
    index: int
    y0: int
    y1: int
    blank: bool = False  # fully uniform: OCR / inpaint / typeset skip it
    forced_cut: bool = False  # the cut at y1 was forced (no uniform band available)
    filtered: bool = False  # removed by slice-level promo filter
    source_files: list[int] = Field(default_factory=list)  # SourceFile indices overlapping this slice


class SlicesArtifact(Artifact):
    strip_width: int
    strip_height: int
    bands: list[Band]
    slices: list[Slice]
    params: dict[str, float | int | str] = Field(default_factory=dict)


# ---------------------------------------------------------------- promo filter


class FilterDecision(Model):
    target: Literal["file", "slice"]
    index: int
    decision: Literal["keep", "filtered", "restored"]
    score: float
    matched_example: str | None = None
    method: Literal["phash", "embed", "manual"] = "phash"


class FilterArtifact(Artifact):
    decisions: list[FilterDecision]


# ---------------------------------------------------------------- detection / OCR

RegionKind = Literal["bubble_text", "free_text", "sfx", "watermark"]


class OcrLine(Model):
    bbox: BBox
    polygon: list[tuple[int, int]] | None = None
    text: str
    score: float
    engine: str  # e.g. "korean_PP-OCRv5_mobile_rec", "PaddleOCR-VL-1.5"


class Region(Model):
    id: str  # stable within the chapter: "r0001", "r0002", ...
    slice_index: int
    kind: RegionKind
    bbox: BBox  # text area
    bubble_bbox: BBox | None = None  # enclosing bubble, if any
    polygon: list[tuple[int, int]] | None = None  # bubble interior outline for typesetting
    reading_order: int = 0
    lang: Lang = "ko"
    orientation: Literal["h", "v"] = "h"
    lines: list[OcrLine] = Field(default_factory=list)
    text: str = ""
    confidence: float = 0.0
    ocr_alt: str | None = None  # second-opinion reading when engines disagree
    text_color: RGB | None = None
    stroke_color: RGB | None = None
    mask_ref: str | None = None  # key inside masks/<slice>.npz
    angle: float = 0.0  # baseline rotation of the lettering, degrees counter-clockwise (measured on sfx)
    weight: float | None = None  # stroke width / letter height of the original lettering (measured on sfx)


class RegionsArtifact(Artifact):
    """Written by `detect` (text empty) and completed by `ocr` (saved as ocr.json)."""

    regions: list[Region]


# ---------------------------------------------------------------- translation


class Candidate(Model):
    region_id: str
    text: str
    notes: str | None = None


class CandidateRun(Artifact):
    run_id: str  # e.g. "translategemma-12b-local@2026-09-18T12:00"
    profile: str
    model: str
    created_at: datetime = Field(default_factory=utcnow)
    candidates: list[Candidate]
    usage: dict[str, float] = Field(default_factory=dict)


class FinalLine(Model):
    region_id: str
    text: str
    decision: Literal["pick", "merge", "rewrite", "manual"]
    sources: list[str] = Field(default_factory=list)  # run_ids used
    rationale: str = ""
    flags: list[str] = Field(default_factory=list)  # e.g. "uncertain", "glossary_violation"


class FinalArtifact(Artifact):
    judge_model: str
    created_at: datetime = Field(default_factory=utcnow)
    lines: list[FinalLine]
    usage: dict[str, float] = Field(default_factory=dict)  # mirrors CandidateRun.usage; see JudgeStats


# ---------------------------------------------------------------- glossary

TermType = Literal["person", "place", "org", "skill", "item", "rank", "title", "honorific", "sfx", "other"]


class GlossaryEntry(Model):
    id: int | None = None
    source: str
    target: str
    type: TermType = "other"
    gender: Literal["male", "female", "unknown", "n/a"] = "unknown"
    pronouns: str | None = None
    aliases: list[str] = Field(default_factory=list)
    notes: str | None = None
    status: Literal["proposed", "locked", "rejected"] = "proposed"
    origin: Literal["llm", "reference", "user"] = "llm"
    first_seen_chapter: float | None = None
    count: int = 0


# ---------------------------------------------------------------- inpainting

InpaintMethod = Literal["flat", "lama", "none"]


class InpaintItem(Model):
    """One cleaned patch of the strip: where it sits and how its pixels were produced.

    The pixels themselves live in `patches.npz` next to `inpaint.json`: key `"<region_id>.pixels"` is a uint8 array
    [h, w, 3] (the region's crop of the strip with the text removed) and `"<region_id>.mask"` a bool array [h, w]
    (True where text was removed); both have exactly the size of `box`. Export replaces only the masked pixels.
    """

    region_id: str
    box: BBox  # patch rectangle in strip space
    method: InpaintMethod  # "flat" = uniform fill, "lama" = model inpainting, "none" = nothing to clean
    fill: RGB | None = None  # the flat fill colour (method "flat"), else None
    needs_lama: bool = False  # flat fill could not clean this region; a LaMa pass should redo it
    mask_px: int = 0  # number of masked pixels


class InpaintArtifact(Artifact):
    """Written by `inpaint` as inpaint.json (+ patches.npz); read by typeset (fill colour) and export (patches)."""

    items: list[InpaintItem]


# ---------------------------------------------------------------- typesetting

FontRole = Literal["dialogue", "thought", "shout", "narration", "free", "sfx"]


class LayoutItem(Model):
    region_id: str
    font_role: FontRole
    font: str
    size_px: int
    lines: list[str]
    box: BBox
    align: Literal["center", "left", "right"] = "center"
    color: RGB = (0, 0, 0)
    stroke_px: int = 0
    stroke_color: RGB = (255, 255, 255)
    overflow: bool = False
    angle: float = (
        0.0  # the lettering is rotated by this many degrees counter-clockwise around the box centre
    )


class LayoutArtifact(Artifact):
    items: list[LayoutItem]


# ---------------------------------------------------------------- export


class ExportFile(Model):
    name: str  # file name inside output_root/<Series>/<Chapter>/ ("0001.jpg", ...)
    slice_index: int
    width: int
    height: int
    bytes: int


class ExportArtifact(Artifact):
    """Written by `export` as export.json in the chapter work dir: what was written to the output folder."""

    quality: int
    subsampling: Literal["444", "422", "420"]
    files: list[ExportFile]


# ---------------------------------------------------------------- manual edits


class RegionEdit(Model):
    """One hand edit of a region (edits.json), re-applied every time the `ocr` stage rewrites ocr.json.

    `anchor` is the region's box as the pipeline produced it when it was first edited. After a re-run the
    edit applies to the region with the same id that still overlaps `anchor` (or `bbox`), else to the region
    overlapping it best — a re-detected chapter whose ids shifted keeps its edits. An `added` region (id
    `m0001`, …) is inserted as given and replaces a pipeline region covering the same box; a `deleted` one
    is dropped. A field left None keeps the pipeline's value. The pipeline's own reading stays in
    `ocr_auto.json` (and the judge's in `final_auto.json`), so dropping an edit reverts it.
    """

    region_id: str
    anchor: BBox
    added: bool = False
    deleted: bool = False
    kind: RegionKind | None = None
    bbox: BBox | None = None  # the text area; replaces the OCR lines with one line of this box
    bubble_bbox: BBox | None = None
    text: str | None = None  # the corrected source text
    lang: Lang | None = None  # an added region's language


class TranslationEdit(Model):
    """One hand-written English line (edits.json), re-applied every time the `judge` stage rewrites final.json."""

    region_id: str
    anchor: BBox  # the region's box when the line was written (matched like RegionEdit.anchor)
    text: str
    source: str  # the region's source text when the line was written; a changed source flags the line
    suggested_by: str | None = None  # the profile whose suggestion was kept as is; None = typed by hand


class ChapterEdits(Artifact):
    """edits.json in the chapter work dir: every hand edit of the chapter. Written only by the editing tools
    (web studio, CLI); the stages read it and re-apply it to what they produce, so edits survive re-runs."""

    regions: list[RegionEdit] = Field(default_factory=list)
    translations: list[TranslationEdit] = Field(default_factory=list)


# ---------------------------------------------------------------- hand cleanup

CleanupMethod = Literal["fill", "inpaint", "clone", "restore"]


class CleanupPatch(Model):
    """One hand-painted cleanup of the strip (cleanup.json), applied by export after every automatic patch.

    The mask — and, except for "restore", the pixels — live in cleanup.npz as `<id>.mask` (bool [h, w]) and
    `<id>.pixels` (uint8 [h, w, 3]), both exactly the size of `box`. "restore" puts the raw page back under
    its mask (undoing an automatic clean there); the other methods replace the masked pixels with the stored
    ones: a flat colour ("fill"), inpainting from the surroundings ("inpaint"), or the raw page `offset` away
    ("clone").
    """

    id: str  # "c0001", "c0002", … in painting order
    box: BBox  # strip space
    method: CleanupMethod
    color: RGB | None = None  # the fill colour ("fill")
    offset: tuple[int, int] | None = None  # clone source minus destination, strip px ("clone")
    mask_px: int = 0


class CleanupArtifact(Artifact):
    """cleanup.json in the chapter work dir: the hand cleanup in painting order (a later patch wins where
    patches overlap). Written only by the editing tools; tied to the strip size it was painted on."""

    strip_width: int
    strip_height: int
    patches: list[CleanupPatch] = Field(default_factory=list)


# ---------------------------------------------------------------- manifest


class StageRecord(Model):
    stage: str
    version: int
    input_hash: str
    config_hash: str
    outputs: list[str]  # file names relative to the chapter work dir
    status: Literal["done", "failed"]
    started_at: datetime
    finished_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class Manifest(Artifact):
    series: str
    chapter: str
    stages: dict[str, StageRecord] = Field(default_factory=dict)
