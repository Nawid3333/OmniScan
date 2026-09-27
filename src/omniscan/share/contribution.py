"""Contribution archives: a series' hand corrections with the pages they were made on (X4 in docs/ROADMAP.md).

A contribution is what a user shares so the models, glossaries and defaults get better. For each page that
carries a hand correction it holds the raw page (strip resolution, re-encoded, so no file name, folder path or
image metadata survives) and every region on it with the pipeline's output next to the editor's
(`core.schemas.Contribution`), plus the series' locked glossary terms. `build` collects it from the chapters'
edits.json and the pipeline's own artifacts (ocr_auto.json, final_auto.json); `write_archive` writes
contribution.json and the page images into one zip file whose entries all carry the same fixed timestamp.
A series that opted out (`[share] enabled = false` in config.toml or its series.toml) builds nothing. Nothing is
uploaded: the archive stays a local file until the upload service exists. Pages are read and encoded on the CPU
(Pillow), like the PSD export: an export of a few edited pages, not a pipeline stage.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Literal

import numpy as np
from PIL import Image

from omniscan.cleanup.strip import strip_crop
from omniscan.core.config import Config, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    ChapterEdits,
    Contribution,
    ContributionChapter,
    ContributionLettering,
    ContributionPage,
    ContributionRegion,
    ContributionTerm,
    FinalArtifact,
    IngestArtifact,
    LayoutEdit,
    Region,
    RegionEdit,
    SourceFile,
    TranslationEdit,
)
from omniscan.edits.apply import match_layout_edits, match_region_edits, match_translation_edits
from omniscan.edits.store import FINAL_AUTO_FILE, auto_regions, current_regions, load_edits
from omniscan.glossary.store import GlossaryStore
from omniscan.translate.on_demand import english_lines
from omniscan.typeset.page_preview import page_box
from omniscan.update.version import current_version

CONTRIBUTION_FILE = "contribution.json"
JPEG_QUALITY = 92
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # every entry's timestamp: when and where the export ran is not shared


class ShareOptOutError(RuntimeError):
    """The series, or this machine, opted out of sharing (`[share] enabled = false`)."""


@dataclass(frozen=True, slots=True)
class PageSource:
    """Where a contributed page's pixels come from; they are read only when the archive is written."""

    member: str
    paths: ChapterPaths
    ingest: IngestArtifact
    page: int


@dataclass(frozen=True, slots=True)
class Summary:
    """What a contribution holds, for the consent line before it is written."""

    chapters: int
    pages: int
    regions: int
    ocr_fixes: int
    kinds: int
    added: int
    deleted: int
    english: int
    lettering: int
    other: int  # region edits that change neither text nor type (a moved box, a speaker)
    terms: int


def digest(text: str) -> str:
    """A short, stable, anonymous id for a name (so the name itself is never shared)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _norm(text: str) -> str:
    """`text` with every run of whitespace collapsed to one space."""
    return " ".join(text.split())


def _page_of(ingest: IngestArtifact, box: BBox) -> SourceFile | None:
    """The page a strip-space box belongs to: the one holding its vertical centre."""
    centre = (box.y0 + box.y1) / 2
    return next((file for file in ingest.files if not file.filtered and file.y0 <= centre < file.y1), None)


def _on_page(box: BBox, page: SourceFile, width: int) -> BBox:
    """A strip-space box in the pixels of `page` (clipped to the page)."""
    height = page.y1 - page.y0
    return BBox(
        x0=min(max(box.x0, 0), width),
        y0=min(max(box.y0 - page.y0, 0), height),
        x1=min(max(box.x1, 0), width),
        y1=min(max(box.y1 - page.y0, 0), height),
    )


def _lettering(edit: LayoutEdit, page: SourceFile, width: int) -> ContributionLettering:
    """A hand-set lettering in page pixels, its font reduced to a file name."""
    return ContributionLettering(
        font=PureWindowsPath(edit.font).name if edit.font else None,  # either separator: never a folder path
        size_px=edit.size_px,
        color=edit.color,
        stroke_px=edit.stroke_px,
        stroke_color=edit.stroke_color,
        align=edit.align,
        angle=edit.angle,
        box=_on_page(edit.box, page, width) if edit.box is not None else None,
        lines=edit.lines,
        hidden=edit.hidden,
    )


def _judged(paths: ChapterPaths) -> dict[str, str]:
    """The judge's own English lines (final_auto.json) by region id; {} when it has none."""
    path = paths.artifact(FINAL_AUTO_FILE)
    return {line.region_id: line.text for line in FinalArtifact.load(path).lines} if path.is_file() else {}


@dataclass(frozen=True, slots=True)
class _Edits:
    """A chapter's hand edits keyed by the id of the region each belongs to."""

    regions: dict[str, RegionEdit]
    deleted: list[tuple[Region, RegionEdit]]
    translations: dict[str, TranslationEdit]
    layout: dict[str, LayoutEdit]

    @classmethod
    def of(cls, edits: ChapterEdits, auto: Sequence[Region], current: Sequence[Region]) -> _Edits:
        """Match every edit to its region the way the stages re-apply them."""
        auto_by_id = {region.id: region for region in auto}
        claims = match_region_edits(auto, edits)
        regions = {region_id: edits.regions[i] for i, region_id in claims.items()}
        regions |= {edit.region_id: edit for edit in edits.regions if edit.added and not edit.deleted}
        deleted = [
            (auto_by_id[region_id], edits.regions[i])
            for i, region_id in claims.items()
            if edits.regions[i].deleted
        ]
        return cls(
            regions=regions,
            deleted=deleted,
            translations={
                region_id: edits.translations[i]
                for i, region_id in match_translation_edits(current, edits).items()
            },
            layout={
                region_id: edits.layout[i] for i, region_id in match_layout_edits(current, edits).items()
            },
        )


def _ocr_text(region: Region, auto: Region | None, edit: RegionEdit | None) -> str | None:
    """The pipeline's reading of a region (when first edited, if it was); None for a region added by hand."""
    if edit is not None and edit.added:
        return None
    if edit is not None and edit.auto_text is not None:
        return edit.auto_text
    return auto.text if auto is not None else region.text


def _english_from(
    line: str, edit: TranslationEdit | None
) -> Literal["machine", "suggestion", "typed"] | None:
    """Who wrote a region's English line: the judge, the editor keeping a suggestion, or the editor typing it."""
    if not line:
        return None
    if edit is None:
        return "machine"
    return "typed" if edit.suggested_by is None else "suggestion"


def _region(
    region: Region,
    page: SourceFile,
    width: int,
    *,
    auto: Region | None,
    edits: _Edits,
    judged: dict[str, str],
    english: dict[str, str],
) -> ContributionRegion:
    """One current region of the chapter, the pipeline's output next to the hand edits."""
    edit = edits.regions.get(region.id)
    added = edit is not None and edit.added
    line_edit = edits.translations.get(region.id)
    layout = edits.layout.get(region.id)
    line = english.get(region.id, "")
    return ContributionRegion(
        id=region.id,
        box=_on_page(region.bbox, page, width),
        bubble_box=_on_page(region.bubble_bbox, page, width) if region.bubble_bbox is not None else None,
        reading_order=region.reading_order,
        lang=region.lang,
        kind=region.kind,
        auto_kind=None if added else (auto.kind if auto is not None else region.kind),
        ocr_text=_ocr_text(region, auto, edit),
        text=region.text,
        added=added,
        edited=edit is not None,
        machine_english=line_edit.auto_text
        if line_edit is not None and line_edit.auto_text is not None
        else judged.get(region.id),
        english=line or None,
        english_from=_english_from(line, line_edit),
        speaker=region.speaker,
        lettering=_lettering(layout, page, width) if layout is not None else None,
    )


def _corrected(region: ContributionRegion) -> bool:
    """Whether a region carries a hand correction (the reason its page is contributed)."""
    return region.edited or region.english_from in ("suggestion", "typed") or region.lettering is not None


def build_chapter(paths: ChapterPaths, order: int) -> tuple[ContributionChapter | None, list[PageSource]]:
    """One chapter's contributed pages (those with a hand correction) and where their pixels come from; None
    when the chapter has no hand edit or no ingest.json to place its regions on pages."""
    chapter_edits = load_edits(paths)
    ingest_path = paths.artifact("ingest.json")
    if (
        not (chapter_edits.regions or chapter_edits.translations or chapter_edits.layout)
        or not ingest_path.is_file()
    ):
        return None, []
    ingest = IngestArtifact.load(ingest_path)
    width = ingest.strip_width
    auto = auto_regions(paths)
    auto_by_id = {region.id: region for region in auto}
    current = current_regions(paths)
    edits = _Edits.of(chapter_edits, auto, current)
    judged, english = _judged(paths), english_lines(paths)
    by_page: dict[int, list[ContributionRegion]] = {}
    files: dict[int, SourceFile] = {}
    for region in current:
        page = _page_of(ingest, region.bbox)
        if page is None:
            continue
        files[page.index] = page
        by_page.setdefault(page.index, []).append(
            _region(
                region,
                page,
                width,
                auto=auto_by_id.get(region.id),
                edits=edits,
                judged=judged,
                english=english,
            )
        )
    for region, edit in edits.deleted:
        page = _page_of(ingest, region.bbox)
        if page is None:
            continue
        files[page.index] = page
        by_page.setdefault(page.index, []).append(
            ContributionRegion(
                id=region.id,
                box=_on_page(region.bbox, page, width),
                bubble_box=_on_page(region.bubble_bbox, page, width)
                if region.bubble_bbox is not None
                else None,
                reading_order=region.reading_order,
                lang=region.lang,
                kind=region.kind,
                auto_kind=region.kind,
                ocr_text=edit.auto_text if edit.auto_text is not None else region.text,
                text=region.text,
                deleted=True,
                edited=True,
                machine_english=judged.get(region.id),
            )
        )
    chapter_id = digest(f"{paths.series}\0{paths.chapter}")
    pages: list[ContributionPage] = []
    sources: list[PageSource] = []
    for index in sorted(by_page):
        regions = by_page[index]
        if not any(_corrected(region) for region in regions):
            continue
        member = f"pages/{chapter_id}/{index:04d}.jpg"
        file = files[index]
        pages.append(
            ContributionPage(
                index=index, image=member, width=width, height=file.y1 - file.y0, regions=regions
            )
        )
        sources.append(PageSource(member=member, paths=paths, ingest=ingest, page=index))
    if not pages:
        return None, []
    return ContributionChapter(id=chapter_id, order=order, pages=pages), sources


def locked_terms(series: SeriesPaths) -> list[ContributionTerm]:
    """The series' locked glossary terms; none when it has no glossary (the database is never created)."""
    if not series.db.is_file():
        return []
    with GlossaryStore(series.db) as db:
        return [
            ContributionTerm(
                source=entry.source, target=entry.target, type=entry.type, aliases=list(entry.aliases)
            )
            for entry in db.list()
            if entry.status == "locked"
        ]


def build(
    series: SeriesPaths, cfg: Config, chapters: Sequence[str] | None = None
) -> tuple[Contribution, list[PageSource]]:
    """The contribution of a series' chapters (all by default) and where its pages' pixels come from.

    ShareOptOutError when the series or this machine opted out; SeriesConfigError for a broken series.toml;
    KeyError for a chapter the series does not have."""
    if not series_config(cfg, series.library_dir).share.enabled:
        raise ShareOptOutError(f"{series.series} is opted out of sharing ([share] enabled = false)")
    order = {name: i for i, name in enumerate(series.chapters())}
    contributed: list[ContributionChapter] = []
    sources: list[PageSource] = []
    for name in chapters if chapters is not None else list(order):
        chapter, pages = build_chapter(series.chapter(name), order[name])
        if chapter is not None:
            contributed.append(chapter)
            sources += pages
    contribution = Contribution(
        app_version=str(current_version()),
        created=datetime.now(UTC).date(),
        series_id=digest(series.series),
        chapters=contributed,
        glossary=locked_terms(series),
    )
    return contribution, sources


def summarize(contribution: Contribution) -> Summary:
    """What `contribution` holds."""
    regions = [
        region for chapter in contribution.chapters for page in chapter.pages for region in page.regions
    ]
    live = [region for region in regions if not region.added and not region.deleted]
    return Summary(
        chapters=len(contribution.chapters),
        pages=sum(len(chapter.pages) for chapter in contribution.chapters),
        regions=len(regions),
        ocr_fixes=sum(1 for r in live if r.ocr_text is not None and _norm(r.ocr_text) != _norm(r.text)),
        kinds=sum(1 for r in live if r.auto_kind is not None and r.auto_kind != r.kind),
        added=sum(1 for r in regions if r.added),
        deleted=sum(1 for r in regions if r.deleted),
        english=sum(1 for r in regions if r.english_from in ("suggestion", "typed")),
        lettering=sum(1 for r in regions if r.lettering is not None),
        other=sum(
            1 for r in live if r.edited and _norm(r.ocr_text or "") == _norm(r.text) and r.auto_kind == r.kind
        ),
        terms=len(contribution.glossary),
    )


def page_jpeg(pixels: np.ndarray) -> bytes:
    """A page (uint8 [h, w, 3]) as a JPEG without any metadata (full-resolution chroma, for the lettering)."""
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    return buffer.getvalue()


def _entry(archive: zipfile.ZipFile, name: str, data: bytes, compression: int) -> None:
    """Write one archive entry with the fixed timestamp and the same system field on every OS."""
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = compression
    info.create_system = 0
    archive.writestr(info, data)


def write_archive(path: Path, contribution: Contribution, pages: Sequence[PageSource]) -> int:
    """Write the contribution archive (contribution.json, then each page image) atomically; returns its size
    in bytes. FileNotFoundError when a page's raw file is gone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w") as archive:
            _entry(
                archive,
                CONTRIBUTION_FILE,
                contribution.model_dump_json(indent=2).encode(),
                zipfile.ZIP_DEFLATED,
            )
            for page in pages:
                pixels = strip_crop(page.paths, page.ingest, page_box(page.ingest, page.page))
                _entry(archive, page.member, page_jpeg(pixels), zipfile.ZIP_STORED)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path.stat().st_size


def read_archive(path: Path) -> Contribution:
    """The contribution.json of an archive (for checking one before it is sent)."""
    with zipfile.ZipFile(path) as archive:
        return Contribution.model_validate_json(archive.read(CONTRIBUTION_FILE))
