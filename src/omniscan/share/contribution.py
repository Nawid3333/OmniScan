"""Contribution archives: a series' hand corrections with the pages they were made on (X4 in docs/ROADMAP.md).

A contribution is what a user shares so the models, glossaries and defaults get better. For each page that
carries a hand correction it holds the raw page (strip resolution, re-encoded, so no file name, folder path or
image metadata survives) and every region on it with the pipeline's output next to the editor's
(`core.schemas.Contribution`), plus the series' locked glossary terms. `build` collects it from the chapters'
edits.json and the pipeline's own artifacts (ocr_auto.json, final_auto.json); `write_archive` writes
contribution.json and the page images into one zip file whose entries all carry the same fixed timestamp, and
contribution.json holds no date. Series and chapters are named by ids salted with a random value kept in this
install's work folder, so hashing known titles cannot reverse them, while a later contribution of the same
series from the same install carries the same ids. A series is built only when both this machine and the series
allow sharing (`[share] enabled`, false in config.toml opts out every series whatever its series.toml says).
Nothing is uploaded: the archive stays a local file until the upload service exists. Pages are read and encoded
on the CPU (Pillow), like the PSD export: an export of a few edited pages, not a pipeline stage.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import secrets
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
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
    IngestArtifact,
    LayoutEdit,
    Region,
    RegionEdit,
    SourceFile,
    TranslationEdit,
)
from omniscan.edits.apply import MATCH_IOU, match_layout_edits, match_region_edits, match_translation_edits
from omniscan.edits.store import auto_regions, current_regions, judged_lines, load_edits
from omniscan.glossary.store import GlossaryStore
from omniscan.learn.harvest import norm
from omniscan.translate.on_demand import english_lines
from omniscan.typeset.page_preview import page_box
from omniscan.update.version import current_version

CONTRIBUTION_FILE = "contribution.json"
SALT_FILE = "contribution-salt"  # in the work root: this install's random salt for the ids (never shared)
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
    redrawn: int = 0  # detected boxes replaced by a box drawn over them by hand


def install_salt(work_root: Path) -> bytes:
    """This install's random salt for anonymous ids, created on first use in `work_root` (never shared)."""
    path = work_root / SALT_FILE
    for _ in range(2):
        try:
            salt = bytes.fromhex(path.read_text(encoding="ascii"))
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("x", encoding="ascii") as out:  # never replace a salt another export made
                    out.write(secrets.token_hex(32))
            except FileExistsError:
                pass
            continue
        except ValueError:
            salt = b""
        if len(salt) >= 16:
            return salt
        raise ValueError(f"{path} is damaged: delete it (the next contribution gets new ids)")
    raise FileNotFoundError(f"could not create {path}")


def digest(text: str, salt: bytes) -> str:
    """A short, stable, anonymous id for a name: an HMAC with this install's salt, so the name cannot be found
    by hashing known titles, and the name itself is never shared."""
    return hmac.new(salt, text.encode(), hashlib.sha256).hexdigest()[:16]


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


@dataclass(frozen=True, slots=True)
class _Edits:
    """A chapter's hand edits keyed by the id of the region each belongs to."""

    regions: dict[str, RegionEdit]
    # pipeline regions deleted by hand (None) or replaced by a box drawn over them (that box's id)
    removed: list[tuple[Region, str | None]]
    translations: dict[str, TranslationEdit]
    layout: dict[str, LayoutEdit]

    @classmethod
    def of(cls, edits: ChapterEdits, auto: Sequence[Region], current: Sequence[Region]) -> _Edits:
        """Match every edit to its region the way the stages re-apply them."""
        auto_by_id = {region.id: region for region in auto}
        claims = match_region_edits(auto, edits)
        regions = {region_id: edits.regions[i] for i, region_id in claims.items()}
        drawn = {edit.region_id for edit in edits.regions if edit.added and not edit.deleted}
        regions |= {edit.region_id: edit for edit in edits.regions if edit.added and not edit.deleted}
        removed: list[tuple[Region, str | None]] = [
            (auto_by_id[region_id], None) for i, region_id in claims.items() if edits.regions[i].deleted
        ]
        claimed = set(claims.values())
        for region in auto:  # a drawn box over a detection no edit claims replaces it (apply_region_edits)
            if region.id in claimed:
                continue
            replaced_by = next(
                (new.id for new in current if new.id in drawn and region.bbox.iou(new.bbox) >= MATCH_IOU),
                None,
            )
            if replaced_by is not None:
                removed.append((region, replaced_by))
        return cls(
            regions=regions,
            removed=removed,
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
    """Who wrote a region's English line: the judge, the editor keeping a suggestion, or the editor typing it
    (also a line cleared by hand); None when the region has no line and no hand-written one."""
    if edit is not None:
        return "typed" if edit.suggested_by is None else "suggestion"
    return "machine" if line else None


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


def build_chapter(
    paths: ChapterPaths, order: int, salt: bytes
) -> tuple[ContributionChapter | None, list[PageSource]]:
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
    judged, english = judged_lines(paths), english_lines(paths)
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
    for region, replaced_by in edits.removed:
        page = _page_of(ingest, region.bbox)
        if page is None:
            continue
        files[page.index] = page
        kept = _region(region, page, width, auto=region, edits=edits, judged=judged, english={})
        by_page.setdefault(page.index, []).append(
            kept.model_copy(update={"deleted": True, "edited": True, "replaced_by": replaced_by})
        )
    chapter_id = digest(f"{paths.series}\0{paths.chapter}", salt)
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

    ShareOptOutError when this machine or the series opted out (the machine's choice covers every series);
    SeriesConfigError for a broken series.toml; KeyError for a chapter the series does not have."""
    if not cfg.share.enabled:
        raise ShareOptOutError(
            "this machine is opted out of sharing ([share] enabled = false in config.toml)"
        )
    if not series_config(cfg, series.library_dir).share.enabled:
        raise ShareOptOutError(f"{series.series} is opted out of sharing ([share] enabled = false)")
    salt = install_salt(cfg.paths.work_root)
    order = {name: i for i, name in enumerate(series.chapters())}
    contributed: list[ContributionChapter] = []
    sources: list[PageSource] = []
    for name in dict.fromkeys(chapters) if chapters is not None else order:
        chapter, pages = build_chapter(series.chapter(name), order[name], salt)
        if chapter is not None:
            contributed.append(chapter)
            sources += pages
    contribution = Contribution(
        app_version=str(current_version()),
        series_id=digest(series.series, salt),
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
    redrawn = sum(1 for r in regions if r.replaced_by is not None)
    return Summary(
        chapters=len(contribution.chapters),
        pages=sum(len(chapter.pages) for chapter in contribution.chapters),
        regions=len(regions),
        ocr_fixes=sum(1 for r in live if r.ocr_text is not None and norm(r.ocr_text) != norm(r.text)),
        kinds=sum(1 for r in live if r.auto_kind is not None and r.auto_kind != r.kind),
        added=sum(1 for r in regions if r.added),
        deleted=sum(1 for r in regions if r.deleted) - redrawn,
        english=sum(1 for r in regions if r.english_from in ("suggestion", "typed")),
        lettering=sum(1 for r in regions if r.lettering is not None),
        other=sum(
            1 for r in live if r.edited and norm(r.ocr_text or "") == norm(r.text) and r.auto_kind == r.kind
        ),
        terms=len(contribution.glossary),
        redrawn=redrawn,
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
