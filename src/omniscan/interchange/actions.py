"""Interchange actions shared by the CLI (`omniscan labelplus|psd|ballons|mit …`) and the desktop Library page.

An export writes a chapter out for another tool and returns what it wrote; an import takes that tool's work back
in as hand edits (edits.json: every re-run keeps them, learning remembers them, one import is one undo step) and
returns what changed and what matched nothing. A refusal (no ingest.json yet, a broken series.toml, an unknown
page, a file the tool did not write) is an InterchangeError with the reason, for the caller to print or show.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from omniscan.cleanup.store import CLEANUP_FILE, current_crop
from omniscan.core.config import Config, SeriesConfigError, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths, jpeg_paths
from omniscan.core.schemas import BBox, IngestArtifact, LayoutArtifact, LayoutItem
from omniscan.edits import store
from omniscan.interchange import ballons, mit
from omniscan.interchange.blocks import Block, join_lines, match_blocks
from omniscan.interchange.labelplus import Label, export_labels, match_labels, parse, strip_pages, write
from omniscan.interchange.psd import page_psd
from omniscan.translate.on_demand import english_lines
from omniscan.typeset.page_preview import page_box

ExportText = Literal["english", "source"]


class InterchangeError(ValueError):
    """An export or import that cannot be done, with the reason."""


@dataclass(frozen=True, slots=True)
class LabelExport:
    """A written LabelPlus file."""

    path: Path
    labels: int
    pages: int


@dataclass(frozen=True, slots=True)
class LabelImport:
    """What a LabelPlus import changed (or, as a dry run, would change)."""

    changed: int  # English lines written
    same: int  # lines the file already agreed with
    joined: int  # labels added to a region another label already pointed into
    unmatched: list[tuple[str, Label]] = field(default_factory=list)  # (page, label) outside every region
    unknown_pages: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PsdExport:
    """Written layered PSD pages."""

    folder: Path
    pages: int
    has_text: bool  # False before typeset: the text layers are empty


@dataclass(frozen=True, slots=True)
class BallonsExport:
    """A written BallonsTranslator project."""

    folder: Path
    blocks: int
    pages: int
    cleaned: bool  # False before any cleaning: no inpainted pages


@dataclass(frozen=True, slots=True)
class BlockImport:
    """What a BallonsTranslator or manga-image-translator import changed (or, as a dry run, would change)."""

    lines: int  # English lines written
    sources: int  # source texts written (with `source`)
    same: int  # lines the project already agreed with
    joined: int  # blocks joined into a region another block is in
    unmatched: list[tuple[Block, BBox, str]] = field(
        default_factory=list
    )  # block, strip box, what became of it
    unknown_pages: list[str] = field(default_factory=list)


def ingest_of(paths: ChapterPaths) -> IngestArtifact:
    """The chapter's ingest.json (the page geometry); InterchangeError before the ingest stage ran."""
    path = paths.artifact("ingest.json")
    if not path.is_file():
        raise InterchangeError("ingest.json not found — run the ingest stage first")
    return IngestArtifact.load(path)


def _layout(paths: ChapterPaths) -> list[LayoutItem]:
    """The chapter's lettering (layout.json), none before typeset."""
    path = paths.artifact("layout.json")
    return LayoutArtifact.load(path).items if path.is_file() else []


def _direction(cfg: Config, series: SeriesPaths) -> tuple[Config, Literal["ltr", "rtl"]]:
    """The series' config and reading direction; InterchangeError for a broken series.toml."""
    try:
        scfg = series_config(cfg, series.library_dir)
    except SeriesConfigError as exc:
        raise InterchangeError(str(exc)) from exc
    return scfg, scfg.detect.reading_direction


def labelplus_path(series: SeriesPaths, chapter: str) -> Path:
    """Where a chapter's LabelPlus file goes by default: <output_root>/<series>/_labelplus/<chapter>.txt."""
    return series.output_dir / "_labelplus" / f"{chapter}.txt"


def psd_folder(series: SeriesPaths, chapter: str) -> Path:
    """Where a chapter's PSD pages go by default: <output_root>/<series>/_psd/<chapter>/."""
    return series.output_dir / "_psd" / chapter


def ballons_folder(series: SeriesPaths, chapter: str) -> Path:
    """Where a chapter's BallonsTranslator project goes by default: <output_root>/<series>/_ballons/<chapter>/."""
    return series.output_dir / "_ballons" / chapter


def export_labelplus(
    series: SeriesPaths, chapter: str, *, text: ExportText = "english", out: Path | None = None
) -> LabelExport:
    """Write the chapter's regions as a LabelPlus file: one label per region, on the raw page it sits on, with its
    English line (or with `text="source"` its source text)."""
    paths = series.chapter(chapter)
    ingest = ingest_of(paths)
    regions = store.current_regions(paths)
    texts = english_lines(paths) if text == "english" else {r.id: r.text for r in regions}
    doc = export_labels(ingest, regions, texts, comment=f"OmniScan: {series.series} / {chapter} ({text})")
    target = out or labelplus_path(series, chapter)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(write(doc), encoding="utf-8-sig")  # LabelPlus itself writes a BOM
    return LabelExport(
        path=target, labels=sum(len(labels) for labels in doc.pages.values()), pages=len(doc.pages)
    )


def import_labelplus(
    cfg: Config, series: SeriesPaths, chapter: str, file: Path, *, dry_run: bool = False
) -> LabelImport:
    """Take a LabelPlus file's texts as the English lines of the regions its labels point into."""
    paths = series.chapter(chapter)
    _scfg, direction = _direction(cfg, series)
    try:
        doc = parse(file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise InterchangeError(str(exc)) from exc
    found = match_labels(doc, ingest_of(paths), store.current_regions(paths))
    current = english_lines(paths)
    changed = {rid: text for rid, text in found.texts.items() if current.get(rid) != text}
    if not dry_run and changed:
        try:  # one write, one rebuild, one undo step: the file lands whole or not at all
            store.set_translations(paths, changed, direction=direction)
        except store.EditNotFoundError as exc:
            raise InterchangeError(str(exc)) from exc
    return LabelImport(
        changed=len(changed),
        same=len(found.texts) - len(changed),
        joined=found.joined,
        unmatched=list(found.unmatched),
        unknown_pages=list(found.unknown_pages),
    )


def export_psd(
    series: SeriesPaths, chapter: str, *, pages: Sequence[int] | None = None, out: Path | None = None
) -> PsdExport:
    """Write one layered PSD per page: raw, clean (cleaning and hand cleanup) and text (the lettering, from
    layout.json), with the finished page as the composite. `pages` (ingest.json indices) limits the export."""
    paths = series.chapter(chapter)
    ingest = ingest_of(paths)
    items = _layout(paths)
    chosen = [f for f in ingest.files if not f.filtered and f.y1 > f.y0]
    if pages:
        unknown = sorted(set(pages) - {f.index for f in chosen})
        if unknown:
            raise InterchangeError(f"no page {unknown[0]} in this chapter")
        chosen = [f for f in chosen if f.index in set(pages)]
    folder = out or psd_folder(series, chapter)
    folder.mkdir(parents=True, exist_ok=True)
    for source in chosen:
        (folder / f"{Path(source.name).stem}.psd").write_bytes(page_psd(paths, ingest, source.index, items))
    return PsdExport(folder=folder, pages=len(chosen), has_text=bool(items))


def _cleaned(paths: ChapterPaths) -> bool:
    """Whether the chapter was cleaned, automatically or by hand."""
    return any(paths.artifact(name).is_file() for name in ("inpaint.json", "inpaint_lama.json", CLEANUP_FILE))


def export_ballons(series: SeriesPaths, chapter: str, *, out: Path | None = None) -> BallonsExport:
    """Write the chapter as a BallonsTranslator project: copies of its pages, the cleaned pages (inpainted/) and
    one text block per region with its source text, English line and lettering size and colours."""
    paths = series.chapter(chapter)
    ingest = ingest_of(paths)
    items = _layout(paths)
    folder = out or ballons_folder(series, chapter)
    folder.mkdir(parents=True, exist_ok=True)
    jpegs = dict(
        zip(
            (f.index for f in ingest.files),
            jpeg_paths(ingest, paths.raw_dir, paths.work_dir / "converted"),
            strict=True,
        )
    )
    pages = strip_pages(ingest)
    cleaned = _cleaned(paths)
    for page in pages:
        name = ballons.page_name(page)
        shutil.copyfile(paths.raw_dir / page.name if name == page.name else jpegs[page.index], folder / name)
        if cleaned:
            pixels = Image.fromarray(current_crop(paths, ingest, page_box(ingest, page.index)))
            if pixels.size != (page.width, page.height):
                pixels = pixels.resize((page.width, page.height), Image.Resampling.LANCZOS)
            (folder / ballons.INPAINTED_DIR).mkdir(exist_ok=True)
            pixels.save(folder / ballons.INPAINTED_DIR / f"{Path(name).stem}.png")
    blocks = ballons.page_blocks(ingest, store.current_regions(paths), english_lines(paths), items)
    project = ballons.write_project(folder.resolve(), ingest, blocks)
    ballons.project_file(folder).write_text(
        json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return BallonsExport(
        folder=folder, blocks=sum(len(b) for b in blocks.values()), pages=len(pages), cleaned=cleaned
    )


def read_ballons(project: Path) -> list[Block]:
    """The text blocks of a BallonsTranslator project: its imgtrans_*.json, or the page folder that holds it."""
    file = ballons.project_file(project) if project.is_dir() else project
    try:
        return ballons.read_project(json.loads(file.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise InterchangeError(f"{file}: {exc}") from exc


def read_mit(files: Sequence[Path]) -> list[Block]:
    """The text regions of manga-image-translator's --save-text files (one *_translations.txt per page)."""
    blocks: list[Block] = []
    for file in files:
        try:
            blocks += mit.parse(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InterchangeError(f"{file}: {exc}") from exc
    return blocks


def import_blocks(
    cfg: Config,
    series: SeriesPaths,
    chapter: str,
    blocks: Sequence[Block],
    *,
    source: bool = False,
    add: bool = False,
    dry_run: bool = False,
) -> BlockImport:
    """Take a foreign project's blocks into the chapter: the translation of the blocks in each region as its
    English line (and with `source` their text as its source text); with `add`, a new region for each block over
    none. The whole import is one undo step."""
    paths = series.chapter(chapter)
    scfg, direction = _direction(cfg, series)
    regions = store.current_regions(paths)
    found = match_blocks(blocks, ingest_of(paths), regions)
    english, texts = english_lines(paths), {r.id: r.text for r in regions}
    lines = {
        rid: " ".join(join_lines([b.translation for b in bs]).split()) for rid, bs in found.matched.items()
    }
    sources = {rid: join_lines([b.text for b in bs]) for rid, bs in found.matched.items()} if source else {}
    new_lines = {rid: line for rid, line in lines.items() if line and english.get(rid) != line}
    new_sources = {rid: text for rid, text in sources.items() if text and texts.get(rid) != text}
    unmatched: list[tuple[Block, BBox, str]] = []
    try:
        with store.edit_group(paths):  # the whole import is one undo step
            if not dry_run:
                for region_id, text in new_sources.items():
                    store.update_region(paths, region_id, direction=direction, text=text)
                for region_id, line in new_lines.items():
                    store.set_translation(paths, region_id, line, direction=direction)
            for block, box in found.unmatched:
                if not add:
                    note = ""
                elif dry_run:
                    note = " — would add a region"
                else:
                    region = store.add_region(
                        paths, box, direction=direction, text=block.text, lang=scfg.ocr.lang
                    )
                    if block.translation.strip():
                        line = " ".join(block.translation.split())
                        store.set_translation(paths, region.id, line, direction=direction)
                    note = f" — added as {region.id}"
                unmatched.append((block, box, note))
    except (store.EditNotFoundError, ValueError) as exc:
        raise InterchangeError(str(exc)) from exc
    return BlockImport(
        lines=len(new_lines),
        sources=len(new_sources),
        same=sum(1 for rid, line in lines.items() if line and english.get(rid) == line),
        joined=sum(len(bs) - 1 for bs in found.matched.values()),
        unmatched=unmatched,
        unknown_pages=list(found.unknown_pages),
    )
