"""Qt-free services behind the desktop Studio: translate, re-read or find regions now, render a page preview,
re-read the finished pages, clean a brush stroke, read the output cuts.

Each call is blocking (it waits on a model) and runs on a worker thread in the GUI; the view injects fakes
in tests. Nothing here writes to the chapter's edits: suggestions and readings go into the Studio's session, where
Save records them like any hand edit (the finished-page re-read writes only its own qa.json, and a cleaned
stroke is a hand-cleanup patch in cleanup.json, as the web Studio's brush makes it).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image, ImageDraw

from omniscan.cleanup import store as cleanup_store
from omniscan.core.config import Config, SeriesConfigError, get_secrets, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths, natural_key
from omniscan.core.schemas import BBox, CleanupPatch, IngestArtifact, QaIssue
from omniscan.edits import store as edit_store
from omniscan.export.segments import cut_crossings
from omniscan.qa.leftover import load_issues
from omniscan.translate.on_demand import translate_now
from omniscan.translate.profiles import default_profile_paths, load_profiles
from omniscan.translate.voices import load_voices
from omniscan.typeset.chapter import chapter_layout
from omniscan.typeset.fonts import fonts_dir
from omniscan.typeset.page_preview import render_page

if TYPE_CHECKING:
    from omniscan.detect.on_demand import Found

PREVIEW_SCHEME = "preview"  # the pseudo paths of rendered preview tiles: preview/<page index>
SNAP_ROWS = 60  # a new or moved output cut snaps into a uniform band this close (as in the web Studio)


@dataclass(frozen=True, slots=True)
class Suggested:
    """One region's suggested English line and the profile that wrote it."""

    region_id: str
    text: str
    profile: str


@dataclass(frozen=True, slots=True)
class PagePreview:
    """One raw page as the release will look, at strip resolution."""

    page: int  # the SourceFile index
    y0: int  # its strip rows
    y1: int
    pixels: np.ndarray  # uint8 [h, w, 3]


def _series_cfg(cfg: Config, paths: ChapterPaths) -> Config:
    """The config with the chapter's series.toml applied."""
    try:
        return series_config(cfg, paths.raw_dir.parent)
    except SeriesConfigError as exc:
        raise ValueError(str(exc)) from exc


def _series_paths(paths: ChapterPaths) -> SeriesPaths:
    """The series folders a chapter belongs to."""
    return SeriesPaths(
        series=paths.series,
        library_dir=paths.raw_dir.parent,
        work_dir=paths.work_dir.parent,
        output_dir=paths.output_dir.parent,
    )


def speaker_names(paths: ChapterPaths) -> list[str]:
    """The series' characters (voices.toml), the names the Speaker column offers; [] without the file.
    ValueError naming what is wrong in a broken voices.toml."""
    return [character.name for character in load_voices(_series_paths(paths))]


def profile_names() -> list[str]:
    """Every known translation profile, enabled ones first."""
    known = load_profiles(default_profile_paths())
    return sorted(known, key=lambda name: (not known[name].enabled, name))


def translate_regions(
    cfg: Config, paths: ChapterPaths, region_ids: list[str], profile: str | None = None
) -> list[Suggested]:
    """Ask the translation model (every enabled profile, or `profile`) for `region_ids` now, with the chapter's
    neighbouring lines, glossary, story and learned memory as context; the first suggestion per region."""
    from omniscan.llm.ollama import OllamaClient  # the LLM stack loads only when a translation is asked

    scfg = _series_cfg(cfg, paths)
    result = translate_now(
        OllamaClient(scfg.ollama, get_secrets()),
        scfg,
        _series_paths(paths),
        paths,
        region_ids,
        load_profiles(default_profile_paths()),
        profile=profile,
    )
    first: dict[str, Suggested] = {}
    for item in result.suggestions:
        first.setdefault(item.region_id, Suggested(item.region_id, item.text, item.profile))
    return [first[region_id] for region_id in region_ids if region_id in first]


def read_region(cfg: Config, paths: ChapterPaths, region_id: str) -> str:
    """Read one region again with the series' OCR engine (the models load for this read); the text."""
    from omniscan.ocr.on_demand import read_region_now  # torch loads only when a region is read

    return read_region_now(paths, region_id, _series_cfg(cfg, paths)).text


def read_finished_pages(cfg: Config, paths: ChapterPaths) -> list[QaIssue]:
    """Re-read the chapter's finished pages with the OCR — the qa stage, as `omniscan qa` runs it (the models load
    for it, with exclusive GPU access; pages unchanged since the last read are not read again) — and return what
    still shows source text or a watermark (qa.json). RuntimeError with the stage's error when it fails."""
    from omniscan.core.stage import run_series  # torch loads only when the pages are read
    from omniscan.gpu.groups import build_vram_manager
    from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock
    from omniscan.qa.stage import QaStage

    scfg = _series_cfg(cfg, paths)
    lock = acquire_gpu_lock() if scfg.gpu.device != "cpu" else None
    try:
        manager = build_vram_manager(scfg)
        try:
            outcomes = run_series([QaStage()], scfg, paths.series, [paths.chapter], gpu=manager)
        finally:
            manager.release()
    finally:
        if lock is not None:
            release_gpu_lock(lock)
    failed = next(
        (outcome for outcome in outcomes.get(paths.chapter, []) if outcome.status == "failed"), None
    )
    if failed is not None:
        raise RuntimeError(failed.error or "the qa stage failed")
    return load_issues(paths)


def finished_page_issues(paths: ChapterPaths) -> list[QaIssue]:
    """What the last re-read of the finished pages found (qa.json); none before one ran or for a damaged file."""
    try:
        return load_issues(paths)
    except OSError, ValueError:
        return []


def find_missed(cfg: Config, paths: ChapterPaths, page: int, threshold: float | None = None) -> list[Found]:
    """Run the detector again on raw page `page` (SourceFile index; the models load for this search): the text
    areas no region covers, read by the series' OCR engine. Suggestions only, nothing is written; `threshold`
    replaces the series' detector score threshold for this search."""
    from omniscan.detect.on_demand import find_on_page_now  # torch loads only when a page is searched

    return find_on_page_now(paths, page, _series_cfg(cfg, paths), threshold)


def page_rows(ingest: IngestArtifact) -> list[tuple[int, int, int]]:
    """(page index, y0, y1) of every raw page of the strip, in strip order."""
    return [(file.index, file.y0, file.y1) for file in sorted(ingest.files, key=lambda f: f.y0)]


def load_ingest(paths: ChapterPaths) -> IngestArtifact | None:
    """The chapter's ingest.json, None before the ingest stage ran."""
    try:
        return cleanup_store.load_ingest(paths)
    except LookupError, OSError, ValueError:
        return None


def render_preview(cfg: Config, paths: ChapterPaths, page: int) -> PagePreview:
    """Page `page` (SourceFile index) as the release will look: automatic cleaning, hand cleanup and the
    lettering with every saved edit, rendered on the CPU (typeset/page_preview.py)."""
    ingest = cleanup_store.load_ingest(paths)
    try:
        items = chapter_layout(paths, _series_cfg(cfg, paths)).items
    except FileNotFoundError:
        items = []  # not typeset yet: the cleaning alone
    source = next(file for file in ingest.files if file.index == page)
    return PagePreview(page=page, y0=source.y0, y1=source.y1, pixels=render_page(paths, ingest, page, items))


def preview_path(page: int) -> Path:
    """The pseudo path a preview tile is keyed by in a StripView (`provide_image`)."""
    return Path(PREVIEW_SCHEME) / str(page)


def font_names() -> list[str]:
    """The lettering fonts in the fonts folder (file names, as a layout edit names them)."""
    folder = fonts_dir()
    if not folder.is_dir():
        return []
    return sorted((p.name for p in folder.iterdir() if p.suffix.lower() in (".ttf", ".otf")), key=natural_key)


CleanMethod = Literal["fill", "inpaint", "lama", "restore"]


@dataclass(frozen=True, slots=True)
class PageStroke:
    """A brush stroke on one page, in that page's own pixels (what cleanup.store.add_patch takes)."""

    page: int  # the SourceFile index
    box: BBox
    mask: np.ndarray  # bool [box.height, box.width]


def stroke_mask(
    points: Sequence[tuple[float, float]], radius: int, strip_width: int, strip_height: int
) -> tuple[BBox, np.ndarray] | None:
    """A round brush of `radius` strip px dragged through `points` (strip space) as a strip box and a bool mask of
    that box, cut to the strip; None when nothing of it lies on the strip."""
    if not points or radius <= 0:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y0 = max(0, math.floor(min(xs) - radius)), max(0, math.floor(min(ys) - radius))
    x1 = min(strip_width, math.ceil(max(xs) + radius) + 1)
    y1 = min(strip_height, math.ceil(max(ys) + radius) + 1)
    if x1 <= x0 or y1 <= y0:  # beside or beyond the strip
        return None
    box = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
    image = Image.new("L", (box.width, box.height), 0)
    draw = ImageDraw.Draw(image)
    local = [(x - box.x0, y - box.y0) for x, y in points]
    if len(local) > 1:
        draw.line(local, fill=255, width=2 * radius, joint="curve")
    for x, y in local:  # round ends (and a single dab)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    mask = np.asarray(image) > 127
    return (box, mask) if mask.any() else None


def stroke_on_page(ingest: IngestArtifact, box: BBox, mask: np.ndarray) -> PageStroke:
    """A strip stroke on the page under its centre, in that page's pixels (the part on other pages is left out).
    ValueError when no page lies under it."""
    centre = (box.y0 + box.y1) / 2
    source = next((f for f in ingest.files if not f.filtered and f.y0 <= centre < f.y1), None)
    if source is None:
        raise ValueError("the stroke is not on a page")
    y0, y1 = max(box.y0, source.y0), min(box.y1, source.y1)
    cut = mask[y0 - box.y0 : y1 - box.y0, :]
    scale = source.scale if source.scale > 0 else 1.0
    full = BBox(
        x0=math.floor(box.x0 / scale),
        y0=math.floor((y0 - source.y0) / scale),
        x1=max(math.ceil(box.x1 / scale), math.floor(box.x0 / scale) + 1),
        y1=max(math.ceil((y1 - source.y0) / scale), math.floor((y0 - source.y0) / scale) + 1),
    )
    scaled = cut
    if (full.width, full.height) != (cut.shape[1], cut.shape[0]):
        resized = Image.fromarray(cut.astype(np.uint8) * 255).resize(
            (full.width, full.height), Image.Resampling.NEAREST
        )
        scaled = np.asarray(resized) > 127
    px0, py0 = max(full.x0, 0), max(full.y0, 0)
    px1, py1 = min(full.x1, source.width), min(full.y1, source.height)
    if px1 <= px0 or py1 <= py0:
        raise ValueError("the stroke is not on a page")
    page_box = BBox(x0=px0, y0=py0, x1=px1, y1=py1)
    page_mask = scaled[
        page_box.y0 - full.y0 : page_box.y1 - full.y0, page_box.x0 - full.x0 : page_box.x1 - full.x0
    ]
    return PageStroke(page=source.index, box=page_box, mask=np.ascontiguousarray(page_mask))


def clean_stroke(cfg: Config, paths: ChapterPaths, stroke: PageStroke, method: CleanMethod) -> CleanupPatch:
    """Clean a brush stroke and add it to the chapter's hand cleanup: "fill" (the colour around it), "inpaint",
    "lama" (the model loads for it) or "restore" (the raw page again). ValueError for a stroke that covers
    nothing."""
    if method == "lama":
        from omniscan.cleanup.lama_now import clean_with_lama  # torch loads only for a LaMa stroke

        return clean_with_lama(
            paths, _series_cfg(cfg, paths), page=stroke.page, box=stroke.box, mask=stroke.mask
        )
    return cleanup_store.add_patch(paths, page=stroke.page, box=stroke.box, mask=stroke.mask, method=method)


def remove_patch(paths: ChapterPaths, patch_id: str) -> None:
    """Take a hand-cleanup patch back (cleanup.store.delete_patch)."""
    cleanup_store.delete_patch(paths, patch_id)


@dataclass(frozen=True, slots=True)
class CutsState:
    """Where a chapter's exported images split."""

    hand: list[int] | None  # the cuts set by hand (strip rows); None: one image per slice
    auto: list[int]  # the slicer's own cuts (where its slices meet)
    strip_height: int
    bands: list[tuple[int, int]]  # the strip's uniform bands [y0, y1): calm places to cut
    crossings: list[tuple[int, str]]  # (cut, region id) for every cut through a region's text or bubble

    @property
    def cuts(self) -> list[int]:
        """The cuts the export applies: the ones set by hand, else the slicer's."""
        return self.auto if self.hand is None else self.hand


def load_cuts(paths: ChapterPaths) -> CutsState:
    """The chapter's output cuts as the export applies them; edits.store.EditNotFoundError before slicing."""
    slices = edit_store.load_slices(paths)
    hand = edit_store.load_edits(paths).cuts
    auto = [piece.y0 for piece in slices.slices[1:]]
    return CutsState(
        hand=hand,
        auto=auto,
        strip_height=slices.strip_height,
        bands=[(band.y0, band.y1) for band in slices.bands],
        crossings=cut_crossings(auto if hand is None else hand, edit_store.current_regions(paths)),
    )


def cut_max_height(cfg: Config, paths: ChapterPaths) -> int:
    """The tallest image the output cuts may leave (the series' slicer.hard_max_height)."""
    return _series_cfg(cfg, paths).slicer.hard_max_height


def snap_to_band(row: int, bands: Sequence[tuple[int, int]], max_distance: int = SNAP_ROWS) -> int:
    """`row` moved to the middle of the nearest uniform band within `max_distance` rows (a clean place to cut,
    between panels); `row` itself when no band is that close."""
    best, best_distance = row, max_distance
    for y0, y1 in bands:
        distance = y0 - row if row < y0 else row - (y1 - 1) if row >= y1 else 0
        if distance <= best_distance:
            best, best_distance = (y0 + y1 + 1) // 2, distance
    return best
