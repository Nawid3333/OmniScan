"""Library service: series/chapter browsing and the tile model behind the reader views.

Qt-free by design (widgets import this, never the other way round). Everything is derived from the
filesystem artifacts the pipeline already writes, with fallbacks so a partially processed chapter
still shows up. All y coordinates are in strip space (see `omniscan.core.schemas`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from omniscan.core.config import Config
from omniscan.core.manifest import load_manifest
from omniscan.core.paths import SeriesPaths, list_chapters, list_images, natural_key
from omniscan.core.schemas import (
    ExportArtifact,
    IngestArtifact,
    Manifest,
    SlicesArtifact,
    StageRecord,
)

TileKind = Literal["image", "filtered", "missing"]
StageState = Literal["done", "stale", "failed", "not run"]


@dataclass(frozen=True, slots=True)
class Tile:
    """One horizontal band of the strip shown as a unit: an image or a gap."""

    y0: int  # strip-space rows [y0, y1)
    y1: int
    path: Path | None  # None for gaps
    label: str  # file name, or e.g. "filtered slice 4"
    kind: TileKind


@dataclass(frozen=True, slots=True)
class ChapterView:
    """Everything the reader needs to display one chapter: raw tiles, output tiles, strip size."""

    series: str
    chapter: str
    strip_width: int
    strip_height: int
    raw: tuple[Tile, ...]  # raw pages in strip space
    output: tuple[Tile, ...]  # output slices in strip space (empty when there is no output)
    has_output: bool


def _series_dirs(root: Path) -> list[str]:
    """Series-candidate folder names under one root: existing dirs, skipping '_'/'.', natural order."""
    if not root.is_dir():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]
    return sorted(names, key=natural_key)


def list_series(cfg: Config) -> list[str]:
    """Series names present in the library or the output root (union, natural order)."""
    names = set(_series_dirs(cfg.paths.library_root)) | set(_series_dirs(cfg.paths.output_root))
    return sorted(names, key=natural_key)


def list_chapter_names(cfg: Config, series: str) -> list[str]:
    """Chapter names of a series: raw folders first, output folders when the library has none."""
    paths = SeriesPaths.from_config(cfg, series)
    chapters = paths.chapters()
    if chapters:
        return chapters
    return [p.name for p in list_chapters(paths.output_dir)]


def _load_ingest(work_dir: Path) -> IngestArtifact | None:
    try:
        return IngestArtifact.load(work_dir / "ingest.json")
    except OSError, ValueError:
        return None


def _load_slices(work_dir: Path) -> SlicesArtifact | None:
    try:
        return SlicesArtifact.load(work_dir / "slices.json")
    except OSError, ValueError:
        return None


def _load_export(work_dir: Path) -> ExportArtifact | None:
    try:
        return ExportArtifact.load(work_dir / "export.json")
    except OSError, ValueError:
        return None


def _image_tiles_from_dir(directory: Path, strip_width: int) -> tuple[tuple[Tile, ...], int, int]:
    """Stack a folder's images from y=0, scaling each height to `strip_width` (or the first image's width).

    Returns (tiles, strip_width, strip_height); strip_width 0 when the folder has no images.
    """
    images = list_images(directory)
    if not images:
        return (), strip_width, 0
    sizes: list[tuple[int, int]] = []
    for path in images:
        with Image.open(path) as img:
            sizes.append(img.size)  # (width, height); header-only read, no pixel data
    if strip_width <= 0:
        strip_width = sizes[0][0]
    y = 0
    tiles: list[Tile] = []
    for path, (width, height) in zip(images, sizes, strict=True):
        scaled = round(height * strip_width / width) if width else height
        tiles.append(Tile(y, y + scaled, path, path.name, "image"))
        y += scaled
    return tuple(tiles), strip_width, y


def _raw_tiles(ingest: IngestArtifact | None, raw_dir: Path) -> tuple[tuple[Tile, ...], int, int]:
    """Raw tiles + strip size from ingest.json (filtered files skipped), else stacked raw images."""
    if ingest is not None:
        files = sorted((f for f in ingest.files if not f.filtered), key=lambda f: f.index)
        tiles = tuple(Tile(f.y0, f.y1, raw_dir / f.name, f.name, "image") for f in files)
        return tiles, ingest.strip_width, ingest.strip_height
    return _image_tiles_from_dir(raw_dir, 0)


def _output_tiles(
    slices: SlicesArtifact | None, export: ExportArtifact | None, output_dir: Path, chapter_width: int
) -> tuple[tuple[Tile, ...], int, int]:
    """Output tiles + strip size from slices.json/export.json, else stacked output images.

    The fallback scales heights to `chapter_width` (the raw strip width when known); with 0 it uses
    the first output image's own width.
    """
    if slices is not None and export is not None:
        by_index = {f.slice_index: f for f in export.files}
        tiles: list[Tile] = []
        for s in sorted(slices.slices, key=lambda s: s.index):
            exported = by_index.get(s.index)
            if exported is not None and (output_dir / exported.name).is_file():
                tiles.append(Tile(s.y0, s.y1, output_dir / exported.name, exported.name, "image"))
            elif s.filtered:
                tiles.append(Tile(s.y0, s.y1, None, f"filtered slice {s.index}", "filtered"))
            else:
                tiles.append(Tile(s.y0, s.y1, None, f"missing slice {s.index}", "missing"))
        return tuple(tiles), slices.strip_width, slices.strip_height
    return _image_tiles_from_dir(output_dir, chapter_width)


def load_chapter_view(cfg: Config, series: str, chapter: str) -> ChapterView:
    """Read one chapter's tiles; falls back on missing/invalid artifacts.

    Raises FileNotFoundError only when neither the raw dir nor the output dir exists.
    """
    paths = SeriesPaths.from_config(cfg, series).chapter(chapter)
    if not paths.raw_dir.is_dir() and not paths.output_dir.is_dir():
        raise FileNotFoundError(f"neither {paths.raw_dir} nor {paths.output_dir} exists")

    ingest = _load_ingest(paths.work_dir)
    slices = _load_slices(paths.work_dir)
    export = _load_export(paths.work_dir)

    raw_tiles, raw_w, raw_h = _raw_tiles(ingest, paths.raw_dir)
    out_tiles, out_w, out_h = _output_tiles(slices, export, paths.output_dir, raw_w)
    if ingest is not None:  # ingest wins for the chapter strip size
        strip_w, strip_h = raw_w, raw_h
    elif slices is not None and export is not None:
        strip_w, strip_h = out_w, out_h
    else:
        strip_w, strip_h = raw_w or out_w, raw_h or out_h

    return ChapterView(
        series=series,
        chapter=chapter,
        strip_width=strip_w,
        strip_height=strip_h,
        raw=raw_tiles,
        output=out_tiles,
        has_output=any(t.kind == "image" for t in out_tiles),
    )


@dataclass(frozen=True, slots=True)
class SeriesSummary:
    """One library series and how many chapters it has (raws and/or output)."""

    name: str
    chapters: int


@dataclass(frozen=True, slots=True)
class ChapterStages:
    """One chapter's state for each of the ten pipeline stages (aligned with `STAGE_ORDER`)."""

    chapter: str
    states: tuple[StageState, ...]


def series_summaries(cfg: Config) -> list[SeriesSummary]:
    """Every series in the library/output roots with its chapter count, natural order."""
    return [SeriesSummary(name, len(list_chapter_names(cfg, name))) for name in list_series(cfg)]


def chapter_stage_states(cfg: Config, series: str) -> list[ChapterStages]:
    """Per chapter, the ten stages' states from the chapter's manifest, in `STAGE_ORDER`.

    `done` = recorded as finished with its output files present. `stale` = done but its recorded
    output is missing, or a stage earlier in `STAGE_ORDER` finished again after it did (the stage
    would re-run). This is the manifest-only view for review — the runner's own hash check decides
    at run time.
    """
    from omniscan.pipeline.stages import STAGE_ORDER  # lazy: keep importing this module light

    paths = SeriesPaths.from_config(cfg, series)
    out = []
    for chapter in list_chapter_names(cfg, series):
        cp = paths.chapter(chapter)
        manifest = load_manifest(cp.manifest, series, chapter)
        out.append(ChapterStages(chapter, _states(manifest, cp.work_dir, STAGE_ORDER)))
    return out


def _states(manifest: Manifest, work_dir: Path, order: Sequence[str]) -> tuple[StageState, ...]:
    """One chapter's stage states: done / stale / failed / not run (see `chapter_stage_states`)."""
    records: dict[str, StageRecord] = manifest.stages
    states: list[StageState] = []
    for index, name in enumerate(order):
        record = records.get(name)
        if record is None:
            states.append("not run")
        elif record.status == "failed":
            states.append("failed")
        elif any(not (work_dir / output).exists() for output in record.outputs) or any(
            earlier.finished_at > record.finished_at
            for earlier in (records.get(before) for before in order[:index])
            if earlier is not None
        ):
            states.append("stale")
        else:
            states.append("done")
    return tuple(states)
