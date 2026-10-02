"""Qt-free services behind the desktop Studio: translate or re-read a few regions now, render a page preview.

Each call is blocking (it waits on a model) and runs on a worker thread in the GUI; the view injects fakes
in tests. Nothing here writes to the chapter: suggestions and readings go into the Studio's session, where
Save records them like any hand edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omniscan.cleanup import store as cleanup_store
from omniscan.core.config import Config, SeriesConfigError, get_secrets, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths, natural_key
from omniscan.core.schemas import IngestArtifact
from omniscan.translate.on_demand import translate_now
from omniscan.translate.profiles import default_profile_paths, load_profiles
from omniscan.typeset.chapter import chapter_layout
from omniscan.typeset.fonts import fonts_dir
from omniscan.typeset.page_preview import render_page

PREVIEW_SCHEME = "preview"  # the pseudo paths of rendered preview tiles: preview/<page index>


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
    series = SeriesPaths(
        series=paths.series,
        library_dir=paths.raw_dir.parent,
        work_dir=paths.work_dir.parent,
        output_dir=paths.output_dir.parent,
    )
    result = translate_now(
        OllamaClient(scfg.ollama, get_secrets()),
        scfg,
        series,
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
