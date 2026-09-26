"""Filesystem layout contract: where raws, work artifacts and outputs of a series/chapter live."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from omniscan.core.config import Config
from omniscan.core.schemas import IngestArtifact

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif", ".tif", ".tiff"})
REFERENCE_DIR = "_reference_en"
FILTERED_DIR = "_filtered"
_CHAPTER_RE = re.compile(r"(?:chapter|chap|ch|episode|ep)?[\s._-]*(\d+(?:\.\d+)?)", re.IGNORECASE)
_NATURAL_RE = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[object, ...]:
    """Sort key so 'Chapter 10' comes after 'Chapter 9' and '002.jpg' after '1.jpg'."""
    return tuple(int(part) if part.isdigit() else part.casefold() for part in _NATURAL_RE.split(name))


def chapter_number(name: str) -> float | None:
    """Parse the chapter number from a folder name ('Chapter 12', 'Ch.12.5', 'ep 7'); None if absent."""
    match = _CHAPTER_RE.search(name)
    return float(match.group(1)) if match else None


def list_chapters(series_dir: Path) -> list[Path]:
    """Chapter folders of a series in reading order (skips '_'-prefixed special folders)."""
    if not series_dir.is_dir():
        return []
    dirs = [p for p in series_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]
    return sorted(
        dirs,
        key=lambda p: (chapter_number(p.name) is None, chapter_number(p.name) or 0.0, natural_key(p.name)),
    )


def list_images(chapter_dir: Path) -> list[Path]:
    """Image files of a chapter folder in natural order."""
    if not chapter_dir.is_dir():
        return []
    files = [p for p in chapter_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: natural_key(p.name))


@dataclass(frozen=True, slots=True)
class ChapterPaths:
    """All locations belonging to one chapter of one series."""

    series: str
    chapter: str
    raw_dir: Path
    work_dir: Path
    output_dir: Path
    filtered_dir: Path

    @property
    def manifest(self) -> Path:
        return self.work_dir / "manifest.json"

    def artifact(self, name: str) -> Path:
        """Path of a JSON/npz artifact inside the chapter work dir (e.g. 'slices.json')."""
        return self.work_dir / name


def jpeg_paths(ingest: IngestArtifact, raw_dir: Path, cache_dir: Path) -> list[Path]:
    """The JPEG file to decode for every entry of ingest.files, in order (raw original or cache copy)."""
    return [
        raw_dir / file.name
        if file.converted_from is None
        else cache_dir / f"{file.index:04d}_{Path(file.name).stem}.jpg"
        for file in ingest.files
    ]


@dataclass(frozen=True, slots=True)
class SeriesPaths:
    """Locations of one series across library / work / output roots."""

    series: str
    library_dir: Path
    work_dir: Path
    output_dir: Path

    @classmethod
    def from_config(cls, cfg: Config, series: str) -> SeriesPaths:
        return cls(
            series=series,
            library_dir=cfg.paths.library_root / series,
            work_dir=cfg.paths.work_root / series,
            output_dir=cfg.paths.output_root / series,
        )

    @property
    def db(self) -> Path:
        return self.work_dir / "series.db"

    @property
    def glossary_yaml(self) -> Path:
        return self.work_dir / "glossary.yaml"

    @property
    def reference_dir(self) -> Path:
        return self.library_dir / REFERENCE_DIR

    @property
    def sources_toml(self) -> Path:
        return self.library_dir / "sources.toml"

    def chapters(self) -> list[str]:
        """Raw chapter folder names in reading order."""
        return [p.name for p in list_chapters(self.library_dir)]

    def chapter(self, chapter: str) -> ChapterPaths:
        return ChapterPaths(
            series=self.series,
            chapter=chapter,
            raw_dir=self.library_dir / chapter,
            work_dir=self.work_dir / chapter,
            output_dir=self.output_dir / chapter,
            filtered_dir=self.output_dir / FILTERED_DIR / chapter,
        )
