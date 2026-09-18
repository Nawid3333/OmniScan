"""Import planning: read-only inspection of a source folder into chapter/file items (three shapes)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from omniscan.core.paths import (
    IMAGE_SUFFIXES,
    chapter_number,
    list_chapters,
    list_images,
    natural_key,
)

# Same keyword set as core.paths._CHAPTER_RE, but the keyword is REQUIRED (not optional). core.paths'
# chapter_number() is deliberately loose for folder names, where a bare "12" IS the chapter — but applying
# that same looseness per-file here would misread an ordinary single chapter of sequentially numbered pages
# ("1.jpg", "2.jpg", "3.jpg" — the most common raw-page naming convention) as three separate one-page
# chapters. Per-file grouping (Case C) only fires on an explicit marker ("Ch1", "Chapter_02", "ep3", ...);
# bare page numbers fall through to Case A instead (single chapter, needs --chapter or a parseable folder name).
_EXPLICIT_CHAPTER_RE = re.compile(r"(?:chapter|chap|ch|episode|ep)[\s._-]*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _explicit_chapter_number(name: str) -> float | None:
    match = _EXPLICIT_CHAPTER_RE.search(name)
    return float(match.group(1)) if match else None


class ImportPlanError(RuntimeError):
    """Raised when a source can't be unambiguously planned, or a destination conflict is found at execute time."""


@dataclass(frozen=True, slots=True)
class ImportPlanItem:
    """One destination chapter and the source files that go into it."""

    chapter: str  # destination chapter folder name, e.g. "Chapter 12" or "Chapter 12.5"
    files: list[Path]  # absolute source file paths, in destination order


@dataclass(frozen=True, slots=True)
class ImportPlan:
    """The result of planning one import: series name, chapter items, and non-fatal warnings."""

    series: str
    items: list[ImportPlanItem]
    warnings: list[str]  # e.g. "skipped non-image file: notes.txt"


def plan_import(source: Path, *, series: str | None = None, chapter: str | None = None) -> ImportPlan:
    """Read-only inspection of `source`; never touches the filesystem outside of listing it.

    Supports three shapes, decided by `source`'s immediate children: a single chapter folder
    (all image files), a folder of chapter folders (all children are directories naming chapters),
    or a flat dump of images whose filenames carry the chapter number.
    """
    source = source.resolve()
    if not source.is_dir():
        raise ImportPlanError(f"source folder not found: {source}")
    entries = list(source.iterdir())
    if not entries:
        raise ImportPlanError(f"source folder is empty: {source}")

    dirs = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    loose_images = [file for file in files if file.suffix.lower() in IMAGE_SUFFIXES]
    warnings = [
        f"skipped non-image file: {file.name}"
        for file in sorted(
            (file for file in files if file.suffix.lower() not in IMAGE_SUFFIXES),
            key=lambda p: natural_key(p.name),
        )
    ]

    if dirs and loose_images:
        raise ImportPlanError(
            f"mixed shape in {source}: subfolder(s) {[d.name for d in dirs]} alongside loose "
            f"image file(s) {[f.name for f in loose_images]} — import either a single chapter "
            "folder, a folder of chapter folders, or a flat folder of images"
        )
    if dirs:
        return _plan_folder_of_folders(source, dirs, series=series, chapter=chapter, warnings=warnings)
    return _plan_flat(source, list_images(source), series=series, chapter=chapter, warnings=warnings)


def _plan_folder_of_folders(
    source: Path,
    dirs: list[Path],
    *,
    series: str | None,
    chapter: str | None,
    warnings: list[str],
) -> ImportPlan:
    """Case B: every child directory names its own chapter; one item per subfolder."""
    unparseable = [d.name for d in dirs if chapter_number(d.name) is None]
    if unparseable:
        raise ImportPlanError(f"subfolder(s) of {source} don't name a chapter: {', '.join(unparseable)}")
    if chapter is not None:
        raise ImportPlanError("--chapter is ambiguous here: every subfolder already names its own chapter")
    if series is None:
        series = source.name
    return ImportPlan(
        series=series,
        items=[
            ImportPlanItem(chapter=subdir.name, files=list_images(subdir)) for subdir in list_chapters(source)
        ],
        warnings=warnings,
    )


def _plan_flat(
    source: Path,
    images: list[Path],
    *,
    series: str | None,
    chapter: str | None,
    warnings: list[str],
) -> ImportPlan:
    """Case A (single chapter) or Case C (flat dump), both all-file shapes."""
    if not images:
        raise ImportPlanError(f"no image files found in {source}")
    if chapter is not None:
        return _plan_single_chapter(source, images, series=series, chapter=chapter, warnings=warnings)
    folder_number = chapter_number(source.name)
    if folder_number is not None:
        return _plan_single_chapter(
            source, images, series=series, chapter=f"Chapter {folder_number:g}", warnings=warnings
        )
    return _plan_flat_dump(source, images, series=series, warnings=warnings)


def _plan_single_chapter(
    source: Path,
    images: list[Path],
    *,
    series: str | None,
    chapter: str,
    warnings: list[str],
) -> ImportPlan:
    """Case A: the whole folder is one chapter (``--chapter`` given or read off the folder name)."""
    if series is None:
        raise ImportPlanError(f"series is required to import {source} — pass --series")
    return ImportPlan(
        series=series,
        items=[ImportPlanItem(chapter=chapter, files=images)],
        warnings=warnings,
    )


def _plan_flat_dump(
    source: Path,
    images: list[Path],
    *,
    series: str | None,
    warnings: list[str],
) -> ImportPlan:
    """Case C: filenames carry an explicit per-file chapter marker; group by number, chapters ascending."""
    numbers = {image.name: _explicit_chapter_number(image.name) for image in images}
    unparsed = [name for name, number in numbers.items() if number is None]
    if len(unparsed) == len(numbers):
        raise ImportPlanError(f"can't tell which chapter the images in {source} belong to — pass --chapter")
    if unparsed:
        raise ImportPlanError(f"no chapter number in filename: {', '.join(unparsed)}")
    if series is None:
        raise ImportPlanError(f"series is required to import {source} — pass --series")
    groups: dict[float, list[Path]] = {}
    for image in images:  # already in natural order, so each group keeps that order
        number = _explicit_chapter_number(image.name)
        if number is not None:
            groups.setdefault(number, []).append(image)
    return ImportPlan(
        series=series,
        items=[
            ImportPlanItem(chapter=f"Chapter {number:g}", files=group)
            for number, group in sorted(groups.items())
        ],
        warnings=warnings,
    )
