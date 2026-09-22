"""Import planning: read-only inspection of a source folder or .zip/.cbz archive into chapter/file items (three shapes)."""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

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

ARCHIVE_SUFFIXES = frozenset({".zip", ".cbz"})
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})


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
    archive: Path | None = None  # the source archive, when the plan was made from one
    temp_dir: tempfile.TemporaryDirectory[str] | None = field(
        default=None, compare=False, repr=False
    )  # where archive members were extracted; `cleanup()` removes it

    def cleanup(self) -> None:
        """Remove the extracted archive's temporary files (no-op when the source was a folder)."""
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def plan_import(source: Path, *, series: str | None = None, chapter: str | None = None) -> ImportPlan:
    """Read-only inspection of `source` (folder, or .zip/.cbz archive extracted to a temp dir).

    Folder sources: three shapes, decided by `source`'s immediate children — a single chapter folder
    (all image files), a folder of chapter folders (all children are directories naming chapters),
    or a flat dump of images whose filenames carry the chapter number.
    """
    source = Path(source)
    if source.suffix.lower() in ARCHIVE_SUFFIXES:
        return _plan_archive(source, series=series, chapter=chapter)
    return _plan_folder(source.resolve(), series=series, chapter=chapter)


def _plan_folder(source: Path, *, series: str | None, chapter: str | None) -> ImportPlan:
    """The folder path: unchanged from before archives existed (all three shapes live here)."""
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


def _plan_archive(source: Path, *, series: str | None, chapter: str | None) -> ImportPlan:
    """Archive shape: extract to a temp dir and plan the extracted folder (same three shapes).

    The series defaults to the archive's own name without extension; a lone top-level wrapper
    folder (how archive tools usually export a series) is descended into. The extraction lives
    until `ImportPlan.cleanup()` — the plan's items point into it.
    """
    if not source.is_file():
        raise ImportPlanError(f"source archive not found: {source.resolve()}")
    temp_dir = tempfile.TemporaryDirectory(prefix="omniscan-import-")
    try:
        root = _unwrap(_extract_archive(source, Path(temp_dir.name) / "archive"))
        plan = _plan_folder(root, series=source.stem if series is None else series, chapter=chapter)
    except BaseException:
        temp_dir.cleanup()  # a failed plan must not leak its extraction
        raise
    return replace(plan, archive=source, temp_dir=temp_dir)


def _extract_archive(source: Path, dest: Path) -> Path:
    """Extract `source`'s file members into `dest` (returning it); unreadable archives raise ImportPlanError."""
    dest.mkdir(parents=True, exist_ok=True)  # an archive with no file members is still an empty root
    dest = dest.resolve()
    try:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                _extract_member(archive, member, dest)
    except ImportPlanError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:  # truncated/corrupt, or encrypted members
        raise ImportPlanError(f"can't read archive {source.resolve()}: {exc}") from exc
    return dest


_UNSAFE_MEMBER_RE = re.compile(r"^[A-Za-z]:|\\")  # a Windows drive prefix, or any backslash


def _extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, dest: Path) -> None:
    """Write one regular-file member into `dest`, refusing names that would escape the tree.

    `member.filename` is attacker-controlled and only *nominally* POSIX-style (the ZIP spec mandates
    forward slashes). A name pattern check on the parsed `PurePosixPath` alone is not sufficient on a
    Windows host: a drive prefix (`C:/evil/file.txt`) parses with an empty `.drive` on `PurePosixPath`
    (POSIX paths have no drive concept, so that check would be a no-op), and a name with an embedded
    backslash (`..\\x`) stays one opaque part under `PurePosixPath` (which never splits on `\\`) yet
    gets split once joined onto a real (Windows) `Path` — whether that actually escapes `dest` then
    depends on which physical drive `dest` happens to live on, which is not something to rely on.
    Neither a drive prefix nor a backslash is ever legitimate in a ZIP member name, so both are
    rejected outright; the resolved target is then also verified to land inside `dest` as a second,
    independent guard (catches plain `..`/absolute members regardless of platform).
    """
    if _UNSAFE_MEMBER_RE.search(member.filename):
        raise ImportPlanError(f"unsafe member name in {archive.filename}: {member.filename!r}")
    name = PurePosixPath(member.filename)
    if member.is_dir() or not name.parts:
        return
    target = dest.joinpath(*name.parts).resolve()
    if target != dest and dest not in target.parents:
        raise ImportPlanError(f"unsafe member name in {archive.filename}: {member.filename!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as member_file, target.open("wb") as out:
        shutil.copyfileobj(member_file, out)


def _unwrap(root: Path) -> Path:
    """Descend through a lone top-level wrapper folder (only when the root holds exactly that one child)."""
    while True:
        entries = list(root.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            root = entries[0]
        else:
            return root


def files_to_convert(plan: ImportPlan) -> list[Path]:
    """Source files a commit will re-encode to JPEG (every image whose suffix is not .jpg/.jpeg)."""
    return [file for item in plan.items for file in item.files if file.suffix.lower() not in JPEG_SUFFIXES]


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
