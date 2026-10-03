"""Import planning for series folders written by manhwa-manga-downloader (`mangadl`).

The downloader saves a series as `downloads/<site>/<series>/` with one `num<N>_<title>` folder per chapter
(`num12_Chapter 12`, `num45_chapter`, `num5.5_Chapter 5.5`) and two bookkeeping files beside them:
`chapter_manifest.json` (page count of every chapter it finished) and `incomplete_chapters.json` (chapters
still missing pages). A chapter it could not number gets `num0_<slug>` or `numunknown_chapter`. Such a folder
imports as `Chapter <N>` per chapter; unfinished and unnumbered chapters are left out with a warning.

A chapter counts as finished only when `chapter_manifest.json` lists it: the downloader writes both files once,
after every chapter of a run, and deletes in-flight `.part` files when a run is stopped, so a run killed at
page 10 of 30 leaves a clean-looking folder that only the missing manifest entry gives away.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from omniscan.core.paths import IMAGE_SUFFIXES, natural_key
from omniscan.importer.plan import ImportPlan, ImportPlanError, ImportPlanItem, chapter_images

MANIFEST_FILE = "chapter_manifest.json"
INCOMPLETE_FILE = "incomplete_chapters.json"
BOOKKEEPING_FILES = frozenset({MANIFEST_FILE, INCOMPLETE_FILE})
PART_SUFFIX = ".part"  # the downloader writes each image to `<name>.part` and renames it once complete

_FOLDER_RE = re.compile(r"num([^_]*)_(.+)")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Series folders the downloader names after the site's id rather than a title: wfwf's numeric toon id, MangaDex's
# manga UUID. Neither makes a usable series name.
_SITE_ID_RE = re.compile(r"\d+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def is_downloader_series(dirs: list[Path]) -> bool:
    """True when every subfolder carries the downloader's `num<N>_<title>` chapter-folder name."""
    return bool(dirs) and all(_FOLDER_RE.fullmatch(d.name) for d in dirs)


def plan_downloader_series(
    source: Path,
    dirs: list[Path],
    files: list[Path],
    *,
    series: str | None,
    chapter: str | None,
) -> ImportPlan:
    """Plan a downloader series folder: one `Chapter <N>` item per finished, numbered chapter folder."""
    if chapter is not None:
        raise ImportPlanError("--chapter is ambiguous here: every subfolder already names its own chapter")
    if series is None:
        series = downloader_series_name(source.name)
    warnings = [
        f"skipped non-image file: {file.name}"
        for file in sorted(files, key=lambda p: natural_key(p.name))
        if file.suffix.lower() not in IMAGE_SUFFIXES and file.name not in BOOKKEEPING_FILES
    ]
    manifest = _read_manifest(source / MANIFEST_FILE)
    incomplete = _read_incomplete(source / INCOMPLETE_FILE)

    chosen: dict[float, tuple[str, list[Path]]] = {}
    for folder in sorted(dirs, key=lambda d: natural_key(d.name)):
        number = _chapter_number(folder.name)
        if number is None:
            warnings.append(
                f"skipped {folder.name}: the downloader recorded no chapter number — "
                "import it on its own with --series and --chapter"
            )
            continue
        images = chapter_images(folder)
        problem = _unfinished(folder, images, manifest, incomplete)
        if problem is not None:
            warnings.append(
                f"skipped {folder.name}: {problem} — finish it with the downloader, then import again"
            )
            continue
        if not images:
            warnings.append(f"skipped {folder.name}: no image files")
            continue
        if number in chosen:
            raise ImportPlanError(
                f"{chosen[number][0]} and {folder.name} are both chapter {number:g} in {source}"
            )
        chosen[number] = (folder.name, images)

    if not chosen:
        raise ImportPlanError(f"nothing to import from {source}: " + "; ".join(warnings))
    return ImportPlan(
        series=series,
        items=[
            ImportPlanItem(chapter=f"Chapter {number:g}", files=images)
            for number, (_, images) in sorted(chosen.items())
        ],
        warnings=warnings,
    )


def downloader_series_name(name: str) -> str:
    """`name` (the downloader's series folder name) as the series name; ImportPlanError when it is the site's id."""
    if _SITE_ID_RE.fullmatch(name):
        raise ImportPlanError(
            f"the downloader named this series after the site's id ({name}) — pass --series"
        )
    return name


def _chapter_number(name: str) -> float | None:
    """The chapter number a `num<N>_<title>` folder name records; None for the downloader's unnumbered fallbacks."""
    match = _FOLDER_RE.fullmatch(name)
    if match is None or _NUMBER_RE.fullmatch(match.group(1)) is None:
        return None
    number = float(match.group(1))
    # `num0_` is also the downloader's "no number" fallback (`num0_<slug>`, `num0_Chapter unknown`); only a
    # folder titled `Chapter 0` is a real chapter zero.
    if number == 0 and match.group(2) != "Chapter 0":
        return None
    return number


def _unfinished(
    folder: Path, images: list[Path], manifest: dict[str, int], incomplete: dict[str, str]
) -> str | None:
    """Why the downloader's own records say `folder` is not a finished chapter, or None when it is."""
    if folder.name in incomplete:
        return incomplete[folder.name]
    if any(entry.name.endswith(PART_SUFFIX) for entry in folder.iterdir()):
        return "the download was interrupted"
    expected = manifest.get(folder.name)
    if expected is None:
        return f"the downloader has not confirmed it finished (no entry in {MANIFEST_FILE})"
    if expected != len(images):
        return f"{len(images)} page(s) on disk, the downloader finished it with {expected}"
    return _numbering_gap(images)


def _numbering_gap(images: list[Path]) -> str | None:
    """The first page missing from the downloader's `0001..N` page names, or None when they have no gap."""
    if not all(image.stem.isdigit() for image in images):
        return None
    numbers = {int(image.stem) for image in images}
    missing = next((n for n in range(1, len(images) + 1) if n not in numbers), None)
    return None if missing is None else f"page {missing:04d} is missing"


def _read_manifest(path: Path) -> dict[str, int]:
    """`chapter_manifest.json`: folder name -> page count; empty when absent or unreadable."""
    data = _read_json(path)
    chapters = data.get("chapters") if isinstance(data, dict) else None
    if not isinstance(chapters, dict):
        return {}
    return {str(name): count for name, count in chapters.items() if isinstance(count, int)}


def _read_incomplete(path: Path) -> dict[str, str]:
    """`incomplete_chapters.json`: folder name -> how much of it was downloaded; empty when absent or unreadable."""
    data = _read_json(path)
    chapters = data.get("chapters") if isinstance(data, dict) else None
    if not isinstance(chapters, list):
        return {}
    result: dict[str, str] = {}
    for entry in chapters:
        if isinstance(entry, dict) and isinstance(entry.get("folder"), str):
            downloaded, total = entry.get("downloaded"), entry.get("total")
            pages = (
                f" ({downloaded}/{total} pages)"
                if isinstance(downloaded, int) and isinstance(total, int)
                else ""
            )
            result[entry["folder"]] = f"incomplete download{pages}"
    return result


def _read_json(path: Path) -> object:
    """Parsed JSON of `path`, or None when it is missing or not valid JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
