"""Importer service: planning, conversion preview, plan editing and commit for the import screen (Qt-free).

`ImporterService` wraps `omniscan.importer` with the configured library root; the plan-editing
helpers below are plain functions over `ImportPlan` (frozen dataclass in, new plan out). Callers
run these on worker threads — none of the methods touch Qt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from omniscan.core.config import Config
from omniscan.core.paths import chapter_number
from omniscan.hw.detect import HardwareInfo, detect_hardware
from omniscan.importer.execute import ImportResult, execute_import
from omniscan.importer.plan import ImportPlan, ImportPlanItem, files_to_convert, plan_import

_Progress = Callable[[int, int | None], None]

# The honest reason the preview gives for the conversion (the pipeline is JPEG end-to-end; the
# encoder is Pillow's CPU libjpeg-turbo — hardware JPEG decode is card C2 and does not exist yet).
CONVERSION_REASON = "for consistent, fast processing"


class ImporterService:
    """Planning, conversion preview and commit of one import into the configured library."""

    def __init__(
        self,
        cfg: Config,
        *,
        hardware: Callable[[], HardwareInfo] | None = None,
    ) -> None:
        """Build the service; the default hardware provider detects this machine (imports torch)."""
        self._cfg = cfg
        self._hardware = hardware or (lambda: detect_hardware(cfg.paths.models_dir))

    def plan(self, source: Path, *, series: str | None = None, chapter: str | None = None) -> ImportPlan:
        """Plan an import from a folder or .zip/.cbz archive (archives extract to a temp dir)."""
        return plan_import(source, series=series, chapter=chapter)

    def execute(
        self, plan: ImportPlan, *, move: bool = False, on_progress: _Progress | None = None
    ) -> ImportResult:
        """Commit `plan` into the library (converting non-JPEG sources); reports (done, total) per file."""
        return execute_import(plan, self._cfg.paths.library_root, move=move, on_progress=on_progress)

    def hardware_line(self) -> str:
        """The conversion notice's context line: what was detected, what actually runs today."""
        hw = self._hardware()
        if hw.gpus:
            gpu = hw.gpus[0]
            detail = f"{gpu.name} · {gpu.vram_gb:g} GB · {gpu.backend}"
        else:
            detail = "no GPU — CPU only"
        return f"machine: {detail} — conversion runs on the CPU (libjpeg-turbo)"


def conversion_text(plan: ImportPlan) -> str:
    """The one-line preview of what will be converted (`""` when every file is already JPEG)."""
    count = len(files_to_convert(plan))
    if not count:
        return ""
    total = sum(len(item.files) for item in plan.items)
    return f"{count} of {total} file(s) will be converted to JPEG {CONVERSION_REASON}."


def set_series(plan: ImportPlan, series: str) -> ImportPlan:
    """A plan whose destination series is `series` (the editable series field)."""
    return replace(plan, series=series)


def rename_chapter(plan: ImportPlan, item: int, name: str) -> ImportPlan:
    """A plan with chapter `item`'s destination folder renamed to `name`."""
    items = list(plan.items)
    items[item] = ImportPlanItem(chapter=name, files=list(items[item].files))
    return _rebuilt(plan, items)


def move_file(
    plan: ImportPlan, from_item: int, file_index: int, to_item: int, to_index: int | None = None
) -> ImportPlan:
    """A plan with one file moved to chapter `to_item` (at `to_index`, or the end); emptied chapters vanish."""
    chapters = [item.chapter for item in plan.items]
    files: list[list[Path]] = [list(item.files) for item in plan.items]
    moved = files[from_item].pop(file_index)
    if to_index is None:
        to_index = len(files[to_item])
    files[to_item].insert(to_index, moved)
    kept = [
        ImportPlanItem(chapter=name, files=group)
        for name, group in zip(chapters, files, strict=True)
        if group
    ]
    return _rebuilt(plan, kept)


def merge_chapters(plan: ImportPlan, item: int, into: int) -> ImportPlan:
    """A plan with chapter `item`'s files appended to chapter `into`'s (after its own); `item` disappears."""
    if item == into:
        raise ValueError("cannot merge a chapter into itself")
    items = list(plan.items)
    target = items[into]
    items[into] = ImportPlanItem(chapter=target.chapter, files=[*target.files, *items[item].files])
    del items[item]
    return _rebuilt(plan, items)


def split_chapter(plan: ImportPlan, item: int, at: int) -> ImportPlan:
    """A plan where chapter `item`'s pages from `at` on form a new chapter (named after the top number)."""
    items = list(plan.items)
    chapter, files = items[item].chapter, list(items[item].files)
    if at < 1 or at >= len(files):
        raise ValueError(f"cannot split chapter {chapter!r} at page {at} of {len(files)}")
    tail = files[at:]
    items[item] = ImportPlanItem(chapter=chapter, files=files[:at])
    items.insert(item + 1, ImportPlanItem(chapter=_next_chapter_name(plan), files=tail))
    return _rebuilt(plan, items)


def _next_chapter_name(plan: ImportPlan) -> str:
    """`"Chapter <n+1>"` for the highest chapter number in the plan (0-based fallback: `Chapter 1`)."""
    numbers = [chapter_number(item.chapter) for item in plan.items]
    top = max((number for number in numbers if number is not None), default=0)
    return f"Chapter {top + 1:g}"


def _rebuilt(plan: ImportPlan, items: list[ImportPlanItem]) -> ImportPlan:
    """A new plan with `items`, keeping everything else (including the archive's temp extraction)."""
    return replace(plan, items=items)
