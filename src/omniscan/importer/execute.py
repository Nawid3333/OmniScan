"""Import execution: apply a planned import into the library layout (idempotent copy/move)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from omniscan.core.manifest import hash_file
from omniscan.importer.plan import ImportPlan, ImportPlanError


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Counts for one executed import; `chapters_written` is in plan order."""

    chapters_written: list[str]  # plan.items chapters, in plan order
    files_copied: int
    files_skipped_duplicate: int


def execute_import(plan: ImportPlan, library_root: Path, *, move: bool = False) -> ImportResult:
    """Apply `plan` under `library_root / plan.series / <chapter>`.

    Fail-fast: stops at the first destination file that exists with different content and raises
    `ImportPlanError`; files already written before that point are left in place.
    """
    chapters_written: list[str] = []
    files_copied = 0
    files_skipped = 0
    for item in plan.items:
        dest_dir = library_root / plan.series / item.chapter
        dest_dir.mkdir(parents=True, exist_ok=True)
        for source_file in item.files:
            dest = dest_dir / source_file.name
            if not dest.exists():
                if move:
                    shutil.move(source_file, dest)
                else:
                    shutil.copy2(source_file, dest)
                files_copied += 1
            elif hash_file(dest) == hash_file(source_file):
                if move:  # identical content already at the destination; drop the redundant source
                    source_file.unlink()
                files_skipped += 1
            else:
                raise ImportPlanError(
                    f"destination exists with different content: {source_file} -> {dest} "
                    f"({files_copied} file(s) already written in this call)"
                )
        chapters_written.append(item.chapter)
    return ImportResult(
        chapters_written=chapters_written,
        files_copied=files_copied,
        files_skipped_duplicate=files_skipped,
    )
