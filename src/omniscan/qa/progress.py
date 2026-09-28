"""Where each chapter of a series stands, for whoever manages the series: `omniscan status` and the web app's
Progress view.

Read-only, from what the chapter already holds: its lines to translate (dialogue and captions with text; sound
effects and watermarks may stay as they are) with how many have English and how many a proofreader checked, the
open problems on the lines not checked yet (qa/problems.py), the last pipeline stage recorded done and a failed
one (manifest.json), and whether the pages were exported — and are out of date because the chapter was edited
by hand or cleaned by hand, or a pipeline stage ran again after the export (the qa re-read does not count).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from omniscan.cleanup.store import CLEANUP_FILE
from omniscan.core.manifest import load_manifest
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.edits import store
from omniscan.qa.problems import chapter_problems
from omniscan.translate.on_demand import english_lines

LINE_KINDS = frozenset({"bubble_text", "free_text"})


@dataclass(frozen=True, slots=True)
class ChapterProgress:
    """One chapter's state."""

    chapter: str
    lines: int  # dialogue and captions with text
    translated: int  # of those, the ones with an English line
    checked: int  # of those, the ones a proofreader checked (the check still holds)
    problems: int  # open problems on lines not checked yet
    last_stage: str | None  # the last pipeline stage recorded done, in pipeline order; None when none ran
    failed: str | None  # the first stage whose last run failed
    exported: bool
    outdated: bool  # exported, but edited by hand or re-run since: the pages need exporting again


def chapter_progress(paths: ChapterPaths) -> ChapterProgress:
    """Where one chapter stands (see the module docstring)."""
    from omniscan.pipeline.stages import STAGE_ORDER  # lazy: the stage table pulls in the pipeline

    targets = [r.id for r in store.current_regions(paths) if r.kind in LINE_KINDS and r.text.strip()]
    english = english_lines(paths)
    status = store.line_statuses(paths)
    stages = load_manifest(paths.manifest, paths.series, paths.chapter).stages
    done = [name for name in STAGE_ORDER if name in stages and stages[name].status == "done"]
    failed = next((name for name in STAGE_ORDER if name in stages and stages[name].status == "failed"), None)
    export = stages.get("export")
    exported = export is not None and export.status == "done"
    outdated = False
    if export is not None and exported:
        rerun = any(stages[name].finished_at > export.finished_at for name in STAGE_ORDER if name in stages)
        hand = [paths.artifact(name) for name in (store.EDITS_FILE, CLEANUP_FILE)]
        edited = [datetime.fromtimestamp(path.stat().st_mtime, UTC) for path in hand if path.is_file()]
        outdated = rerun or any(when > export.finished_at for when in edited)
    return ChapterProgress(
        chapter=paths.chapter,
        lines=len(targets),
        translated=sum(1 for region_id in targets if english.get(region_id, "").strip()),
        checked=sum(1 for region_id in targets if status.get(region_id) == "checked"),
        problems=sum(1 for problem in chapter_problems(paths) if problem.status != "checked"),
        last_stage=done[-1] if done else None,
        failed=failed,
        exported=exported,
        outdated=outdated,
    )


def series_progress(series: SeriesPaths) -> list[ChapterProgress]:
    """Every chapter of the series in reading order."""
    return [chapter_progress(series.chapter(name)) for name in series.chapters()]
