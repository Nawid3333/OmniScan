"""Qt-free series checks behind the desktop Studio: find and replace across chapters, and the consistency report.

Find and replace is `omniscan edit replace` (edits/replace.py): `plan_replace` lists every line a rule changes,
`apply_replace` records the kept ones as hand edits, one undo step per chapter. The consistency report is `omniscan
consistency` (qa/consistency.py): one source line translated in several ways, and lines that miss a locked
glossary term. Each call reads (or writes) the chapters' files and runs on a worker thread in the GUI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omniscan.core.config import Config, SeriesConfigError, series_config
from omniscan.core.paths import SeriesPaths
from omniscan.edits.replace import Change, FindReplace, Target, apply_changes, plan
from omniscan.qa.consistency import Divergence, TermMiss, divergences, glossary, series_lines, term_misses


@dataclass(frozen=True, slots=True)
class Consistency:
    """A series' consistency report."""

    divergences: list[Divergence]  # one source line, several English renderings
    misses: list[TermMiss]  # lines whose English lacks a locked term's target


def plan_replace(
    cfg: Config, series: str, rule: FindReplace, target: Target, chapters: Sequence[str] | None = None
) -> list[Change]:
    """What `rule` would change in the English lines (or source texts) of `chapters` (default: every chapter),
    in chapter and reading order. ValueError for an empty search or a bad regular expression."""
    paths = SeriesPaths.from_config(cfg, series)
    names = list(chapters) if chapters is not None else paths.chapters()
    return [change for name in names for change in plan(paths.chapter(name), rule, target)]


def apply_replace(cfg: Config, series: str, changes: Sequence[Change], target: Target) -> int:
    """Record `changes` as hand edits (each chapter's are one undo step); how many lines changed. ValueError for a
    broken series.toml."""
    paths = SeriesPaths.from_config(cfg, series)
    try:
        direction = series_config(cfg, paths.library_dir).detect.reading_direction
    except SeriesConfigError as exc:
        raise ValueError(str(exc)) from exc
    by_chapter: dict[str, list[Change]] = {}
    for change in changes:
        by_chapter.setdefault(change.chapter, []).append(change)
    for chapter, group in by_chapter.items():
        apply_changes(paths.chapter(chapter), group, target, direction=direction)
    return len(changes)


def consistency(cfg: Config, series: str) -> Consistency:
    """The series' consistency report over every chapter's current lines."""
    paths = SeriesPaths.from_config(cfg, series)
    lines = series_lines(paths)
    return Consistency(divergences=divergences(lines), misses=term_misses(lines, glossary(paths)))
