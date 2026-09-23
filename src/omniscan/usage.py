"""Reading LLM usage counters back out of the work tree and summing them (`omniscan usage`).

A read-only report over the artifacts the text stages already wrote: every `translations/*.json`
run (its `CandidateRun.usage`) and every chapter's `final.json` (the judge's `FinalArtifact.usage`).
Artifacts written before usage recording existed simply carry no counters and count as zeros.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omniscan.core.config import Config
from omniscan.core.paths import natural_key
from omniscan.core.schemas import CandidateRun, FinalArtifact


@dataclass(frozen=True, slots=True)
class UsageRow:
    """One chapter artifact's LLM usage: one translation run or the judge's final.json."""

    series: str
    chapter: str
    source: str  # the run's profile name, or "judge" for final.json
    model: str
    usage: dict[str, float]  # the raw usage dict exactly as stored in the artifact


def collect_usage(cfg: Config, series: str | None = None) -> list[UsageRow]:
    """One row per chapter artifact carrying usage (translation runs then judge), in reading order."""
    root = cfg.paths.work_root
    names = [series] if series is not None else _series_names(root)
    rows: list[UsageRow] = []
    for name in names:
        series_dir = root / name
        if not series_dir.is_dir():
            continue
        for chapter_dir in _chapter_dirs(series_dir):
            rows.extend(_chapter_rows(name, chapter_dir))
    return rows


def totals(rows: Sequence[UsageRow]) -> dict[str, float]:
    """Element-wise sum of every usage key present in any row (generic, not key-set-limited)."""
    sums: dict[str, float] = {}
    for row in rows:
        for key, value in row.usage.items():
            sums[key] = sums.get(key, 0.0) + value
    return sums


def _series_names(root: Path) -> list[str]:
    """Series directory names under `root` in name order, skipping the '_'/'.-prefixed ones."""
    if not root.is_dir():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]
    return sorted(names, key=lambda name: (name.casefold(), name))


def _chapter_dirs(series_dir: Path) -> list[Path]:
    """Chapter directories of a series in reading order, skipping the '_'/'.-prefixed ones."""
    dirs = [p for p in series_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]
    return sorted(dirs, key=lambda p: natural_key(p.name))


def _chapter_rows(series: str, chapter_dir: Path) -> list[UsageRow]:
    """The usage rows of one chapter: every `translations/*.json` run first, then `final.json`."""
    rows: list[UsageRow] = []
    translations_dir = chapter_dir / "translations"
    if translations_dir.is_dir():
        for path in sorted(translations_dir.glob("*.json")):
            if path.name.startswith("."):
                continue  # dot-prefixed partial files are never candidate runs
            try:
                run = CandidateRun.load(path)
            except Exception:
                continue  # a corrupt file is skipped: `usage` is a read-only report, never a crash
            rows.append(
                UsageRow(
                    series=series,
                    chapter=chapter_dir.name,
                    source=run.profile,
                    model=run.model,
                    usage=run.usage,
                )
            )
    final_path = chapter_dir / "final.json"
    if final_path.is_file():
        try:
            final = FinalArtifact.load(final_path)
        except Exception:
            final = None  # same as above: a corrupt final.json is skipped, never a crash
        if final is not None:
            rows.append(
                UsageRow(
                    series=series,
                    chapter=chapter_dir.name,
                    source="judge",
                    model=final.judge_model,
                    usage=final.usage,
                )
            )
    return rows
