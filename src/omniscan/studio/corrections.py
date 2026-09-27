"""The local log of every manual correction (work/<series>/<chapter>/corrections.jsonl).

One JSON object per line, append-only, never uploaded by itself: it is what an opt-in contribution (docs/ROADMAP.md
X4) would later be built from, and a per-series memory of how the translator fixes the machine's output.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

CORRECTIONS_NAME = "corrections.jsonl"
CorrectionField = Literal["source", "translation", "removed"]


@dataclass(frozen=True, slots=True)
class Correction:
    """One manual change: what the machine had, what the person made of it, and the line's context."""

    series: str
    chapter: str
    region_id: str
    field: CorrectionField
    before: str
    after: str
    source_text: str  # the region's source text after the change (context for a translation fix)
    kind: str  # region kind: bubble_text | free_text | sfx | watermark
    at: str  # ISO-8601 UTC time of the save


def now_iso() -> str:
    """The current UTC time as ISO-8601 (seconds)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def append_corrections(path: Path, corrections: Iterable[Correction]) -> int:
    """Append corrections to the log (created on first use); returns how many were written."""
    lines = [json.dumps(asdict(item), ensure_ascii=False) for item in corrections]
    if not lines:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def load_corrections(path: Path) -> list[Correction]:
    """Every correction in the log, oldest first; unreadable lines are skipped."""
    if not path.is_file():
        return []
    items: list[Correction] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            items.append(Correction(**json.loads(line)))
        except ValueError, TypeError:
            continue
    return items
