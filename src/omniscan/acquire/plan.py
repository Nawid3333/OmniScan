"""The per-batch acquire plan (resume state + credit estimate) and the per-chapter acceptance record."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from omniscan.acquire.completeness import Finding
from omniscan.acquire.extractor import Mode, credit_cost
from omniscan.acquire.sources import ChapterSource

ACCEPTED_NAME = "accepted.json"
_DONE_STATUSES = frozenset({"ok", "review"})


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One chapter in an acquire plan: its source, resume state and credit cost."""

    source: ChapterSource
    state: Literal["todo", "done"]
    credits: int


@dataclass(frozen=True, slots=True)
class AcquirePlan:
    """The whole batch: items in order, the extraction mode, total credits and todo/done counts."""

    items: tuple[PlanItem, ...]
    mode: Mode
    credits_total: int
    todo: int
    done: int


def chapter_state(chapter_dir: Path) -> Literal["todo", "done"]:
    """'done' when the chapter has an accepted.json with status ok/review; else 'todo'."""
    record = read_accepted(chapter_dir)
    return "done" if record is not None and record.get("status") in _DONE_STATUSES else "todo"


def build_plan(
    sources: Sequence[ChapterSource], series_dir: Path, *, mode: Mode = "basic", force: bool = False
) -> AcquirePlan:
    """One plan item per source; done chapters cost 0 credits unless force re-runs them."""
    items: list[PlanItem] = []
    for source in sources:
        state = "todo" if force else chapter_state(series_dir / source.name)
        items.append(
            PlanItem(source=source, state=state, credits=0 if state == "done" else credit_cost(1, mode))
        )
    todo = sum(1 for item in items if item.state == "todo")
    return AcquirePlan(
        items=tuple(items),
        mode=mode,
        credits_total=sum(item.credits for item in items),
        todo=todo,
        done=len(items) - todo,
    )


def write_accepted(
    chapter_dir: Path,
    status: Literal["ok", "review"],
    findings: Sequence[Finding],
    pages: int,
    *,
    source_url: str,
) -> None:
    """Atomically record a chapter's acceptance verdict in accepted.json (tmp file, then replace)."""
    record = {
        "status": status,
        "pages": pages,
        "source_url": source_url,
        "accepted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "findings": [
            {"level": finding.level, "code": finding.code, "message": finding.message} for finding in findings
        ],
    }
    tmp = chapter_dir / f"{ACCEPTED_NAME}.tmp"
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(chapter_dir / ACCEPTED_NAME)


def read_accepted(chapter_dir: Path) -> dict[str, Any] | None:
    """The parsed accepted.json of a chapter; None when missing, corrupt or not a JSON object."""
    try:
        record = json.loads((chapter_dir / ACCEPTED_NAME).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return record if isinstance(record, dict) else None


def to_json(plan: AcquirePlan) -> dict[str, Any]:
    """The GUI-facing JSON of a plan: mode, credit total, counts and one row per item."""
    return {
        "mode": plan.mode,
        "credits_total": plan.credits_total,
        "todo": plan.todo,
        "done": plan.done,
        "items": [
            {"name": item.source.name, "url": item.source.url, "state": item.state, "credits": item.credits}
            for item in plan.items
        ],
    }
