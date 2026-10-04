"""Where a chapter stands in a group's workflow: chapter_status.json in its work folder (#38).

A group passes a chapter along: translated → proofread → cleaned → lettered → QC passed. Each step is marked done
(or undone) by whoever did it, and the chapter is handed over to the next person; every change is kept as an event
with the name of whoever made it (`[user] name`, when set). The file lives in the chapter's work folder, so a
chapter project file (`omniscan project pack`) carries it to the next person; the pipeline never reads it.
"""

from __future__ import annotations

import threading
from typing import Literal

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import WORKFLOW_STEPS, ChapterStatus, StatusEvent, WorkflowStep, utcnow

STATUS_FILE = "chapter_status.json"
Role = Literal["translator", "proofreader", "cleaner", "typesetter", "qc"]
ROLES: tuple[Role, ...] = ("translator", "proofreader", "cleaner", "typesetter", "qc")
ROLE_STEP: dict[Role, WorkflowStep] = {
    "translator": "translated",
    "proofreader": "proofread",
    "cleaner": "cleaned",
    "typesetter": "lettered",
    "qc": "qc_passed",
}
STEP_LABELS: dict[WorkflowStep, str] = {
    "translated": "translated",
    "proofread": "proofread",
    "cleaned": "cleaned",
    "lettered": "lettered",
    "qc_passed": "QC passed",
}

_LOCK = threading.Lock()  # one read-modify-write of a status file at a time in this process


def load_status(paths: ChapterPaths) -> ChapterStatus:
    """The chapter's status (nothing done, nobody holding it when the file does not exist); ValueError when the
    file is damaged."""
    path = paths.artifact(STATUS_FILE)
    return ChapterStatus.load(path) if path.is_file() else ChapterStatus()


def mark_step(
    paths: ChapterPaths,
    step: WorkflowStep,
    *,
    done: bool = True,
    by: str | None = None,
    note: str | None = None,
) -> ChapterStatus:
    """Mark a step done (or not done any more) and record who did it; returns the new status. Marking a step that
    already is so changes nothing."""
    if step not in WORKFLOW_STEPS:
        raise ValueError(f"unknown step {step!r} (one of {', '.join(WORKFLOW_STEPS)})")
    with _LOCK:
        status = load_status(paths)
        if (step in status.done) == done:
            return status
        steps = {*status.done, step} if done else set(status.done) - {step}
        event = StatusEvent(
            kind="done" if done else "undone", step=step, by=by or None, note=note or None, at=utcnow()
        )
        status = status.model_copy(
            update={"done": [s for s in WORKFLOW_STEPS if s in steps], "events": [*status.events, event]}
        )
        status.save(paths.artifact(STATUS_FILE))
        return status


def hand_over(
    paths: ChapterPaths, to: str | None, *, by: str | None = None, note: str | None = None
) -> ChapterStatus:
    """Hand the chapter to `to` (None or "": to nobody in particular) and record it; returns the new status."""
    with _LOCK:
        status = load_status(paths)
        event = StatusEvent(kind="handed", to=to or None, by=by or None, note=note or None, at=utcnow())
        status = status.model_copy(update={"holder": to or None, "events": [*status.events, event]})
        status.save(paths.artifact(STATUS_FILE))
        return status


def next_step(status: ChapterStatus) -> WorkflowStep | None:
    """The first step not done yet, or None when every step is."""
    return next((step for step in WORKFLOW_STEPS if step not in status.done), None)


def describe(status: ChapterStatus) -> str:
    """One short line: the last step done in order, how many of the five, and who has the chapter."""
    done: list[WorkflowStep] = [step for step in WORKFLOW_STEPS if step in status.done]
    text = f"{STEP_LABELS[done[-1]]} ({len(done)}/{len(WORKFLOW_STEPS)})" if done else "not started"
    return text + (f" · with {status.holder}" if status.holder else "")
