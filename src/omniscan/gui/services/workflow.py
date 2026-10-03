"""Qt-free services behind the group workflow in the desktop app (#38): a chapter's steps and holder for the
Library, each role's to-do list and the region notes for the Studio. Names come from `[user] name`."""

from __future__ import annotations

from collections.abc import Sequence

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import RegionNote, WorkflowStep
from omniscan.workflow import notes, status
from omniscan.workflow.status import Role
from omniscan.workflow.todo import Todo, todo

ROLE_LABELS: dict[Role, str] = {
    "translator": "Translator",
    "proofreader": "Proofreader",
    "cleaner": "Cleaner",
    "typesetter": "Typesetter",
    "qc": "Quality check",
}


def _paths(cfg: Config, series: str, chapter: str) -> ChapterPaths:
    """The chapter's paths."""
    return SeriesPaths.from_config(cfg, series).chapter(chapter)


def chapter_workflow(cfg: Config, series: str, chapter: str) -> str:
    """The chapter's workflow in one line (steps done, who has it); the problem when its file is damaged."""
    try:
        return status.describe(status.load_status(_paths(cfg, series, chapter)))
    except ValueError as exc:
        return f"damaged chapter_status.json: {exc}"


def done_steps(cfg: Config, series: str, chapter: str) -> list[WorkflowStep]:
    """The chapter's steps marked done ([] when its status file is damaged)."""
    try:
        return status.load_status(_paths(cfg, series, chapter)).done
    except ValueError:
        return []


def mark_step(cfg: Config, series: str, chapter: str, step: WorkflowStep, done: bool) -> str:
    """Mark a step of the chapter done or not done, signed with `[user] name`; the new one-line status."""
    return status.describe(status.mark_step(_paths(cfg, series, chapter), step, done=done, by=cfg.user.name))


def hand_over(cfg: Config, series: str, chapter: str, to: str, note: str = "") -> str:
    """Hand the chapter to `to`, signed with `[user] name`; the new one-line status."""
    return status.describe(status.hand_over(_paths(cfg, series, chapter), to, by=cfg.user.name, note=note))


def role_todo(paths: ChapterPaths, role: Role) -> list[Todo]:
    """What `role` still has to look at in the chapter."""
    return todo(paths, role)


def open_notes(paths: ChapterPaths) -> dict[str, list[RegionNote]]:
    """Every current region's notes that nobody resolved yet ({} when the notes file is damaged)."""
    try:
        found = notes.notes_by_region(paths)
    except ValueError:
        return {}
    return {
        region_id: unresolved
        for region_id, region_notes in found.items()
        if (unresolved := [note for note in region_notes if not note.resolved])
    }


def add_region_note(cfg: Config, paths: ChapterPaths, region_id: str, text: str) -> RegionNote:
    """Leave a note on a region, signed with `[user] name` (KeyError: no such region; ValueError: no text)."""
    return notes.add_note(paths, region_id, text, by=cfg.user.name)


def resolve_region_notes(paths: ChapterPaths, region_ids: Sequence[str]) -> int:
    """Mark every open note of these regions resolved; returns how many."""
    count = 0
    for region_id in region_ids:
        for note in open_notes(paths).get(region_id, []):
            notes.resolve_note(paths, note.id)
            count += 1
    return count
