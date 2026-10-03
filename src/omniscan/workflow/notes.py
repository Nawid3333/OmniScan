"""Notes on regions: notes.json in the chapter work folder (#38).

A proofreader questions a line, a cleaner flags a spot for the typesetter: a note sits on a region without
changing it, until someone marks it resolved. A note follows its region like an edit does (the same id still
overlapping the box it was written on, else the region overlapping that box best), so it survives a re-run that
renumbers the regions. The file travels in chapter project files with the rest of the work folder; names
(`[user] name`) are recorded only when set and never go into a contribution.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import NotesArtifact, Region, RegionNote, utcnow
from omniscan.edits.apply import match_region
from omniscan.edits.store import current_regions

NOTES_FILE = "notes.json"
MAX_NOTE_CHARS = 2000

_LOCK = threading.Lock()


def load_notes(paths: ChapterPaths) -> list[RegionNote]:
    """The chapter's notes, oldest first ([] without a notes file); ValueError when the file is damaged."""
    path = paths.artifact(NOTES_FILE)
    return NotesArtifact.load(path).notes if path.is_file() else []


def add_note(paths: ChapterPaths, region_id: str, text: str, *, by: str | None = None) -> RegionNote:
    """Add a note to a current region; KeyError for an unknown region, ValueError for an empty or too long text."""
    body = text.strip()
    if not body:
        raise ValueError("a note needs some text")
    if len(body) > MAX_NOTE_CHARS:
        raise ValueError(f"a note holds at most {MAX_NOTE_CHARS} characters")
    region = next((r for r in current_regions(paths) if r.id == region_id), None)
    if region is None:
        raise KeyError(region_id)
    with _LOCK:
        notes = load_notes(paths)
        number = 1 + max((int(note.id[1:]) for note in notes if note.id[1:].isdigit()), default=0)
        note = RegionNote(
            id=f"n{number:04d}",
            region_id=region.id,
            anchor=region.bbox,
            text=body,
            by=by or None,
            at=utcnow(),
        )
        NotesArtifact(notes=[*notes, note]).save(paths.artifact(NOTES_FILE))
        return note


def resolve_note(paths: ChapterPaths, note_id: str, *, resolved: bool = True) -> RegionNote:
    """Mark a note resolved (or open again); KeyError for an unknown note."""
    with _LOCK:
        notes = load_notes(paths)
        index = next((i for i, note in enumerate(notes) if note.id == note_id), None)
        if index is None:
            raise KeyError(note_id)
        notes[index] = notes[index].model_copy(update={"resolved": resolved})
        NotesArtifact(notes=notes).save(paths.artifact(NOTES_FILE))
        return notes[index]


def notes_by_region(
    paths: ChapterPaths, regions: Sequence[Region] | None = None
) -> dict[str, list[RegionNote]]:
    """The notes of each current region (region id -> its notes, oldest first); a note whose region is gone is
    left out."""
    current = list(regions) if regions is not None else current_regions(paths)
    found: dict[str, list[RegionNote]] = {}
    for note in load_notes(paths):
        region = match_region(note.region_id, [note.anchor], current)
        if region is not None:
            found.setdefault(region.id, []).append(note)
    return found
