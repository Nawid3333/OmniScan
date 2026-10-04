"""What each role has left on a chapter (#38): the lines a translator, proofreader, cleaner, typesetter or quality
check still has to look at, from the chapter's own artifacts.

- translator: translatable lines with no English yet;
- proofreader: lines with English that nobody checked (`edits.store.line_statuses`);
- cleaner: regions whose original text or watermark still shows on the finished page (the last `qa` re-read);
- typesetter: lettering that does not fit its balloon (layout.json);
- qc: every finished-page issue and every open note.

Open notes on a region show for every role. Nothing is computed that a stage has not already written: a chapter
not lettered yet has no typesetter items, one never re-read by `qa` no cleaner items.
"""

from __future__ import annotations

from dataclasses import dataclass

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import LayoutArtifact
from omniscan.edits.store import current_regions, final_lines, line_statuses
from omniscan.qa.leftover import load_issues
from omniscan.translate.prompts import translatable
from omniscan.workflow.notes import notes_by_region
from omniscan.workflow.status import Role


@dataclass(frozen=True, slots=True)
class Todo:
    """One thing a role still has to do: the region and why."""

    region_id: str
    page: int  # the region's slice index
    what: str


def todo(paths: ChapterPaths, role: Role) -> list[Todo]:
    """What `role` has left on the chapter, in reading order (one entry per reason)."""
    regions = translatable(current_regions(paths))
    page = {region.id: region.slice_index for region in regions}
    english = final_lines(paths)
    items: list[Todo] = []
    if role == "translator":
        items = [
            Todo(r.id, r.slice_index, "no translation yet")
            for r in regions
            if not english.get(r.id, "").strip()
        ]
    elif role == "proofreader":
        statuses = line_statuses(paths)
        items = [
            Todo(r.id, r.slice_index, f"not checked ({statuses.get(r.id, 'todo')})")
            for r in regions
            if english.get(r.id, "").strip() and statuses.get(r.id) != "checked"
        ]
    elif role in ("cleaner", "qc"):
        items = [
            Todo(issue.region_id, page.get(issue.region_id, -1), issue.message)
            for issue in load_issues(paths)
            if role == "qc" or issue.kind in ("source_left", "watermark_left")
        ]
    if role in ("typesetter", "qc"):
        layout = paths.artifact("layout.json")
        if layout.is_file():
            items += [
                Todo(item.region_id, page.get(item.region_id, -1), "lettering does not fit")
                for item in LayoutArtifact.load(layout).items
                if item.overflow
            ]
    for region_id, notes in notes_by_region(paths).items():
        items += [
            Todo(region_id, page.get(region_id, -1), f"note {note.id}: {note.text}")
            for note in notes
            if not note.resolved
        ]
    order = {region.id: index for index, region in enumerate(regions)}
    return sorted(items, key=lambda item: (order.get(item.region_id, len(order)), item.what))
