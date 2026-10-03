"""One chapter open in the desktop Translator Studio, on the edits.json layer (edits/store.py).

The session holds the Studio's changes in memory until `save`, which records each one through the same edit
operations as every other editing tool: the change goes into edits.json and is applied to ocr.json / final.json
at once, so a pipeline re-run keeps it, even one that renumbers the regions. Learning (learn/) reads the
corrections back from edits.json. "Machine" is the judge's own line (final_auto.json, else final.json before any
hand edit); "English" is what will be lettered.

A chapter that has been detected but not OCR'd yet (regions.json only) can be edited too: the first save starts
its ocr.json from regions.json, and the OCR stage re-applies the edits over its own reading when it runs.
"""

from __future__ import annotations

import importlib
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from omniscan.core.config import get_config, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    Artifact,
    BBox,
    FinalArtifact,
    LayoutArtifact,
    Region,
    RegionKind,
    RegionsArtifact,
)
from omniscan.edits import store
from omniscan.edits.apply import match_layout_edits

LayoutFields = dict[
    str, object
]  # LayoutEdit's style fields (font, size_px, color, …) as the Studio sets them


def _load[T: Artifact](cls: type[T], path: Path) -> T | None:
    """An artifact, or None when it is missing or unreadable (the Studio then shows what it can)."""
    try:
        return cls.load(path)
    except OSError, ValueError:
        return None


class Issue(Protocol):
    """A QA finding as the Studio shows it (omniscan.studio.qa.Issue has this shape)."""

    region_id: str
    kind: str
    message: str
    word: str  # the unknown word of a "typo" issue, "" otherwise


@dataclass(frozen=True, slots=True)
class StudioRow:
    """One region as the Studio's table shows it."""

    region_id: str
    page: int  # slice index
    kind: str
    source: str
    machine: str  # the judge's own line, "" before translation
    english: str  # what will be lettered: the hand-written line if there is one, else the machine's
    edited: bool  # the region or its English was changed by hand (saved or not)
    speaker: str = ""  # who says the line (translate/voices.py), "" when nobody is set
    status: str = "todo"  # the line's review state: todo, edited or checked (edits.store.line_statuses)
    lettered: bool = False  # the lettering was set by hand (saved or not)


class StudioSession:
    """Edits of one chapter, held in memory until `save`.

    `direction` (the reading direction, used when an edit re-sorts regions) defaults to the series'
    `detect.reading_direction` (series.toml applied), read when saving."""

    def __init__(self, paths: ChapterPaths, *, direction: store.Direction | None = None) -> None:
        self.paths = paths
        self._direction: store.Direction | None = direction
        self._reload()

    def _reload(self) -> None:
        """Read the chapter's artifacts and drop every unsaved change."""
        ocr_path = self.paths.artifact("ocr.json")
        regions = _load(
            RegionsArtifact, ocr_path if ocr_path.is_file() else self.paths.artifact("regions.json")
        )
        self._all: list[Region] = regions.regions if regions is not None else []
        self._ids = {region.id for region in self._all}
        final = _load(FinalArtifact, self.paths.artifact("final.json"))
        self._final = final.lines if final is not None else []
        judged = _load(FinalArtifact, self.paths.artifact(store.FINAL_AUTO_FILE))
        self._machine = {
            line.region_id: line.text for line in (judged.lines if judged is not None else self._final)
        }
        self._current = {line.region_id: line.text for line in self._final}  # hand lines applied
        layout = _load(LayoutArtifact, self.paths.artifact("layout.json"))
        self._layout = layout.items if layout is not None else []
        # an unreadable ocr.json loads as no regions (like _load), so there is nothing edited to look up
        readable = ocr_path.is_file() and regions is not None
        edited, hand = store.edited_ids(self.paths) if readable else ([], [])
        self._edited, self._hand = set(edited) | set(hand), set(hand)
        self._status: dict[str, store.LineStatus] = (
            store.line_statuses(self.paths, self._edited) if readable else {}
        )
        self._saved_layouts: dict[str, LayoutFields] = self._read_layouts() if readable else {}
        self._sources: dict[str, str] = {}
        self._boxes: dict[str, BBox] = {}
        self._kinds: dict[str, RegionKind] = {}
        self._layouts: dict[str, LayoutFields | None] = {}  # None: back to the typesetter's lettering
        self._speakers: dict[str, str] = {}  # "": no speaker
        self._english: dict[str, str | None] = {}  # None: back to the judge's line
        self._removed: list[str] = []
        self._checks: dict[str, bool] = {}  # True: mark the line checked on save, False: unmark it
        # (only the ones that change something are saved: `_pending_checks`)

    @property
    def has_regions(self) -> bool:
        """Whether detection has run for this chapter (there is something to edit)."""
        return bool(self._all)

    def _read_layouts(self) -> dict[str, LayoutFields]:
        """region id -> its saved hand lettering's style fields (edits.json's layout edits, matched to the
        current regions like the typeset stage matches them)."""
        try:
            edits = store.load_edits(self.paths)
        except OSError, ValueError:
            return {}
        claims = match_layout_edits(self._all, edits)
        return {
            region_id: edits.layout[index].model_dump(exclude={"region_id", "anchor"}, exclude_none=True)
            for index, region_id in claims.items()
        }

    @property
    def dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return bool(
            self._sources
            or self._speakers
            or self._english
            or self._removed
            or self._boxes
            or self._kinds
            or self._layouts
            or self._pending_checks()
        )

    def regions(self) -> list[Region]:
        """The chapter's regions with every edit applied (saved and unsaved), in reading order."""
        removed = set(self._removed)
        return [self._pending(region) for region in self._all if region.id not in removed]

    def _pending(self, region: Region) -> Region:
        """`region` with its unsaved source text, speaker, box and kind applied."""
        update: dict[str, object] = {}
        if region.id in self._sources:
            update["text"] = self._sources[region.id]
        if region.id in self._speakers:
            update["speaker"] = self._speakers[region.id] or None
        if region.id in self._boxes:
            update["bbox"] = self._boxes[region.id]
        if region.id in self._kinds:
            update["kind"] = self._kinds[region.id]
        return region.model_copy(update=update) if update else region

    def pages(self) -> list[int]:
        """The slice indices that hold a remaining region, ascending."""
        return sorted({region.slice_index for region in self.regions()})

    def counts(self) -> dict[str, int]:
        """How many lines are todo, edited and checked (as a save would leave them)."""
        counts = {"todo": 0, "edited": 0, "checked": 0}
        for row in self.rows():
            counts[row.status] += 1
        return counts

    def machine_line(self, region_id: str) -> str:
        """The judge's own English line for a region ("" before translation)."""
        return self._machine.get(region_id, "")

    def translations(self) -> dict[str, str]:
        """The effective English per region: the judge's lines with the hand-written ones on top."""
        lines = dict(self._current)
        for region_id, text in self._english.items():
            if text is not None:
                lines[region_id] = text
            elif region_id in self._machine:
                lines[region_id] = self._machine[region_id]
            else:
                lines.pop(region_id, None)
        return lines

    def rows(self) -> list[StudioRow]:
        """One row per remaining region."""
        english = self.translations()
        pending = self._pending_checks()
        rows: list[StudioRow] = []
        for region in self.regions():
            unsaved = self._unsaved(region.id)
            rows.append(
                StudioRow(
                    region_id=region.id,
                    page=region.slice_index,
                    kind=region.kind,
                    source=region.text,
                    machine=self._machine.get(region.id, ""),
                    english=english.get(region.id, ""),
                    edited=region.id in self._edited or unsaved,
                    speaker=region.speaker or "",
                    status=self._row_status(region.id, unsaved, pending),
                    lettered=self.layout_of(region.id) is not None,
                )
            )
        return rows

    def _row_status(self, region_id: str, unsaved: bool, pending: Mapping[str, bool]) -> str:
        """A row's review state as a save would leave it: a pending check wins; a saved check holds unless the
        line is unchecked or its source text or English changes (a new speaker keeps it); else the line
        is edited when it carries a saved or unsaved hand edit, or todo."""
        check = pending.get(region_id)
        if check:
            return "checked"
        if check is None and self._status.get(region_id) == "checked" and not self._rewrites(region_id):
            return "checked"
        return "edited" if region_id in self._edited or unsaved else "todo"

    def issues(self) -> list[Issue]:
        """The automatic QA pass over the edited chapter (omniscan.studio.qa, the desktop Studio's checks),
        typos included (qa/typos.py: the series' glossary, character names and "not a typo" words are known)."""
        from omniscan.qa import typos  # the dictionary and the translate helpers load on the first check only

        qa = importlib.import_module("omniscan.studio.qa")
        checker = typos.checker_for(self._series())
        return qa.check_chapter(self.regions(), self.translations(), self._final, self._layout, checker)

    def allow_word(self, word: str) -> list[str]:
        """Mark a word "not a typo" for the whole series (saved at once); returns the series' list.
        ValueError for anything but a single word."""
        from omniscan.qa import typos

        return typos.allow_word(self._series(), word)

    def _series(self) -> SeriesPaths:
        """The paths of the chapter's series (its library, work and output folders)."""
        return SeriesPaths(
            series=self.paths.series,
            library_dir=self.paths.raw_dir.parent,
            work_dir=self.paths.work_dir.parent,
            output_dir=self.paths.output_dir.parent,
        )

    def set_source(self, region_id: str, text: str) -> None:
        """Correct a region's source (OCR) text; a re-run OCR re-applies the fix over its own reading."""
        self._require(region_id)
        saved = next(region.text for region in self._all if region.id == region_id)
        if text == saved:
            self._sources.pop(region_id, None)
        else:
            self._sources[region_id] = text

    def set_speaker(self, region_id: str, name: str) -> None:
        """Say who speaks a region's line (a character of the series' voices.toml, or any name; "" for nobody)."""
        self._require(region_id)
        saved = next(region.speaker or "" for region in self._all if region.id == region_id)
        if name.strip() == saved:
            self._speakers.pop(region_id, None)
        else:
            self._speakers[region_id] = name.strip()

    def set_bbox(self, region_id: str, bbox: BBox) -> None:
        """Move or resize a region's text box (strip space); the OCR lines become one line of this box."""
        self._require(region_id)
        saved = next(region.bbox for region in self._all if region.id == region_id)
        if bbox == saved:
            self._boxes.pop(region_id, None)
        else:
            self._boxes[region_id] = bbox

    def set_kind(self, region_id: str, kind: RegionKind) -> None:
        """Make a region bubble text, free text, a sound effect or a watermark."""
        self._require(region_id)
        saved = next(region.kind for region in self._all if region.id == region_id)
        if kind == saved:
            self._kinds.pop(region_id, None)
        else:
            self._kinds[region_id] = kind

    def layout_of(self, region_id: str) -> LayoutFields | None:
        """A region's hand-set lettering (style fields, saved and unsaved merged), None when the typesetter
        letters it."""
        if region_id in self._layouts:
            return self._layouts[region_id]
        return self._saved_layouts.get(region_id)

    def set_layout(self, region_id: str, fields: LayoutFields) -> None:
        """Set some of a region's lettering by hand (LayoutEdit's style fields: font, size_px, color, stroke_px,
        stroke_color, align, angle, box, lines, hidden); fields not given keep their hand-set value, so one
        dialog can restyle many regions at once. A field set to None goes back to the typesetter's choice."""
        self._require(region_id)
        merged = dict(self.layout_of(region_id) or {})
        for key, value in fields.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        if merged == self._saved_layouts.get(region_id):
            self._layouts.pop(region_id, None)
        else:
            self._layouts[region_id] = merged

    def revert_layout(self, region_id: str) -> None:
        """Give a region back to the typesetter (drops its hand lettering on save)."""
        self._require(region_id)
        if region_id in self._saved_layouts:
            self._layouts[region_id] = None
        else:
            self._layouts.pop(region_id, None)

    def add_region(self, bbox: BBox, kind: RegionKind = "bubble_text", text: str = "") -> str:
        """Draw a new region (text the detector missed); unsaved changes are saved first and the new region
        is written at once, as its own undo step. Returns its id (m0001, …)."""
        self.save()
        direction = self._direction if self._direction is not None else self._series_direction()
        self._start_ocr_json()
        region = store.add_region(self.paths, bbox, direction=direction, kind=kind, text=text)
        self._reload()
        return region.id

    def set_cuts(self, cuts: list[int] | None, *, max_height: int | None = None) -> list[int] | None:
        """Set where the exported images split (strip rows; None or none at all: one image per slice); unsaved
        changes are saved first and the cuts are written at once, as their own undo step. Returns what was stored;
        ValueError for a cut outside the strip or an image taller than `max_height` rows."""
        self.save()
        try:
            return store.set_cuts(self.paths, cuts, max_height=max_height)
        finally:
            self._reload()

    def set_translation(self, region_id: str, text: str) -> None:
        """Set a region's English line; setting it back to the machine's line clears the hand-written one."""
        self._require(region_id)
        wanted = None if text == self._machine.get(region_id, "") else text
        saved = self._current.get(region_id) if region_id in self._hand else None
        if wanted == saved:
            self._english.pop(region_id, None)
        else:
            self._english[region_id] = wanted

    def set_checked(self, region_id: str, checked: bool = True) -> None:
        """Mark a region's line checked (its source and English as they are after this save) or unchecked;
        KeyError for an unknown region or one removed in this session."""
        self._require(region_id)
        if region_id in self._removed:
            raise KeyError(f"region {region_id!r} is removed")
        self._checks[region_id] = checked

    def _unsaved(self, region_id: str) -> bool:
        """Whether the region has an unsaved source text, speaker, box, kind or English line."""
        return (
            region_id in self._sources
            or region_id in self._speakers
            or region_id in self._boxes
            or region_id in self._kinds
            or region_id in self._english
        )

    def _rewrites(self, region_id: str) -> bool:
        """Whether the region has an unsaved source text or English line (which a saved check does not approve)."""
        return region_id in self._sources or region_id in self._english

    def _pending_checks(self) -> dict[str, bool]:
        """The checks and unchecks a save makes: checking a line whose saved check still approves it, or
        unchecking one without a check, changes nothing and is left out."""
        return {
            region_id: checked
            for region_id, checked in self._checks.items()
            if (self._status.get(region_id) == "checked") != checked
            or (checked and self._rewrites(region_id))
        }

    def remove_region(self, region_id: str) -> None:
        """Drop a region (a false detection: nothing is erased or lettered there)."""
        self._require(region_id)
        if region_id not in self._removed:
            self._removed.append(region_id)
        self._sources.pop(region_id, None)
        self._speakers.pop(region_id, None)
        self._boxes.pop(region_id, None)
        self._kinds.pop(region_id, None)
        self._layouts.pop(region_id, None)
        self._english.pop(region_id, None)
        self._checks.pop(region_id, None)

    def save(self) -> int:
        """Record every change in edits.json (applied to ocr.json / final.json at once) as one undo step
        (`undo`); returns how many."""
        if not self.dirty:
            return 0
        direction: store.Direction = (
            self._direction if self._direction is not None else self._series_direction()
        )
        self._start_ocr_json()
        with store.edit_group(self.paths):  # one save is one undo step
            for region_id in {*self._sources, *self._speakers, *self._boxes, *self._kinds}:
                store.update_region(
                    self.paths,
                    region_id,
                    direction=direction,
                    kind=self._kinds.get(region_id),
                    bbox=self._boxes.get(region_id),
                    text=self._sources.get(region_id),
                    speaker=self._speakers.get(region_id),
                )
            for region_id, fields in self._layouts.items():
                if fields is None:
                    store.revert_layout(self.paths, region_id)
                else:
                    store.set_layout(self.paths, region_id, dict(fields))
            for region_id, line in self._english.items():
                if line is None:
                    store.revert_translation(self.paths, region_id, direction=direction)
                else:
                    store.set_translation(self.paths, region_id, line, direction=direction)
            for region_id in self._removed:
                store.delete_region(self.paths, region_id, direction=direction)
            pending = self._pending_checks()
            for checked in (True, False):  # after the text edits: a check approves the saved lines
                ids = [region_id for region_id, value in pending.items() if value is checked]
                if ids:
                    store.set_checked(self.paths, ids, checked=checked)
        count = (
            len(self._sources)
            + len(self._speakers)
            + len(self._boxes)
            + len(self._kinds)
            + len(self._layouts)
            + len(self._english)
            + len(self._removed)
            + len(pending)
        )
        self._reload()
        return count

    def undo(self) -> bool:
        """Take back the last saved change of the chapter (a whole save is one step; unsaved changes are
        dropped); False when there is nothing to undo."""
        return self._history_step(store.undo)

    def redo(self) -> bool:
        """Make the last undone change again (unsaved changes are dropped); False when there is nothing to redo."""
        return self._history_step(store.redo)

    def _history_step(self, step: Callable[..., object]) -> bool:
        """Run `store.undo` or `store.redo`, then read the chapter again."""
        direction = self._direction if self._direction is not None else self._series_direction()
        try:
            step(self.paths, direction=direction)
        except store.EditNotFoundError:
            return False
        finally:
            self._reload()
        return True

    def _series_direction(self) -> store.Direction:
        """The series' reading direction (series.toml applied)."""
        return series_config(get_config(), self.paths.raw_dir.parent).detect.reading_direction

    def _start_ocr_json(self) -> None:
        """Before the OCR ran, the edits start from the detected regions (regions.json)."""
        ocr_path, detected = self.paths.artifact("ocr.json"), self.paths.artifact("regions.json")
        if not ocr_path.is_file() and detected.is_file():
            shutil.copyfile(detected, ocr_path)

    def _require(self, region_id: str) -> None:
        """KeyError when the chapter has no region `region_id`."""
        if region_id not in self._ids:
            raise KeyError(f"no region {region_id!r} in {self.paths.chapter}")
