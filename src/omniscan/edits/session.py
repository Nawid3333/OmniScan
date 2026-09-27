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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from omniscan.core.config import get_config, series_config
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import Artifact, FinalArtifact, LayoutArtifact, Region, RegionsArtifact
from omniscan.edits import store


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
        edited, hand = store.edited_ids(self.paths) if ocr_path.is_file() else ([], [])
        self._edited, self._hand = set(edited) | set(hand), set(hand)
        self._sources: dict[str, str] = {}
        self._speakers: dict[str, str] = {}  # "": no speaker
        self._english: dict[str, str | None] = {}  # None: back to the judge's line
        self._removed: list[str] = []

    @property
    def has_regions(self) -> bool:
        """Whether detection has run for this chapter (there is something to edit)."""
        return bool(self._all)

    @property
    def dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return bool(self._sources or self._speakers or self._english or self._removed)

    def regions(self) -> list[Region]:
        """The chapter's regions with every edit applied (saved and unsaved), in reading order."""
        removed = set(self._removed)
        return [self._pending(region) for region in self._all if region.id not in removed]

    def _pending(self, region: Region) -> Region:
        """`region` with its unsaved source text and speaker applied."""
        update: dict[str, object] = {}
        if region.id in self._sources:
            update["text"] = self._sources[region.id]
        if region.id in self._speakers:
            update["speaker"] = self._speakers[region.id] or None
        return region.model_copy(update=update) if update else region

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
        return [
            StudioRow(
                region_id=region.id,
                page=region.slice_index,
                kind=region.kind,
                source=region.text,
                machine=self._machine.get(region.id, ""),
                english=english.get(region.id, ""),
                edited=region.id in self._edited
                or region.id in self._sources
                or region.id in self._speakers
                or region.id in self._english,
                speaker=region.speaker or "",
            )
            for region in self.regions()
        ]

    def issues(self) -> list[Issue]:
        """The automatic QA pass over the edited chapter (omniscan.studio.qa, the desktop Studio's checks)."""
        qa = importlib.import_module("omniscan.studio.qa")
        return qa.check_chapter(self.regions(), self.translations(), self._final, self._layout)

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

    def set_translation(self, region_id: str, text: str) -> None:
        """Set a region's English line; setting it back to the machine's line clears the hand-written one."""
        self._require(region_id)
        wanted = None if text == self._machine.get(region_id, "") else text
        saved = self._current.get(region_id) if region_id in self._hand else None
        if wanted == saved:
            self._english.pop(region_id, None)
        else:
            self._english[region_id] = wanted

    def remove_region(self, region_id: str) -> None:
        """Drop a region (a false detection: nothing is erased or lettered there)."""
        self._require(region_id)
        if region_id not in self._removed:
            self._removed.append(region_id)
        self._sources.pop(region_id, None)
        self._speakers.pop(region_id, None)
        self._english.pop(region_id, None)

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
            for region_id in {*self._sources, *self._speakers}:
                store.update_region(
                    self.paths,
                    region_id,
                    direction=direction,
                    text=self._sources.get(region_id),
                    speaker=self._speakers.get(region_id),
                )
            for region_id, line in self._english.items():
                if line is None:
                    store.revert_translation(self.paths, region_id, direction=direction)
                else:
                    store.set_translation(self.paths, region_id, line, direction=direction)
            for region_id in self._removed:
                store.delete_region(self.paths, region_id, direction=direction)
        count = len(self._sources) + len(self._speakers) + len(self._english) + len(self._removed)
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
