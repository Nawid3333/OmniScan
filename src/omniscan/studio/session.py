"""One chapter open in the Translator Studio: its regions, the machine's lines and the translator's edits.

Loads ocr.json (regions.json before OCR ran), final.json, layout.json and studio.json. Saving writes source fixes
and removed regions into ocr.json (so translate/inpaint/typeset see them on the next run), every manual line into
studio.json (typeset applies it on top of final.json) and one record per change into corrections.jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import Artifact, FinalArtifact, LayoutArtifact, Region, RegionsArtifact
from omniscan.studio.corrections import (
    CORRECTIONS_NAME,
    Correction,
    CorrectionField,
    append_corrections,
    now_iso,
)
from omniscan.studio.edits import STUDIO_NAME, apply_source_edits, load_edits
from omniscan.studio.qa import Issue, check_chapter


def _load[T: Artifact](cls: type[T], path: Path) -> T | None:
    """An artifact, or None when it is missing or unreadable (the Studio then shows what it can)."""
    try:
        return cls.load(path)
    except OSError, ValueError:
        return None


@dataclass(frozen=True, slots=True)
class StudioRow:
    """One region as the Studio's table shows it."""

    region_id: str
    page: int  # slice index
    kind: str
    source: str
    machine: str  # the judge's line (final.json), "" before translation
    english: str  # what will be lettered: the manual line if there is one, else the machine's
    edited: bool  # the source or the English was changed by hand


class StudioSession:
    """Edits of one chapter, held in memory until `save`."""

    def __init__(self, paths: ChapterPaths) -> None:
        self.paths = paths
        ocr_path = paths.artifact("ocr.json")
        self._regions_name = "ocr.json" if ocr_path.is_file() else "regions.json"
        regions = _load(RegionsArtifact, paths.artifact(self._regions_name))
        self._all: list[Region] = regions.regions if regions is not None else []
        final = _load(FinalArtifact, paths.artifact("final.json"))
        self._final = final.lines if final is not None else []
        layout = _load(LayoutArtifact, paths.artifact("layout.json"))
        self._layout = layout.items if layout is not None else []
        self._machine = {line.region_id: line.text for line in self._final}
        self._saved = load_edits(paths)
        self._edits = self._saved.model_copy(deep=True)
        self._original = {region.id: region.text for region in self._all}

    @property
    def has_regions(self) -> bool:
        """Whether detection has run for this chapter (there is something to edit)."""
        return bool(self._all)

    @property
    def dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return self._edits != self._saved

    def regions(self) -> list[Region]:
        """The chapter's regions with the edits applied, in reading order."""
        return apply_source_edits(self._all, self._edits)

    def translations(self) -> dict[str, str]:
        """The effective English per region: the machine's lines with the manual ones on top."""
        return self._machine | self._edits.translations

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
                edited=region.id in self._edits.sources or region.id in self._edits.translations,
            )
            for region in self.regions()
        ]

    def issues(self) -> list[Issue]:
        """The automatic QA pass over the edited chapter."""
        return check_chapter(self.regions(), self.translations(), self._final, self._layout)

    def set_source(self, region_id: str, text: str) -> None:
        """Correct a region's source (OCR) text; it is kept in studio.json so a re-run OCR does not undo it."""
        self._require(region_id)
        if text != self._edits.sources.get(region_id, self._original[region_id]):
            self._edits.sources = self._edits.sources | {region_id: text}

    def set_translation(self, region_id: str, text: str) -> None:
        """Set a region's English line; setting it back to the machine's line clears the edit."""
        self._require(region_id)
        translations = dict(self._edits.translations)
        if text == self._machine.get(region_id):
            translations.pop(region_id, None)
        else:
            translations[region_id] = text
        self._edits.translations = translations

    def remove_region(self, region_id: str) -> None:
        """Drop a region (a false detection): nothing is erased or lettered there."""
        self._require(region_id)
        if region_id not in self._edits.removed:
            self._edits.removed = [*self._edits.removed, region_id]

    def save(self) -> int:
        """Write ocr.json, studio.json and the corrections log; returns how many corrections were logged."""
        corrections = self._corrections()
        if not corrections:
            return 0
        RegionsArtifact(regions=self.regions()).save(self.paths.artifact(self._regions_name))
        self._edits.save(self.paths.artifact(STUDIO_NAME))
        append_corrections(self.paths.artifact(CORRECTIONS_NAME), corrections)
        self._saved = self._edits.model_copy(deep=True)
        return len(corrections)

    def _require(self, region_id: str) -> None:
        if region_id not in self._original:
            raise KeyError(f"no region {region_id!r} in {self.paths.chapter}")

    def _corrections(self) -> list[Correction]:
        at = now_iso()
        before, after = self._saved, self._edits
        saved_english = self._machine | before.translations
        items: list[Correction] = []
        for region in self._all:
            old_source = before.sources.get(region.id, region.text)
            source = after.sources.get(region.id, region.text)
            changes: list[tuple[CorrectionField, str, str]] = []
            if region.id in after.removed and region.id not in before.removed:
                changes.append(("removed", old_source, ""))
            else:
                if source != old_source:
                    changes.append(("source", old_source, source))
                old_english = saved_english.get(region.id, "")
                new_english = after.translations.get(region.id, self._machine.get(region.id, ""))
                if new_english != old_english:
                    changes.append(("translation", old_english, new_english))
            items.extend(
                Correction(
                    self.paths.series, self.paths.chapter, region.id, field, old, new, source, region.kind, at
                )
                for field, old, new in changes
            )
        return items
