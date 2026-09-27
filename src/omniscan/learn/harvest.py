"""What a chapter's hand edits corrected: each edit compared with what the pipeline had produced.

Every edit records the pipeline's text it replaced (`auto_text`), so every correction is a pair — the OCR's
reading and the editor's text, the machine's English and the editor's — that stays the same after a re-run
already applies the lesson (else the lesson would erase its own evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import FinalArtifact, RegionKind
from omniscan.edits.apply import match_region_edits, match_translation_edits
from omniscan.edits.store import FINAL_AUTO_FILE, auto_regions, current_regions, load_edits


def norm(text: str) -> str:
    """`text` with every run of whitespace collapsed to one space."""
    return " ".join(text.split())


@dataclass(slots=True)
class Corrections:
    """Every correction found in a series' chapters, in reading order."""

    ocr_fixes: list[tuple[str, str]] = field(default_factory=list)  # (OCR reading, corrected text)
    deleted: list[str] = field(default_factory=list)  # texts of pipeline regions deleted by hand
    kinds: list[tuple[str, RegionKind]] = field(default_factory=list)  # (region text, kind set by hand)
    translations: list[tuple[str, str, bool, str]] = field(
        default_factory=list
    )  # (source, english, typed, chapter)
    english_fixes: list[tuple[str, str]] = field(default_factory=list)  # (machine line, line typed by hand)

    def extend(self, other: Corrections) -> None:
        """Append every correction of `other`."""
        self.ocr_fixes.extend(other.ocr_fixes)
        self.deleted.extend(other.deleted)
        self.kinds.extend(other.kinds)
        self.translations.extend(other.translations)
        self.english_fixes.extend(other.english_fixes)


def harvest_chapter(paths: ChapterPaths) -> Corrections:
    """The corrections one chapter's edits.json makes to its pipeline output.

    Each edit is compared with the pipeline's text it records (`auto_text`); an edit made before that was
    recorded is compared with the pipeline's current output (ocr_auto.json, final_auto.json) instead."""
    found = Corrections()
    edits = load_edits(paths)
    if not edits.regions and not edits.translations:
        return found
    auto = auto_regions(paths)
    text_of = {region.id: region.text for region in auto}
    read = {index: text_of[region_id] for index, region_id in match_region_edits(auto, edits).items()}
    for index, edit in enumerate(edits.regions):
        before = (
            "" if edit.added else norm(edit.auto_text if edit.auto_text is not None else read.get(index, ""))
        )
        if edit.deleted:
            if before:
                found.deleted.append(before)
            continue
        text = norm(edit.text) if edit.text is not None else before
        if before and text and text != before:
            found.ocr_fixes.append((before, text))
        if edit.kind in ("sfx", "watermark") and text:
            found.kinds.append((text, edit.kind))
    final_auto = paths.artifact(FINAL_AUTO_FILE)
    judged = (
        {line.region_id: line.text for line in FinalArtifact.load(final_auto).lines}
        if final_auto.is_file()
        else {}
    )
    claims = match_translation_edits(current_regions(paths), edits)
    for index, edit in enumerate(edits.translations):
        typed = edit.suggested_by is None
        if norm(edit.source) and norm(edit.text):
            found.translations.append((norm(edit.source), edit.text.strip(), typed, paths.chapter))
        before = edit.auto_text if edit.auto_text is not None else judged.get(claims.get(index, ""), "")
        if typed and norm(before) and norm(before) != norm(edit.text):
            found.english_fixes.append((norm(before), norm(edit.text)))
    return found


def harvest_series(series: SeriesPaths) -> Corrections:
    """The corrections of every chapter of the series (reading order)."""
    found = Corrections()
    for chapter in series.chapters():
        found.extend(harvest_chapter(series.chapter(chapter)))
    return found
