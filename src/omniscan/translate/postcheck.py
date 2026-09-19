"""Glossary post-check: locked terms present in the Korean source but missing from the English line."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from omniscan.core.schemas import GlossaryEntry, Region
from omniscan.glossary.match import find_terms
from omniscan.translate.prompts import source_text, translatable


@dataclass(frozen=True, slots=True)
class TermViolation:
    """A locked glossary term whose target is absent from the translated line."""

    entry_id: int
    source: str  # entry.source (never the alias that matched)
    expected: str  # entry.target


def normalize_for_check(text: str) -> str:
    """NFKC-normalised casefolded text (no whitespace or punctuation change)."""
    return unicodedata.normalize("NFKC", text).casefold()


def check_locked_terms(source: str, target: str, entries: Sequence[GlossaryEntry]) -> list[TermViolation]:
    """Violations for locked entries hit in `source` whose target is missing from `target`, in source order."""
    locked = [e for e in entries if e.status == "locked"]
    if not locked:
        return []
    by_id = {e.id: e for e in locked}
    first: dict[int, GlossaryEntry] = {}  # entry id -> entry, in order of first occurrence in source
    for match in find_terms(source, locked):
        first.setdefault(match.entry_id, by_id[match.entry_id])
    norm_target = normalize_for_check(target)
    return [
        TermViolation(entry_id=eid, source=entry.source, expected=entry.target)
        for eid, entry in first.items()
        if normalize_for_check(entry.target) not in norm_target
    ]


def check_regions(
    regions: Sequence[Region], texts: Mapping[str, str], entries: Sequence[GlossaryEntry]
) -> dict[str, list[TermViolation]]:
    """Per-region violations (only regions with them) in reading order; regions without a text are skipped."""
    result: dict[str, list[TermViolation]] = {}
    for region in translatable(regions):
        text = texts.get(region.id)
        if text is None:
            continue
        violations = check_locked_terms(source_text(region), text, entries)
        if violations:
            result[region.id] = violations
    return result
