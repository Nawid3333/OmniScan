"""Consistency of a series' translation: lines said again but translated differently, and glossary terms whose
agreed English is missing from a line.

A proofreader's report, never a stage: `series_lines` reads every chapter's current regions and English lines
(watermarks left out), `divergences` groups repeated source lines (sound effects and one-character lines left out)
whose English differs beyond case, spacing and surrounding punctuation, and `term_misses` lists the regions
where a locked glossary term appears in the source but its target not in the English.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import GlossaryEntry, RegionKind
from omniscan.edits import store
from omniscan.glossary.match import find_terms, term_present
from omniscan.glossary.store import GlossaryStore
from omniscan.translate.on_demand import english_lines

MIN_SOURCE_CHARS = 2  # a one-character line ("헉", "?") may well read differently each time
_EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$")


@dataclass(frozen=True, slots=True)
class Line:
    """One region of the series with its source text and English line ("" before translation)."""

    chapter: str
    region_id: str
    kind: RegionKind
    lang: str
    source: str
    english: str


@dataclass(frozen=True, slots=True)
class Rendering:
    """One way a repeated line was translated, and where."""

    english: str
    places: list[tuple[str, str]] = field(default_factory=list)  # (chapter, region id)


@dataclass(frozen=True, slots=True)
class Divergence:
    """A source line said more than once with different English, most used rendering first."""

    source: str
    renderings: list[Rendering]


@dataclass(frozen=True, slots=True)
class TermMiss:
    """A region whose source has a locked glossary term but whose English lacks the term's target."""

    chapter: str
    region_id: str
    term: str
    target: str
    english: str


def series_lines(series: SeriesPaths, chapters: Sequence[str] | None = None) -> list[Line]:
    """Every region of the series' chapters (all by default) in reading order, watermarks left out."""
    lines: list[Line] = []
    for chapter in chapters if chapters is not None else series.chapters():
        paths = series.chapter(chapter)
        english = english_lines(paths)
        lines += [
            Line(chapter, r.id, r.kind, r.lang, r.text, english.get(r.id, ""))
            for r in store.current_regions(paths)
            if r.kind != "watermark"
        ]
    return lines


def _source_key(text: str) -> str:
    """A source line compared across the series: whitespace collapsed."""
    return " ".join(text.split())


def _english_key(text: str) -> str:
    """An English line compared across the series: case, spacing and surrounding punctuation ignored."""
    return _EDGE_PUNCTUATION.sub("", " ".join(text.split()).casefold())


def divergences(lines: Sequence[Line]) -> list[Divergence]:
    """Repeated source lines translated differently, the most repeated first."""
    groups: dict[str, dict[str, Rendering]] = {}
    for line in lines:
        source = _source_key(line.source)
        if line.kind == "sfx" or not line.english.strip() or len(source.replace(" ", "")) < MIN_SOURCE_CHARS:
            continue
        renderings = groups.setdefault(source, {})
        rendering = renderings.setdefault(_english_key(line.english), Rendering(line.english.strip()))
        rendering.places.append((line.chapter, line.region_id))
    found = [
        Divergence(source, sorted(renderings.values(), key=lambda r: -len(r.places)))
        for source, renderings in groups.items()
        if len(renderings) > 1
    ]
    return sorted(found, key=lambda d: -sum(len(r.places) for r in d.renderings))


def term_misses(lines: Sequence[Line], entries: Sequence[GlossaryEntry]) -> list[TermMiss]:
    """Translated regions whose source has a locked term (`entries` with ids) missing from their English."""
    locked = [entry for entry in entries if entry.status == "locked"]
    by_id = {entry.id: entry for entry in locked}
    misses: list[TermMiss] = []
    for line in lines:
        if not line.english.strip() or not locked:
            continue
        seen: set[int] = set()
        for match in find_terms(line.source, locked, line.lang):
            entry = by_id[match.entry_id]
            if match.entry_id in seen or term_present(line.english, entry):
                continue
            seen.add(match.entry_id)
            misses.append(TermMiss(line.chapter, line.region_id, match.source, entry.target, line.english))
    return misses


def glossary(series: SeriesPaths) -> list[GlossaryEntry]:
    """The series' glossary entries; none when it has no glossary yet (the database is never created)."""
    if not series.db.is_file():
        return []
    with GlossaryStore(series.db) as db:
        return db.list()
