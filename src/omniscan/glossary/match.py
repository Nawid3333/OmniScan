"""Glossary term matcher: exact-string matching over source text with language-aware particle stripping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omniscan.core.schemas import GlossaryEntry

# Conventional Korean particles, including the glued pair forms ("이가", "은는", "을를") in addition to their
# component halves. Particle lookup tries candidates longest-first so "에게" is consumed before "에".
KOREAN_PARTICLES: tuple[str, ...] = (
    "이가",
    "은는",
    "을를",
    "의",
    "아",
    "야",
    "도",
    "에게",
    "에서",
    "에",
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
)

# Japanese case particles that attach directly after a noun, the same way KOREAN_PARTICLES does. Longer
# compound forms first so e.g. "とは" is tried before its component "は" (the lookup below sorts by length
# anyway, but the tuple is written longest-first for readability, matching KOREAN_PARTICLES's style).
JAPANESE_PARTICLES: tuple[str, ...] = (
    "とは",
    "には",
    "での",
    "から",
    "まで",
    "は",
    "が",
    "を",
    "に",
    "で",
    "と",
    "も",
    "の",
    "へ",
    "や",
)

# Particle set to try after a matched term, by the text's source language. Chinese and English have no
# equivalent attached particles, so they get an empty tuple (no stripping — always correct for them, not a
# gap: see docs/reports/GL2.md's Why section).
PARTICLES_BY_LANG: dict[str, tuple[str, ...]] = {
    "ko": KOREAN_PARTICLES,
    "ja": JAPANESE_PARTICLES,
    "zh": (),
    "en": (),
}


def _particles_longest_first(lang: str) -> tuple[str, ...]:
    """`PARTICLES_BY_LANG[lang]` sorted longest-first; unknown `lang` -> no particles (empty tuple)."""
    return tuple(sorted(PARTICLES_BY_LANG.get(lang, ()), key=len, reverse=True))


@dataclass(frozen=True, slots=True)
class Match:
    """One glossary-term occurrence in source text, including an attached particle if any."""

    entry_id: int
    source: str  # the glossary term matched (entry.source or one of its aliases)
    start: int  # character offset into the searched text
    end: int  # end offset, exclusive; end - start may exceed len(source) when a particle was stripped
    particle: str | None  # the particle consumed after the term, if any


def _candidates(entries: Sequence[GlossaryEntry]) -> tuple[tuple[str, int], ...]:
    """(candidate, entry index) pairs ordered longest-first; ties by entry order, source before its aliases."""
    ordered: list[tuple[str, int]] = []
    for priority, entry in enumerate(entries):
        ordered.append((entry.source, priority))
        ordered.extend((alias, priority) for alias in entry.aliases)
    return tuple(sorted(ordered, key=lambda candidate: -len(candidate[0])))


def find_terms(text: str, entries: Sequence[GlossaryEntry], lang: str) -> list[Match]:
    """Greedy left-to-right, non-overlapping scan for glossary terms with an optional attached particle.

    Entries must carry an id (as returned by `GlossaryStore.add`); a None id raises ValueError.
    `lang` selects the particle set — see PARTICLES_BY_LANG; an unrecognised code strips no particles instead of raising.
    """
    candidates = _candidates(entries)
    particles = _particles_longest_first(lang)
    matches: list[Match] = []
    i = 0
    while i < len(text):
        hit = next((c for c in candidates if text.startswith(c[0], i)), None)
        if hit is None:
            i += 1
            continue
        candidate, priority = hit
        term_end = i + len(candidate)
        particle = next((p for p in particles if text.startswith(p, term_end)), None)
        end = term_end + len(particle) if particle else term_end
        entry_id = entries[priority].id
        if entry_id is None:
            raise ValueError(f"entry {candidate!r} has no id; find_terms needs store entries")
        matches.append(Match(entry_id=entry_id, source=candidate, start=i, end=end, particle=particle))
        i = end
    return matches


def term_present(text: str, entry: GlossaryEntry) -> bool:
    """True if the entry's English target appears verbatim in the (English) text."""
    return entry.target in text
