"""Automatic quality check of a translated chapter: the problems a proofreader would look for first.

It reads the stage artifacts only (no GPU, no model): lines that were never translated, source script left in the
English, lettering that overflowed its balloon, and lines the judge marked uncertain or as breaking the glossary.
Re-reading the finished pages with OCR is the next step (docs/ROADMAP.md X3).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from omniscan.core.schemas import FinalLine, LayoutItem, Region

IssueKind = Literal["untranslated", "source_left", "overflow", "uncertain", "glossary", "too_long"]

# Hangul, CJK ideographs, Hiragana/Katakana (incl. half-width) — none belongs in an English line
_SOURCE_SCRIPT = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-鿿가-힯ｦ-ﾟ]")
_TOO_LONG_RATIO = (
    4.0  # English this many times longer than the source (in characters) reads as a hallucination
)
_TOO_LONG_MIN = 60  # ...but only past this many characters


@dataclass(frozen=True, slots=True)
class Issue:
    """One problem on one region, with a sentence the Studio shows."""

    region_id: str
    kind: IssueKind
    message: str


def check_chapter(
    regions: Sequence[Region],
    translations: Mapping[str, str],
    final_lines: Sequence[FinalLine] = (),
    layout: Sequence[LayoutItem] = (),
) -> list[Issue]:
    """All issues of a chapter, in reading order; `translations` are the effective lines (manual edits applied)."""
    flags = {line.region_id: set(line.flags) for line in final_lines}
    overflow = {item.region_id for item in layout if item.overflow}
    issues: list[Issue] = []
    for region in regions:
        if region.kind == "watermark" or not region.text.strip():
            continue
        english = translations.get(region.id, "").strip()
        if not english:
            if region.kind != "sfx":  # an effect may be kept on purpose
                issues.append(Issue(region.id, "untranslated", "no English line"))
            continue
        if _SOURCE_SCRIPT.search(english):
            issues.append(Issue(region.id, "source_left", "source-language characters left in the English"))
        if region.id in overflow:
            issues.append(Issue(region.id, "overflow", "the lettering does not fit its balloon"))
        if "uncertain" in flags.get(region.id, set()):
            issues.append(Issue(region.id, "uncertain", "the judge was not sure about this line"))
        if "glossary_violation" in flags.get(region.id, set()):
            issues.append(Issue(region.id, "glossary", "a locked glossary term is not used"))
        source_len = len("".join(region.text.split()))
        if len(english) >= _TOO_LONG_MIN and len(english) > _TOO_LONG_RATIO * max(1, source_len):
            issues.append(
                Issue(region.id, "too_long", "much longer than the source; check for invented text")
            )
    return issues
