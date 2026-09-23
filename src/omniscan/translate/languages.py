"""Naming each source language in the LLM prompts: one `SourceLanguage` per `Region.lang` code.

Every prompt module assembles its Korean rendering from these values (`*_system("ko")` stays the
tuned text measured in docs/benchmarks/translation-probe.md); callers pass the chapter's
`region_language` so a Japanese or Chinese chapter is never told its text is Korean.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import Region, RegionsArtifact


@dataclass(frozen=True, slots=True)
class SourceLanguage:
    """How the prompts name one source language."""

    code: str  # "ko", "zh", "ja", "en"
    name: str  # "Korean", "Chinese", "Japanese", "English"
    work: str  # "manhwa", "manhua", "manga", "comic"
    honorifics: str  # the whole honorifics sentence incl. its trailing space, or ""
    particle_hint: str  # e.g. " (with or without a particle such as 이/가/은/는/을/를/의)", or ""


_LANGUAGES: dict[str, SourceLanguage] = {
    "ko": SourceLanguage(
        code="ko",
        name="Korean",
        work="manhwa",
        honorifics=(
            "Keep Korean honorific suffixes and titles romanized when they address a person "
            "(-nim, -ssi, hyung, noona, sunbae) unless that reads badly in English. "
        ),
        particle_hint=" (with or without a particle such as 이/가/은/는/을/를/의)",
    ),
    "ja": SourceLanguage(
        code="ja",
        name="Japanese",
        work="manga",
        honorifics=(
            "Keep Japanese honorific suffixes romanized when they address a person "
            "(-san, -kun, -chan, -sama, senpai) unless that reads badly in English. "
        ),
        particle_hint=" (with or without a particle such as は/が/を/に/の)",
    ),
    "zh": SourceLanguage(code="zh", name="Chinese", work="manhua", honorifics="", particle_hint=""),
    "en": SourceLanguage(code="en", name="English", work="comic", honorifics="", particle_hint=""),
}


def source_language(code: str) -> SourceLanguage:
    """The SourceLanguage for a Region.lang code; an unknown code raises ValueError."""
    language = _LANGUAGES.get(code)
    if language is None:
        raise ValueError(f"unknown source language code: {code!r}")
    return language


def region_language(regions: Sequence[Region]) -> str:
    """The most common `lang` among `regions` (ties: the first region's), "ko" when `regions` is empty."""
    if not regions:
        return "ko"
    counts = Counter(region.lang for region in regions)
    return max(counts, key=lambda code: counts[code])  # max keeps the first of equal counts


def chapter_language(paths: ChapterPaths) -> str:
    """region_language of the chapter's ocr.json regions; "ko" when ocr.json is missing."""
    path = paths.artifact("ocr.json")
    return region_language(RegionsArtifact.load(path).regions) if path.is_file() else "ko"
