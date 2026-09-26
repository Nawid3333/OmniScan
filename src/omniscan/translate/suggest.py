"""On-demand translation of a few regions: the Studio's *Translate* button and `omniscan edit translate`.

Nothing is written: every profile's reading comes back as a suggestion, and the user keeps one as the
region's hand-written English line (edits.json). The model sees the neighbouring lines of the chapter —
source and current English — so a single bubble reads like the dialogue around it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from omniscan.core.schemas import GlossaryEntry, Region
from omniscan.llm.ollama import OllamaRateLimitError
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import context_lines, translatable
from omniscan.translate.run import ChatClient, run_profile

if TYPE_CHECKING:
    from omniscan.learn.apply import TranslationHints


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One profile's translation of one region."""

    region_id: str
    profile: str
    model: str
    text: str


def suggest(
    client: ChatClient,
    profiles: Sequence[TranslationProfile],
    regions: Sequence[Region],
    target_ids: Sequence[str],
    entries: Sequence[GlossaryEntry],
    english: Mapping[str, str],
    *,
    fallbacks: Mapping[str, TranslationProfile] | None = None,
    story_summary: str | None = None,
    hints: TranslationHints | None = None,
) -> list[Suggestion]:
    """Translate the target regions with every profile; a profile that hits the Ollama rate limit is
    replaced by its fallback. `hints` (the series' learned memory) shows the models similar lines the editor
    translated and their preferred wording, never an old line as the answer. ValueError names a target that
    is unknown or has nothing to translate."""
    ordered = translatable(regions)
    known = {region.id for region in regions}
    translatable_ids = {region.id for region in ordered}
    for region_id in target_ids:
        if region_id not in known:
            raise ValueError(f"region {region_id!r} not found in ocr.json")
        if region_id not in translatable_ids:
            raise ValueError(f"region {region_id!r} has no source text to translate (or is a watermark)")
    targets = [region for region in ordered if region.id in set(target_ids)]
    context = context_lines(ordered, set(target_ids), english)
    hints = replace(hints, exact={}) if hints is not None else None  # a fresh suggestion, not the kept line
    suggestions: list[Suggestion] = []
    for profile in profiles:
        used = profile
        try:
            run = run_profile(
                client, profile, targets, entries, story_summary=story_summary, context=context, hints=hints
            )
        except OllamaRateLimitError:
            fallback = (fallbacks or {}).get(profile.name)
            if fallback is None:
                raise
            used = fallback
            run = run_profile(
                client, fallback, targets, entries, story_summary=story_summary, context=context, hints=hints
            )
        suggestions.extend(
            Suggestion(region_id=c.region_id, profile=used.name, model=used.model, text=c.text)
            for c in run.candidates
        )
    return suggestions
