"""On-demand translation of a few regions: the Studio's *Translate* button and `omniscan edit translate`.

Nothing is written: every profile's reading comes back as a suggestion, and the user keeps one as the
region's hand-written English line (edits.json). The model sees the neighbouring lines of the chapter —
source and current English — so a single bubble reads like the dialogue around it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from omniscan.core.schemas import GlossaryEntry, Region
from omniscan.llm.ollama import OllamaRateLimitError
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import ContextLine, source_text, translatable
from omniscan.translate.run import ChatClient, run_profile

CONTEXT_BEFORE = 6  # neighbouring lines shown before the first region translated
CONTEXT_AFTER = 3  # and after the last one


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One profile's translation of one region."""

    region_id: str
    profile: str
    model: str
    text: str


def context_lines(
    ordered: Sequence[Region],
    target_ids: Collection[str],
    english: Mapping[str, str],
    *,
    before: int = CONTEXT_BEFORE,
    after: int = CONTEXT_AFTER,
) -> list[ContextLine]:
    """The lines around the targets in reading order (`ordered` = the chapter's translatable regions): up to
    `before` lines before the first target and `after` after the last, targets themselves excluded."""
    positions = [i for i, region in enumerate(ordered) if region.id in target_ids]
    if not positions:
        return []
    window = ordered[max(0, positions[0] - before) : positions[-1] + after + 1]
    return [
        ContextLine(id=region.id, text=source_text(region), english=english.get(region.id, ""))
        for region in window
        if region.id not in target_ids
    ]


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
) -> list[Suggestion]:
    """Translate the target regions with every profile; a profile that hits the Ollama rate limit is
    replaced by its fallback. ValueError names a target that is unknown or has nothing to translate."""
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
    suggestions: list[Suggestion] = []
    for profile in profiles:
        used = profile
        try:
            run = run_profile(client, profile, targets, entries, story_summary=story_summary, context=context)
        except OllamaRateLimitError:
            fallback = (fallbacks or {}).get(profile.name)
            if fallback is None:
                raise
            used = fallback
            run = run_profile(
                client, fallback, targets, entries, story_summary=story_summary, context=context
            )
        suggestions.extend(
            Suggestion(region_id=c.region_id, profile=used.name, model=used.model, text=c.text)
            for c in run.candidates
        )
    return suggestions
