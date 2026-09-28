"""Translating a few regions of a chapter on demand — the Studio's Translate and `omniscan edit translate`.

The chosen profiles translate the regions with everything the pipeline gives a translation (neighbouring lines,
glossary, story so far, the series' learned memory and character voices); nothing is written unless `apply` keeps each region's
first suggestion as its English line (recorded in edits.json with the profile that wrote it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import FinalArtifact, FinalLine
from omniscan.edits import store as edit_store
from omniscan.learn.apply import translation_hints
from omniscan.learn.memory import current_memory
from omniscan.pipeline.stages import series_entries, series_story_context
from omniscan.translate.images import PageImages
from omniscan.translate.profiles import TranslationProfile, resolve_fallbacks
from omniscan.translate.run import ChatClient
from omniscan.translate.suggest import Suggestion, suggest
from omniscan.translate.voices import load_voices


@dataclass(frozen=True, slots=True)
class OnDemand:
    """Every profile's suggestion, and the English lines kept (with `apply`)."""

    suggestions: list[Suggestion]
    applied: list[FinalLine]


def pick_profiles(known: Mapping[str, TranslationProfile], name: str | None) -> list[TranslationProfile]:
    """The named profile (a disabled one too), or every enabled one; ValueError when there is none."""
    if name is not None:
        if name not in known:
            raise ValueError(f"unknown profile {name!r}")
        return [known[name]]
    enabled = [profile for profile in known.values() if profile.enabled]
    if not enabled:
        raise ValueError("no translation profile is enabled")
    return enabled


def english_lines(paths: ChapterPaths) -> dict[str, str]:
    """region id -> the chapter's current English line (final.json); {} when it is missing or unreadable."""
    path = paths.artifact("final.json")
    if not path.is_file():
        return {}
    try:
        final = FinalArtifact.load(path)
    except OSError, ValueError:
        return {}
    lines: dict[str, str] = {}
    for line in final.lines:
        lines.setdefault(line.region_id, line.text)
    return lines


def translate_now(
    client: ChatClient,
    cfg: Config,
    series: SeriesPaths,
    paths: ChapterPaths,
    region_ids: Sequence[str],
    known: Mapping[str, TranslationProfile],
    *,
    profile: str | None = None,
    apply: bool = False,
) -> OnDemand:
    """Translate `region_ids` of the chapter now with `profile` (or every enabled one); `cfg` has the series'
    settings applied. ValueError names an unknown profile or region; Ollama errors pass through."""
    profiles = pick_profiles(known, profile)
    suggestions = suggest(
        client,
        profiles,
        edit_store.current_regions(paths),
        region_ids,
        series_entries(series),
        english_lines(paths),
        fallbacks=resolve_fallbacks(profiles, dict(known)),
        story_summary=series_story_context(series, paths.chapter),
        hints=translation_hints(current_memory(series), cfg.learn) if cfg.learn.enabled else None,
        characters=load_voices(series),
        images=PageImages(paths),
    )
    applied: list[FinalLine] = []
    if apply:
        first: dict[str, Suggestion] = {}
        for item in suggestions:
            first.setdefault(item.region_id, item)
        applied = edit_store.set_translations(  # one write, one rebuild, one undo step
            paths,
            {item.region_id: item.text for item in first.values()},
            direction=cfg.detect.reading_direction,
            suggested_by={item.region_id: item.profile for item in first.values()},
        )
    return OnDemand(suggestions=suggestions, applied=applied)
