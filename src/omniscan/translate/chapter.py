"""Chapter-level translation: one profile over one chapter's ocr.json, with skip/resume behaviour."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import Candidate, CandidateRun, GlossaryEntry, RegionsArtifact
from omniscan.translate.incremental import translation_key
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import translatable
from omniscan.translate.run import ChatClient, run_profile


def translate_chapter(
    client: ChatClient,
    paths: ChapterPaths,
    profile: TranslationProfile,
    entries: Sequence[GlossaryEntry],
    *,
    force: bool = False,
    story_summary: str | None = None,
    reuse: bool = False,
) -> tuple[Literal["done", "skipped"], CandidateRun | None]:
    """Run one translation profile over a chapter; write `translations/<profile>.json` (skip if present).

    Every candidate is stamped with its key (translate/incremental.py). With `reuse`, the candidates of the
    existing run whose key still matches are kept and only the other regions are sent — the pipeline's
    way of re-translating just what a hand edit or a glossary change touched."""
    output = paths.artifact(f"translations/{profile.name}.json")
    partial = paths.artifact(f"translations/.{profile.name}.partial.json")
    if output.is_file() and not force:
        return "skipped", None
    ocr_path = paths.artifact("ocr.json")
    if not ocr_path.is_file():
        raise FileNotFoundError("ocr.json missing — run the ocr stage first")
    if force:
        partial.unlink(missing_ok=True)  # a leftover partial from another attempt is stale under --force
    artifact = RegionsArtifact.load(ocr_path)
    keys = {region.id: translation_key(region, entries, profile) for region in translatable(artifact.regions)}
    reused: dict[str, Candidate] = {}
    if reuse and output.is_file():
        previous = CandidateRun.load(output)
        reused = {
            c.region_id: c
            for c in previous.candidates
            if c.key is not None and c.key == keys.get(c.region_id)
        }
    run = run_profile(
        client,
        profile,
        artifact.regions,
        entries,
        partial_path=partial,
        story_summary=story_summary,
        reused=reused,
    )
    run = run.model_copy(
        update={"candidates": [c.model_copy(update={"key": keys.get(c.region_id)}) for c in run.candidates]}
    )
    run.save(output)
    partial.unlink(missing_ok=True)
    return "done", run
