"""Chapter-level translation: one profile over one chapter's ocr.json, with skip/resume behaviour."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import CandidateRun, GlossaryEntry, RegionsArtifact
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.run import ChatClient, run_profile


def translate_chapter(
    client: ChatClient,
    paths: ChapterPaths,
    profile: TranslationProfile,
    entries: Sequence[GlossaryEntry],
    *,
    force: bool = False,
    story_summary: str | None = None,
) -> tuple[Literal["done", "skipped"], CandidateRun | None]:
    """Run one translation profile over a chapter; write `translations/<profile>.json` (skip if present)."""
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
    run = run_profile(
        client, profile, artifact.regions, entries, partial_path=partial, story_summary=story_summary
    )
    run.save(output)
    partial.unlink(missing_ok=True)
    return "done", run
