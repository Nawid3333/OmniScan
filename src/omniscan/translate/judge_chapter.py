"""Chapter-level judging: a chapter's ocr.json plus every candidate run judged into final.json.

The judge's own lines are kept as final_auto.json; final.json is them with the chapter's hand-written lines
(edits.json) applied, so a re-run never loses them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import CandidateRun, FinalArtifact, FinalLine, GlossaryEntry, RegionsArtifact
from omniscan.edits.apply import apply_translation_edits
from omniscan.edits.store import FINAL_AUTO_FILE, load_edits
from omniscan.translate.incremental import REJUDGE_FLAGS, judge_key
from omniscan.translate.judge import JudgeStats, judge_regions
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.prompts import translatable
from omniscan.translate.run import ChatClient


def judge_chapter(
    client: ChatClient,
    paths: ChapterPaths,
    cfg: JudgeConfig,
    entries: Sequence[GlossaryEntry],
    *,
    run_ids: Sequence[str] | None = None,
    force: bool = False,
    story_summary: str | None = None,
    rate_limit_fallback: bool = False,
    reuse: bool = False,
) -> tuple[Literal["done", "skipped"], FinalArtifact | None, JudgeStats | None]:
    """Judge one chapter's candidate runs into `final.json` (skipped when it exists, unless forced); the
    returned artifact is final.json's, hand-written lines included.

    Every line is stamped with its key (translate/incremental.py). With `reuse`, the judge's previous line
    of a region whose key still matches — same source, same candidates, same glossary, same judge — is
    kept, and only the other regions are judged (`stats.reused` counts the kept ones)."""
    output = paths.artifact("final.json")
    if output.is_file() and not force:
        return "skipped", None, None
    ocr_path = paths.artifact("ocr.json")
    if not ocr_path.is_file():
        raise FileNotFoundError("ocr.json missing — run the ocr stage first")
    runs = _load_runs(paths, run_ids)
    artifact = RegionsArtifact.load(ocr_path)
    targets = translatable(artifact.regions)
    keys = {region.id: judge_key(region, runs, entries, cfg) for region in targets}
    kept = _reusable(paths, keys) if reuse else {}
    lines, stats = judge_regions(
        client,
        cfg,
        [region for region in artifact.regions if region.id not in kept],
        runs,
        entries,
        story_summary=story_summary,
        rate_limit_fallback=rate_limit_fallback,
    )
    judged = {line.region_id: line for line in lines}
    lines = [
        (kept.get(region.id) or judged[region.id]).model_copy(update={"key": keys[region.id]})
        for region in targets
        if region.id in kept or region.id in judged
    ]
    stats = dataclasses.replace(stats, regions=stats.regions + len(kept), reused=len(kept))
    result = FinalArtifact(
        judge_model=cfg.model,
        lines=lines,
        usage={  # mirrors CandidateRun.usage; the judge's pipeline metrics stay out of it
            "prompt_tokens": float(stats.prompt_tokens),
            "completion_tokens": float(stats.completion_tokens),
            "requests": float(stats.requests),
            "repair_requests": float(stats.repair_requests),
            "regions": float(stats.regions),
            "seconds": float(stats.seconds),
            "rate_limited": float(stats.rate_limited),
            "reused": float(stats.reused),
        },
    )
    result.save(paths.artifact(FINAL_AUTO_FILE))
    edits = load_edits(paths)
    if edits.translations:
        lines, _orphans = apply_translation_edits(result.lines, edits, artifact.regions)
        result = result.model_copy(update={"lines": lines})
    result.save(output)
    return "done", result, stats


def _reusable(paths: ChapterPaths, keys: dict[str, str]) -> dict[str, FinalLine]:
    """The judge's previous lines (final_auto.json) whose key still matches, except lines it failed on."""
    path = paths.artifact(FINAL_AUTO_FILE)
    if not path.is_file():
        return {}
    return {
        line.region_id: line
        for line in FinalArtifact.load(path).lines
        if line.key is not None
        and line.key == keys.get(line.region_id)
        and not REJUDGE_FLAGS & set(line.flags)
    }


def _load_runs(paths: ChapterPaths, run_ids: Sequence[str] | None) -> dict[str, dict[str, str]]:
    """region_id -> text per run from every non-hidden `translations/*.json` (empty texts dropped)."""
    runs: dict[str, dict[str, str]] = {}
    translations_dir = paths.artifact("translations")
    if translations_dir.is_dir():
        for path in sorted(translations_dir.glob("*.json")):
            if path.name.startswith("."):
                continue  # dot-prefixed partial files are never candidate runs
            run = CandidateRun.load(path)
            runs[run.run_id] = {c.region_id: c.text.strip() for c in run.candidates if c.text.strip()}
    if not runs:
        raise FileNotFoundError("no translation runs — run `omniscan translate` first")
    if run_ids is None:
        return runs
    unknown = next((run_id for run_id in run_ids if run_id not in runs), None)
    if unknown is not None:
        raise ValueError(f"unknown run {unknown!r} (known: {', '.join(runs)})")
    return {run_id: runs[run_id] for run_id in run_ids}
