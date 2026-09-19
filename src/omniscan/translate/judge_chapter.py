"""Chapter-level judging: a chapter's ocr.json plus every candidate run judged into final.json."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import CandidateRun, FinalArtifact, GlossaryEntry, RegionsArtifact
from omniscan.translate.judge import JudgeStats, judge_regions
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.run import ChatClient


def judge_chapter(
    client: ChatClient,
    paths: ChapterPaths,
    cfg: JudgeConfig,
    entries: Sequence[GlossaryEntry],
    *,
    run_ids: Sequence[str] | None = None,
    force: bool = False,
) -> tuple[Literal["done", "skipped"], FinalArtifact | None, JudgeStats | None]:
    """Judge one chapter's candidate runs into `final.json` (skipped when it exists, unless forced)."""
    output = paths.artifact("final.json")
    if output.is_file() and not force:
        return "skipped", None, None
    ocr_path = paths.artifact("ocr.json")
    if not ocr_path.is_file():
        raise FileNotFoundError("ocr.json missing — run the ocr stage first")
    runs = _load_runs(paths, run_ids)
    artifact = RegionsArtifact.load(ocr_path)
    lines, stats = judge_regions(client, cfg, artifact.regions, runs, entries)
    result = FinalArtifact(judge_model=cfg.model, lines=lines)
    result.save(output)
    return "done", result, stats


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
