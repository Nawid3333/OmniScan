"""Step-mode previews: one-line summaries (plus sample lines) of a finished stage's artifacts.

`describe` reads only the chapter's artifact JSON files — a missing or invalid artifact becomes a
`Preview` saying so, never an exception, so the CLI gate can always print something.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    Artifact,
    CandidateRun,
    ExportArtifact,
    FinalArtifact,
    InpaintArtifact,
    IngestArtifact,
    LayoutArtifact,
    RegionsArtifact,
    SlicesArtifact,
)

_REGION_KINDS = ("bubble_text", "free_text", "sfx", "watermark")

_T = TypeVar("_T", bound=Artifact)


@dataclass(frozen=True, slots=True)
class Preview:
    """What the step-mode gate shows for one finished stage: a one-line summary plus sample lines."""

    stage: str
    summary: str  # one line
    details: tuple[str, ...]  # up to 6 short lines (samples), possibly empty


def _clip(text: str, limit: int = 60) -> str:
    """The first `limit` characters of `text`, an ellipsis marking a longer one."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _load(cls: type[_T], path: Path) -> _T | None:
    """The artifact saved at `path`, or None when the file is missing or does not validate."""
    try:
        return cls.load(path)
    except (OSError, ValueError):
        return None


def _ingest(paths: ChapterPaths) -> Preview:
    """Preview of a finished `ingest.json`."""
    artifact = _load(IngestArtifact, paths.artifact("ingest.json"))
    if artifact is None:
        return Preview("ingest", "no ingest.json found", ())
    n = sum(1 for file in artifact.files if not file.filtered)
    return Preview("ingest", f"{n} file(s), strip {artifact.strip_width}x{artifact.strip_height} px", ())


def _slice(paths: ChapterPaths) -> Preview:
    """Preview of a finished `slices.json`."""
    artifact = _load(SlicesArtifact, paths.artifact("slices.json"))
    if artifact is None:
        return Preview("slice", "no slices.json found", ())
    slices = artifact.slices
    summary = (
        f"{len(slices)} slice(s): {sum(s.blank for s in slices)} blank, "
        f"{sum(s.forced_cut for s in slices)} forced cut(s), {sum(s.filtered for s in slices)} filtered"
    )
    return Preview("slice", summary, ())


def _detect(paths: ChapterPaths) -> Preview:
    """Preview of a finished `regions.json`."""
    artifact = _load(RegionsArtifact, paths.artifact("regions.json"))
    if artifact is None:
        return Preview("detect", "no regions.json found", ())
    regions = artifact.regions
    if not regions:
        return Preview("detect", "0 regions", ())
    parts = ", ".join(
        f"{kind} {sum(1 for region in regions if region.kind == kind)}"
        for kind in _REGION_KINDS
        if any(region.kind == kind for region in regions)
    )
    return Preview("detect", f"{len(regions)} region(s): {parts}", ())


def _ocr(paths: ChapterPaths, cfg: Config) -> Preview:
    """Preview of a finished `ocr.json`."""
    artifact = _load(RegionsArtifact, paths.artifact("ocr.json"))
    if artifact is None:
        return Preview("ocr", "no ocr.json found", ())
    regions = artifact.regions
    n = sum(1 for region in regions if region.text)
    low = sum(1 for region in regions if region.confidence < cfg.ocr.low_conf)
    return Preview(
        "ocr",
        f"{n} region(s) with text, {low} low-confidence (< {cfg.ocr.low_conf})",
        tuple(_clip(region.text) for region in regions[:5]),
    )


def _translate(paths: ChapterPaths) -> Preview:
    """Preview of the chapter's candidate runs under `translations/`."""
    runs_dir = paths.artifact("translations")
    run_files = (
        sorted(path for path in runs_dir.glob("*.json") if not path.name.startswith("."))
        if runs_dir.is_dir()
        else []
    )
    if not run_files:
        return Preview("translate", "no translations found", ())
    runs: list[CandidateRun] = []
    for path in run_files:
        run = _load(CandidateRun, path)
        if run is None:
            return Preview("translate", "no translations found", ())
        runs.append(run)
    regions = _load(RegionsArtifact, paths.artifact("ocr.json"))
    source_of = {region.id: region.text for region in regions.regions} if regions is not None else {}
    details = tuple(
        f"{_clip(source_of.get(candidate.region_id, ''))} → {_clip(candidate.text)}"
        for run in runs
        for candidate in run.candidates
    )[:3]
    return Preview("translate", f"{len(runs)} candidate run(s)", details)


def _judge(paths: ChapterPaths) -> Preview:
    """Preview of a finished `final.json`."""
    artifact = _load(FinalArtifact, paths.artifact("final.json"))
    if artifact is None:
        return Preview("judge", "no final.json found", ())
    return Preview(
        "judge",
        f"{len(artifact.lines)} final line(s)",
        tuple(_clip(line.text) for line in artifact.lines[:5]),
    )


def _inpaint(stage: str, paths: ChapterPaths) -> Preview:
    """Preview of a finished `inpaint.json` (or the LaMa stage's own `inpaint_lama.json`)."""
    name = "inpaint_lama.json" if stage == "inpaint_lama" else "inpaint.json"
    artifact = _load(InpaintArtifact, paths.artifact(name))
    if artifact is None:
        return Preview(stage, f"no {name} found", ())
    return Preview(stage, f"{len(artifact.items)} item(s) inpainted", ())


def _typeset(paths: ChapterPaths) -> Preview:
    """Preview of a finished `layout.json`."""
    artifact = _load(LayoutArtifact, paths.artifact("layout.json"))
    if artifact is None:
        return Preview("typeset", "no layout.json found", ())
    unfitted = sum(1 for item in artifact.items if item.overflow)
    return Preview("typeset", f"{len(artifact.items)} layout item(s), {unfitted} unfitted", ())


def _export(paths: ChapterPaths) -> Preview:
    """Preview of a finished `export.json`."""
    artifact = _load(ExportArtifact, paths.artifact("export.json"))
    if artifact is None:
        return Preview("export", "no export.json found", ())
    return Preview("export", f"{len(artifact.files)} file(s) written to {paths.output_dir}", ())


def describe(stage: str, paths: ChapterPaths, cfg: Config) -> Preview:
    """Summarise the named stage's finished artifacts of one chapter; missing ones never raise."""
    if stage == "ingest":
        return _ingest(paths)
    if stage == "slice":
        return _slice(paths)
    if stage == "detect":
        return _detect(paths)
    if stage == "ocr":
        return _ocr(paths, cfg)
    if stage == "translate":
        return _translate(paths)
    if stage == "judge":
        return _judge(paths)
    if stage == "inpaint" or stage == "inpaint_lama":
        return _inpaint(stage, paths)
    if stage == "typeset":
        return _typeset(paths)
    if stage == "export":
        return _export(paths)
    return Preview(stage, f"no {stage} artifact found", ())