"""Executor that runs a queued job's pipeline stages through the core stage runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.stage import run_series
from omniscan.queue.store import Job
from omniscan.queue.worker import Executor, PermanentJobError


def _ingest_stages() -> list[Any]:
    from omniscan.ingest.stage import IngestStage

    return [IngestStage()]


def _slice_stages() -> list[Any]:
    from omniscan.ingest.stage import IngestStage
    from omniscan.slicer.stage import SliceStage

    return [IngestStage(), SliceStage()]


# Stage table: name -> factory building the stage list to run, mirroring the CLI. Stages are built per
# executor call (not at import) so importing this module — and `omniscan --help` — stays fast.
STAGE_TABLE: dict[str, Callable[[], list[Any]]] = {
    "ingest": _ingest_stages,
    "slice": _slice_stages,
}


def stage_executor(cfg: Config) -> Executor:
    """Run a queued job's stages over its series via `run_series`; unimplemented stages never retry."""

    def _run(job: Job) -> None:
        unknown = next((name for name in job.stages if name not in STAGE_TABLE), None)
        if unknown is not None:
            raise PermanentJobError(f"stage {unknown!r} is not implemented yet")
        if job.chapters is None and not SeriesPaths.from_config(cfg, job.series).chapters():
            raise PermanentJobError(f"no chapters found for series {job.series!r}")
        for name in job.stages:
            results = run_series(
                STAGE_TABLE[name](),
                cfg,
                job.series,
                list(job.chapters) if job.chapters is not None else None,
                force=job.force,
            )
            failures = [
                f"{chapter}/{outcome.stage}: {outcome.error}"
                for chapter, outcomes in results.items()
                for outcome in outcomes
                if outcome.status == "failed"
            ]
            if failures:
                raise RuntimeError("; ".join(failures))  # later stage names are not run

    return _run
