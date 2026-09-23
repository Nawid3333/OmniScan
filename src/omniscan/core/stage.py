"""Stage contract + runner.

A *stage* turns input files/artifacts of one chapter into output artifacts. The runner decides per chapter whether
a stage is up to date (manifest: version + input hash + config hash + outputs exist), acquires the stage's GPU model
group (models stay loaded while consecutive stages/chapters use the same group), runs it, and records the result.

Stages share in-memory data within one chapter through `ChapterContext.lazy()` — e.g. the decoded strip tensor is
produced once by whichever stage first needs it and reused by later stages in the same pass.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from omniscan.core.config import Config, series_config
from omniscan.core.manifest import hash_inputs, hash_json, is_up_to_date, load_manifest
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import Manifest, StageRecord, utcnow

log = logging.getLogger(__name__)


class GpuScheduler(Protocol):
    """What the runner needs from the VRAM manager (implemented by omniscan.gpu.vram.VramManager)."""

    def acquire(self, group: str) -> Mapping[str, Any]: ...
    def reset_peak(self) -> None: ...
    def peak_gib(self) -> float: ...


@dataclass
class ChapterContext:
    cfg: Config
    series: SeriesPaths
    paths: ChapterPaths
    manifest: Manifest
    gpu: GpuScheduler | None = None
    _memo: dict[str, Any] = field(default_factory=dict, repr=False)

    def lazy[T](self, key: str, factory: Callable[[], T]) -> T:
        """Return the in-memory value for `key`, creating it once per chapter pass."""
        if key not in self._memo:
            self._memo[key] = factory()
        return self._memo[key]

    def put(self, key: str, value: Any) -> None:
        self._memo[key] = value

    def drop(self, key: str) -> None:
        """Release an in-memory value (e.g. free the strip tensor when the pass is done)."""
        self._memo.pop(key, None)

    def clear(self) -> None:
        self._memo.clear()


@runtime_checkable
class Stage(Protocol):
    name: ClassVar[str]
    version: ClassVar[int]
    gpu_group: ClassVar[str | None]  # model group to hold in VRAM while running, or None

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        ...

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        ...

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        ...

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        ...


@dataclass(frozen=True, slots=True)
class StageOutcome:
    stage: str
    status: Literal["done", "skipped", "failed"]
    seconds: float = 0.0
    metrics: Mapping[str, float] = field(default_factory=dict)
    error: str | None = None


def make_context(
    cfg: Config,
    series: str,
    chapter: str,
    gpu: GpuScheduler | None = None,
    *,
    merge_series_config: bool = True,
) -> ChapterContext:
    """Context of one chapter; the series' `series.toml` overrides are already applied to `ctx.cfg`.

    `merge_series_config=False` skips that merge: the caller is asserting `cfg` is already exactly the
    config it wants every stage to see (the OCR qualification suite needs this — see bug O1e — because a
    series' own `series.toml` can otherwise silently overwrite the OCR engine it deliberately varies)."""
    sp = SeriesPaths.from_config(cfg, series)
    cp = sp.chapter(chapter)
    return ChapterContext(
        cfg=series_config(cfg, sp.library_dir) if merge_series_config else cfg,
        series=sp,
        paths=cp,
        manifest=load_manifest(cp.manifest, series, chapter),
        gpu=gpu,
    )


def run_stage(stage: Stage, ctx: ChapterContext, *, force: bool = False) -> StageOutcome:
    """Run one stage for one chapter unless it is up to date."""
    work = ctx.paths.work_dir
    input_hash = hash_inputs(stage.inputs(ctx))
    config_hash = hash_json(stage.config_subset(ctx.cfg))
    if not force and is_up_to_date(
        ctx.manifest,
        stage.name,
        version=stage.version,
        input_hash=input_hash,
        config_hash=config_hash,
        work_dir=work,
    ):
        log.debug("%s/%s: %s up to date", ctx.paths.series, ctx.paths.chapter, stage.name)
        return StageOutcome(stage.name, "skipped")

    work.mkdir(parents=True, exist_ok=True)
    models: Mapping[str, Any] = {}
    if stage.gpu_group is not None:
        if ctx.gpu is None:
            raise RuntimeError(
                f"stage {stage.name} needs GPU group {stage.gpu_group!r} but no scheduler given"
            )
        models = ctx.gpu.acquire(stage.gpu_group)
        ctx.gpu.reset_peak()

    started = utcnow()
    t0 = time.perf_counter()
    status: Literal["done", "failed"] = "done"
    error: str | None = None
    metrics: dict[str, float] = {}
    try:
        metrics = dict(stage.run(ctx, models))
    except Exception as exc:  # recorded in the manifest, re-raised by the caller's policy
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        log.error(
            "%s/%s: %s failed\n%s", ctx.paths.series, ctx.paths.chapter, stage.name, traceback.format_exc()
        )
    seconds = time.perf_counter() - t0
    metrics["seconds"] = round(seconds, 4)
    if stage.gpu_group is not None and ctx.gpu is not None:
        metrics["peak_vram_gib"] = round(ctx.gpu.peak_gib(), 3)

    ctx.manifest.stages[stage.name] = StageRecord(
        stage=stage.name,
        version=stage.version,
        input_hash=input_hash,
        config_hash=config_hash,
        outputs=list(stage.outputs(ctx)),
        status=status,
        started_at=started,
        finished_at=utcnow(),
        metrics=metrics,
        error=error,
    )
    ctx.manifest.save(ctx.paths.manifest)
    return StageOutcome(stage.name, status, seconds, metrics, error)


class RunAbortedError(Exception):
    """An `after_stage` hook returned False: the run stops after the stage that was just recorded."""

    def __init__(self, chapter: str, stage: str) -> None:
        super().__init__(f"stopped after {stage} of {chapter}")
        self.chapter = chapter
        self.stage = stage
        self.outcomes: list[StageOutcome] = []  # outcomes of the chapter that was being run
        self.results: dict[str, list[StageOutcome]] = {}  # chapters finished before it (set by run_series)


type AfterStage = Callable[[ChapterContext, StageOutcome], bool]


def run_chapter(
    stages: Sequence[Stage],
    ctx: ChapterContext,
    *,
    force: bool = False,
    stop_on_error: bool = True,
    after_stage: AfterStage | None = None,
) -> list[StageOutcome]:
    """Run stages in order for one chapter; later stages see earlier stages' in-memory values.

    `after_stage(ctx, outcome)` runs after every stage that finished (also skipped/failed ones) while the chapter's
    in-memory values are still alive; returning False raises RunAbortedError (the manifest is already saved).
    """
    outcomes: list[StageOutcome] = []
    try:
        for stage in stages:
            outcome = run_stage(stage, ctx, force=force)
            outcomes.append(outcome)
            if after_stage is not None and not after_stage(ctx, outcome):
                aborted = RunAbortedError(ctx.paths.chapter, stage.name)
                aborted.outcomes = outcomes
                raise aborted
            if outcome.status == "failed" and stop_on_error:
                break
    finally:
        ctx.clear()  # never keep a chapter's tensors alive past its pass
    return outcomes


def run_series(
    stages: Sequence[Stage],
    cfg: Config,
    series: str,
    chapters: Sequence[str] | None = None,
    *,
    gpu: GpuScheduler | None = None,
    force: bool = False,
    after_stage: AfterStage | None = None,
    merge_series_config: bool = True,
) -> dict[str, list[StageOutcome]]:
    """Run a pass (ordered stages) over chapters in reading order. Model groups stay loaded across chapters.

    Raises RunAbortedError (with `.results` = the chapters finished before it) when `after_stage` returns False.
    `merge_series_config` is forwarded to `make_context` for every chapter (see its docstring)."""
    names = list(chapters) if chapters is not None else SeriesPaths.from_config(cfg, series).chapters()
    results: dict[str, list[StageOutcome]] = {}
    for chapter in names:
        ctx = make_context(cfg, series, chapter, gpu, merge_series_config=merge_series_config)
        try:
            results[chapter] = run_chapter(stages, ctx, force=force, after_stage=after_stage)
        except RunAbortedError as aborted:
            aborted.results = results
            raise
    return results
