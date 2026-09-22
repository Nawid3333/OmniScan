"""The pipeline runner: ordered passes over all chapters, so each model group is loaded once.

`plan_passes` splits the selected stages into three passes (vision, text, render) and `run_pipeline`
executes them with the core stage runner — one `run_series` call per pass, so a model group stays
resident across every chapter of its pass. Chapters whose stage fails are dropped from the later
passes; an Ollama rate limit stops the whole run. Step mode (`mode="step"`) runs the preview chapter
through all passes behind a gate first, then the remaining chapters automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from omniscan.core.config import series_config
from omniscan.core.paths import SeriesPaths
from omniscan.core.stage import (
    AfterStage,
    ChapterContext,
    GpuScheduler,
    RunAbortedError,
    Stage,
    StageOutcome,
    run_series,
)
from omniscan.gpu.timeline import mark
from omniscan.gpu.vram import OLLAMA_GROUP
from omniscan.pipeline.stages import PASS_OF, STAGE_ORDER, build_stage

if TYPE_CHECKING:
    from omniscan.core.config import Config
    from omniscan.translate.run import ChatClient

ReportFn = Callable[[str, StageOutcome], None]

type RunMode = Literal["auto", "step"]


@runtime_checkable
class PrefetchScheduler(Protocol):
    """A GpuScheduler that can preload a group's models on a background thread (omniscan.gpu.vram.VramManager)."""

    def prefetch(self, group: str) -> bool: ...


class TimedStage:
    """Stage proxy that brackets `run()` with timeline marks; everything else delegates."""

    def __init__(self, stage: Stage) -> None:
        self._stage = stage

    @property
    def name(self) -> str:
        return self._stage.name

    @property
    def version(self) -> int:
        return self._stage.version

    @property
    def gpu_group(self) -> str | None:
        return self._stage.gpu_group

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        return self._stage.inputs(ctx)

    def outputs(self, ctx: ChapterContext) -> list[str]:
        return self._stage.outputs(ctx)

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        return self._stage.config_subset(cfg)

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        name = self._stage.name
        mark(f"stage {name} begin")
        try:
            return self._stage.run(ctx, models)
        finally:
            mark(f"stage {name} end")


def _timed(stages: Sequence[str], cfg: Config, client: ChatClient | None) -> list[Stage]:
    """The pass's freshly built stages, each wrapped in a TimedStage."""
    return [TimedStage(build_stage(name, cfg, client=client)) for name in stages]  # pyright: ignore[reportReturnType]


@dataclass(frozen=True, slots=True)
class GateEvent:
    """One stage outcome of the preview chapter, offered to the step-mode gate."""

    series: str
    chapter: str  # the preview chapter
    stage: str  # the stage that just finished for it
    outcome: StageOutcome  # done | skipped | failed
    position: int  # 1-based number of this stage among the selected stages
    total: int  # number of selected stages (after --no-lama filtering)


type Gate = Callable[[GateEvent], bool]  # True = continue, False = stop the run


def plan_passes(stages: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group stage names into consecutive passes sharing a `PASS_OF` label, ordered by `STAGE_ORDER`."""
    unknown = next((name for name in stages if name not in STAGE_ORDER), None)
    if unknown is not None:
        raise ValueError(f"unknown stage {unknown!r} (known: {', '.join(STAGE_ORDER)})")
    selected = set(stages)
    passes: list[tuple[str, list[str]]] = []
    for name in (name for name in STAGE_ORDER if name in selected):
        label = PASS_OF[name]
        if passes and passes[-1][0] == label:
            passes[-1][1].append(name)
        else:
            passes.append((label, [name]))
    return passes


@dataclass(slots=True)
class PipelineResult:
    """What one `run_pipeline` call did: per-chapter outcomes, first failures, rate-limit abort."""

    outcomes: dict[str, list[StageOutcome]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)  # chapter -> "<stage>: <error>"
    aborted: str | None = None  # "rate limit", "stopped" (the gate said stop) or "preview failed"

    @property
    def ok(self) -> bool:
        """True when no chapter failed and the run was not aborted."""
        return not self.failed and self.aborted is None


def _record_pass_results(
    result: PipelineResult,
    pass_results: Mapping[str, list[StageOutcome]],
    report: ReportFn | None,
) -> None:
    """Store one pass's outcomes, report them and record each chapter's first failure."""
    for chapter, outcomes in pass_results.items():
        result.outcomes.setdefault(chapter, []).extend(outcomes)
        if report is not None:
            for outcome in outcomes:
                report(chapter, outcome)
        first_failure = next((o for o in outcomes if o.status == "failed"), None)
        if first_failure is not None and first_failure.error is not None:
            result.failed.setdefault(chapter, f"{first_failure.stage}: {first_failure.error}")
            if first_failure.error.startswith("OllamaRateLimitError"):
                result.aborted = "rate limit"


def _run_preview(
    result: PipelineResult,
    cfg: Config,
    series: str,
    preview_chapter: str,
    passes: list[tuple[str, list[str]]],
    *,
    client: ChatClient | None,
    gpu: GpuScheduler | None,
    force: bool,
    report: ReportFn | None,
    gate: Gate,
    after_stage: AfterStage | None,
    total: int,
) -> bool:
    """Step-mode phase A: run the preview chapter through all passes behind `gate`; False stops the run.

    The preview chapter's outcomes and failures are stored into `result` exactly as the auto loop does;
    a gate (or `after_stage`, checked first — a cancel while paused doubles as "no") that answers False
    raises RunAbortedError out of `run_series` and is recorded as `aborted = "stopped"`, a failed preview
    stage as `aborted = "preview failed"`.
    """
    position = 0

    def hook(ctx: ChapterContext, outcome: StageOutcome) -> bool:
        nonlocal position
        position += 1
        if after_stage is not None and not after_stage(ctx, outcome):
            return False
        return gate(
            GateEvent(
                series=series,
                chapter=ctx.paths.chapter,
                stage=outcome.stage,
                outcome=outcome,
                position=position,
                total=total,
            )
        )

    for _label, pass_stages in passes:
        stage_objects = _timed(pass_stages, cfg, client)
        try:
            pass_results = run_series(
                stage_objects, cfg, series, [preview_chapter], gpu=gpu, force=force, after_stage=hook
            )
        except RunAbortedError as error:
            result.outcomes.setdefault(error.chapter, []).extend(error.outcomes)
            if report is not None:
                for outcome in error.outcomes:
                    report(error.chapter, outcome)
            result.aborted = "stopped"
            return False
        _record_pass_results(result, pass_results, report)
        if preview_chapter in result.failed:
            rate_limited = "OllamaRateLimitError" in result.failed[preview_chapter]
            result.aborted = "rate limit" if rate_limited else "preview failed"
            return False
    return True


def _prefetch_groups(
    gpu: GpuScheduler | None, passes: list[tuple[str, list[str]]], cfg: Config, client: ChatClient | None
) -> None:
    """Queue background loads for every GPU model group except the first, unless `gpu.warmup` is off.

    The first group loads synchronously at its acquire: started early it competes with the pipeline's own
    startup work (strip decode, GPU library init) and slows both. Later groups hide behind the vision pass
    instead. A scheduler without prefetch (e.g. a test double) is skipped; OLLAMA_GROUP is not a registered
    torch group, and a prefetch that would not fit free VRAM is skipped inside the scheduler anyway; failed
    prefetches fall back to the synchronous load in `acquire`.
    """
    if gpu is None or not cfg.gpu.warmup:
        return
    seen: set[str] = set()
    first = True
    for _label, pass_stages in passes:
        for name in pass_stages:
            group = build_stage(name, cfg, client=client).gpu_group
            if (
                group is None
                or group == OLLAMA_GROUP
                or group in seen
                or not isinstance(gpu, PrefetchScheduler)
            ):
                continue
            seen.add(group)
            if first:
                first = False
                continue
            gpu.prefetch(group)


def run_pipeline(
    cfg: Config,
    series: str,
    chapters: Sequence[str] | None = None,
    *,
    stages: Sequence[str] | None = None,
    lama: bool = True,
    force: bool = False,
    client: ChatClient | None = None,
    gpu: GpuScheduler | None = None,
    report: ReportFn | None = None,
    mode: RunMode = "auto",
    preview_chapter: str | None = None,
    gate: Gate | None = None,
    after_stage: AfterStage | None = None,
) -> PipelineResult:
    """Run the selected stages over the chapters in passes; failed chapters skip the later passes.

    In step mode (`mode="step"`) `preview_chapter` (default: the first chapter) runs through all
    passes first, with `gate` asked after every stage; when it gets through, the remaining chapters
    run automatically as before. The caller owns `client` and `gpu`: the runner never builds either.
    `report` is called once per outcome after its pass finished (in chapter order, then stage order).
    `after_stage`, when given, is checked after every stage in every mode (in step mode, before the
    gate): it returning False cancels the run (`PipelineResult.aborted == "stopped"`) after the stage
    that just finished — the same one-stage-granularity cancellation `core.stage.run_series` already
    gives step mode, now available to a plain run too (a GUI Cancel button's hook, for example).
    """
    cfg = series_config(cfg, SeriesPaths.from_config(cfg, series).library_dir)
    names = list(stages) if stages is not None else list(STAGE_ORDER)
    if not lama:
        names = [name for name in names if name != "inpaint_lama"]
    passes = plan_passes(names)  # validates the names and puts them in STAGE_ORDER
    all_chapters = list(chapters) if chapters is not None else SeriesPaths.from_config(cfg, series).chapters()
    if not all_chapters:
        raise ValueError(f"no chapters found for series {series!r}")
    preview: str | None = None
    step_gate: Gate | None = None
    if mode == "step":
        if gate is None:
            raise ValueError("step mode needs a gate")
        candidate = preview_chapter if preview_chapter is not None else all_chapters[0]
        if candidate not in all_chapters:
            raise ValueError(f"unknown preview chapter {candidate!r}")
        preview = candidate
        step_gate = gate  # narrowed to Gate by the check above
    needs_client = next(
        (name for _label, pass_stages in passes for name in pass_stages if PASS_OF[name] == "text"), None
    )
    if needs_client is not None and client is None:
        raise ValueError(f"stage {needs_client!r} needs a chat client")
    _prefetch_groups(gpu, passes, cfg, client)

    result = PipelineResult()
    if preview is not None and step_gate is not None:
        completed = _run_preview(
            result,
            cfg,
            series,
            preview,
            passes,
            client=client,
            gpu=gpu,
            force=force,
            report=report,
            gate=step_gate,
            after_stage=after_stage,
            total=len(names),
        )
        if not completed:
            return result
        active = [chapter for chapter in all_chapters if chapter != preview]
    else:
        active = all_chapters
    for _label, pass_stages in passes:
        stage_objects = _timed(pass_stages, cfg, client)
        try:
            pass_results = run_series(
                stage_objects, cfg, series, active, gpu=gpu, force=force, after_stage=after_stage
            )
        except RunAbortedError as error:
            for chapter, outcomes in {**error.results, error.chapter: error.outcomes}.items():
                result.outcomes.setdefault(chapter, []).extend(outcomes)
                if report is not None:
                    for outcome in outcomes:
                        report(chapter, outcome)
            result.aborted = "stopped"
            break
        _record_pass_results(result, pass_results, report)
        if result.aborted is not None:
            break
        active = [chapter for chapter in active if chapter not in result.failed]
    return result


def needs_gpu(names: Sequence[str], cfg: Config, client: ChatClient | None) -> bool:
    """True when any stage built for `names` needs a GPU model group (translate/judge need a client)."""
    return any(build_stage(name, cfg, client=client).gpu_group is not None for name in names)
