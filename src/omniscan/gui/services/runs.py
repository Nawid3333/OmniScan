"""Run service: one pipeline run driven through callbacks (Qt-free; the Run page's worker drives it).

The controller is the GUI's single path into `pipeline.runner.run_pipeline`: stage outcomes arrive
live via `on_stage` (the runner's `after_stage` hook fires after every finished stage in every
mode), step mode blocks on a `GateChannel` after every stage of the preview chapter until the GUI
answers Continue/Abort, and Cancel is a `threading.Event` the same hook checks — the run then stops
after the stage that just finished (`PipelineResult.aborted == "stopped"`). The controller owns the
chat client and GPU scheduler lifetime exactly like the CLI `run` command (built on demand,
released in `finally`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any, Literal

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.stage import StageOutcome
from omniscan.pipeline.preview import describe
from omniscan.pipeline.runner import GateEvent, PipelineResult, needs_gpu, run_pipeline
from omniscan.pipeline.stages import PASS_OF, STAGE_ORDER

type RunMode = Literal["full", "subset", "step", "auto"]
type RunFn = Callable[..., PipelineResult]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """What the Run page asked for: series, mode, chapters, stages and the step-mode preview."""

    series: str
    mode: RunMode = "full"
    chapters: tuple[str, ...] = ()  # empty = every chapter of the series
    stages: tuple[str, ...] = STAGE_ORDER
    lama: bool = True
    force: bool = False
    preview_chapter: str | None = None  # step mode; the runner defaults to the first chapter


@dataclass(frozen=True, slots=True)
class StageUpdate:
    """One finished stage outcome, live from the runner's `after_stage` hook."""

    chapter: str
    stage: str
    status: str  # done | skipped | failed
    seconds: float
    error: str | None
    chapter_number: int  # 1-based position among the run's chapters
    chapter_total: int
    stage_number: int  # 1-based count of stages already reported for this chapter
    stage_total: int


@dataclass(frozen=True, slots=True)
class StepPreview:
    """What the step-mode gate shows for one finished stage of the preview chapter."""

    stage: str
    status: str
    summary: str
    details: tuple[str, ...]
    position: int  # 1-based number of this stage among the selected stages
    total: int
    error: str | None


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Summary of a finished `run_pipeline` call, as the log view and status bar show it."""

    ok: bool
    aborted: str | None  # None | "stopped" (cancelled) | "rate limit" | "preview failed"
    failed: tuple[tuple[str, str], ...]  # (chapter, "<stage>: <error>")
    chapters_run: int


@dataclass(slots=True)
class GateChannel:
    """The Continue/Abort handshake between the worker's gate hook and the GUI.

    The worker clears `resume`, emits the preview and blocks on `resume.wait()`; the GUI calls
    `respond` exactly once per preview (the Run page disables its buttons between previews, so a
    stale release cannot arm the next gate).
    """

    resume: threading.Event = field(default_factory=threading.Event)
    answer: str = "abort"  # the value the GUI wrote before releasing

    def respond(self, answer: Literal["continue", "abort"]) -> None:
        """Write the answer and release the blocked gate."""
        self.answer = answer
        self.resume.set()


def validate_spec(cfg: Config, spec: RunSpec) -> None:
    """Refuse a spec the runner cannot run: unknown series/stages, empty selection, bad preview chapter.

    Raises ValueError with a one-line message the Run page shows inline.
    """
    available = tuple(SeriesPaths.from_config(cfg, spec.series).chapters())
    if not available:
        raise ValueError(f"series {spec.series!r} has no chapters")
    unknown_stages = [name for name in spec.stages if name not in STAGE_ORDER]
    if unknown_stages:
        raise ValueError(f"unknown stage(s): {', '.join(unknown_stages)}")
    if not spec.stages:
        raise ValueError("select at least one stage")
    unknown_chapters = [name for name in spec.chapters if name not in available]
    if unknown_chapters:
        raise ValueError(f"unknown chapter(s): {', '.join(unknown_chapters)}")
    if spec.mode == "step" and spec.preview_chapter is not None and spec.preview_chapter not in available:
        raise ValueError(f"unknown preview chapter {spec.preview_chapter!r}")


class RunController:
    """One pipeline run with the GUI's progress/cancel/gate hooks wired in."""

    def __init__(
        self,
        cfg: Config,
        spec: RunSpec,
        *,
        run_fn: RunFn | None = None,
        client_factory: Callable[[Config], Any] | None = None,
        gpu_factory: Callable[[Config], Any] | None = None,  # the CLI's manager also has release()
    ) -> None:
        """Build the controller; the factories default to the CLI's (OllamaClient / build_vram_manager)."""
        self._cfg = cfg
        self._spec = spec
        self._run_fn = run_fn or run_pipeline
        self._client_factory = client_factory or _default_client_factory
        self._gpu_factory = gpu_factory or _default_gpu_factory

    def run(
        self,
        *,
        on_stage: Callable[[StageUpdate], None],
        on_preview: Callable[[StepPreview], None],
        cancel_event: threading.Event,
        channel: GateChannel,
    ) -> RunOutcome:
        """Run the spec on the calling thread; returns the summary (a cancel ends with `aborted="stopped"`)."""
        cfg, spec = self._cfg, self._spec
        names = list(spec.stages)
        chapters = list(spec.chapters) or list(SeriesPaths.from_config(cfg, spec.series).chapters())
        if not chapters:
            raise ValueError(f"series {spec.series!r} has no chapters")
        chapter_of = {name: index + 1 for index, name in enumerate(chapters)}
        stages_seen: dict[str, int] = {}
        is_step = spec.mode == "step"

        def report_update(chapter: str, outcome: StageOutcome) -> None:
            number = stages_seen.get(chapter, 0) + 1
            stages_seen[chapter] = number
            on_stage(
                StageUpdate(
                    chapter=chapter,
                    stage=outcome.stage,
                    status=outcome.status,
                    seconds=outcome.seconds,
                    error=outcome.error,
                    chapter_number=chapter_of[chapter],
                    chapter_total=len(chapters),
                    stage_number=number,
                    stage_total=len(names),
                )
            )

        def after_stage(ctx: Any, outcome: StageOutcome) -> bool:
            """Live progress after every finished stage; False = the user pressed Cancel."""
            report_update(ctx.paths.chapter, outcome)
            return not cancel_event.is_set()

        def gate(event: GateEvent) -> bool:
            """Show the preview block for the just-finished stage and wait for the GUI's answer."""
            if cancel_event.is_set():
                return False
            paths = SeriesPaths.from_config(cfg, spec.series).chapter(event.chapter)
            described = describe(event.stage, paths, cfg)
            channel.resume.clear()  # arm before showing: a respond() can only follow this preview
            on_preview(
                StepPreview(
                    stage=event.stage,
                    status=event.outcome.status,
                    summary=described.summary,
                    details=described.details,
                    position=event.position,
                    total=event.total,
                    error=event.outcome.error,
                )
            )
            channel.resume.wait()
            return channel.answer == "continue" and not cancel_event.is_set()

        # Auto is Full today (the card's placeholder): when Auto gains its own behaviour it diverges
        # on `spec.mode == "auto"` here (e.g. its own chapter policy); run_pipeline stays mode="auto".
        mode = "step" if is_step else "auto"
        client = self._client_factory(cfg) if any(PASS_OF[name] == "text" for name in names) else None
        gpu: Any = None  # opaque: the factory's product goes to run_pipeline and is released here
        hw_lock: IO[bytes] | None = None
        try:
            if needs_gpu(names, cfg, client):
                if cfg.gpu.device != "cpu":
                    # exclusive real-GPU access before the manager's warm-up touches the device,
                    # exactly like the CLI `run` command (the manager's warm-up thread needs it too)
                    from omniscan.gpu.lock import acquire_gpu_lock

                    hw_lock = acquire_gpu_lock()
                gpu = self._gpu_factory(cfg)
            result = self._run_fn(
                cfg,
                spec.series,
                chapters=chapters,
                stages=names,
                lama=spec.lama,
                force=spec.force,
                client=client,
                gpu=gpu,
                report=None,  # after_stage already reports every outcome live, including aborts
                mode=mode,
                preview_chapter=spec.preview_chapter if is_step else None,
                gate=gate if is_step else None,
                after_stage=after_stage,
            )
        finally:
            if gpu is not None:
                gpu.release()  # the models leave VRAM when the run ends (as the CLI does)
            if hw_lock is not None:
                from omniscan.gpu.lock import release_gpu_lock

                release_gpu_lock(hw_lock)
            if client is not None:
                client.close()
        return RunOutcome(
            ok=result.ok,
            aborted=result.aborted,
            failed=tuple(result.failed.items()),
            chapters_run=len(result.outcomes),
        )


def _default_client_factory(cfg: Config) -> Any:
    """The CLI's chat client (built lazily so importing this module never pulls the LLM stack)."""
    from omniscan.core.config import get_secrets
    from omniscan.llm.ollama import OllamaClient

    return OllamaClient(cfg.ollama, get_secrets())


def _default_gpu_factory(cfg: Config) -> Any:
    """The CLI's VRAM manager (built lazily: importing `omniscan.gpu.groups` pulls torch)."""
    from omniscan.gpu.groups import build_vram_manager

    return build_vram_manager(cfg)
