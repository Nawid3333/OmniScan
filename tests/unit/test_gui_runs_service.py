"""RunService tests (Qt-free): spec validation, live progress, cancel, step gate, client/GPU lifetimes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from omniscan.core.config import Config
from omniscan.core.stage import RunAbortedError, StageOutcome
from omniscan.gui.services.runs import (
    GateChannel,
    RunController,
    RunSpec,
    StageUpdate,
    StepPreview,
    validate_spec,
)
from omniscan.pipeline.runner import GateEvent, PipelineResult
from tests.fixtures.gui_library import SERIES, build_library


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library (real series/chapter names, no pipeline runs)."""
    return build_library(tmp_path / "lib")


class FakeRunner:
    """run_pipeline stand-in: drives the controller's hooks exactly as the real runner does."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        cfg: Config,
        series: str,
        chapters: list[str] | None = None,
        *,
        stages: list[str],
        lama: bool,
        force: bool,
        client: object,
        gpu: object,
        report: object,
        mode: str,
        preview_chapter: str | None,
        gate: object,
        after_stage: object,
    ) -> PipelineResult:
        self.calls.append(
            {
                "series": series,
                "chapters": list(chapters or []),
                "stages": list(stages or []),
                "lama": lama,
                "force": force,
                "client": client,
                "gpu": gpu,
                "mode": mode,
                "preview_chapter": preview_chapter,
            }
        )
        outcomes: dict[str, list[StageOutcome]] = {}
        try:
            for chapter in chapters or []:
                chapter_outcomes: list[StageOutcome] = []
                for position, stage in enumerate(stages or [], start=1):
                    outcome = StageOutcome(stage=stage, status="done", seconds=0.1)
                    assert callable(after_stage)
                    ctx = SimpleNamespace(paths=SimpleNamespace(chapter=chapter))
                    if not after_stage(ctx, outcome):
                        error = RunAbortedError(chapter, stage)
                        error.results = dict(outcomes)  # chapters finished before this one
                        error.outcomes = chapter_outcomes
                        raise error
                    chapter_outcomes.append(outcome)
                    outcomes[chapter] = chapter_outcomes
                    preview_chapter_name = (
                        preview_chapter if preview_chapter is not None else (chapters or [None])[0]
                    )
                    if mode == "step" and chapter == preview_chapter_name:
                        assert callable(gate)
                        if not gate(
                            GateEvent(
                                series=series,
                                chapter=chapter,
                                stage=stage,
                                outcome=outcome,
                                position=position,
                                total=len(stages or []),
                            )
                        ):
                            return PipelineResult(outcomes=outcomes, failed={}, aborted="preview failed")
        except RunAbortedError as error:  # the runner merges the partial results, it never re-raises
            merged = {**error.results, error.chapter: error.outcomes}
            return PipelineResult(outcomes=merged, failed={}, aborted="stopped")
        return PipelineResult(outcomes=outcomes, failed={}, aborted=None)


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll `predicate` until it is true (the step-gate handshake needs a second thread)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------- validate_spec


def test_validate_spec_accepts_a_real_spec(cfg: Config) -> None:
    """A full run and a narrowed subset both pass."""
    validate_spec(cfg, RunSpec(series=SERIES))
    validate_spec(cfg, RunSpec(series=SERIES, chapters=("Episode 02",), stages=("ocr", "export")))


def test_validate_spec_rejects_bad_specs(cfg: Config) -> None:
    """Every refusal names the problem."""
    with pytest.raises(ValueError, match="has no chapters"):
        validate_spec(cfg, RunSpec(series="Nope"))
    with pytest.raises(ValueError, match="unknown stage"):
        validate_spec(cfg, RunSpec(series=SERIES, stages=("bogus",)))
    with pytest.raises(ValueError, match="select at least one stage"):
        validate_spec(cfg, RunSpec(series=SERIES, stages=()))
    with pytest.raises(ValueError, match="unknown chapter"):
        validate_spec(cfg, RunSpec(series=SERIES, chapters=("Episode 99",)))
    with pytest.raises(ValueError, match="unknown preview chapter"):
        validate_spec(cfg, RunSpec(series=SERIES, mode="step", preview_chapter="Episode 99"))


# ---------------------------------------------------------------------- progress + lifetimes


def test_reports_every_stage_and_needs_no_worker_for_vision_only_stages(cfg: Config) -> None:
    """Every stage outcome arrives live; ingest/slice runs build neither a client nor a GPU."""
    updates: list[StageUpdate] = []
    runner = FakeRunner()
    built_clients: list[str] = []
    built_gpus: list[str] = []

    controller = RunController(
        cfg,
        RunSpec(series=SERIES, chapters=("Episode 01", "Episode 03"), stages=("ingest", "slice")),
        run_fn=runner,
        client_factory=lambda _cfg: built_clients.append("built") or None,
        gpu_factory=lambda _cfg: built_gpus.append("built") or None,
    )
    outcome = controller.run(
        on_stage=updates.append,
        on_preview=lambda _p: None,
        cancel_event=threading.Event(),
        channel=GateChannel(),
    )

    assert [
        (u.chapter, u.stage, u.chapter_number, u.chapter_total, u.stage_number, u.stage_total)
        for u in updates
    ] == [
        ("Episode 01", "ingest", 1, 2, 1, 2),
        ("Episode 01", "slice", 1, 2, 2, 2),
        ("Episode 03", "ingest", 2, 2, 1, 2),
        ("Episode 03", "slice", 2, 2, 2, 2),
    ]
    assert outcome.ok and outcome.aborted is None and outcome.chapters_run == 2
    assert runner.calls[0]["client"] is None
    assert runner.calls[0]["gpu"] is None
    assert built_clients == [] and built_gpus == []


def test_text_stage_builds_and_closes_the_chat_client(cfg: Config) -> None:
    """A text stage builds the client once and closes it in `finally`, even when the run raises."""
    runner = FakeRunner()
    closed: list[str] = []

    class Client:
        def close(self) -> None:
            closed.append("closed")

    controller = RunController(
        cfg,
        RunSpec(series=SERIES, chapters=("Episode 01",), stages=("ingest", "translate")),
        run_fn=runner,
        client_factory=lambda _cfg: Client(),
        gpu_factory=lambda _cfg: None,
    )
    outcome = controller.run(
        on_stage=lambda _u: None,
        on_preview=lambda _p: None,
        cancel_event=threading.Event(),
        channel=GateChannel(),
    )
    assert outcome.ok
    assert closed == ["closed"]
    assert runner.calls[0]["mode"] == "auto"  # Full/Subset/Auto all run in auto mode today
    assert runner.calls[0]["lama"] is True and runner.calls[0]["force"] is False


def test_failed_chapter_reaches_the_outcome(cfg: Config) -> None:
    """The runner's failed dict maps to (chapter, error) tuples."""
    runner = FakeRunner()

    def run(cfg: Config, series: str, chapters: list[str] | None = None, **_kw: object) -> PipelineResult:
        runner.calls.append({"series": series})
        return PipelineResult(outcomes={}, failed={"Episode 01": "ocr: boom"}, aborted=None)

    controller = RunController(
        cfg,
        RunSpec(series=SERIES, chapters=("Episode 01",), stages=("ingest",)),
        run_fn=run,
        client_factory=lambda _cfg: None,
        gpu_factory=lambda _cfg: None,
    )
    outcome = controller.run(
        on_stage=lambda _u: None,
        on_preview=lambda _p: None,
        cancel_event=threading.Event(),
        channel=GateChannel(),
    )
    assert outcome.failed == (("Episode 01", "ocr: boom"),)
    assert not outcome.ok


def test_gpu_run_takes_and_releases_the_exclusive_gpu_lock(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU spec acquires the exclusive lock before the manager is built and releases it in `finally`."""
    import omniscan.gpu.lock as lock_module
    import omniscan.gui.services.runs as runs_module

    events: list[str] = []

    def fake_acquire(*, poll_seconds: float = 2.0, on_wait: object = None) -> object:
        events.append("acquire")
        return ("handle",)

    def fake_release(handle: object) -> None:
        events.append("release")

    class FakeGpu:
        released = False

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(lock_module, "acquire_gpu_lock", fake_acquire)
    monkeypatch.setattr(lock_module, "release_gpu_lock", fake_release)
    monkeypatch.setattr(runs_module, "needs_gpu", lambda *_a: True)
    gpu = FakeGpu()
    runner = FakeRunner()

    controller = RunController(
        cfg,
        RunSpec(series=SERIES, chapters=("Episode 01",), stages=("ingest",)),
        run_fn=runner,
        client_factory=lambda _cfg: None,
        gpu_factory=lambda _cfg: gpu,
    )
    outcome = controller.run(
        on_stage=lambda _u: None,
        on_preview=lambda _p: None,
        cancel_event=threading.Event(),
        channel=GateChannel(),
    )

    assert outcome.ok
    assert events == ["acquire", "release"]
    assert gpu.released


# ---------------------------------------------------------------------- cancel


def test_cancel_stops_after_the_current_stage(cfg: Config) -> None:
    """Cancel after stage two: the finished stage is still reported, the run ends `aborted="stopped"`."""
    updates: list[StageUpdate] = []
    runner = FakeRunner()
    cancel_event = threading.Event()

    def on_stage(update: StageUpdate) -> None:
        updates.append(update)
        if len(updates) == 2:  # the user pressed Cancel while stage two ran
            cancel_event.set()

    controller = RunController(
        cfg,
        RunSpec(series=SERIES, chapters=("Episode 01",), stages=("ingest", "slice")),
        run_fn=runner,
        client_factory=lambda _cfg: None,
        gpu_factory=lambda _cfg: None,
    )
    outcome = controller.run(
        on_stage=on_stage, on_preview=lambda _p: None, cancel_event=cancel_event, channel=GateChannel()
    )
    assert outcome.aborted == "stopped" and not outcome.ok
    assert len(updates) == 2  # ingest and slice reported; nothing else ran


# ---------------------------------------------------------------------- step gate


def test_step_gate_continue_then_abort(cfg: Config) -> None:
    """The worker blocks in the gate until answered; Continue proceeds, Abort ends the run."""
    previews: list[StepPreview] = []
    runner = FakeRunner()
    channel = GateChannel()
    controller = RunController(
        cfg,
        RunSpec(series=SERIES, chapters=("Episode 01",), stages=("ingest", "slice"), mode="step"),
        run_fn=runner,
        client_factory=lambda _cfg: None,
        gpu_factory=lambda _cfg: None,
    )
    done: dict[str, object] = {}

    def run() -> None:
        done["outcome"] = controller.run(
            on_stage=lambda _u: None,
            on_preview=previews.append,
            cancel_event=threading.Event(),
            channel=channel,
        )

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert _wait(lambda: len(previews) >= 1)
        assert previews[0].position == 1 and previews[0].total == 2 and previews[0].stage == "ingest"
        channel.respond("continue")
        assert _wait(lambda: len(previews) >= 2)
        assert previews[1].position == 2 and previews[1].stage == "slice"
        channel.respond("abort")
        thread.join(5)
    finally:
        if thread.is_alive():
            channel.respond("abort")
            thread.join(5)
    assert not thread.is_alive()
    assert done["outcome"].aborted == "preview failed"  # type: ignore[union-attr]
    assert runner.calls[0]["preview_chapter"] is None  # unset: the runner picks the first chapter
