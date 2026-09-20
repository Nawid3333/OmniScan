from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from omniscan.core.config import Config, PathsConfig
from omniscan.core.manifest import hash_inputs, load_manifest
from omniscan.core.stage import (
    ChapterContext,
    RunAbortedError,
    StageOutcome,
    make_context,
    run_chapter,
    run_series,
    run_stage,
)


class FakeGpu:
    def __init__(self) -> None:
        self.acquired: list[str] = []

    def acquire(self, group: str) -> Mapping[str, Any]:
        self.acquired.append(group)
        return {"model": group}

    def reset_peak(self) -> None: ...

    def peak_gib(self) -> float:
        return 1.5


class CopyStage:
    """Upper-cases input.txt into out.txt; counts runs."""

    name: ClassVar[str] = "copy"
    version: ClassVar[int] = 1
    gpu_group: ClassVar[str | None] = None
    runs = 0

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        return [ctx.paths.raw_dir / "input.txt"]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        return ["out.txt"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        return {"tol": cfg.slicer.uniform_tol}

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        type(self).runs += 1
        text = (ctx.paths.raw_dir / "input.txt").read_text()
        ctx.put("text", text)
        ctx.paths.artifact("out.txt").write_text(text.upper())
        return {"chars": float(len(text))}


class GpuStage(CopyStage):
    name: ClassVar[str] = "gpu"
    gpu_group: ClassVar[str | None] = "vision"
    seen_models: ClassVar[list[Mapping[str, Any]]] = []

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        return [ctx.paths.artifact("out.txt")]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        return ["gpu.txt"]

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        type(self).seen_models.append(models)
        # value produced by the previous stage in the same pass
        ctx.paths.artifact("gpu.txt").write_text(ctx.lazy("text", lambda: "recomputed"))
        return {}


class BoomStage(CopyStage):
    name: ClassVar[str] = "boom"

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        raise RuntimeError("boom")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    raw = tmp_path / "lib" / "S" / "Chapter 1"
    raw.mkdir(parents=True)
    (raw / "input.txt").write_text("hello")
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


def test_stage_runs_then_skips_then_reruns_on_change(cfg: Config) -> None:
    CopyStage.runs = 0
    ctx = make_context(cfg, "S", "Chapter 1")
    assert run_stage(CopyStage(), ctx).status == "done"
    assert run_stage(CopyStage(), make_context(cfg, "S", "Chapter 1")).status == "skipped"
    (ctx.paths.raw_dir / "input.txt").write_text("changed")
    assert run_stage(CopyStage(), make_context(cfg, "S", "Chapter 1")).status == "done"
    assert CopyStage.runs == 2


def test_config_change_invalidates(cfg: Config) -> None:
    run_stage(CopyStage(), make_context(cfg, "S", "Chapter 1"))
    cfg2 = cfg.model_copy(update={"slicer": cfg.slicer.model_copy(update={"uniform_tol": 99})})
    assert run_stage(CopyStage(), make_context(cfg2, "S", "Chapter 1")).status == "done"


def test_missing_output_forces_rerun(cfg: Config) -> None:
    ctx = make_context(cfg, "S", "Chapter 1")
    run_stage(CopyStage(), ctx)
    ctx.paths.artifact("out.txt").unlink()
    assert run_stage(CopyStage(), make_context(cfg, "S", "Chapter 1")).status == "done"


def test_manifest_records_metrics_and_hash(cfg: Config) -> None:
    ctx = make_context(cfg, "S", "Chapter 1")
    run_stage(CopyStage(), ctx)
    manifest = load_manifest(ctx.paths.manifest, "S", "Chapter 1")
    rec = manifest.stages["copy"]
    assert rec.status == "done"
    assert rec.metrics["chars"] == 5.0 and "seconds" in rec.metrics
    assert rec.input_hash == hash_inputs([ctx.paths.raw_dir / "input.txt"])


def test_gpu_group_acquired_and_memo_shared(cfg: Config) -> None:
    gpu = FakeGpu()
    GpuStage.seen_models = []
    ctx = make_context(cfg, "S", "Chapter 1", gpu)
    outcomes = run_chapter([CopyStage(), GpuStage()], ctx, force=True)
    assert [o.status for o in outcomes] == ["done", "done"]
    assert gpu.acquired == ["vision"]
    assert GpuStage.seen_models == [{"model": "vision"}]
    assert ctx.paths.artifact("gpu.txt").read_text() == "hello"
    assert outcomes[1].metrics["peak_vram_gib"] == 1.5
    assert ctx._memo == {}  # cleared after the pass


def test_gpu_stage_without_scheduler_raises(cfg: Config) -> None:
    ctx = make_context(cfg, "S", "Chapter 1")
    run_stage(CopyStage(), ctx)
    with pytest.raises(RuntimeError, match="needs GPU group"):
        run_stage(GpuStage(), ctx)


def test_failure_recorded_and_stops_chapter(cfg: Config) -> None:
    ctx = make_context(cfg, "S", "Chapter 1")
    outcomes = run_chapter([BoomStage(), CopyStage()], ctx)
    assert [o.status for o in outcomes] == ["failed"]
    assert outcomes[0].error == "RuntimeError: boom"
    manifest = load_manifest(ctx.paths.manifest, "S", "Chapter 1")
    assert manifest.stages["boom"].status == "failed"
    # a failed stage is never "up to date"
    assert run_stage(BoomStage(), make_context(cfg, "S", "Chapter 1")).status == "failed"


def test_run_series_discovers_chapters(cfg: Config, tmp_path: Path) -> None:
    raw2 = tmp_path / "lib" / "S" / "Chapter 2"
    raw2.mkdir()
    (raw2 / "input.txt").write_text("two")
    results = run_series([CopyStage()], cfg, "S", force=True)
    assert list(results) == ["Chapter 1", "Chapter 2"]


def test_after_stage_sees_every_outcome_and_can_abort(cfg: Config) -> None:
    seen: list[tuple[str, str]] = []

    def hook(ctx: ChapterContext, outcome: StageOutcome) -> bool:
        seen.append((ctx.paths.chapter, outcome.stage))
        assert ctx.lazy("text", lambda: "gone") == "hello"  # in-memory values are still alive
        return outcome.stage != "gpu"

    gpu = FakeGpu()
    with pytest.raises(RunAbortedError) as info:
        run_chapter(
            [CopyStage(), GpuStage(), CopyStage()], make_context(cfg, "S", "Chapter 1", gpu), after_stage=hook
        )
    assert seen == [("Chapter 1", "copy"), ("Chapter 1", "gpu")]
    assert (info.value.chapter, info.value.stage) == ("Chapter 1", "gpu")
    assert [o.stage for o in info.value.outcomes] == ["copy", "gpu"]
    manifest = load_manifest(make_context(cfg, "S", "Chapter 1").paths.manifest, "S", "Chapter 1")
    assert manifest.stages["gpu"].status == "done"  # the aborting stage is recorded


def test_run_series_abort_carries_finished_chapters(cfg: Config) -> None:
    second = cfg.paths.library_root / "S" / "Chapter 2"
    second.mkdir(parents=True)
    (second / "input.txt").write_text("again")
    calls = 0

    def hook(ctx: ChapterContext, outcome: StageOutcome) -> bool:
        nonlocal calls
        calls += 1
        return calls < 2  # the second stage run (chapter 2) aborts

    with pytest.raises(RunAbortedError) as info:
        run_series([CopyStage()], cfg, "S", ["Chapter 1", "Chapter 2"], after_stage=hook)
    assert info.value.chapter == "Chapter 2"
    assert list(info.value.results) == ["Chapter 1"]
    assert info.value.results["Chapter 1"][0].status == "done"


def test_after_stage_runs_for_failed_stages_too(cfg: Config) -> None:
    seen: list[str] = []

    def hook(ctx: ChapterContext, outcome: StageOutcome) -> bool:
        seen.append(f"{outcome.stage}:{outcome.status}")
        return True

    outcomes = run_chapter([BoomStage(), CopyStage()], make_context(cfg, "S", "Chapter 1"), after_stage=hook)
    assert seen == ["boom:failed"]
    assert [o.stage for o in outcomes] == ["boom"]
