"""Tests for the stage executor (card R1): run_pipeline wiring, unknown stages, client/manager lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import omniscan.queue.executor as executor_module
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.pipeline.runner import PipelineResult, run_pipeline
from omniscan.pipeline.stages import STAGE_ORDER
from omniscan.queue.executor import stage_executor
from omniscan.queue.store import KNOWN_STAGES, Job
from omniscan.queue.worker import PermanentJobError

SERIES = "S"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """CPU-only config with all paths under tmp_path."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


@dataclass
class Recorder:
    """Monkeypatched run_pipeline collecting its calls and returning a scripted result."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    result: PipelineResult = field(default_factory=PipelineResult)


def recorder(monkeypatch: pytest.MonkeyPatch, result: PipelineResult | None = None) -> Recorder:
    record = Recorder(result=result if result is not None else PipelineResult())

    def fake_run_pipeline(
        cfg: Config,
        series: str,
        chapters: list[str] | None,
        *,
        stages: list[str] | None = None,
        lama: bool = True,
        force: bool = False,
        client: Any = None,
        gpu: Any = None,
        report: Any = None,
    ) -> PipelineResult:
        record.calls.append(
            {
                "cfg": cfg,
                "series": series,
                "chapters": chapters,
                "stages": stages,
                "force": force,
                "client": client,
                "gpu": gpu,
            }
        )
        return record.result

    monkeypatch.setattr(executor_module, "run_pipeline", fake_run_pipeline)
    return record


def make_chapter(cfg: Config, series: str = SERIES, chapter: str = "Chapter 1") -> Path:
    """Create an empty raw chapter folder so `chapters()` finds the series."""
    path = cfg.paths.library_root / series / chapter
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_job(stages: tuple[str, ...], chapters: tuple[str, ...] | None = None, force: bool = False) -> Job:
    return Job(
        id=1,
        series=SERIES,
        chapters=chapters,
        stages=stages,
        force=force,
        priority=0,
        status="running",
        attempts=1,
        max_attempts=2,
        error=None,
        created_at=None,  # type: ignore[arg-type]
        started_at=None,
        finished_at=None,
    )


def test_slice_job_runs_only_the_slice_stage(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)
    stage_executor(cfg)(make_job(("slice",), chapters=("Chapter 1",), force=True))

    assert len(record.calls) == 1
    call = record.calls[0]
    assert call["stages"] == ["slice"]
    assert call["series"] == SERIES
    assert call["chapters"] == ("Chapter 1",)
    assert call["force"] is True
    assert call["client"] is None
    assert call["gpu"] is None


def test_ingest_and_slice_run_in_one_pipeline_call(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    record = recorder(monkeypatch)
    stage_executor(cfg)(make_job(("ingest", "slice")))

    assert len(record.calls) == 1
    assert record.calls[0]["stages"] == ["ingest", "slice"]
    assert record.calls[0]["chapters"] is None


def test_unknown_stage_fails_permanently_before_running(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)

    with pytest.raises(PermanentJobError, match="stage 'nope' is not implemented yet"):
        stage_executor(cfg)(make_job(("slice", "nope")))

    assert record.calls == []  # nothing ran at all


def test_failed_chapter_raises_with_chapter_stage_and_error(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_chapter(cfg)
    failure = PipelineResult(failed={"Chapter 1": "slice: FileNotFoundError: ingest.json missing"})
    recorder(monkeypatch, failure)

    with pytest.raises(RuntimeError) as excinfo:
        stage_executor(cfg)(make_job(("slice",)))

    assert "Chapter 1: slice: FileNotFoundError: ingest.json missing" in str(excinfo.value)


def test_rate_limit_abort_raises_the_retryable_error(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    recorder(monkeypatch, PipelineResult(aborted="rate limit"))

    with pytest.raises(RuntimeError, match="Ollama rate limit reached"):
        stage_executor(cfg)(make_job(("translate",), chapters=("Chapter 1",)))


def test_no_chapters_found_is_permanent(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)  # tmp library has no series at all

    with pytest.raises(PermanentJobError, match="no chapters found for series 'S'"):
        stage_executor(cfg)(make_job(("slice",)))

    assert record.calls == []


def test_named_chapters_skip_the_chapter_check(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)

    stage_executor(cfg)(make_job(("slice",), chapters=("Chapter 9",)))

    assert len(record.calls) == 1
    assert record.calls[0]["chapters"] == ("Chapter 9",)


def test_queue_knows_exactly_the_pipeline_stages_in_order() -> None:
    assert KNOWN_STAGES == STAGE_ORDER


def test_run_pipeline_is_the_module_level_import() -> None:
    """Tests monkeypatch omniscan.queue.executor.run_pipeline; the module must own that name."""
    assert executor_module.run_pipeline is run_pipeline


def test_text_stages_build_and_close_a_client(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    record = recorder(monkeypatch)
    built: list[Any] = []
    closed: list[bool] = []

    class FakeClient:
        def __init__(self, ollama_cfg: Any, secrets: Any) -> None:
            self.args = (ollama_cfg, secrets)
            built.append(self)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(executor_module, "OllamaClient", FakeClient)
    monkeypatch.setattr(executor_module, "get_secrets", lambda: "secrets")
    stage_executor(cfg)(make_job(("translate",), chapters=("Chapter 1",)))

    assert len(built) == 1
    assert built[0].args == (cfg.ollama, "secrets")
    assert closed == [True]
    assert record.calls[0]["client"] is built[0]


def test_vision_stages_build_no_client(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    record = recorder(monkeypatch)
    built: list[Any] = []

    class FakeClient:
        def __init__(self, *args: Any) -> None:
            built.append(args)

        def close(self) -> None:
            raise AssertionError("no client to close")

    monkeypatch.setattr(executor_module, "OllamaClient", FakeClient)
    stage_executor(cfg)(make_job(("slice",), chapters=("Chapter 1",)))

    assert built == []
    assert record.calls[0]["client"] is None


def test_gpu_jobs_build_and_release_a_vram_manager(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    record = recorder(monkeypatch)
    released: list[bool] = []

    class FakeManager:
        def release(self) -> None:
            released.append(True)

    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", lambda _cfg: FakeManager())
    stage_executor(cfg)(make_job(("detect",), chapters=("Chapter 1",)))  # real needs_gpu: detect has a group

    assert released == [True]
    assert record.calls[0]["gpu"] is not None


def test_non_gpu_jobs_build_no_vram_manager(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    record = recorder(monkeypatch)
    built: list[Any] = []
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", lambda _cfg: built.append(1))
    stage_executor(cfg)(make_job(("typeset",), chapters=("Chapter 1",)))

    assert built == []
    assert record.calls[0]["gpu"] is None


class _FakeManager:
    def release(self) -> None:
        return None


def test_gpu_jobs_build_the_vram_manager_from_series_merged_config(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the executor used to call `build_vram_manager` with the un-merged config, so a
    per-series `ocr.engine` override picked the right stage behaviour (via make_context, which does
    merge) but the wrong model group — the OCR stage would then crash with KeyError('reader')."""
    make_chapter(cfg)
    (cfg.paths.library_root / SERIES / "series.toml").write_text(
        '[ocr]\nengine = "manga_ocr"\n', encoding="utf-8"
    )
    record = recorder(monkeypatch)
    seen: list[Config] = []
    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager", lambda cfg: seen.append(cfg) or _FakeManager()
    )
    stage_executor(cfg)(make_job(("detect",), chapters=("Chapter 1",)))

    assert seen and seen[0].ocr.engine == "manga_ocr"
    assert record.calls[0]["cfg"].ocr.engine == "manga_ocr"  # run_pipeline sees it too
