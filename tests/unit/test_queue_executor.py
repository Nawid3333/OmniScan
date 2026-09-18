"""Tests for the stage executor (card B23): run_series wiring, unknown stages, failed outcomes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import omniscan.queue.executor as executor_module
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.stage import Stage, StageOutcome, run_series
from omniscan.queue.executor import stage_executor
from omniscan.queue.store import Job
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
    """Monkeypatched run_series collecting its calls and returning scripted outcomes."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    outcome: Callable[[list[Stage]], StageOutcome] = lambda stages: StageOutcome(stages[-1].name, "done")


def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    record = Recorder()

    def fake_run_series(
        stages: list[Stage],
        cfg: Config,
        series: str,
        chapters: list[str] | None,
        **kwargs: bool,
    ) -> dict[str, list[StageOutcome]]:
        record.calls.append(
            {"stages": stages, "series": series, "chapters": chapters, "force": kwargs.get("force", False)}
        )
        return {chapter: [record.outcome(stages)] for chapter in (chapters or ["Chapter 1"])}

    monkeypatch.setattr(executor_module, "run_series", fake_run_series)
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


def test_slice_job_runs_ingest_then_slice(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)
    stage_executor(cfg)(make_job(("slice",), chapters=("Chapter 1",), force=True))

    assert len(record.calls) == 1
    call = record.calls[0]
    assert [stage.name for stage in call["stages"]] == ["ingest", "slice"]
    assert call["series"] == SERIES
    assert call["chapters"] == ["Chapter 1"]
    assert call["force"] is True


def test_ingest_and_slice_run_as_two_calls(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    record = recorder(monkeypatch)
    stage_executor(cfg)(make_job(("ingest", "slice")))

    assert [stage.name for call in record.calls for stage in call["stages"]] == [
        "ingest",
        "ingest",
        "slice",
    ]
    assert [call["chapters"] for call in record.calls] == [None, None]


def test_unknown_stage_fails_permanently_before_running(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)

    with pytest.raises(PermanentJobError, match="stage 'detect' is not implemented yet"):
        stage_executor(cfg)(make_job(("slice", "detect")))

    assert record.calls == []  # nothing ran at all


def test_failed_outcome_raises_and_skips_later_stages(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(cfg)
    record = recorder(monkeypatch)
    record.outcome = lambda stages: StageOutcome(stages[-1].name, "failed", error="ingest.json is missing")

    with pytest.raises(RuntimeError) as excinfo:
        stage_executor(cfg)(make_job(("slice", "ingest")))

    assert "Chapter 1/slice" in str(excinfo.value)
    assert "ingest.json is missing" in str(excinfo.value)
    assert len(record.calls) == 1  # the later stage name ('ingest') was not run


def test_no_chapters_found_is_permanent(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)  # tmp library has no series at all

    with pytest.raises(PermanentJobError, match="no chapters found for series 'S'"):
        stage_executor(cfg)(make_job(("slice",)))

    assert record.calls == []


def test_named_chapters_skip_the_chapter_check(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    record = recorder(monkeypatch)

    stage_executor(cfg)(make_job(("slice",), chapters=("Chapter 9",)))

    assert len(record.calls) == 1
    assert record.calls[0]["chapters"] == ["Chapter 9"]


def test_stage_table_covers_the_known_running_stages() -> None:
    assert set(executor_module.STAGE_TABLE) == {"ingest", "slice"}


def test_run_series_is_the_module_level_import() -> None:
    """Tests monkeypatch omniscan.queue.executor.run_series; the module must own that name."""
    assert executor_module.run_series is run_series
