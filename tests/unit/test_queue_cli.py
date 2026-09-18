"""Tests for the `omniscan queue` command group (card B23)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig

SERIES = "S"
CHAPTER = "Chapter 1"
runner = CliRunner()


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


@pytest.fixture
def patched_cfg(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    return cfg


def raw_dir(cfg: Config, series: str = SERIES, chapter: str = CHAPTER) -> Path:
    """Create and return the raw chapter dir."""
    path = cfg.paths.library_root / series / chapter
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_slicables(cfg: Config, series: str = SERIES) -> None:
    """Write two art pages into one chapter of `series` (real art blocks between white bands)."""
    raw = raw_dir(cfg, series)
    rng = np.random.default_rng(0)
    for i in range(2):
        rows = [
            np.full((1700, 400, 3), 255, dtype=np.uint8),
            rng.integers(0, 256, size=(2000, 400, 3), dtype=np.uint8),
            np.full((1700, 400, 3), 255, dtype=np.uint8),
        ]
        Image.fromarray(np.concatenate(rows)).save(raw / f"{i + 1:03d}.jpg", format="JPEG", quality=95)


# ---------------------------------------------------------------- add / list


def test_add_then_list(patched_cfg: Config) -> None:
    result = runner.invoke(
        app, ["queue", "add", SERIES, "--stage", "ingest", "--stage", "slice", "-c", CHAPTER, "-p", "3"]
    )
    assert result.exit_code == 0
    assert f"queued job 1: {SERIES} stages=ingest,slice chapters={CHAPTER} priority=3" in result.output

    result = runner.invoke(app, ["queue", "list"])
    assert result.exit_code == 0
    assert "queued" in result.output
    assert "pri=3" in result.output

    result = runner.invoke(app, ["queue", "list", "--status", "done"])
    assert result.exit_code == 0
    assert "queue is empty" in result.output


def test_add_rejects_unknown_stage(patched_cfg: Config) -> None:
    result = runner.invoke(app, ["queue", "add", SERIES, "--stage", "nope"])
    assert result.exit_code == 2
    assert "unknown stage" in result.output


def test_list_rejects_unknown_status(patched_cfg: Config) -> None:
    result = runner.invoke(app, ["queue", "list", "--status", "finished"])
    assert result.exit_code == 2
    assert "unknown status" in result.output


# ---------------------------------------------------------------- end to end run


def test_queue_run_slices_a_chapter(patched_cfg: Config) -> None:
    write_slicables(patched_cfg)

    result = runner.invoke(app, ["queue", "add", SERIES, "--stage", "slice"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["queue", "run"])
    assert result.exit_code == 0
    assert "job 1 done" in result.output
    assert "done=1 failed=0 retried=0" in result.output
    assert (patched_cfg.paths.work_root / SERIES / CHAPTER / "slices.json").is_file()

    result = runner.invoke(app, ["queue", "add", SERIES, "--stage", "slice"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["queue", "run"])
    assert result.exit_code == 0  # the manifest makes the repeated ingest/slice a skipped
    assert "job 2 done" in result.output
    assert "done=1 failed=0 retried=0" in result.output


def test_queue_run_fails_permanent_stage(patched_cfg: Config) -> None:
    result = runner.invoke(app, ["queue", "add", SERIES, "--stage", "detect"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["queue", "run"])
    assert result.exit_code == 1
    assert "job 1 failed: PermanentJobError: stage 'detect' is not implemented yet" in result.output
    assert "failed=1" in result.output

    result = runner.invoke(app, ["queue", "list"])
    assert "failed" in result.output

    result = runner.invoke(app, ["queue", "retry", "1"])
    assert result.exit_code == 0
    assert "job 1: queued" in result.output


# ---------------------------------------------------------------- pause / resume / cancel / clear


def test_pause_resume_cancel_round_trip(patched_cfg: Config) -> None:
    runner.invoke(app, ["queue", "add", SERIES])
    result = runner.invoke(app, ["queue", "pause", "1"])
    assert result.exit_code == 0
    assert "job 1: paused" in result.output
    result = runner.invoke(app, ["queue", "resume", "1"])
    assert result.exit_code == 0
    assert "job 1: queued" in result.output
    result = runner.invoke(app, ["queue", "cancel", "1"])
    assert result.exit_code == 0
    assert "job 1: cancelled" in result.output


def test_cancel_of_a_done_job_fails(patched_cfg: Config) -> None:
    write_slicables(patched_cfg)
    runner.invoke(app, ["queue", "add", SERIES, "--stage", "slice"])
    assert runner.invoke(app, ["queue", "run"]).exit_code == 0

    result = runner.invoke(app, ["queue", "cancel", "1"])
    assert result.exit_code == 2
    assert "cannot cancel" in result.output


def test_pause_unknown_job(patched_cfg: Config) -> None:
    result = runner.invoke(app, ["queue", "pause", "999"])
    assert result.exit_code == 2
    assert "no such job 999" in result.output


def test_clear_removes_finished_jobs(patched_cfg: Config) -> None:
    write_slicables(patched_cfg)
    runner.invoke(app, ["queue", "add", SERIES, "--stage", "slice"])
    assert runner.invoke(app, ["queue", "run"]).exit_code == 0  # job 1 -> done
    runner.invoke(app, ["queue", "add", SERIES])
    runner.invoke(app, ["queue", "pause", "2"])
    runner.invoke(app, ["queue", "cancel", "2"])

    result = runner.invoke(app, ["queue", "clear"])
    assert result.exit_code == 0
    assert "removed 2 finished job(s)" in result.output
    result = runner.invoke(app, ["queue", "list"])
    assert "queue is empty" in result.output


def test_run_on_empty_queue(patched_cfg: Config) -> None:
    result = runner.invoke(app, ["queue", "run"])
    assert result.exit_code == 0
    assert "queue is empty" in result.output


# ---------------------------------------------------------------- webhook wiring


def test_webhook_is_created_with_the_given_url_and_receives_events(
    patched_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_slicables(patched_cfg)
    runner.invoke(app, ["queue", "add", SERIES, "--stage", "slice"])

    created: list[str] = []
    events: list[str] = []

    def fake_webhook_notifier(url: str, *, timeout: float = 5.0, client: Any = None) -> Any:
        created.append(url)

        def notifier(event: str, job: Any) -> None:
            events.append(event)

        return notifier

    monkeypatch.setattr(omniscan.cli, "webhook_notifier", fake_webhook_notifier)

    result = runner.invoke(app, ["queue", "run", "--webhook", "http://example/hook"])
    assert result.exit_code == 0
    assert created == ["http://example/hook"]
    assert events == ["job_done", "queue_empty"]
