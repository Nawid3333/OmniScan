"""Tests for `omniscan run` (card R1): selection, exit codes, client and VRAM-manager lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.cli
import omniscan.pipeline.runner as runner_module
import omniscan.pipeline.stages as stages_module
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


def make_chapter(cfg: Config, chapter: str = CHAPTER) -> Path:
    """Create an empty raw chapter folder so `chapters()` finds the series."""
    path = cfg.paths.library_root / SERIES / chapter
    path.mkdir(parents=True, exist_ok=True)
    return path


class FakeStage:
    """A minimal Stage double that writes its output, records its calls, and can fail."""

    def __init__(self, name: str, *, gpu_group: str | None, calls: list[str]) -> None:
        self.name = name
        self.version = 1
        self.gpu_group = gpu_group
        self._calls = calls
        self.error: Exception | None = None

    def inputs(self, ctx: Any) -> list[Path]:
        return [ctx.paths.artifact(f"{self.name}.in.json")]

    def outputs(self, ctx: Any) -> list[str]:
        return [f"{self.name}.out.json"]

    def config_subset(self, cfg: Any) -> dict[str, str]:
        return {"name": self.name}

    def run(self, ctx: Any, models: Any) -> dict[str, float]:
        self._calls.append(f"{self.name}({ctx.paths.chapter})")
        if self.error is not None:
            raise self.error
        ctx.paths.artifact(f"{self.name}.out.json").write_text("{}", encoding="utf-8")
        return {"n": 1.0}


class FakeChatClient:
    """OllamaClient double; the fake stages never chat."""

    def chat(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError("the fake stages never chat")

    def close(self) -> None:
        self.closed = True


class SpyManager:
    """VramManager double mirroring its sticky acquire; records acquires and releases."""

    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.released = 0
        self._resident: str | None = None

    def acquire(self, group: str) -> dict[str, Any]:
        if group != self._resident:
            self._resident = group
            self.acquired.append(group)
        return {}

    def reset_peak(self) -> None:
        return None

    def peak_gib(self) -> float:
        return 0.0

    def release(self) -> None:
        self.released += 1


@pytest.fixture
def fake_pipeline(patched_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake stages wired into runner.build_stage, plus spy client/manager factories in the cli namespace."""
    calls: list[str] = []
    groups = {
        "detect": "vision",
        "ocr": "vision",
        "translate": "ollama_local",
        "judge": "ollama_local",
        "inpaint_lama": "inpaint",
    }
    stages = {
        name: FakeStage(name, gpu_group=groups.get(name), calls=calls) for name in stages_module.STAGE_ORDER
    }
    monkeypatch.setattr(runner_module, "build_stage", lambda name, cfg, *, client=None: stages[name])

    clients: list[FakeChatClient] = []
    monkeypatch.setattr(omniscan.cli, "get_secrets", lambda: None)
    monkeypatch.setattr(
        omniscan.cli,
        "OllamaClient",
        lambda ollama_cfg, secrets: clients.append(FakeChatClient()) or clients[-1],
    )

    managers: list[SpyManager] = []
    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager", lambda cfg: managers.append(SpyManager()) or managers[-1]
    )
    return {"calls": calls, "clients": clients, "managers": managers, "stages": stages}


def test_run_prints_outcomes_and_summary(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/{CHAPTER} ingest: done (0.00s)" in result.output
    assert "1 chapter(s) ok, 0 failed" in result.output
    assert len(fake_pipeline["calls"]) == 10
    assert len(fake_pipeline["clients"]) == 1  # one client for the text pass
    assert fake_pipeline["clients"][0].closed
    assert len(fake_pipeline["managers"]) == 1  # the vision group needs a manager
    assert fake_pipeline["managers"][0].released == 1
    assert fake_pipeline["managers"][0].acquired == ["vision", "ollama_local", "inpaint"]


def test_run_chapter_and_stage_selection(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    make_chapter(patched_cfg, "Chapter 2")
    result = runner.invoke(app, ["run", SERIES, "-c", CHAPTER, "-s", "ingest"])
    assert result.exit_code == 0
    assert fake_pipeline["calls"] == [f"ingest({CHAPTER})"]
    assert "Chapter 2" not in result.output
    assert fake_pipeline["clients"] == []  # a vision-only selection builds no client
    assert fake_pipeline["managers"] == []  # ingest needs no GPU group


def test_run_no_lama_drops_inpaint_lama(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "--no-lama"])
    assert result.exit_code == 0
    assert not any(call.startswith("inpaint_lama(") for call in fake_pipeline["calls"])
    assert len(fake_pipeline["calls"]) == 9


def test_run_force_reruns_up_to_date_stages(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    assert runner.invoke(app, ["run", SERIES]).exit_code == 0
    first = len(fake_pipeline["calls"])
    assert first == 10
    result = runner.invoke(app, ["run", SERIES])
    assert result.exit_code == 0
    assert len(fake_pipeline["calls"]) == first  # up to date: skipped
    assert f"{SERIES}/{CHAPTER} ingest: skipped" in result.output
    result = runner.invoke(app, ["run", SERIES, "--force"])
    assert result.exit_code == 0
    assert len(fake_pipeline["calls"]) == 2 * first


def test_run_unknown_series_exits_2(patched_cfg: Config) -> None:
    result = runner.invoke(app, ["run", "NoSuch"])
    assert result.exit_code == 2
    assert "run: no chapters found for series 'NoSuch'" in result.output


def test_run_unknown_stage_exits_2(patched_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "-s", "nope"])
    assert result.exit_code == 2
    assert (
        "run: unknown stage 'nope' (known: ingest, slice, detect, ocr, translate, judge, inpaint, "
        "inpaint_lama, typeset, export)" in result.output
    )


def test_run_failing_chapter_exits_1(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    fake_pipeline["stages"]["slice"].error = RuntimeError("slice boom")
    result = runner.invoke(app, ["run", SERIES, "-s", "slice"])
    assert result.exit_code == 1
    assert f"{SERIES}/{CHAPTER} slice: failed" in result.output
    assert "    RuntimeError: slice boom" in result.output
    assert "0 chapter(s) ok, 1 failed" in result.output


def test_run_rate_limit_exits_3(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    from omniscan.llm.ollama import OllamaRateLimitError

    make_chapter(patched_cfg)
    fake_pipeline["stages"]["detect"].error = OllamaRateLimitError("session cap")
    result = runner.invoke(app, ["run", SERIES])
    assert result.exit_code == 3
    assert "run: Ollama rate limit reached — re-run later" in result.output
    assert "chapter(s) ok" not in result.output
    assert fake_pipeline["managers"][0].released == 1  # released even on abort
    assert not any(call.startswith("translate(") for call in fake_pipeline["calls"])


def test_run_builds_the_vram_manager_from_series_merged_config(
    patched_cfg: Config, fake_pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `omniscan run` used to call `build_vram_manager` with the un-merged user config,
    so a per-series `ocr.engine` override picked the right stage behaviour (via make_context, which
    does merge) but the wrong model group — the OCR stage then crashed with KeyError('reader')
    because the VRAM manager had loaded ppocr's models, not the crop reader the override asked for."""
    make_chapter(patched_cfg)
    series_dir = patched_cfg.paths.library_root / SERIES
    (series_dir / "series.toml").write_text('[ocr]\nengine = "manga_ocr"\n', encoding="utf-8")
    seen: list[Config] = []
    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager", lambda cfg: seen.append(cfg) or SpyManager()
    )
    result = runner.invoke(app, ["run", SERIES, "-s", "detect"])
    assert result.exit_code == 0
    assert seen and seen[0].ocr.engine == "manga_ocr"


def test_run_stages_subcommand_builds_the_vram_manager_from_series_merged_config(
    patched_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same regression as above, for the `_run_stages` path used by `omniscan detect`/`ocr`/etc.

    `detect` wires real Stage objects (not the `fake_pipeline` fixture's fakes), so it fails past
    `ingest` on this empty chapter — irrelevant here: `build_vram_manager` is called, with the
    series-merged config, before any stage runs."""
    make_chapter(patched_cfg)
    series_dir = patched_cfg.paths.library_root / SERIES
    (series_dir / "series.toml").write_text('[ocr]\nengine = "manga_ocr"\n', encoding="utf-8")
    seen: list[Config] = []
    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager", lambda cfg: seen.append(cfg) or SpyManager()
    )
    runner.invoke(app, ["detect", SERIES])
    assert seen and seen[0].ocr.engine == "manga_ocr"


# ---------------------------------------------------------------- step mode


def test_run_step_answers_y_and_finishes_normally(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "--step"], input="y\n" * 10)
    assert result.exit_code == 0
    assert f"Preview [1/10] {CHAPTER} ingest: done" in result.output
    assert result.output.count("Preview [") == 10
    assert result.output.count("Continue?") == 10
    assert "1 chapter(s) ok, 0 failed" in result.output
    assert len(fake_pipeline["calls"]) == 10


def test_run_step_n_stops_after_the_stage(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "--step"], input="y\nn\n")
    assert result.exit_code == 4
    assert f"run: stopped after slice of {CHAPTER}" in result.output
    assert "chapter(s) ok" not in result.output
    assert fake_pipeline["calls"] == [f"ingest({CHAPTER})", f"slice({CHAPTER})"]


def test_run_step_all_answer_runs_the_rest_without_asking(
    patched_cfg: Config, fake_pipeline: dict[str, Any]
) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "--step"], input="a\n")
    assert result.exit_code == 0
    assert result.output.count("Continue?") == 1
    assert result.output.count("Preview [") == 1  # later gates pass without even printing
    assert len(fake_pipeline["calls"]) == 10
    assert "1 chapter(s) ok, 0 failed" in result.output


def test_run_step_eof_at_the_prompt_stops(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "--step"], input="")
    assert result.exit_code == 4
    assert f"run: stopped after ingest of {CHAPTER}" in result.output
    assert fake_pipeline["calls"] == [f"ingest({CHAPTER})"]


def test_run_step_three_invalid_answers_count_as_no(
    patched_cfg: Config, fake_pipeline: dict[str, Any]
) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "--step"], input="x\nx\nx\n")
    assert result.exit_code == 4
    assert result.output.count("Continue?") == 3
    assert f"run: stopped after ingest of {CHAPTER}" in result.output


def test_run_step_failing_preview_stage_exits_1(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    fake_pipeline["stages"]["slice"].error = RuntimeError("slice boom")
    result = runner.invoke(app, ["run", SERIES, "--step", "-s", "ingest", "-s", "slice"], input="y\ny\n")
    assert result.exit_code == 1
    assert "run: the preview chapter failed — nothing else was run" in result.output
    assert "    RuntimeError: slice boom" in result.output
    assert fake_pipeline["calls"] == [f"ingest({CHAPTER})", f"slice({CHAPTER})"]


def test_run_step_honours_preview_chapter(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    make_chapter(patched_cfg, "Chapter 2")
    result = runner.invoke(app, ["run", SERIES, "--step", "--preview-chapter", "Chapter 2"], input="y\n" * 10)
    assert result.exit_code == 0
    assert fake_pipeline["calls"][0] == "ingest(Chapter 2)"
    assert "Preview [1/10] Chapter 2 ingest: done" in result.output
    assert "2 chapter(s) ok, 0 failed" in result.output


def test_run_preview_chapter_without_step_exits_2(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES, "--preview-chapter", "Chapter 2"])
    assert result.exit_code == 2
    assert "run: --preview-chapter needs --step" in result.output


def test_run_without_step_never_prompts(patched_cfg: Config, fake_pipeline: dict[str, Any]) -> None:
    make_chapter(patched_cfg)
    result = runner.invoke(app, ["run", SERIES])
    assert result.exit_code == 0
    assert "Continue?" not in result.output
    assert "Preview [" not in result.output
