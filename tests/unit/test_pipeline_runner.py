"""Tests for omniscan.pipeline.runner — pass planning, per-pass execution, failure handling (card R1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import omniscan.pipeline.runner as runner_module
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.pipeline.runner import Gate, GateEvent, PipelineResult, needs_gpu, plan_passes, run_pipeline
from omniscan.pipeline.stages import STAGE_ORDER
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.profiles import TranslationProfile

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


class FakeStage:
    """A minimal Stage double that writes its output and records its run calls."""

    def __init__(
        self,
        name: str,
        *,
        gpu_group: str | None = None,
        error: Exception | None = None,
        fail_chapters: frozenset[str] = frozenset(),
        calls: list[str] | None = None,
    ) -> None:
        self.name = name
        self.version = 1
        self.gpu_group = gpu_group
        self._error = error
        self._fail_chapters = fail_chapters
        self._calls = calls if calls is not None else []

    def inputs(self, ctx: Any) -> list[Path]:
        return [ctx.paths.artifact(f"{self.name}.in.json")]

    def outputs(self, ctx: Any) -> list[str]:
        return [f"{self.name}.out.json"]

    def config_subset(self, cfg: Any) -> dict[str, str]:
        return {"name": self.name}

    def run(self, ctx: Any, models: Any) -> dict[str, float]:
        self._calls.append(f"{self.name}({ctx.paths.chapter})")
        if ctx.paths.chapter in self._fail_chapters:
            raise self._error or RuntimeError(f"{self.name} boom")
        ctx.paths.artifact(f"{self.name}.out.json").write_text("{}", encoding="utf-8")
        return {"n": 1.0}


class FakeScheduler:
    """GpuScheduler double mirroring VramManager's sticky acquire; records actual group switches."""

    def __init__(self) -> None:
        self.acquired: list[str] = []
        self._resident: str | None = None

    def acquire(self, group: str) -> dict[str, Any]:
        if group == self._resident:
            return {}
        self._resident = group
        self.acquired.append(group)
        return {}

    def reset_peak(self) -> None:
        return None

    def peak_gib(self) -> float:
        return 0.0


class FakeClient:
    """ChatClient double; the fake stages never chat."""

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the fake stages never chat")


@pytest.fixture
def fake_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeStage]:
    """Fake stages for all ten names, wired into runner.build_stage; groups mirror the real stages."""
    groups = {
        "detect": "vision",
        "ocr": "vision",
        "translate": "ollama_local",
        "judge": "ollama_local",
        "inpaint_lama": "inpaint",
    }
    stages = {name: FakeStage(name, gpu_group=groups.get(name)) for name in STAGE_ORDER}
    monkeypatch.setattr(runner_module, "build_stage", lambda name, cfg, *, client=None: stages[name])
    return stages


def wire_calls(stages: dict[str, FakeStage]) -> list[str]:
    """Make every fake stage append to one shared call list."""
    calls: list[str] = []
    for stage in stages.values():
        stage._calls = calls
    return calls


# ---------------------------------------------------------------- plan_passes


def test_plan_passes_goldens() -> None:
    assert plan_passes(STAGE_ORDER) == [
        ("vision", ["ingest", "slice", "detect", "ocr"]),
        ("text", ["translate", "judge"]),
        ("render", ["inpaint", "inpaint_lama", "typeset", "export"]),
    ]
    assert plan_passes(["export", "ingest"]) == [("vision", ["ingest"]), ("render", ["export"])]
    assert plan_passes([]) == []


def test_plan_passes_deduplicates_and_orders_by_stage_order() -> None:
    assert plan_passes(["judge", "translate", "judge"]) == [("text", ["translate", "judge"])]
    assert plan_passes(["export", "ocr", "slice", "ingest"]) == [
        ("vision", ["ingest", "slice", "ocr"]),
        ("render", ["export"]),
    ]


def test_plan_passes_unknown_name_raises_naming_it_and_the_known_ones() -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown stage 'nope' \(known: ingest, slice, detect, ocr, translate, judge, inpaint, "
        r"inpaint_lama, typeset, export\)",
    ):
        plan_passes(["ingest", "nope"])


# ---------------------------------------------------------------- run_pipeline


def test_full_plan_runs_pass_by_pass_with_groups_acquired_once_per_pass(
    cfg: Config, fake_stages: dict[str, FakeStage]
) -> None:
    calls = wire_calls(fake_stages)
    scheduler = FakeScheduler()
    reports: list[tuple[str, str]] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B"],
        client=FakeClient(),
        gpu=scheduler,
        report=lambda c, o: reports.append((c, o.stage)),
    )
    assert result.ok
    assert calls == [
        "ingest(A)",
        "slice(A)",
        "detect(A)",
        "ocr(A)",
        "ingest(B)",
        "slice(B)",
        "detect(B)",
        "ocr(B)",
        "translate(A)",
        "judge(A)",
        "translate(B)",
        "judge(B)",
        "inpaint(A)",
        "inpaint_lama(A)",
        "typeset(A)",
        "export(A)",
        "inpaint(B)",
        "inpaint_lama(B)",
        "typeset(B)",
        "export(B)",
    ]
    assert scheduler.acquired == ["vision", "ollama_local", "inpaint"]  # once per pass, never per chapter
    expected = [
        *[("A", name) for name in ("ingest", "slice", "detect", "ocr")],
        *[("B", name) for name in ("ingest", "slice", "detect", "ocr")],
        *[("A", name) for name in ("translate", "judge")],
        *[("B", name) for name in ("translate", "judge")],
        *[("A", name) for name in ("inpaint", "inpaint_lama", "typeset", "export")],
        *[("B", name) for name in ("inpaint", "inpaint_lama", "typeset", "export")],
    ]
    assert reports == expected
    assert [outcome.stage for outcome in result.outcomes["A"]] == list(STAGE_ORDER)
    assert len(result.outcomes["B"]) == len(STAGE_ORDER)


def test_failed_chapter_is_dropped_from_later_passes(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    fake_stages["detect"] = FakeStage(
        "detect",
        gpu_group="vision",
        fail_chapters=frozenset({"A"}),
        error=RuntimeError("detect boom"),
        calls=calls,
    )
    result = run_pipeline(cfg, SERIES, ["A", "B"], client=FakeClient(), gpu=FakeScheduler())
    assert not result.ok
    assert result.aborted is None
    assert result.failed == {"A": "detect: RuntimeError: detect boom"}
    assert [outcome.stage for outcome in result.outcomes["A"]] == ["ingest", "slice", "detect"]
    assert [outcome.stage for outcome in result.outcomes["B"]] == list(STAGE_ORDER)
    assert calls.count("ingest(A)") == 1  # A is never re-run by the later passes
    assert "translate(A)" not in calls


def test_rate_limit_aborts_before_the_next_pass(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    from omniscan.llm.ollama import OllamaRateLimitError

    calls = wire_calls(fake_stages)
    fake_stages["detect"] = FakeStage(
        "detect",
        gpu_group="vision",
        fail_chapters=frozenset({"A"}),
        error=OllamaRateLimitError("session cap"),
        calls=calls,
    )
    result = run_pipeline(cfg, SERIES, ["A", "B"], client=FakeClient(), gpu=FakeScheduler())
    assert not result.ok
    assert result.aborted == "rate limit"
    assert result.failed == {"A": "detect: OllamaRateLimitError: session cap"}
    # the rest of the vision pass still ran; the text and render passes never started
    assert "translate(A)" not in calls
    assert "translate(B)" not in calls
    assert "export(A)" not in calls
    assert "ingest(B)" in calls


def test_stage_selection_runs_only_the_selected_stage(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    result = run_pipeline(cfg, SERIES, ["A"], stages=["translate"], client=FakeClient(), gpu=FakeScheduler())
    assert result.ok
    assert calls == ["translate(A)"]
    assert [outcome.stage for outcome in result.outcomes["A"]] == ["translate"]


def test_lama_false_drops_inpaint_lama(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    result = run_pipeline(cfg, SERIES, ["A"], lama=False, client=FakeClient(), gpu=FakeScheduler())
    assert result.ok
    assert calls == [f"{name}(A)" for name in STAGE_ORDER if name != "inpaint_lama"]
    assert [outcome.stage for outcome in result.outcomes["A"]] == [
        name for name in STAGE_ORDER if name != "inpaint_lama"
    ]


def test_force_reruns_up_to_date_stages(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    scheduler = FakeScheduler()
    run_pipeline(cfg, SERIES, ["A"], client=FakeClient(), gpu=scheduler)
    assert len(calls) == len(STAGE_ORDER)
    run_pipeline(cfg, SERIES, ["A"], client=FakeClient(), gpu=scheduler)  # up to date: skipped, no calls
    assert len(calls) == len(STAGE_ORDER)
    result = run_pipeline(cfg, SERIES, ["A"], force=True, client=FakeClient(), gpu=scheduler)
    assert result.ok
    assert len(calls) == 2 * len(STAGE_ORDER)
    assert [outcome.status for outcome in result.outcomes["A"]] == ["done"] * len(STAGE_ORDER)


def test_report_is_called_once_per_outcome_after_each_pass(
    cfg: Config, fake_stages: dict[str, FakeStage]
) -> None:
    calls = wire_calls(fake_stages)
    fake_stages["slice"] = FakeStage(
        "slice", fail_chapters=frozenset({"A"}), error=RuntimeError("slice boom"), calls=calls
    )
    reports: list[tuple[str, str, str]] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A"],
        stages=["ingest", "slice"],
        report=lambda c, o: reports.append((c, o.stage, o.status)),
    )
    assert not result.ok
    assert reports == [("A", "ingest", "done"), ("A", "slice", "failed")]


def test_no_chapters_raises_value_error(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    with pytest.raises(ValueError, match="no chapters found for series 'S'"):
        run_pipeline(cfg, SERIES)


def test_missing_client_raises_before_anything_runs(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    with pytest.raises(ValueError, match="stage 'translate' needs a chat client"):
        run_pipeline(cfg, SERIES, ["A"], stages=["ingest", "translate"])
    with pytest.raises(ValueError, match="stage 'judge' needs a chat client"):
        run_pipeline(cfg, SERIES, ["A"], stages=["judge"])
    assert calls == []


# ---------------------------------------------------------------- needs_gpu


def test_needs_gpu_follows_the_stage_gpu_groups(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniscan.translate.profiles.default_profile_paths", lambda: [])
    monkeypatch.setattr("omniscan.translate.profiles.load_profiles", lambda _paths: {})
    monkeypatch.setattr("omniscan.translate.judge_config.default_judge_paths", lambda: [])
    monkeypatch.setattr(
        "omniscan.translate.judge_config.load_judge_config",
        lambda _paths: JudgeConfig(model="m:cloud"),
    )
    assert needs_gpu(list(STAGE_ORDER), cfg, FakeClient()) is True  # detect/ocr carry the vision group
    assert needs_gpu(["ingest", "slice", "typeset"], cfg, None) is False
    monkeypatch.setattr(
        "omniscan.translate.profiles.load_profiles",
        lambda _paths: {
            "p": TranslationProfile(name="p", endpoint="local", model="translategemma:12b", style="chat_json")
        },
    )
    assert needs_gpu(["translate"], cfg, FakeClient()) is True
    assert needs_gpu(["judge"], cfg, FakeClient()) is False  # cloud judge needs no VRAM group


def test_pipeline_result_ok_property() -> None:
    assert PipelineResult().ok
    assert not PipelineResult(failed={"A": "detect: boom"}).ok
    assert not PipelineResult(aborted="rate limit").ok


# ---------------------------------------------------------------- step mode


def always(event: GateEvent) -> bool:
    """A gate that records nothing and continues."""
    return True


def recording(events: list[GateEvent]) -> Gate:
    """A gate that records every event and continues."""

    def gate(event: GateEvent) -> bool:
        events.append(event)
        return True

    return gate


def test_auto_mode_ignores_gate_and_preview_chapter(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    events: list[GateEvent] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B"],
        client=FakeClient(),
        gpu=FakeScheduler(),
        mode="auto",
        preview_chapter="B",
        gate=lambda event: events.append(event) or False,
    )
    assert result.ok
    assert events == []  # the gate is never called in auto mode
    assert calls == [
        *[f"{name}(A)" for name in list(STAGE_ORDER)[:4]],
        *[f"{name}(B)" for name in list(STAGE_ORDER)[:4]],
        *[f"{name}(A)" for name in list(STAGE_ORDER)[4:6]],
        *[f"{name}(B)" for name in list(STAGE_ORDER)[4:6]],
        *[f"{name}(A)" for name in list(STAGE_ORDER)[6:]],
        *[f"{name}(B)" for name in list(STAGE_ORDER)[6:]],
    ]


def test_step_mode_runs_the_preview_chapter_through_all_passes_first(
    cfg: Config, fake_stages: dict[str, FakeStage]
) -> None:
    calls = wire_calls(fake_stages)
    events: list[GateEvent] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B", "C"],
        client=FakeClient(),
        gpu=FakeScheduler(),
        mode="step",
        gate=recording(events),
    )
    assert result.ok
    assert (
        calls
        == [
            *[f"{name}(A)" for name in STAGE_ORDER],  # phase A: all passes of A first
            *[f"{name}(B)" for name in list(STAGE_ORDER)[:4]],
            *[f"{name}(C)" for name in list(STAGE_ORDER)[:4]],
            *[f"{name}(B)" for name in list(STAGE_ORDER)[4:6]],
            *[f"{name}(C)" for name in list(STAGE_ORDER)[4:6]],
            *[f"{name}(B)" for name in list(STAGE_ORDER)[6:]],
            *[f"{name}(C)" for name in list(STAGE_ORDER)[6:]],
        ]
    )
    assert [(event.chapter, event.stage, event.position, event.total) for event in events] == [
        ("A", name, index, len(STAGE_ORDER)) for index, name in enumerate(STAGE_ORDER, 1)
    ]
    assert [outcome.stage for outcome in result.outcomes["A"]] == list(STAGE_ORDER)  # never duplicated


def test_step_mode_positions_follow_the_selection(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    events: list[GateEvent] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B"],
        stages=["ingest", "inpaint_lama", "export"],
        lama=False,
        client=FakeClient(),
        mode="step",
        gate=recording(events),
    )
    assert result.ok
    assert calls == ["ingest(A)", "export(A)", "ingest(B)", "export(B)"]
    assert [(event.stage, event.position, event.total) for event in events] == [
        ("ingest", 1, 2),
        ("export", 2, 2),
    ]


def test_step_mode_stops_when_the_gate_answers_no(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    reports: list[tuple[str, str]] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B", "C"],
        client=FakeClient(),
        gpu=FakeScheduler(),
        mode="step",
        gate=lambda event: event.position < 4,
        report=lambda chapter, outcome: reports.append((chapter, outcome.stage)),
    )
    assert result.aborted == "stopped"
    assert not result.ok
    assert result.failed == {}
    assert [outcome.stage for outcome in result.outcomes["A"]] == list(STAGE_ORDER)[:4]
    assert reports == [("A", name) for name in list(STAGE_ORDER)[:4]]
    assert calls == [f"{name}(A)" for name in list(STAGE_ORDER)[:4]]  # no later stage ran
    assert "B" not in result.outcomes
    assert "C" not in result.outcomes


def test_step_mode_reports_a_rate_limit_in_the_preview_as_such(
    cfg: Config, fake_stages: dict[str, FakeStage]
) -> None:
    from omniscan.llm.ollama import OllamaRateLimitError

    calls = wire_calls(fake_stages)
    fake_stages["detect"] = FakeStage(
        "detect",
        gpu_group="vision",
        fail_chapters=frozenset({"A"}),
        error=OllamaRateLimitError("session cap"),
        calls=calls,
    )
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B"],
        client=FakeClient(),
        gpu=FakeScheduler(),
        mode="step",
        gate=recording([]),
    )
    assert result.aborted == "rate limit"
    assert "B" not in result.outcomes


def test_step_mode_stops_after_a_failed_preview_stage(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    calls = wire_calls(fake_stages)
    fake_stages["detect"] = FakeStage(
        "detect",
        gpu_group="vision",
        fail_chapters=frozenset({"A"}),
        error=RuntimeError("detect boom"),
        calls=calls,
    )
    events: list[GateEvent] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B"],
        client=FakeClient(),
        gpu=FakeScheduler(),
        mode="step",
        gate=recording(events),
    )
    assert result.aborted == "preview failed"
    assert not result.ok
    assert result.failed == {"A": "detect: RuntimeError: detect boom"}
    assert (events[2].stage, events[2].outcome.status) == ("detect", "failed")  # the gate still saw it
    assert calls == [f"{name}(A)" for name in list(STAGE_ORDER)[:3]]  # nothing for B ran
    assert "B" not in result.outcomes


def test_step_mode_requires_a_gate(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    with pytest.raises(ValueError, match="step mode needs a gate"):
        run_pipeline(cfg, SERIES, ["A"], mode="step")


def test_step_mode_rejects_unknown_preview_chapter(cfg: Config, fake_stages: dict[str, FakeStage]) -> None:
    with pytest.raises(ValueError, match="unknown preview chapter 'Z'"):
        run_pipeline(cfg, SERIES, ["A", "B"], mode="step", preview_chapter="Z", gate=always)


def test_step_mode_selected_preview_chapter_runs_first(
    cfg: Config, fake_stages: dict[str, FakeStage]
) -> None:
    calls = wire_calls(fake_stages)
    result = run_pipeline(
        cfg,
        SERIES,
        ["A", "B", "C"],
        preview_chapter="B",
        mode="step",
        gate=always,
        client=FakeClient(),
        gpu=FakeScheduler(),
    )
    assert result.ok
    assert calls[0] == "ingest(B)"
    assert calls[len(STAGE_ORDER)] == "ingest(A)"  # phase B starts with the remaining chapters
    assert calls[len(STAGE_ORDER) + 4] == "ingest(C)"
    assert [outcome.stage for outcome in result.outcomes["B"]] == list(STAGE_ORDER)


def test_step_mode_gates_skipped_stages_and_force_reruns_phase_a(
    cfg: Config, fake_stages: dict[str, FakeStage]
) -> None:
    run_pipeline(cfg, SERIES, ["A"], client=FakeClient(), gpu=FakeScheduler())  # all up to date afterwards
    calls = wire_calls(fake_stages)
    statuses: list[str] = []
    result = run_pipeline(
        cfg,
        SERIES,
        ["A"],
        client=FakeClient(),
        gpu=FakeScheduler(),
        mode="step",
        gate=lambda event: statuses.append(event.outcome.status) or True,
    )
    assert result.ok
    assert statuses == ["skipped"] * len(STAGE_ORDER)  # up-to-date stages still trigger the gate
    assert calls == []
    statuses.clear()
    result = run_pipeline(
        cfg,
        SERIES,
        ["A"],
        client=FakeClient(),
        gpu=FakeScheduler(),
        mode="step",
        force=True,
        gate=lambda event: statuses.append(event.outcome.status) or True,
    )
    assert result.ok
    assert statuses == ["done"] * len(STAGE_ORDER)  # force reaches phase A
    assert len(calls) == len(STAGE_ORDER)
