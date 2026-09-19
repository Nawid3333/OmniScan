"""Tests for omniscan.pipeline.stages — the stage registry and the translate/judge adapters (card R1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.manifest import load_manifest
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    BBox,
    Candidate,
    CandidateRun,
    FinalArtifact,
    Region,
    RegionsArtifact,
)
from omniscan.core.stage import ChapterContext, StageOutcome, make_context, run_stage
from omniscan.gpu.vram import OLLAMA_GROUP
from omniscan.llm.ollama import ChatResponse
from omniscan.pipeline.stages import PASS_OF, STAGE_ORDER, JudgeStage, TranslateStage, build_stage
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.profiles import TranslationProfile

SERIES = "S"
CHAPTER = "Chapter 1"
MODEL = "test-model"


def run_adapter(stage: Any, ctx: ChapterContext, *, force: bool = False) -> StageOutcome:
    """run_stage with the adapter typed as Any (the card mandates an instance-attr gpu_group)."""
    return run_stage(stage, ctx, force=force)


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


def region(rid: str, text: str) -> Region:
    return Region(id=rid, slice_index=0, kind="bubble_text", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text=text)


def write_ocr(paths: ChapterPaths, *regions: tuple[str, str]) -> None:
    RegionsArtifact(regions=[region(rid, text) for rid, text in regions]).save(paths.artifact("ocr.json"))


def write_run(paths: ChapterPaths, run_id: str, texts: dict[str, str]) -> None:
    CandidateRun(
        run_id=run_id,
        profile=run_id,
        model="run-model",
        candidates=[Candidate(region_id=rid, text=text) for rid, text in texts.items()],
    ).save(paths.artifact(f"translations/{run_id}.json"))


def profile(**overrides: object) -> TranslationProfile:
    fields: dict[str, object] = {
        "name": "test-profile",
        "endpoint": "local",
        "model": MODEL,
        "style": "chat_json",
    }
    fields.update(overrides)
    return TranslationProfile(**fields)  # type: ignore[arg-type]


def cloud_profile(**overrides: object) -> TranslationProfile:
    """A profile whose model needs no local VRAM, so run_stage works without a scheduler."""
    return profile(model=f"{MODEL}:cloud", **overrides)


class FakeClient:
    """One scripted reply per chat call; counts calls."""

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        cloud: bool = False,
        format: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
        max_retries: int = 5,
    ) -> ChatResponse:
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResponse(
            content=reply,
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=1,
            eval_count=1,
            raw={},
        )


def json_reply(texts: dict[str, str]) -> str:
    return json.dumps(
        {"translations": [{"id": rid, "text": text} for rid, text in texts.items()]}, ensure_ascii=False
    )


def judgements_reply(*judgements: dict[str, str]) -> str:
    return json.dumps({"judgements": list(judgements)}, ensure_ascii=False)


# ---------------------------------------------------------------- registry


def test_stage_order_and_pass_of_cover_the_ten_stages() -> None:
    assert STAGE_ORDER == (
        "ingest",
        "slice",
        "detect",
        "ocr",
        "translate",
        "judge",
        "inpaint",
        "inpaint_lama",
        "typeset",
        "export",
    )
    assert set(PASS_OF) == set(STAGE_ORDER)
    assert {name for name, label in PASS_OF.items() if label == "vision"} == {
        "ingest",
        "slice",
        "detect",
        "ocr",
    }
    assert {name for name, label in PASS_OF.items() if label == "text"} == {"translate", "judge"}
    assert {name for name, label in PASS_OF.items() if label == "render"} == {
        "inpaint",
        "inpaint_lama",
        "typeset",
        "export",
    }


def test_build_stage_returns_the_right_stage_classes(cfg: Config) -> None:
    from omniscan.detect.stage import DetectStage
    from omniscan.export.stage import ExportStage
    from omniscan.ingest.stage import IngestStage
    from omniscan.inpaint.lama_stage import LamaStage
    from omniscan.inpaint.stage import InpaintStage
    from omniscan.ocr.stage import OcrStage
    from omniscan.slicer.stage import SliceStage
    from omniscan.typeset.stage import TypesetStage

    assert type(build_stage("ingest", cfg)) is IngestStage
    assert type(build_stage("slice", cfg)) is SliceStage
    assert type(build_stage("detect", cfg)) is DetectStage
    assert type(build_stage("ocr", cfg)) is OcrStage
    assert type(build_stage("inpaint", cfg)) is InpaintStage
    assert type(build_stage("inpaint_lama", cfg)) is LamaStage
    assert type(build_stage("typeset", cfg)) is TypesetStage
    assert type(build_stage("export", cfg)) is ExportStage


def test_build_stage_unknown_name_raises_with_the_known_ones(cfg: Config) -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown stage 'nope' \(known: ingest, slice, detect, ocr, translate, judge, inpaint, "
        r"inpaint_lama, typeset, export\)",
    ):
        build_stage("nope", cfg)


def test_build_stage_text_stages_need_a_client(cfg: Config) -> None:
    with pytest.raises(ValueError, match="stage 'translate' needs a chat client"):
        build_stage("translate", cfg)
    with pytest.raises(ValueError, match="stage 'judge' needs a chat client"):
        build_stage("judge", cfg)


def test_build_stage_translate_uses_enabled_profiles_only(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = {"on": profile(), "off": profile(name="off", enabled=False)}
    monkeypatch.setattr("omniscan.translate.profiles.default_profile_paths", lambda: [])
    monkeypatch.setattr("omniscan.translate.profiles.load_profiles", lambda _paths: profiles)
    stage = build_stage("translate", cfg, client=FakeClient([]))
    assert isinstance(stage, TranslateStage)
    ctx = make_context(cfg, SERIES, CHAPTER)
    assert stage.outputs(ctx) == ["translations/test-profile.json"]


# ---------------------------------------------------------------- TranslateStage


def test_translate_stage_inputs_include_the_db_only_when_it_exists(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    stage = TranslateStage(FakeClient([]), [profile()])
    assert stage.inputs(ctx) == [ctx.paths.artifact("ocr.json")]
    ctx.series.db.parent.mkdir(parents=True, exist_ok=True)
    ctx.series.db.write_bytes(b"")
    assert stage.inputs(ctx) == [ctx.paths.artifact("ocr.json"), ctx.series.db]


def test_translate_stage_config_subset_hashes_the_profiles(cfg: Config) -> None:
    stage = TranslateStage(FakeClient([]), [profile()])
    other = TranslateStage(FakeClient([]), [profile(temperature=0.9)])
    assert stage.config_subset(cfg) != other.config_subset(cfg)
    assert TranslateStage(FakeClient([]), [profile()]).config_subset(cfg) == stage.config_subset(cfg)


def test_translate_stage_gpu_group_needs_a_local_model(cfg: Config) -> None:
    client = FakeClient([])
    assert TranslateStage(client, [profile(model="translategemma:12b")]).gpu_group == OLLAMA_GROUP
    assert TranslateStage(client, [profile(model="m:cloud")]).gpu_group is None
    assert TranslateStage(client, [profile(endpoint="cloud", model="m")]).gpu_group is None
    assert (
        TranslateStage(client, [profile(model="m:cloud"), profile(model="translategemma:12b")]).gpu_group
        == OLLAMA_GROUP
    )
    assert TranslateStage(client, []).gpu_group is None
    # the shipped gemma4-31b-cloud profile is a local-endpoint "-cloud" model: per the rule it still
    # declares OLLAMA_GROUP (harmless — it runs on the cloud through the local Ollama proxy)
    assert TranslateStage(client, [profile(model="gemma4:31b-cloud")]).gpu_group == OLLAMA_GROUP


def test_translate_stage_run_writes_runs_and_metrics(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"), ("r0002", "반가워"))
    client = FakeClient([json_reply({"r0001": "Hello", "r0002": "Hi"})])
    stage = TranslateStage(client, [cloud_profile()])
    outcome = run_adapter(stage, ctx, force=True)
    assert outcome.status == "done"
    loaded = CandidateRun.load(ctx.paths.artifact("translations/test-profile.json"))
    assert [candidate.text for candidate in loaded.candidates] == ["Hello", "Hi"]
    assert outcome.metrics["profiles"] == 1.0
    assert outcome.metrics["regions"] == 2.0


def test_translate_stage_metrics_sum_over_profiles(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    client = FakeClient([json_reply({"r0001": "Hello"}), json_reply({"r0001": "Hi"})])
    stage = TranslateStage(client, [cloud_profile(), cloud_profile(name="second")])
    outcome = run_adapter(stage, ctx, force=True)
    assert outcome.metrics["profiles"] == 2.0
    assert outcome.metrics["regions"] == 2.0


def test_translate_stage_regenerates_an_existing_run(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    write_run(ctx.paths, "test-profile", {"r0001": "stale"})
    stage = TranslateStage(FakeClient([json_reply({"r0001": "fresh"})]), [cloud_profile()])
    assert run_adapter(stage, ctx).status == "done"  # stage-level force over the existing file
    loaded = CandidateRun.load(ctx.paths.artifact("translations/test-profile.json"))
    assert loaded.candidates[0].text == "fresh"


def test_translate_stage_client_error_propagates(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    stage = TranslateStage(FakeClient([ValueError("boom")]), [profile()])
    with pytest.raises(ValueError, match="boom"):
        stage.run(ctx, {})


# ---------------------------------------------------------------- JudgeStage


def test_judge_stage_inputs_are_ocr_sorted_runs_and_db(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    write_run(ctx.paths, "run2", {"r0001": "Hi"})
    write_run(ctx.paths, "run1", {"r0001": "Hello"})
    CandidateRun(
        run_id="run1",
        profile="run1",
        model="m",
        candidates=[Candidate(region_id="r0001", text="partial")],
    ).save(ctx.paths.artifact("translations/.run1.partial.json"))
    stage = JudgeStage(FakeClient([]), JudgeConfig(model=MODEL))
    ocr = ctx.paths.artifact("ocr.json")
    runs = [ctx.paths.artifact("translations/run1.json"), ctx.paths.artifact("translations/run2.json")]
    assert stage.inputs(ctx) == [ocr, *runs]
    ctx.series.db.parent.mkdir(parents=True, exist_ok=True)
    ctx.series.db.write_bytes(b"")
    assert stage.inputs(ctx) == [ocr, *runs, ctx.series.db]


def test_judge_stage_gpu_group_needs_a_local_model(cfg: Config) -> None:
    client = FakeClient([])
    assert JudgeStage(client, JudgeConfig(model="judge-model")).gpu_group == OLLAMA_GROUP
    assert JudgeStage(client, JudgeConfig(model="m:cloud")).gpu_group is None
    assert JudgeStage(client, JudgeConfig(model="m", endpoint="cloud")).gpu_group is None
    # the shipped judge model is a local-endpoint "-cloud" model: per the rule it still declares
    # OLLAMA_GROUP (it runs on the cloud through the local Ollama proxy)
    assert JudgeStage(client, JudgeConfig()).gpu_group == OLLAMA_GROUP


def test_judge_stage_run_writes_final_metrics_and_is_resumable(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"), ("r0002", "반가워"))
    write_run(ctx.paths, "run1", {"r0001": "Hello", "r0002": "Hi"})
    write_run(ctx.paths, "run2", {"r0001": "Goodbye", "r0002": "Hi"})
    client = FakeClient([judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})])
    stage = JudgeStage(client, JudgeConfig(model=f"{MODEL}:cloud"))
    outcome = run_adapter(stage, ctx)
    assert outcome.status == "done"
    artifact = FinalArtifact.load(ctx.paths.artifact("final.json"))
    assert [(line.text, line.sources) for line in artifact.lines] == [("Hello", ["run1"]), ("Hi", ["run1"])]
    assert client.calls == 1  # only the disagreeing region is judged
    assert {key: value for key, value in outcome.metrics.items() if key != "seconds"} == {
        "regions": 2.0,
        "judged": 1.0,
        "auto_picked": 1.0,
        "untranslated": 0.0,
        "violations_left": 0.0,
        "requests": 1.0,
    }
    assert "seconds" in outcome.metrics
    manifest = load_manifest(ctx.paths.manifest, SERIES, CHAPTER)
    assert manifest.stages["judge"].status == "done"
    assert run_adapter(stage, ctx).status == "skipped"  # same inputs, config and outputs
    assert client.calls == 1
