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
    GlossaryEntry,
    Region,
    RegionsArtifact,
)
from omniscan.core.stage import ChapterContext, StageOutcome, make_context, run_stage
from omniscan.glossary.store import GlossaryStore
from omniscan.gpu.vram import OLLAMA_GROUP
from omniscan.llm.ollama import ChatResponse, OllamaRateLimitError
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
        self.messages: list[list[dict[str, Any]]] = []

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
        self.messages.append(messages)
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
    # the shipped gemma4-31b-cloud profile is a local-endpoint "-cloud" model: it runs on Ollama's cloud
    assert TranslateStage(client, [profile(model="gemma4:31b-cloud")]).gpu_group is None


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


class EvictingGpu:
    """A GpuScheduler double that records every evict_ollama(keep=...) call."""

    def __init__(self) -> None:
        self.kept: list[str | None] = []

    def acquire(self, group: str) -> dict[str, Any]:
        return {}

    def reset_peak(self) -> None:
        pass

    def peak_gib(self) -> float:
        return 0.0

    def evict_ollama(self, keep: str | None = None) -> list[str]:
        self.kept.append(keep)
        return []


def test_translate_stage_makes_each_local_model_the_only_resident_one(cfg: Config) -> None:
    gpu = EvictingGpu()
    ctx = make_context(cfg, SERIES, CHAPTER, gpu)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    replies: list[str | Exception] = [json_reply({"r0001": "Hi"})] * 3
    profiles = [profile(name="a", model="a:12b"), cloud_profile(name="c"), profile(name="b", model="b:12b")]
    stage = TranslateStage(FakeClient(replies), profiles)
    assert run_adapter(stage, ctx, force=True).status == "done"
    assert gpu.kept == ["a:12b", "b:12b"]  # the cloud profile never evicts


def rate_limit() -> OllamaRateLimitError:
    return OllamaRateLimitError("session cap", status_code=429)


def test_translate_stage_falls_back_on_rate_limit_for_the_rest_of_the_pass(cfg: Config) -> None:
    first = make_context(cfg, SERIES, "Chapter 1")
    second = make_context(cfg, SERIES, "Chapter 2")
    for ctx in (first, second):
        write_ocr(ctx.paths, ("r0001", "안녕"))
    write_run(first.paths, "test-profile", {"r0001": "stale cloud line"})
    primary = cloud_profile(fallback="fb")
    fallback = cloud_profile(name="fb")
    client = FakeClient([rate_limit(), json_reply({"r0001": "Hi 1"}), json_reply({"r0001": "Hi 2"})])
    stage = TranslateStage(client, [primary], {"test-profile": fallback})

    outcome = run_adapter(stage, first, force=True)
    assert outcome.status == "done"
    assert outcome.metrics["fallbacks_used"] == 1.0
    assert CandidateRun.load(first.paths.artifact("translations/fb.json")).candidates[0].text == "Hi 1"
    assert not first.paths.artifact("translations/test-profile.json").exists()  # stale, would be judged

    assert run_adapter(stage, second, force=True).status == "done"
    assert client.calls == 3  # chapter 2 went straight to the fallback: no second rate-limited attempt
    assert first.manifest.stages["translate"].outputs == [
        "translations/test-profile.json"
    ]  # retried next run


def test_translate_stage_primary_success_removes_a_stale_fallback_run(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    write_run(ctx.paths, "fb", {"r0001": "old fallback line"})
    stage = TranslateStage(
        FakeClient([json_reply({"r0001": "Hello"})]),
        [cloud_profile(fallback="fb")],
        {"test-profile": cloud_profile(name="fb")},
    )
    outcome = run_adapter(stage, ctx, force=True)
    assert outcome.metrics["fallbacks_used"] == 0.0
    assert ctx.paths.artifact("translations/test-profile.json").is_file()
    assert not ctx.paths.artifact("translations/fb.json").exists()


def test_translate_stage_rate_limit_without_fallback_propagates(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    with pytest.raises(OllamaRateLimitError):
        TranslateStage(FakeClient([rate_limit()]), [cloud_profile()]).run(ctx, {})


def test_translate_stage_gpu_group_counts_a_local_fallback() -> None:
    stage = TranslateStage(
        FakeClient([]), [cloud_profile(fallback="fb")], {"test-profile": profile(name="fb")}
    )
    assert stage.gpu_group == OLLAMA_GROUP


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
    # the shipped judge model is a local-endpoint "-cloud" model: it runs on Ollama's cloud
    assert JudgeStage(client, JudgeConfig()).gpu_group is None


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
        "rate_limited": 0.0,
        "reused": 0.0,
    }
    assert "seconds" in outcome.metrics
    manifest = load_manifest(ctx.paths.manifest, SERIES, CHAPTER)
    assert manifest.stages["judge"].status == "done"
    assert run_adapter(stage, ctx).status == "skipped"  # same inputs, config and outputs
    assert client.calls == 1


# ---------------------------------------------------------------- glossary, requests and re-runs (mutation gaps)


def add_locked_term(ctx: ChapterContext, source: str, target: str) -> None:
    ctx.series.db.parent.mkdir(parents=True, exist_ok=True)
    with GlossaryStore(ctx.series.db) as store:
        store.add(GlossaryEntry(source=source, target=target, status="locked"))


def test_translate_stage_sends_the_series_glossary_to_the_model(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "성진아 안녕"))
    add_locked_term(ctx, "성진", "Sungjin")
    client = FakeClient([json_reply({"r0001": "Hello Sungjin"})])
    run_adapter(TranslateStage(client, [cloud_profile()]), ctx, force=True)
    assert "Sungjin" in json.dumps(client.messages[0], ensure_ascii=False)


def test_judge_stage_judges_only_its_own_runs(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    write_run(ctx.paths, "cloud", {"r0001": "Hello"})
    write_run(ctx.paths, "old-local", {"r0001": "Not hello"})  # left by an earlier profile set
    client = FakeClient([])
    stage = JudgeStage(client, JudgeConfig(model=f"{MODEL}:cloud"), ["cloud", "fb"])  # fb: not on disk
    assert run_adapter(stage, ctx).status == "done"
    assert client.calls == 0  # one candidate left: auto-picked, the stale run never reached the judge
    assert [line.sources for line in FinalArtifact.load(ctx.paths.artifact("final.json")).lines] == [
        ["cloud"]
    ]


def test_judge_stage_rate_limit_keeps_the_deterministic_pick(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    write_run(ctx.paths, "run1", {"r0001": "Hello"})
    write_run(ctx.paths, "run2", {"r0001": "Goodbye"})
    outcome = run_adapter(JudgeStage(FakeClient([rate_limit()]), JudgeConfig(model=f"{MODEL}:cloud")), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["rate_limited"] == 1.0
    line = FinalArtifact.load(ctx.paths.artifact("final.json")).lines[0]
    assert (line.text, line.flags) == ("Hello", ["judge_failed"])


def test_judge_stage_reports_locked_term_violations_from_the_series_glossary(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "성진아 안녕"))
    write_run(ctx.paths, "run1", {"r0001": "Hello there"})
    add_locked_term(ctx, "성진", "Sungjin")
    client = FakeClient([judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})] * 3)
    outcome = run_adapter(JudgeStage(client, JudgeConfig(model=f"{MODEL}:cloud")), ctx)
    assert outcome.status == "done"
    assert outcome.metrics["violations_left"] == 1.0


def test_judge_stage_counts_requests_apart_from_judged_regions(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"), ("r0002", "반가워"))
    write_run(ctx.paths, "run1", {"r0001": "Hello", "r0002": "Hi"})
    write_run(ctx.paths, "run2", {"r0001": "Goodbye", "r0002": "Bye"})
    client = FakeClient(
        [
            judgements_reply(
                {"id": "r0001", "decision": "pick", "pick": "A"},
                {"id": "r0002", "decision": "pick", "pick": "B"},
            )
        ]
    )
    outcome = run_adapter(JudgeStage(client, JudgeConfig(model=f"{MODEL}:cloud")), ctx)
    assert outcome.metrics["judged"] == 2.0
    assert outcome.metrics["requests"] == 1.0


def test_judge_stage_rejudges_when_a_candidate_run_changed(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    write_run(ctx.paths, "run1", {"r0001": "Hello"})
    write_run(ctx.paths, "run2", {"r0001": "Goodbye"})
    pick_a = judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})
    client = FakeClient([pick_a, pick_a])
    stage = JudgeStage(client, JudgeConfig(model=f"{MODEL}:cloud"))
    assert run_adapter(stage, ctx).status == "done"
    write_run(ctx.paths, "run1", {"r0001": "Howdy"})  # changes an input: the runner must re-run the stage
    assert run_adapter(stage, ctx).status == "done"
    assert FinalArtifact.load(ctx.paths.artifact("final.json")).lines[0].text == "Howdy"
    assert client.calls == 2


# ---------------------------------------------------------------- story memory (card C5c)


def make_prior_chapters(cfg: Config) -> None:
    """Library folders so ctx's chapter has a prior chapter in series.chapters() order."""
    (cfg.paths.library_root / SERIES / "Chapter 0").mkdir(parents=True, exist_ok=True)
    (cfg.paths.library_root / SERIES / CHAPTER).mkdir(parents=True, exist_ok=True)


def testseries_story_context_joins_the_last_summaries_oldest_first(cfg: Config) -> None:
    from omniscan.core.paths import SeriesPaths
    from omniscan.pipeline.stages import series_story_context
    from omniscan.story.store import SummaryStore

    sp = SeriesPaths.from_config(cfg, "S")
    for name in ("Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4", "Chapter 5"):
        (sp.library_dir / name).mkdir(parents=True, exist_ok=True)
    sp.work_dir.mkdir(parents=True, exist_ok=True)  # a db only ever exists beside a work dir
    assert series_story_context(sp, "Chapter 5") is None  # no db yet
    with SummaryStore(sp.db) as store:
        for name, text in (
            ("Chapter 1", "One."),
            ("Chapter 2", "Two."),
            ("Chapter 3", "Three."),
            ("Chapter 4", "Four."),
        ):
            store.set(name, text, "m")
    # more than max_chapters exist: only the most recent three, still oldest-first
    assert (
        series_story_context(sp, "Chapter 5") == "- Chapter 2: Two.\n- Chapter 3: Three.\n- Chapter 4: Four."
    )
    # a prior chapter with no summary is skipped without breaking the order of the others
    with SummaryStore(sp.db) as store:
        assert store.delete("Chapter 3")
    assert series_story_context(sp, "Chapter 5") == "- Chapter 1: One.\n- Chapter 2: Two.\n- Chapter 4: Four."
    assert series_story_context(sp, "Chapter 99") is None  # chapter not in series.chapters()
    assert series_story_context(sp, "Chapter 1") is None  # no prior chapters


def test_translate_stage_passes_stored_story_summaries_through(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    make_prior_chapters(cfg)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    from omniscan.story.store import SummaryStore

    with SummaryStore(ctx.series.db) as store:
        store.set("Chapter 0", "The previous chapter.", "m")
    client = FakeClient([json_reply({"r0001": "Hello"})])
    run_adapter(TranslateStage(client, [cloud_profile()]), ctx, force=True)
    assert "Story so far:\n- Chapter 0: The previous chapter." in client.messages[0][1]["content"]


def test_judge_stage_passes_stored_story_summaries_through(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    make_prior_chapters(cfg)
    write_ocr(ctx.paths, ("r0001", "안녕"), ("r0002", "반가워"))
    write_run(ctx.paths, "run1", {"r0001": "Hello", "r0002": "Hi"})
    write_run(ctx.paths, "run2", {"r0001": "Goodbye"})
    from omniscan.story.store import SummaryStore

    with SummaryStore(ctx.series.db) as store:
        store.set("Chapter 0", "The previous chapter.", "m")
    client = FakeClient([judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"})])
    run_adapter(JudgeStage(client, JudgeConfig(model=f"{MODEL}:cloud")), ctx)
    assert "Story so far:\n- Chapter 0: The previous chapter." in client.messages[0][1]["content"]


def test_stages_pass_none_without_a_db_and_never_create_it(cfg: Config) -> None:
    ctx = make_context(cfg, SERIES, CHAPTER)
    write_ocr(ctx.paths, ("r0001", "안녕"))
    client = FakeClient([json_reply({"r0001": "Hello"})])
    run_adapter(TranslateStage(client, [cloud_profile()]), ctx, force=True)
    assert "Story so far" not in client.messages[0][1]["content"]
    assert not ctx.series.db.exists()  # mirrors _series_entries' own "never created here" rule
