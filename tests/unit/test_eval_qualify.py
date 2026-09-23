"""Tests for omniscan.eval.qualify and scripts/qualify_ocr.py (card O1c): CPU fakes only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import omniscan.eval.qualify as qual
from omniscan.core.config import Config, GpuConfig, OcrConfig, PathsConfig
from omniscan.core.schemas import BBox, IngestArtifact, Region, RegionsArtifact, SourceFile
from omniscan.eval.qualify import (
    Candidate,
    Dataset,
    Measurement,
    Summary,
    candidate_config,
    default_run_candidate,
    load_plan,
    missing_models,
    recommend,
    render_markdown,
    run_qualification,
    select,
    summarize,
)
from omniscan.eval.truth import TruthBox, TruthStats
from omniscan.models.catalog import ModelEntry, default_catalog_path, load_catalog
from omniscan.pipeline.runner import PipelineResult

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "config" / "qualification.toml"

C1 = Candidate("ppocr-v5-ko", "ppocr", "d1", "r1", ("ko",))
C2 = Candidate("manga-ocr-2025", "manga_ocr", None, "r2", ("ja",))
C3 = Candidate("ppocr-v6-tiny", "ppocr", "d3", "r3", ("zh", "en", "ja", "ko"))
D_KO = Dataset("ko", "PepperCarrotKR", ("Episode 06", "Episode 09"), "kr")
D_JA = Dataset("ja", "PepperCarrotJA", ("Episode 06",), "ja")


def make_cfg(tmp_path: Path, *, work_root: str = "work") -> Config:
    return Config(
        gpu=GpuConfig(device="cpu"),
        ocr=OcrConfig(lang="ko"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / work_root,
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


def write_plan(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "plan.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def meas(
    candidate: str,
    lang: str,
    chapter: str,
    *,
    chrf: float | None = None,
    recall: float | None = None,
    cer: float | None = None,
    seconds: float = 1.0,
    vram: float | None = None,
    series: str = "PepperCarrotKR",
    error: str | None = None,
) -> Measurement:
    return Measurement(
        candidate=candidate,
        lang=lang,
        series=series,
        chapter=chapter,
        recall_chars=recall,
        chrf=chrf,
        cer_micro=cer,
        seconds=seconds,
        peak_vram_gib=vram,
        error=error,
    )


def summary(
    lang: str,
    candidate: str,
    chrf: float | None,
    recall: float | None,
    *,
    seconds: float = 1.0,
    errors: int = 0,
    chapters: int = 2,
) -> Summary:
    return Summary(
        lang=lang,
        candidate=candidate,
        series="PepperCarrotKR",
        chapters=chapters,
        errors=errors,
        recall_chars=recall,
        chrf=chrf,
        cer_micro=None,
        seconds=seconds,
        peak_vram_gib=None,
    )


# ---------------------------------------------------------------- load_plan


def test_load_plan_shipped_plan_matches_the_catalog() -> None:
    candidates, datasets = load_plan(PLAN_PATH)
    assert [c.id for c in candidates] == [
        "ppocr-v5-ko",
        "ppocr-v5-server-multi",
        "ppocr-v6-tiny",
        "ppocr-v6-small",
        "ppocr-v6-medium",
        "manga-ocr-2025",
        "manga-ocr-base",
        "paddleocr-vl-1.6",
    ]
    assert [(d.lang, d.series, d.truth, d.chapters) for d in datasets] == [
        ("ko", "PepperCarrotKR", "kr", ("Episode 06", "Episode 09")),
        ("ja", "PepperCarrotJA", "ja", ("Episode 06", "Episode 09", "Episode 12", "Episode 22")),
        ("zh", "PepperCarrotCN", "cn", ("Episode 06", "Episode 09", "Episode 12", "Episode 22")),
    ]
    catalog = {entry.id: entry for entry in load_catalog(default_catalog_path())}
    rec_role = {"ppocr": "recognizer", "manga_ocr": "recognizer", "paddleocr_vl": "vlm_ocr"}
    for cand in candidates:
        assert cand.langs, f"{cand.id}: no langs"
        for field, role in (("det_model", "text_line_detector"), ("rec_model", rec_role[cand.engine])):
            model_id = getattr(cand, field)
            if model_id is not None:
                assert model_id in catalog, f"{cand.id}: {field} {model_id!r} is not in the catalog"
                assert catalog[model_id].role == role, f"{cand.id}: {model_id!r} has the wrong role"


GOOD_PLAN = """
[[candidate]]
id = "c1"
engine = "ppocr"
det_model = "d"
rec_model = "r"
langs = ["ko"]

[[dataset]]
lang = "ko"
series = "S"
chapters = ["Episode 06"]
truth = "kr"
"""


def test_load_plan_good_file(tmp_path: Path) -> None:
    candidates, datasets = load_plan(write_plan(tmp_path, GOOD_PLAN))
    assert [c.id for c in candidates] == ["c1"]
    assert datasets[0].chapters == ("Episode 06",)


def test_load_plan_bad_entries_name_the_entry(tmp_path: Path) -> None:
    cases = [
        # (plan body, expected error fragment)
        (
            GOOD_PLAN + '\n[[candidate]]\nid = "c1"\nengine = "ppocr"\nlangs = ["ko"]\n',
            "duplicate candidate id 'c1'",
        ),
        ('[[candidate]]\nid = "c1"\nlangs = ["ko"]\n', "candidate #1 ('c1'): engine must be one of"),
        ('[[candidate]]\nid = "c1"\nengine = "nope"\nlangs = ["ko"]\n', "unknown engine 'nope'"),
        ('[[candidate]]\nid = "c1"\nengine = "ppocr"\nlangs = "ko"\n', "langs must be a list"),
        (
            '[[candidate]]\nid = 5\nengine = "ppocr"\nlangs = ["ko"]\n',
            "candidate #1: id must be a non-empty string",
        ),
        ('[[candidate]]\nid = "c1"\nengine = "ppocr"\nlangs = ["ko"]\noops = 1\n', "unknown field(s): oops"),
        (
            '[[dataset]]\nlang = "ko"\nseries = "S"\ntruth = "kr"\nchapters = []\n',
            "chapters must be a non-empty list",
        ),
        (
            '[[dataset]]\nlang = "ko"\nseries = "S"\ntruth = "kr"\nchapters = "Episode 06"\n',
            "chapters must be",
        ),
        ('[[dataset]]\nlang = "ko"\nseries = "S"\nchapters = ["C1"]\n', "truth must be a non-empty string"),
        (
            '[[dataset]]\nlang = "ko"\nseries = "S"\ntruth = "kr"\nchapters = ["C1"]\nextra = true\n',
            "unknown field",
        ),
        ("not toml at all ][\n", "invalid TOML"),
    ]
    for index, (body, fragment) in enumerate(cases):
        with pytest.raises(ValueError, match=f".*{fragment.replace('(', '\\(').replace(')', '\\)')}"):
            load_plan(write_plan(tmp_path / f"case{index}", body))


def test_load_plan_toml_syntax_errors_become_value_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid TOML"):
        load_plan(write_plan(tmp_path, "candidate = ["))


# ---------------------------------------------------------------- select


def test_select_pairs_candidates_with_their_datasets_in_order() -> None:
    assert select([C1, C2, C3], [D_KO, D_JA], lang=None, only=None) == [
        (C1, D_KO),
        (C2, D_JA),
        (C3, D_KO),
        (C3, D_JA),
    ]


def test_select_filters_by_language_and_only() -> None:
    assert select([C1, C2, C3], [D_KO, D_JA], lang="ja", only=None) == [(C2, D_JA), (C3, D_JA)]
    assert select([C1, C2, C3], [D_KO, D_JA], lang="ja", only=["manga-ocr-2025"]) == [(C2, D_JA)]
    assert select([C1, C2, C3], [D_KO, D_JA], lang="zh", only=None) == []  # no zh dataset
    assert select([C1], [D_KO], lang=None, only=[]) == [(C1, D_KO)]  # empty only = no filter
    assert select([C1], [D_KO], lang=None, only=["nosuch"]) == []
    assert select([C2], [D_KO], lang=None, only=None) == []  # ja candidate never applies to a ko dataset


# ---------------------------------------------------------------- candidate_config


def test_candidate_config_applies_the_candidate_onto_a_copy(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    dataset = Dataset("zh", "PepperCarrotCN", ("Episode 06",), "cn")
    cand = Candidate("ppocr-v6-tiny", "ppocr", "det-id", "rec-id", ("zh",))
    out = candidate_config(cfg, cand, dataset, tmp_path / "qual")
    assert out is not cfg and out.paths is not cfg.paths and out.ocr is not cfg.ocr
    assert out.paths.work_root == tmp_path / "qual"
    assert out.ocr.engine == "ppocr"
    assert (out.ocr.det_model, out.ocr.rec_model) == ("det-id", "rec-id")
    assert out.ocr.lang == "zh"  # the dataset language, not the truth folder (`cn`)
    assert cfg.paths.work_root == tmp_path / "work"  # the input config is untouched
    assert cfg.ocr.engine == "ppocr" and cfg.ocr.det_model is None and cfg.ocr.lang == "ko"


def test_candidate_config_keeps_its_lang_when_the_dataset_language_is_unknown(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    dataset = Dataset("fr", "PepperCarrotFR", ("Episode 06",), "fr")
    out = candidate_config(cfg, Candidate("m", "manga_ocr", None, "rec", ("fr",)), dataset, tmp_path)
    assert out.ocr.engine == "manga_ocr" and out.ocr.rec_model == "rec"
    assert out.ocr.lang == "ko"  # untouched default: fr is not an OcrConfig language


# ---------------------------------------------------------------- missing_models


def test_missing_models_lists_uninstalled_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        ModelEntry(
            id=model_id,
            name=model_id,
            kind="ocr",
            format="hf",
            size_mb=10,
            license="Apache-2.0",
            description="d",
        )
        for model_id in ("d1", "r1", "r2")
    ]
    monkeypatch.setattr(qual, "load_catalog", lambda: entries)
    monkeypatch.setattr(
        qual,
        "model_status",
        lambda entry, models_dir, *, ollama_names, deep=False: "installed" if entry.id == "d1" else "missing",
    )
    cfg = make_cfg(tmp_path)
    assert missing_models(Candidate("a", "ppocr", "d1", "r1", ("ko",)), cfg) == ["r1"]
    assert missing_models(Candidate("b", "manga_ocr", None, "r2", ("ja",)), cfg) == ["r2"]
    assert missing_models(Candidate("c", "ppocr", "d1", None, ("ko",)), cfg) == []
    assert missing_models(Candidate("d", "ppocr", "not-in-catalog", None, ("ko",)), cfg) == ["not-in-catalog"]


# ---------------------------------------------------------------- run_qualification


def test_run_qualification_runs_every_chapter_in_pair_order(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def run(cfg: Config, cand: Candidate, dataset: Dataset, chapter: str, *, gpu: Any = None) -> Measurement:
        calls.append((cand.id, chapter))
        return Measurement(
            candidate=cand.id,
            lang=dataset.lang,
            series=dataset.series,
            chapter=chapter,
            recall_chars=0.8,
            chrf=0.5,
            cer_micro=0.1,
            seconds=2.5,
            regions=4,
        )

    logs: list[str] = []
    measurements = run_qualification(
        [(C1, D_KO), (C2, D_JA)], make_cfg(tmp_path), run_candidate=run, log=logs.append
    )
    assert calls == [
        ("ppocr-v5-ko", "Episode 06"),
        ("ppocr-v5-ko", "Episode 09"),
        ("manga-ocr-2025", "Episode 06"),
    ]
    assert [m.error for m in measurements] == [None, None, None]
    assert [m.regions for m in measurements] == [4, 4, 4]
    assert any("ppocr-v5-ko ko Episode 06: done in 2.5s" in line for line in logs)


def test_run_qualification_isolates_failures(tmp_path: Path) -> None:
    def run(cfg: Config, cand: Candidate, dataset: Dataset, chapter: str, *, gpu: Any = None) -> Measurement:
        if cand.id == "manga-ocr-2025":
            raise FileNotFoundError("weights missing")
        return Measurement(
            candidate=cand.id, lang=dataset.lang, series=dataset.series, chapter=chapter, chrf=0.5
        )

    logs: list[str] = []
    measurements = run_qualification(
        [(C1, D_KO), (C2, D_JA)], make_cfg(tmp_path), run_candidate=run, log=logs.append
    )
    assert measurements[2].error == "FileNotFoundError: weights missing"
    assert measurements[2].chapter == "Episode 06"
    assert measurements[0].chrf == 0.5
    assert any("error: FileNotFoundError: weights missing" in line for line in logs)


def test_run_qualification_skips_pairs_with_missing_models(tmp_path: Path) -> None:
    calls: list[str] = []

    def run(cfg: Config, cand: Candidate, dataset: Dataset, chapter: str, *, gpu: Any = None) -> Measurement:
        calls.append(cand.id)
        return Measurement(
            candidate=cand.id, lang=dataset.lang, series=dataset.series, chapter=chapter, chrf=0.5
        )

    def installed(cand: Candidate) -> list[str]:
        return ["ocr-rec-manga-ocr-2025"] if cand.id == "manga-ocr-2025" else []

    logs: list[str] = []
    measurements = run_qualification(
        [(C1, D_KO), (C2, D_JA)],
        make_cfg(tmp_path),
        run_candidate=run,
        models_installed=installed,
        log=logs.append,
    )
    assert calls == ["ppocr-v5-ko", "ppocr-v5-ko"]  # the manga-ocr pair never ran
    assert len(measurements) == 3
    skipped = measurements[2]
    assert skipped.error == "missing model: ocr-rec-manga-ocr-2025"
    assert skipped.chapter == "Episode 06"  # the dataset's first chapter stands in
    assert any("skip manga-ocr-2025/ja: missing model: ocr-rec-manga-ocr-2025" in line for line in logs)


def test_run_qualification_shares_one_gpu_manager_per_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate's GPU manager is built once and reused across its chapters AND its datasets/
    languages; only a candidate change releases it and builds a new one."""
    built: list[FakeManager] = []

    def fake_build(_cfg: Config) -> FakeManager:
        manager = FakeManager()
        built.append(manager)
        return manager

    monkeypatch.setattr(qual, "build_vram_manager", fake_build)

    seen_gpu: list[Any] = []

    def run(cfg: Config, cand: Candidate, dataset: Dataset, chapter: str, *, gpu: Any = None) -> Measurement:
        seen_gpu.append(gpu)
        return Measurement(candidate=cand.id, lang=dataset.lang, series=dataset.series, chapter=chapter)

    # C3 (ppocr-v6-tiny) applies to both ko and ja: D_KO (2 chapters) then D_JA (1 chapter) must
    # share one manager (3 calls, same object); C1 before and C2 after must each get their own.
    run_qualification([(C1, D_KO), (C3, D_KO), (C3, D_JA), (C2, D_JA)], make_cfg(tmp_path), run_candidate=run)
    assert len(built) == 3  # one per candidate (C1, C3, C2) — not one per (candidate, dataset) pair
    # C1/D_KO: 2 chapters; C3/D_KO: 2 chapters; C3/D_JA: 1 chapter (same manager as C3/D_KO); C2/D_JA: 1 chapter
    assert seen_gpu == [built[0], built[0], built[1], built[1], built[1], built[2]]
    assert all(manager.released for manager in built)  # including the last one, via `finally`


# ---------------------------------------------------------------- summarize + recommend


def test_summarize_averages_ok_chapters_and_counts_errors() -> None:
    measurements = [
        meas("a", "ko", "Episode 06", chrf=0.4, recall=0.8, cer=0.1, seconds=2.0, vram=1.0),
        meas("a", "ko", "Episode 09", chrf=0.6, recall=0.9, cer=0.3, seconds=4.0, vram=3.0),
        meas("a", "ko", "Episode 12", error="RuntimeError: boom"),
    ]
    s = summarize(measurements)[("ko", "a")]
    assert (s.lang, s.candidate, s.series) == ("ko", "a", "PepperCarrotKR")
    assert (s.chapters, s.errors) == (3, 1)
    assert s.chrf == pytest.approx(0.5)
    assert s.recall_chars == pytest.approx(0.85)
    assert s.cer_micro == pytest.approx(0.2)
    assert s.seconds == 3.0 and s.peak_vram_gib == 3.0  # VRAM: the max across chapters


def test_summarize_all_errors_has_no_means() -> None:
    measurements = [meas("a", "ko", "Episode 06", error="x"), meas("a", "ko", "Episode 09", error="y")]
    s = summarize(measurements)[("ko", "a")]
    assert s.chrf is None and s.recall_chars is None and s.cer_micro is None
    assert s.seconds is None and s.peak_vram_gib is None
    assert (s.chapters, s.errors) == (2, 2)


def test_recommend_keeps_when_the_default_is_best() -> None:
    summaries = {
        ("ko", "base"): summary("ko", "base", 0.50, 0.90),
        ("ko", "new"): summary("ko", "new", 0.48, 0.95),
    }
    rec = recommend(summaries, lang="ko", current_default="base")
    assert (rec.verdict, rec.default, rec.candidate, rec.reason) == (
        "KEEP",
        "base",
        "base",
        "the default is the best candidate",
    )


def test_recommend_promotes_a_clear_winner() -> None:
    summaries = {
        ("ko", "base"): summary("ko", "base", 0.50, 0.90),
        ("ko", "new"): summary("ko", "new", 0.52, 0.90),
    }
    rec = recommend(summaries, lang="ko", current_default="base")
    assert (rec.verdict, rec.candidate) == ("PROMOTE", "new")
    assert rec.reason.startswith("gains 0.020 chrF")


def test_recommend_keeps_on_small_gain_or_recall_loss() -> None:
    small_gain = {
        ("ko", "base"): summary("ko", "base", 0.50, 0.90),
        ("ko", "new"): summary("ko", "new", 0.519, 0.90),
    }
    rec = recommend(small_gain, lang="ko", current_default="base")
    assert rec.verdict == "KEEP" and rec.reason == "best candidate new gains only 0.019 chrF"

    recall_loss = {
        ("ko", "base"): summary("ko", "base", 0.50, 0.90),
        ("ko", "new"): summary("ko", "new", 0.90, 0.86),
    }
    rec = recommend(recall_loss, lang="ko", current_default="base")
    assert rec.verdict == "KEEP" and rec.reason == "best candidate new loses 0.040 char recall"


def test_recommend_tie_breaks_on_recall_then_seconds() -> None:
    recall_tie = {
        ("ko", "base"): summary("ko", "base", 0.45, 0.95),
        ("ko", "x"): summary("ko", "x", 0.50, 0.90),
        ("ko", "y"): summary("ko", "y", 0.50, 0.95),
    }
    rec = recommend(recall_tie, lang="ko", current_default="base")
    assert (rec.verdict, rec.candidate) == ("PROMOTE", "y")  # equal chrF: the higher recall wins

    seconds_tie = {
        ("ko", "base"): summary("ko", "base", 0.45, 0.90),
        ("ko", "x"): summary("ko", "x", 0.50, 0.90, seconds=2.0),
        ("ko", "y"): summary("ko", "y", 0.50, 0.90, seconds=1.0),
    }
    rec = recommend(seconds_tie, lang="ko", current_default="base")
    assert (rec.verdict, rec.candidate) == ("PROMOTE", "y")  # equal chrF and recall: the faster wins


def test_recommend_never_replaces_a_default_without_data() -> None:
    no_default = {("ko", "new"): summary("ko", "new", 0.90, 0.95)}
    rec = recommend(no_default, lang="ko", current_default="base")
    assert (rec.verdict, rec.candidate, rec.reason) == ("KEEP", "base", "no data for the default")

    all_errors = {
        ("ko", "base"): summary("ko", "base", None, None, errors=2, chapters=2),
        ("ko", "new"): summary("ko", "new", 0.90, 0.95),
    }
    rec = recommend(all_errors, lang="ko", current_default="base")
    assert (rec.verdict, rec.candidate, rec.reason) == ("KEEP", "base", "no data for the default")

    other_lang = {("ja", "new"): summary("ja", "new", 0.90, 0.95)}
    rec = recommend(other_lang, lang="ko", current_default="base")
    assert rec.verdict == "KEEP" and rec.reason == "no data for the default"


# ---------------------------------------------------------------- render_markdown

GOLDEN = """\
# OCR qualification

## ja (PepperCarrotJA, 1 chapter(s))

| candidate | chrF | char recall | CER (box, not comparable) | s/chapter | VRAM GiB | errors |
|---|---:|---:|---:|---:|---:|---:|
| manga-ocr-2025 | 0.690 | 0.895 | 0.050 | 6.000 | 1.600 | 0 |
| ppocr-v5-server-multi ★ | 0.600 | 0.900 | 0.300 | 7.000 | 2.900 | 0 |

Recommendation: PROMOTE manga-ocr-2025 (gains 0.090 chrF with 0.005 char recall loss)

## ko (PepperCarrotKR, 2 chapter(s))

| candidate | chrF | char recall | CER (box, not comparable) | s/chapter | VRAM GiB | errors |
|---|---:|---:|---:|---:|---:|---:|
| ppocr-v5-ko ★ | 0.475 | 0.855 | 0.115 | 10.500 | 2.900 | 0 |
| ppocr-v6-medium | 0.105 | 0.745 | 0.115 | 8.500 | 2.500 | 0 |
| paddleocr-vl-1.6 | n/a | n/a | n/a | n/a | n/a | 2 |

Recommendation: KEEP ppocr-v5-ko (the default is the best candidate)

Skipped, models not installed: paddleocr-vl-1.6 — install with `omniscan models download <model id>`
"""


def test_render_markdown_golden() -> None:
    measurements = [
        meas(
            "manga-ocr-2025",
            "ja",
            "Episode 06",
            chrf=0.69,
            recall=0.895,
            cer=0.05,
            seconds=6.0,
            vram=1.6,
            series="PepperCarrotJA",
        ),
        meas(
            "ppocr-v5-server-multi",
            "ja",
            "Episode 06",
            chrf=0.60,
            recall=0.90,
            cer=0.30,
            seconds=7.0,
            vram=2.9,
            series="PepperCarrotJA",
        ),
        meas("ppocr-v5-ko", "ko", "Episode 06", chrf=0.47, recall=0.85, cer=0.11, seconds=10.0, vram=2.9),
        meas("ppocr-v5-ko", "ko", "Episode 09", chrf=0.48, recall=0.86, cer=0.12, seconds=11.0, vram=2.9),
        meas("ppocr-v6-medium", "ko", "Episode 06", chrf=0.10, recall=0.74, cer=0.11, seconds=8.0, vram=2.5),
        meas("ppocr-v6-medium", "ko", "Episode 09", chrf=0.11, recall=0.75, cer=0.12, seconds=9.0, vram=2.5),
        meas("paddleocr-vl-1.6", "ko", "Episode 06", error="missing model: ocr-vl-1.6"),
        meas("paddleocr-vl-1.6", "ko", "Episode 09", error="missing model: ocr-vl-1.6"),
    ]
    summaries = summarize(measurements)
    recommendations = [
        recommend(summaries, lang="ko", current_default="ppocr-v5-ko"),
        recommend(summaries, lang="ja", current_default="ppocr-v5-server-multi"),
    ]
    text = render_markdown(
        summaries,
        recommendations,
        installed={
            "ppocr-v5-ko": True,
            "ppocr-v6-medium": True,
            "paddleocr-vl-1.6": False,
            "manga-ocr-2025": True,
            "ppocr-v5-server-multi": True,
        },
    )
    assert text == GOLDEN


def test_render_markdown_omits_the_skip_note_without_installed() -> None:
    summaries = summarize([meas("a", "ko", "Episode 06", chrf=0.5, recall=0.9)])
    text = render_markdown(summaries, [recommend(summaries, lang="ko", current_default="a")])
    assert "Skipped" not in text
    assert "Recommendation: KEEP a (the default is the best candidate)" in text


# ---------------------------------------------------------------- default_run_candidate


class FakeManager:
    """build_vram_manager stand-in that records release()."""

    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def write_chapter_artifacts(cfg: Config, series: str, chapter: str) -> None:
    """Real tiny ingest/ocr artifacts in the tmp work dir, as the ocr stage would leave them."""
    work = cfg.paths.work_root / series / chapter
    work.mkdir(parents=True)
    IngestArtifact(
        series=series,
        chapter=chapter,
        strip_width=1000,
        strip_height=3000,
        files=[SourceFile(index=0, name="01.jpg", sha256="0" * 64, width=1000, height=3000, y0=0, y1=3000)],
    ).save(work / "ingest.json")
    RegionsArtifact(
        regions=[
            Region(
                id=f"r{i + 1:04d}",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=100, y0=100 + 100 * i, x1=400, y1=180 + 100 * i),
                text="안녕 세상",
            )
            for i in range(2)
        ]
    ).save(work / "ocr.json")


def test_default_run_candidate_measures_one_chapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(tmp_path)
    write_chapter_artifacts(cfg, "PepperCarrotKR", "Episode 06")
    manager = FakeManager()
    seen: dict[str, Any] = {}

    def fake_pipeline(
        run_cfg: Config, series: str, chapters: Any, *, stages: Any, gpu: Any, force: bool
    ) -> Any:
        seen["call"] = (series, list(chapters), list(stages), force, gpu, run_cfg.paths.work_root)
        return PipelineResult()

    def fake_load_truth(check_dir: Path, truth: str, ingest: Any) -> Any:
        seen["check_dir"] = check_dir
        seen["truth"] = truth
        return (
            [TruthBox(page=1, bbox=BBox(x0=100, y0=100, x1=400, y1=180), lines=("안녕 세상",))],
            TruthStats(pages=1, pages_without_truth=0, dropped=0),
        )

    monkeypatch.setattr(qual, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(qual, "build_vram_manager", lambda _cfg: manager)
    monkeypatch.setattr(qual, "load_truth", fake_load_truth)
    monkeypatch.setattr(qual, "load_english_pages", lambda _check_dir, _ingest: {1: ""})
    monkeypatch.setattr(
        qual,
        "score_chapter",
        lambda *args, **_kw: SimpleNamespace(recall_chars=0.9, ocr_chrf_mean=0.42, cer_micro=0.07),
    )

    m = default_run_candidate(cfg, C1, D_KO, "Episode 06")
    assert m.error is None
    assert (m.recall_chars, m.chrf, m.cer_micro) == (0.9, 0.42, 0.07)
    assert m.regions == 2
    assert m.seconds >= 0.0
    assert m.peak_vram_gib is None  # cpu config: no VRAM tracking
    assert (m.candidate, m.lang, m.series, m.chapter) == ("ppocr-v5-ko", "ko", "PepperCarrotKR", "Episode 06")
    assert seen["call"] == (
        "PepperCarrotKR",
        ["Episode 06"],
        ["ingest", "slice", "detect", "ocr"],
        False,
        manager,
        tmp_path / "work",
    )
    assert seen["check_dir"] == tmp_path / "translated-check" / "PepperCarrotKR" / "Episode 06" / "truth"
    assert seen["truth"] == "kr"
    assert manager.released


def test_default_run_candidate_reports_errors_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    write_chapter_artifacts(cfg, "PepperCarrotKR", "Episode 06")
    manager = FakeManager()
    monkeypatch.setattr(qual, "build_vram_manager", lambda _cfg: manager)
    monkeypatch.setattr(qual, "run_pipeline", lambda *args, **kwargs: PipelineResult())

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("scoring blew up")

    monkeypatch.setattr(qual, "load_truth", boom)
    m = default_run_candidate(cfg, C1, D_KO, "Episode 06")
    assert m.error == "ValueError: scoring blew up"
    assert m.chrf is None and m.recall_chars is None and m.cer_micro is None
    assert m.seconds == 0.0 and m.regions == 0
    assert manager.released


def test_default_run_candidate_reports_a_failed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path)
    manager = FakeManager()
    monkeypatch.setattr(qual, "build_vram_manager", lambda _cfg: manager)
    monkeypatch.setattr(
        qual,
        "run_pipeline",
        lambda *args, **kwargs: PipelineResult(failed={"Episode 06": "ocr: model missing"}),
    )
    m = default_run_candidate(cfg, C1, D_KO, "Episode 06")
    assert m.error == "RuntimeError: pipeline failed: ocr: model missing"
    assert manager.released


# ---------------------------------------------------------------- scripts/qualify_ocr.py


_SCRIPT = REPO_ROOT / "scripts" / "qualify_ocr.py"
_script_spec = importlib.util.spec_from_file_location("test_qualify_ocr_script", _SCRIPT)
assert _script_spec is not None and _script_spec.loader is not None
script = importlib.util.module_from_spec(_script_spec)
_script_spec.loader.exec_module(script)

SCRIPT_PLAN = """
[[candidate]]
id = "cand-a"
engine = "ppocr"
det_model = "d1"
rec_model = "r1"
langs = ["ko"]

[[candidate]]
id = "cand-b"
engine = "manga_ocr"
rec_model = "r2"
langs = ["ko"]

[[dataset]]
lang = "ko"
series = "PepperCarrotKR"
chapters = ["Episode 06", "Episode 09"]
truth = "kr"
"""


def patch_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the script at tmp paths and pretend every model is installed."""
    monkeypatch.setattr(script, "load_config", lambda: make_cfg(tmp_path))
    monkeypatch.setattr(script, "missing_models", lambda cand, cfg: [])


def script_plan(tmp_path: Path) -> Path:
    return write_plan(tmp_path / "script", SCRIPT_PLAN)


def fake_run_factory(
    seen: list[tuple[str, str, Path]],
) -> Any:
    def fake_run(
        cfg: Config, cand: Candidate, dataset: Dataset, chapter: str, *, gpu: Any = None
    ) -> Measurement:
        seen.append((cand.id, chapter, cfg.paths.work_root))
        chrf = 0.5 if cand.id == "cand-a" else 0.6
        return Measurement(
            candidate=cand.id,
            lang=dataset.lang,
            series=dataset.series,
            chapter=chapter,
            recall_chars=0.9,
            chrf=chrf,
            cer_micro=0.1,
            seconds=2.0,
            regions=3,
        )

    return fake_run


def test_main_runs_reports_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    patch_script(tmp_path, monkeypatch)
    seen: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(script, "default_run_candidate", fake_run_factory(seen))
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"

    rc = script.main(
        [
            "--plan",
            str(script_plan(tmp_path)),
            "--default",
            "ko=cand-a",
            "--json",
            str(json_path),
            "--markdown",
            str(md_path),
        ]
    )

    assert rc == 0
    assert len(seen) == 4  # 2 candidates x 2 chapters
    assert {work for _cand, _chapter, work in seen} == {tmp_path / "work_qual"}
    out = capsys.readouterr()
    assert "# OCR qualification" in out.out
    assert "PROMOTE cand-b" in out.out  # cand-b (0.600) beats the --default cand-a (0.500)
    measurements = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(measurements) == 4
    assert {m["candidate"] for m in measurements} == {"cand-a", "cand-b"}
    assert "PROMOTE cand-b" in md_path.read_text(encoding="utf-8")


def test_main_dry_run_prints_pairs_without_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    patch_script(tmp_path, monkeypatch)
    monkeypatch.setattr(
        script, "missing_models", lambda cand, cfg: ["ocr-rec-manga-ocr-2025"] if cand.id == "cand-b" else []
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dry run must not run the pipeline")

    monkeypatch.setattr(script, "default_run_candidate", fail)
    rc = script.main(["--plan", str(script_plan(tmp_path)), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 pair(s)" in out  # 2 candidates x 1 dataset
    assert "cand-a ko PepperCarrotKR: Episode 06, Episode 09" in out
    assert "missing models for cand-b: ocr-rec-manga-ocr-2025" in out


def test_main_only_and_chapters_restrict_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    patch_script(tmp_path, monkeypatch)
    rc = script.main(
        ["--plan", str(script_plan(tmp_path)), "--only", "cand-b", "--chapters", "1", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 pair(s)" not in out and "1 pair(s)" in out
    assert "cand-b ko PepperCarrotKR: Episode 06" in out
    assert "cand-a" not in out and "Episode 09" not in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--default", "ko"],  # not LANG=ID
        ["--default", "ko=nosuch-candidate"],  # id not in the plan
        ["--chapters", "0"],  # N >= 1
        ["--plan", "does/not/exist.toml"],  # unreadable plan
        ["--only", "nosuch-candidate", "--dry-run"],  # no pairs selected
    ],
)
def test_main_bad_arguments_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    patch_script(tmp_path, monkeypatch)
    assert script.main(["--plan", str(script_plan(tmp_path)), *argv]) == 2


def test_main_exit_1_when_every_measurement_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_script(tmp_path, monkeypatch)

    def failing_run(cfg: Config, cand: Candidate, dataset: Dataset, chapter: str) -> Measurement:
        return Measurement(
            candidate=cand.id,
            lang=dataset.lang,
            series=dataset.series,
            chapter=chapter,
            error="RuntimeError: boom",
        )

    monkeypatch.setattr(script, "default_run_candidate", failing_run)
    assert script.main(["--plan", str(script_plan(tmp_path))]) == 1
