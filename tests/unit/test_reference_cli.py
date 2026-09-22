"""Tests for `omniscan reference` (card GL1): orchestration, series-config merge, pseudo paths, CLI surface.

The real chapter matcher (CM1) runs over the synthetic chapter-set fixtures; the OCR pipeline and the
chat client are faked — the fake pipeline writes the exact artifacts `run_reference` reads (ingest.json
+ ocr.json), on whichever side it was invoked for, which is what makes the read/write locations
assertable without a GPU."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.cli
import omniscan.glossary.reference as reference_module
import omniscan.gpu.lock as lock_module
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.paths import REFERENCE_DIR, SeriesPaths
from omniscan.core.schemas import BBox, IngestArtifact, Region, RegionsArtifact, SourceFile
from omniscan.glossary.reference import MergeReport, ReferenceSummary, run_reference
from omniscan.glossary.store import GlossaryStore
from omniscan.llm.ollama import ChatResponse
from omniscan.pipeline.runner import PipelineResult
from tests.fixtures.chapter_sets import page_seeds, write_raw_series, write_translated_series

SERIES = "S"
runner = CliRunner()

RAW = {"Chapter 001": page_seeds(1001, 3), "Chapter 002": page_seeds(2001, 3)}
REF = {"ch_01": RAW["Chapter 001"], "ch_02": RAW["Chapter 002"]}
RAW_TEXTS = ("민준이", "검은성으로")
REF_TEXTS = ("Minjun", "to the Black Citadel")
EXTRACT_REPLY = json.dumps(
    {
        "terms": [
            {"source": "민준", "target": "Minjun", "type": "person"},
            {"source": "검은성", "target": "Black Citadel", "type": "place"},
        ]
    },
    ensure_ascii=False,
)


def make_cfg(tmp_path: Path, device: str = "cpu") -> Config:
    return Config(
        gpu=GpuConfig(device=device),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


def make_series(
    cfg: Config,
    *,
    raw: dict[str, list[int]] | None = RAW,
    ref: dict[str, list[int]] | None = REF,
) -> SeriesPaths:
    """Build the series' raw chapters and (by default) its imported reference chapters."""
    sp = SeriesPaths.from_config(cfg, SERIES)
    if raw is not None:
        write_raw_series(sp.library_dir, raw)
    if ref is not None:
        write_translated_series(sp.reference_dir, ref)
    return sp


# ---------------------------------------------------------------- fakes


class FakeOcrPipeline:
    """run_pipeline double writing the ingest/OCR artifacts `run_reference` reads, side-aware."""

    def __init__(self, failed: set[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failed = failed or set()

    def __call__(
        self,
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
        self.calls.append(
            {
                "cfg": cfg,
                "series": series,
                "chapters": list(chapters or []),
                "stages": stages,
                "force": force,
            }
        )
        texts = REF_TEXTS if series.endswith(REFERENCE_DIR) else RAW_TEXTS
        sp = SeriesPaths.from_config(cfg, series)
        failed = {name for name in (chapters or []) if name in self.failed}
        for chapter in chapters or []:
            if chapter in failed:
                continue
            paths = sp.chapter(chapter)
            paths.work_dir.mkdir(parents=True, exist_ok=True)
            files = [
                SourceFile(
                    index=0,
                    name="000.jpg",
                    sha256=f"sha-{series}-{chapter}",
                    width=400,
                    height=200,
                    y0=0,
                    y1=200,
                )
            ]
            IngestArtifact(
                series=series, chapter=chapter, strip_width=400, strip_height=200, files=files
            ).save(paths.artifact("ingest.json"))
            regions = [
                Region(
                    id=f"r{i:04d}",
                    slice_index=0,
                    kind="bubble_text",
                    bbox=BBox(x0=10, y0=10 + 90 * i, x1=200, y1=50 + 90 * i),
                    reading_order=i,
                    text=text,
                    confidence=0.9,
                )
                for i, text in enumerate(texts)
            ]
            RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
        return PipelineResult(failed={name: "ocr: boom" for name in failed})


@dataclass
class FakeClient:
    """ChatClient double: scripted replies, every call recorded."""

    replies: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def close(self) -> None:
        self.closed = True

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
        self.calls.append(
            {"model": model, "messages": messages, "cloud": cloud, "format": format, "options": options}
        )
        return ChatResponse(
            content=self.replies.pop(0),
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=None,
            eval_count=None,
            raw={},
        )


# ---------------------------------------------------------------- run_reference: config + paths


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> FakeOcrPipeline:
    fake = FakeOcrPipeline()
    monkeypatch.setattr(reference_module, "run_pipeline", fake)
    return fake


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_both_sides_ocr_with_the_series_merged_config(
    tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (the config trap): run_pipeline re-merges against the series name it is given, and
    the pseudo reference series has no series.toml — so without the explicit merge in run_reference
    the reference side would silently OCR with the base config's engine."""
    cfg = make_cfg(tmp_path, device=device)
    make_series(cfg)
    (cfg.paths.library_root / SERIES / "series.toml").write_text(
        '[ocr]\nengine = "manga_ocr"\n', encoding="utf-8"
    )
    fake = FakeOcrPipeline()
    monkeypatch.setattr(reference_module, "run_pipeline", fake)
    run_reference(cfg, SERIES, client=FakeClient([EXTRACT_REPLY, EXTRACT_REPLY]))

    assert len(fake.calls) == 2
    assert all(call["cfg"].ocr.engine == "manga_ocr" for call in fake.calls)
    assert {call["series"] for call in fake.calls} == {SERIES, f"{SERIES}/{REFERENCE_DIR}"}


def test_pseudo_series_paths_land_reference_artifacts_in_the_reference_dir(tmp_path: Path) -> None:
    """The pseudo series name carries the reference folder: its library_dir IS reference_dir and its
    work artifacts land under work_root/<series>/_reference_en/ — next to the raw side's, not inside."""
    cfg = make_cfg(tmp_path)
    sp = make_series(cfg)
    ref_sp = SeriesPaths.from_config(cfg, f"{SERIES}/{REFERENCE_DIR}")

    assert ref_sp.library_dir == sp.reference_dir
    assert ref_sp.chapters() == ["ch_01", "ch_02"]
    assert ref_sp.work_dir == cfg.paths.work_root / SERIES / REFERENCE_DIR
    assert sp.chapters() == ["Chapter 001", "Chapter 002"]  # the raw side never sees _reference_en


def test_reference_side_reads_english_and_raw_side_reads_korean(
    tmp_path: Path, fake_pipeline: FakeOcrPipeline
) -> None:
    """The pairing prompt must carry the raw chapter's Korean against the reference chapter's English
    — which proves which side's artifacts each pipeline call read."""
    cfg = make_cfg(tmp_path)
    make_series(cfg)
    client = FakeClient([EXTRACT_REPLY, EXTRACT_REPLY])
    summary = run_reference(cfg, SERIES, client=client)

    assert (cfg.paths.work_root / SERIES / "Chapter 001" / "ocr.json").is_file()
    assert (cfg.paths.work_root / SERIES / REFERENCE_DIR / "ch_01" / "ocr.json").is_file()
    prompt = client.calls[0]["messages"][1]["content"]
    assert "민준이" in prompt and "Minjun" in prompt
    assert summary.lines_paired == 4  # 2 regions x 2 matched chapters
    assert summary.chapters_extracted == 2


# ---------------------------------------------------------------- run_reference: orchestration


def test_unmatched_raw_chapters_are_reported_and_not_ocrd(
    tmp_path: Path, fake_pipeline: FakeOcrPipeline
) -> None:
    cfg = make_cfg(tmp_path)
    make_series(cfg, raw={**RAW, "Chapter 003": page_seeds(3001, 3)})
    summary = run_reference(cfg, SERIES, client=FakeClient([EXTRACT_REPLY, EXTRACT_REPLY]))

    assert summary.matched == (("Chapter 001", "ch_01"), ("Chapter 002", "ch_02"))
    assert summary.raw_only == ("Chapter 003",)  # normal, not an error
    assert summary.reference_only == ()
    raw_call = next(call for call in fake_pipeline.calls if call["series"] == SERIES)
    assert raw_call["chapters"] == ["Chapter 001", "Chapter 002"]  # Chapter 003 never OCR'd


def test_dry_run_writes_no_glossary(tmp_path: Path, fake_pipeline: FakeOcrPipeline) -> None:
    cfg = make_cfg(tmp_path)
    make_series(cfg)
    summary = run_reference(cfg, SERIES, client=FakeClient([EXTRACT_REPLY, EXTRACT_REPLY]), dry_run=True)

    assert summary.dry_run
    assert summary.merge.proposed == 2  # the report is still computed
    assert not (cfg.paths.work_root / SERIES / "series.db").exists()
    assert not (cfg.paths.work_root / SERIES / "glossary.yaml").exists()


def test_run_writes_the_store_and_reexports_yaml(tmp_path: Path, fake_pipeline: FakeOcrPipeline) -> None:
    cfg = make_cfg(tmp_path)
    make_series(cfg)
    summary = run_reference(cfg, SERIES, client=FakeClient([EXTRACT_REPLY, EXTRACT_REPLY]))

    assert not summary.dry_run
    with GlossaryStore(cfg.paths.work_root / SERIES / "series.db") as store:
        rows = store.list()
    assert len(rows) == 2
    assert all(row.status == "proposed" and row.origin == "reference" for row in rows)
    assert (cfg.paths.work_root / SERIES / "glossary.yaml").is_file()


def test_ocr_failure_is_reported_and_skips_that_chapter(
    tmp_path: Path, fake_pipeline: FakeOcrPipeline
) -> None:
    cfg = make_cfg(tmp_path)
    make_series(cfg)
    fake_pipeline.failed = {"Chapter 002"}
    summary = run_reference(cfg, SERIES, client=FakeClient([EXTRACT_REPLY]))

    assert summary.ocr_failed == (("raw", "Chapter 002", "ocr: boom"),)
    assert summary.chapters_extracted == 1 and summary.lines_paired == 2


def test_force_reruns_up_to_date_ocr_stages(tmp_path: Path, fake_pipeline: FakeOcrPipeline) -> None:
    cfg = make_cfg(tmp_path)
    make_series(cfg)
    replies = [EXTRACT_REPLY, EXTRACT_REPLY]
    run_reference(cfg, SERIES, client=FakeClient(list(replies)))
    assert all(call["force"] is False for call in fake_pipeline.calls)
    run_reference(cfg, SERIES, client=FakeClient(list(replies)), force=True)
    assert all(call["force"] is True for call in fake_pipeline.calls[-2:])


# ---------------------------------------------------------------- CLI


def fake_summary(**overrides: Any) -> ReferenceSummary:
    values: dict[str, Any] = {
        "matched": (("Chapter 001", "ch_01"),),
        "raw_only": ("Chapter 009",),
        "reference_only": (),
        "chapters_extracted": 1,
        "pages_skipped": 0,
        "lines_paired": 2,
        "merge": MergeReport(locked=1, proposed=1, conflicts=(), rejected=()),
        "ocr_failed": (),
        "dry_run": False,
    }
    values.update(overrides)
    return ReferenceSummary(**values)


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """CPU CLI env: faked config/secrets/client/manager; lock functions recorded as events."""
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "get_secrets", lambda: None)

    clients: list[Any] = []

    class FakeCliClient:
        def __init__(self, ollama_cfg: Any, secrets: Any) -> None:
            self.args = (ollama_cfg, secrets)
            self.closed = False

        def chat(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the CLI never chats directly")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(omniscan.cli, "OllamaClient", lambda o, s: clients.append(FakeCliClient(o, s)) or clients[-1])

    events: list[str] = []

    class FakeManager:
        def release(self) -> None:
            events.append("release")

    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager", lambda c: events.append("manager") or FakeManager()
    )
    monkeypatch.setattr(
        lock_module,
        "acquire_gpu_lock",
        lambda *, poll_seconds=2.0, on_wait=None: events.append("lock") or object(),
    )
    monkeypatch.setattr(lock_module, "release_gpu_lock", lambda handle: events.append("unlock"))
    return {"cfg": cfg, "events": events, "clients": clients}


def patch_run_reference(monkeypatch: pytest.MonkeyPatch, summary: ReferenceSummary) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(
        cfg: Config,
        series: str,
        *,
        client: Any,
        model: str,
        min_locks: int,
        dry_run: bool,
        force: bool,
        gpu: Any,
    ) -> ReferenceSummary:
        calls.append(
            {
                "cfg": cfg,
                "series": series,
                "client": client,
                "model": model,
                "min_locks": min_locks,
                "dry_run": dry_run,
                "force": force,
                "gpu": gpu,
            }
        )
        return summary

    monkeypatch.setattr(reference_module, "run_reference", fake)
    return calls


def test_reference_cli_unknown_series_exits_2(cli_env: dict[str, Any]) -> None:
    result = runner.invoke(app, ["reference", "NoSuch"])
    assert result.exit_code == 2
    assert "reference: no chapters found for series 'NoSuch'" in result.output


def test_reference_cli_without_reference_chapters_exits_2(cli_env: dict[str, Any]) -> None:
    make_series(cli_env["cfg"], ref=None)
    result = runner.invoke(app, ["reference", SERIES])
    assert result.exit_code == 2
    assert "_reference_en" in result.output and "import the official releases first" in result.output


def test_reference_cli_prints_the_summary_and_closes_the_client(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_series(cli_env["cfg"])
    calls = patch_run_reference(monkeypatch, fake_summary())
    result = runner.invoke(app, ["reference", SERIES])

    assert result.exit_code == 0
    assert "reference: 1 matched chapter pair(s), 1 raw-only, 0 reference-only" in result.output
    assert "raw chapter without reference match: Chapter 009" in result.output
    assert "reference: 1 term(s) locked, 1 proposed" in result.output
    assert cli_env["clients"][0].closed
    assert cli_env["events"] == ["manager", "release"]  # cpu: no lock, manager still built and released
    call = calls[0]
    assert call["series"] == SERIES
    assert (call["min_locks"], call["model"], call["dry_run"], call["force"]) == (3, "gemma4:31b-cloud", False, False)
    assert call["client"] is cli_env["clients"][0]


def test_reference_cli_passes_flags_through(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_series(cli_env["cfg"])
    calls = patch_run_reference(monkeypatch, fake_summary())
    result = runner.invoke(
        app, ["reference", SERIES, "--min-locks", "2", "--model", "translategemma:12b", "--dry-run", "--force"]
    )
    assert result.exit_code == 0
    call = calls[0]
    assert (call["min_locks"], call["model"], call["dry_run"], call["force"]) == (
        2,
        "translategemma:12b",
        True,
        True,
    )


def test_reference_cli_exit_1_when_ocr_failed(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_series(cli_env["cfg"])
    patch_run_reference(
        monkeypatch, fake_summary(ocr_failed=(("raw", "Chapter 001", "ocr: out of memory"),))
    )
    result = runner.invoke(app, ["reference", SERIES])
    assert result.exit_code == 1
    assert "raw chapter Chapter 001 OCR failed: ocr: out of memory" in result.output


def test_reference_cli_acquires_the_gpu_lock_before_the_vram_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a real GPU the lock is acquired before build_vram_manager (its warm-up thread touches the
    device) and released after the run — the standing order after the 2026-09-22 driver crash."""
    cfg = make_cfg(tmp_path, device="cuda")
    make_series(cfg)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "get_secrets", lambda: None)

    clients: list[Any] = []

    class FakeCliClient:
        def close(self) -> None:
            return None

    monkeypatch.setattr(omniscan.cli, "OllamaClient", lambda o, s: clients.append(FakeCliClient()) or clients[-1])
    events: list[str] = []

    class FakeManager:
        def release(self) -> None:
            events.append("release")

    monkeypatch.setattr(
        "omniscan.gpu.groups.build_vram_manager", lambda c: events.append("manager") or FakeManager()
    )
    monkeypatch.setattr(
        lock_module,
        "acquire_gpu_lock",
        lambda *, poll_seconds=2.0, on_wait=None: events.append("lock") or object(),
    )
    monkeypatch.setattr(lock_module, "release_gpu_lock", lambda handle: events.append("unlock"))

    def fake_run(
        cfg: Config,
        series: str,
        *,
        client: Any,
        model: str,
        min_locks: int,
        dry_run: bool,
        force: bool,
        gpu: Any,
    ) -> ReferenceSummary:
        events.append("run")
        return fake_summary()

    monkeypatch.setattr(reference_module, "run_reference", fake_run)

    result = runner.invoke(app, ["reference", SERIES])
    assert result.exit_code == 0
    assert events == ["lock", "manager", "run", "release", "unlock"]


def test_reference_cli_dry_run_writes_nothing(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full stack through the CLI: faked OCR pipeline + faked chat, real run_reference."""
    cfg = cli_env["cfg"]
    make_series(cfg)
    fake = FakeOcrPipeline()
    monkeypatch.setattr(reference_module, "run_pipeline", fake)
    monkeypatch.setattr(
        omniscan.cli,
        "OllamaClient",
        lambda o, s: cli_env["clients"].append(FakeClient(replies=[EXTRACT_REPLY, EXTRACT_REPLY]))
        or cli_env["clients"][-1],
    )

    result = runner.invoke(app, ["reference", SERIES, "--dry-run"])
    assert result.exit_code == 0
    assert "dry run" in result.output
    assert "2 term(s) locked" not in result.output
    assert not (cfg.paths.work_root / SERIES / "series.db").exists()
    assert not (cfg.paths.work_root / SERIES / "glossary.yaml").exists()


def test_reference_cli_full_run_locks_and_exports(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg)
    fake = FakeOcrPipeline()
    monkeypatch.setattr(reference_module, "run_pipeline", fake)

    # the CLI builds its own client; hand the scripted extraction replies to that one
    monkeypatch.setattr(
        omniscan.cli,
        "OllamaClient",
        lambda o, s: cli_env["clients"].append(FakeClient(replies=[EXTRACT_REPLY, EXTRACT_REPLY]))
        or cli_env["clients"][-1],
    )
    result = runner.invoke(app, ["reference", SERIES])
    assert result.exit_code == 0
    assert "reference: 0 term(s) locked, 2 proposed" in result.output
    with GlossaryStore(cfg.paths.work_root / SERIES / "series.db") as store:
        rows = store.list()
    assert [(r.source, r.target, r.status) for r in rows] == [
        ("검은성", "Black Citadel", "proposed"),
        ("민준", "Minjun", "proposed"),
    ]
    assert (cfg.paths.work_root / SERIES / "glossary.yaml").is_file()