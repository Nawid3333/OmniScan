"""Tests for `omniscan judge` — fake Ollama client monkeypatched onto omniscan.cli."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.core.schemas import BBox, Candidate, CandidateRun, GlossaryEntry, Region, RegionsArtifact
from omniscan.glossary.store import GlossaryStore
from omniscan.llm.ollama import ChatResponse, OllamaError, OllamaRateLimitError

SERIES = "S"
runner = CliRunner()

JUDGE_TOML = '[judge]\nmodel = "judge-model"\n'


class FakeOllamaClient:
    """Stands in for OllamaClient (the CLI builds exactly one); `script` is one FIFO reply queue."""

    script: ClassVar[list[str | Exception]] = []
    instances: ClassVar[list[FakeOllamaClient]] = []

    def __init__(self, cfg: Any, secrets: Any) -> None:
        self.replies = FakeOllamaClient.script
        self.calls: list[list[dict[str, Any]]] = []
        FakeOllamaClient.instances.append(self)

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
        self.calls.append(messages)
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

    def close(self) -> None:
        pass

    def __enter__(self) -> FakeOllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def judgements_reply(*judgements: dict[str, str]) -> str:
    return json.dumps({"judgements": list(judgements)}, ensure_ascii=False)


def region(rid: str, text: str) -> Region:
    return Region(id=rid, slice_index=0, kind="bubble_text", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text=text)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        )
    )


@pytest.fixture
def judge_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "judge.toml"
    path.write_text(JUDGE_TOML, encoding="utf-8")
    import omniscan.translate.judge_config as judge_config_module

    monkeypatch.setattr(judge_config_module, "default_judge_paths", lambda: [path])
    return path


@pytest.fixture
def patched(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "OllamaClient", FakeOllamaClient)
    FakeOllamaClient.instances = []
    FakeOllamaClient.script = []
    return cfg


def setup_chapter(
    cfg: Config, chapter: str, *, ocr: bool = True, texts: dict[str, str] | None = None
) -> None:
    (cfg.paths.library_root / SERIES / chapter).mkdir(parents=True)
    if ocr:
        work = cfg.paths.work_root / SERIES / chapter
        work.mkdir(parents=True, exist_ok=True)
        regions = [region(rid, text) for rid, text in (texts or {"r0001": "안녕", "r0002": "반가워"}).items()]
        RegionsArtifact(regions=regions).save(work / "ocr.json")


def write_run(cfg: Config, chapter: str, run_id: str, texts: dict[str, str]) -> None:
    CandidateRun(
        run_id=run_id,
        profile=run_id,
        model="run-model",
        candidates=[Candidate(region_id=rid, text=text) for rid, text in texts.items()],
    ).save(cfg.paths.work_root / SERIES / chapter / "translations" / f"{run_id}.json")


def test_agreeing_runs_are_auto_picked_and_final_json_written(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello", "r0002": "Hi"})
    write_run(patched, "Chapter 1", "run2", {"r0001": "Hello", "r0002": "Hi"})
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 0
    assert re.search(
        rf"{re.escape(SERIES)}/Chapter 1: done \(2 regions, 0 judged, 2 auto, "
        rf"0 untranslated, 0 violations left, \d+\.\d+s\)",
        result.output,
    )
    loaded = json.loads((patched.paths.work_root / SERIES / "Chapter 1" / "final.json").read_text("utf-8"))
    assert loaded["judge_model"] == "judge-model"
    assert [line["region_id"] for line in loaded["lines"]] == ["r0001", "r0002"]


def test_disagreeing_runs_are_judged(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello", "r0002": "Well met"})
    write_run(patched, "Chapter 1", "run2", {"r0001": "Goodbye", "r0002": "See you"})
    FakeOllamaClient.script = [
        judgements_reply(
            {"id": "r0001", "decision": "pick", "pick": "A"},
            {"id": "r0002", "decision": "pick", "pick": "B"},
        )
    ]
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 0
    assert re.search(
        rf"{re.escape(SERIES)}/Chapter 1: done \(2 regions, 2 judged, 0 auto, "
        rf"0 untranslated, 0 violations left, \d+\.\d+s\)",
        result.output,
    )


def test_existing_final_json_is_skipped(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello", "r0002": "Hi"})
    assert runner.invoke(app, ["judge", SERIES]).exit_code == 0
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1: skipped" in result.output


def test_force_flag_reruns(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello", "r0002": "Hi"})
    assert runner.invoke(app, ["judge", SERIES]).exit_code == 0
    result = runner.invoke(app, ["judge", SERIES, "--force"])
    assert result.exit_code == 0
    assert f"{SERIES}/Chapter 1: done" in result.output


def test_no_chapters_exits_2(patched: Config, judge_toml: Path) -> None:
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 2
    assert "judge: no chapters found for series 'S'" in result.output


def test_missing_ocr_for_one_chapter_fails_while_other_judges(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1", ocr=False)
    setup_chapter(patched, "Chapter 2")
    write_run(patched, "Chapter 2", "run1", {"r0001": "Hi", "r0002": "Ho"})
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 1
    assert f"{SERIES}/Chapter 1: ocr.json missing — run the ocr stage first" in result.output
    assert f"{SERIES}/Chapter 2: done" in result.output


def test_no_runs_for_one_chapter_fails_while_other_judges(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    setup_chapter(patched, "Chapter 2")
    write_run(patched, "Chapter 2", "run1", {"r0001": "Hi", "r0002": "Ho"})
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 1
    assert f"{SERIES}/Chapter 1: no translation runs — run `omniscan translate` first" in result.output
    assert f"{SERIES}/Chapter 2: done" in result.output


def test_rate_limit_exits_3(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello"})
    write_run(patched, "Chapter 1", "run2", {"r0001": "Goodbye"})
    FakeOllamaClient.script = [OllamaRateLimitError("429", status_code=429)]
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 3
    assert "judge: Ollama rate limit reached — re-run later" in result.output


def test_other_ollama_error_counts_as_failure_and_continues(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello"})
    write_run(patched, "Chapter 1", "run2", {"r0001": "Goodbye"})
    setup_chapter(patched, "Chapter 2")
    write_run(patched, "Chapter 2", "run1", {"r0001": "Hi", "r0002": "Ho"})
    FakeOllamaClient.script = [OllamaError("Ollama HTTP 404 on POST", status_code=404)]
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 1
    assert "judge: Ollama HTTP 404" in result.output
    assert f"{SERIES}/Chapter 2: done" in result.output


def test_unknown_run_flag_exits_2(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello", "r0002": "Hi"})
    result = runner.invoke(app, ["judge", SERIES, "--run", "nope"])
    assert result.exit_code == 2
    assert "judge: unknown run 'nope' (known: run1)" in result.output


def test_no_series_db_runs_without_glossary_and_does_not_create_it(patched: Config, judge_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    write_run(patched, "Chapter 1", "run1", {"r0001": "Hello", "r0002": "Hi"})
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 0
    assert not (patched.paths.work_root / SERIES / "series.db").exists()


def test_invalid_judge_config_exits_2(
    patched: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_chapter(patched, "Chapter 1")
    path = tmp_path / "judge.toml"
    path.write_text("[judge]\ntemperature = 3.0\n", encoding="utf-8")
    import omniscan.translate.judge_config as judge_config_module

    monkeypatch.setattr(judge_config_module, "default_judge_paths", lambda: [path])
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 2
    assert "judge: " in result.output
    assert "judge.toml" in result.output


def test_glossary_reaches_the_judge_prompt_and_violations_are_counted(
    patched: Config, judge_toml: Path
) -> None:
    work = patched.paths.work_root / SERIES / "Chapter 1"
    (patched.paths.library_root / SERIES / "Chapter 1").mkdir(parents=True)
    work.mkdir(parents=True)
    RegionsArtifact(regions=[region("r0001", "성진이가 간다")]).save(work / "ocr.json")
    with GlossaryStore(patched.paths.work_root / SERIES / "series.db") as store:
        store.add(GlossaryEntry(source="성진", target="Seong-jin", status="locked", type="person"))
    write_run(patched, "Chapter 1", "run1", {"r0001": "He goes"})
    write_run(patched, "Chapter 1", "run2", {"r0001": "Away he goes"})
    FakeOllamaClient.script = [
        judgements_reply({"id": "r0001", "decision": "pick", "pick": "A"}),
        judgements_reply({"id": "r0001", "decision": "rewrite", "text": "He goes again"}),
    ]
    result = runner.invoke(app, ["judge", SERIES])
    assert result.exit_code == 0
    client = FakeOllamaClient.instances[0]
    user = client.calls[0][1]["content"]
    assert "Glossary (binding):\n- 성진 -> Seong-jin (person)" in user
    assert re.search(
        rf"{re.escape(SERIES)}/Chapter 1: done \(1 regions, 1 judged, 0 auto, "
        rf"0 untranslated, 1 violations left, \d+\.\d+s\)",
        result.output,
    )
