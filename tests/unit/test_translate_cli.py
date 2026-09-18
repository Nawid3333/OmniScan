"""Tests for `omniscan translate` — fake Ollama client monkeypatched onto omniscan.cli."""

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
from omniscan.core.schemas import BBox, GlossaryEntry, Region, RegionsArtifact
from omniscan.glossary.store import GlossaryStore
from omniscan.llm.ollama import ChatResponse, OllamaError, OllamaRateLimitError

SERIES = "S"
runner = CliRunner()

PROFILES_TOML = """
[profiles.p1]
endpoint = "local"
model = "m1"
style = "chat_json"

[profiles.p2]
enabled = false
endpoint = "local"
model = "m2"
style = "chat_json"

[profiles.p3]
enabled = false
endpoint = "local"
model = "m3"
style = "translategemma"
"""


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


def json_reply(texts: dict[str, str]) -> str:
    return json.dumps(
        {"translations": [{"id": rid, "text": text} for rid, text in texts.items()]}, ensure_ascii=False
    )


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
def profile_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "translation_profiles.toml"
    path.write_text(PROFILES_TOML, encoding="utf-8")
    import omniscan.translate.profiles as profiles_module

    monkeypatch.setattr(profiles_module, "default_profile_paths", lambda: [path])
    return path


@pytest.fixture
def patched(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "OllamaClient", FakeOllamaClient)
    FakeOllamaClient.instances = []
    FakeOllamaClient.script = []
    return cfg


def setup_chapter(cfg: Config, chapter: str, *, ocr: bool = True, regions: int = 2) -> None:
    (cfg.paths.library_root / SERIES / chapter).mkdir(parents=True)
    if ocr:
        work = cfg.paths.work_root / SERIES / chapter
        work.mkdir(parents=True, exist_ok=True)
        RegionsArtifact(regions=[region(f"r{i:04d}", f"문장 {i}") for i in range(1, regions + 1)]).save(
            work / "ocr.json"
        )


def test_default_profiles_are_enabled_ones_and_files_written(patched: Config, profile_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    FakeOllamaClient.script = [json_reply({"r0001": "Hello", "r0002": "Hi"})]
    result = runner.invoke(app, ["translate", SERIES])
    assert result.exit_code == 0
    line = re.search(
        rf"{re.escape(SERIES)}/Chapter 1 p1: done \(2 regions, 0 missing, \d+\.\d+s\)", result.output
    )
    assert line is not None
    assert "p2" not in result.output
    loaded = json.loads(
        (patched.paths.work_root / SERIES / "Chapter 1" / "translations" / "p1.json").read_text("utf-8")
    )
    assert loaded["run_id"] == "p1"
    assert [c["region_id"] for c in loaded["candidates"]] == ["r0001", "r0002"]


def test_profile_flag_selects(patched: Config, profile_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    FakeOllamaClient.script = [json_reply({"r0001": "A", "r0002": "B"})]
    result = runner.invoke(app, ["translate", SERIES, "--profile", "p2"])
    assert result.exit_code == 0
    assert "p2: done" in result.output


def test_unknown_profile_exits_2(patched: Config, profile_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    result = runner.invoke(app, ["translate", SERIES, "--profile", "nope"])
    assert result.exit_code == 2
    assert "translate: unknown profile 'nope'" in result.output
    assert "(known: p1, p2, p3)" in result.output


def test_no_chapters_exits_2(patched: Config, profile_toml: Path) -> None:
    result = runner.invoke(app, ["translate", SERIES])
    assert result.exit_code == 2
    assert "translate: no chapters found for series 'S'" in result.output


def test_missing_ocr_for_one_chapter_fails_while_other_runs(patched: Config, profile_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1", ocr=False)
    setup_chapter(patched, "Chapter 2")
    FakeOllamaClient.script = [json_reply({"r0001": "A", "r0002": "B"})]
    result = runner.invoke(app, ["translate", SERIES])
    assert result.exit_code == 1
    assert f"{SERIES}/Chapter 1: ocr.json missing — run the ocr stage first" in result.output
    assert f"{SERIES}/Chapter 2 p1: done" in result.output


def test_rate_limit_exits_3_with_message(patched: Config, profile_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    FakeOllamaClient.script = [OllamaRateLimitError("429", status_code=429)]
    result = runner.invoke(app, ["translate", SERIES])
    assert result.exit_code == 3
    assert "translate: Ollama rate limit reached — partial results are kept; re-run later" in result.output


def test_other_ollama_error_counts_as_failure_and_continues(patched: Config, profile_toml: Path) -> None:
    setup_chapter(patched, "Chapter 1")
    setup_chapter(patched, "Chapter 2")
    FakeOllamaClient.script = [
        OllamaError("Ollama HTTP 404 on POST", status_code=404),
        json_reply({"r0001": "A", "r0002": "B"}),
    ]
    result = runner.invoke(app, ["translate", SERIES])
    assert result.exit_code == 1
    assert "translate: Ollama HTTP 404" in result.output
    assert f"{SERIES}/Chapter 2 p1: done" in result.output


def test_no_series_db_runs_without_glossary_and_does_not_create_it(
    patched: Config, profile_toml: Path
) -> None:
    setup_chapter(patched, "Chapter 1")
    FakeOllamaClient.script = [json_reply({"r0001": "Hello", "r0002": "Hi"})]
    result = runner.invoke(app, ["translate", SERIES])
    assert result.exit_code == 0
    assert not (patched.paths.work_root / SERIES / "series.db").exists()


def test_glossary_db_entries_reach_the_prompt(patched: Config, profile_toml: Path) -> None:
    work = patched.paths.work_root / SERIES / "Chapter 1"
    work.mkdir(parents=True)
    (patched.paths.library_root / SERIES / "Chapter 1").mkdir(parents=True)
    RegionsArtifact(regions=[region("r0001", "성진이가 간다")]).save(work / "ocr.json")
    with GlossaryStore(patched.paths.work_root / SERIES / "series.db") as store:
        store.add(GlossaryEntry(source="성진", target="Seong-jin", status="locked", type="person"))
    FakeOllamaClient.script = [json_reply({"r0001": "Hello", "r0002": "Hi"})]
    result = runner.invoke(app, ["translate", SERIES])
    assert result.exit_code == 0
    client = FakeOllamaClient.instances[0]
    user = client.calls[0][1]["content"]
    assert "Glossary (binding):\n- 성진 -> Seong-jin (person)" in user
