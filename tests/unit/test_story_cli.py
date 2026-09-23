"""Tests for `omniscan story summarize` (card C5c): CLI surface, faked chat client, no GPU.

The GPU-freedom proof: with a cuda config and acquire_gpu_lock/build_vram_manager patched to blow
up, the command still exits 0 — it must never import or call either (card: "no GPU lock, no VRAM
manager")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, FinalArtifact, FinalLine, Region, RegionsArtifact
from omniscan.llm.ollama import ChatResponse
from omniscan.story.store import SummaryStore

SERIES = "S"
runner = CliRunner()


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


def make_series(cfg: Config, *chapters: str) -> SeriesPaths:
    sp = SeriesPaths.from_config(cfg, SERIES)
    for name in chapters:
        (sp.library_dir / name).mkdir(parents=True, exist_ok=True)
    return sp


def ready(cfg: Config, chapter: str) -> None:
    """final.json + ocr.json for one chapter."""
    paths: ChapterPaths = SeriesPaths.from_config(cfg, SERIES).chapter(chapter)
    RegionsArtifact(
        regions=[Region(id="r0001", slice_index=0, kind="bubble_text", bbox=BBox(x0=0, y0=0, x1=10, y1=10))]
    ).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[FinalLine(region_id="r0001", text="첫 문장", decision="pick", sources=["run"])],
    ).save(paths.artifact("final.json"))


class FakeCliClient:
    """OllamaClient double: scripted replies, close tracked."""

    def __init__(self, ollama_cfg: Any, secrets: Any) -> None:
        self.args = (ollama_cfg, secrets)
        self.closed = False
        self.replies: list[str] = []

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
        return ChatResponse(
            content=self.replies.pop(0),
            model=model,
            done=True,
            total_duration_ns=None,
            prompt_eval_count=None,
            eval_count=None,
            raw={},
        )

    def close(self) -> None:
        self.closed = True


def _boom(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the story commands must never touch the GPU lock or the VRAM manager")


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """CPU CLI env: faked config/secrets/client; every GPU entry point booby-trapped."""
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "get_secrets", lambda: None)
    clients: list[FakeCliClient] = []
    replies: list[str] = []

    def factory(o: Any, s: Any) -> FakeCliClient:
        client = FakeCliClient(o, s)
        client.replies = list(replies)
        clients.append(client)
        return client

    monkeypatch.setattr(omniscan.cli, "OllamaClient", factory)
    monkeypatch.setattr("omniscan.gpu.lock.acquire_gpu_lock", _boom)
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", _boom)
    return {"cfg": cfg, "clients": clients, "replies": replies}


def test_story_summarize_happy_path(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1")
    ready(cfg, "Chapter 1")
    cli_env["replies"].append('{"summary": "Minjun enters the gate."}')
    result = runner.invoke(app, ["story", "summarize", SERIES])
    assert result.exit_code == 0
    assert "story: Chapter 1: summarized" in result.output
    assert "story: 1 summarized, 0 without final.json, 0 already summarized" in result.output
    client = cli_env["clients"][-1]
    assert client.closed
    with SummaryStore(SeriesPaths.from_config(cfg, SERIES).db) as store:
        row = store.get("Chapter 1")
        assert row is not None and row.summary == "Minjun enters the gate."
        assert row.model == "gemma4:31b-cloud"


def test_story_summarize_json_output(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1", "Chapter 2")
    ready(cfg, "Chapter 1")  # Chapter 2 has no final.json
    cli_env["replies"].append('{"summary": "One."}')
    result = runner.invoke(app, ["story", "summarize", SERIES, "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert set(parsed) == {"series", "done", "skipped_no_final", "skipped_existing"}
    assert (parsed["series"], parsed["done"], parsed["skipped_no_final"], parsed["skipped_existing"]) == (
        SERIES,
        ["Chapter 1"],
        ["Chapter 2"],
        [],
    )


def test_story_summarize_no_chapters_exits_2(cli_env: dict[str, Any]) -> None:
    result = runner.invoke(app, ["story", "summarize", "NoSuch"])
    assert result.exit_code == 2
    assert "story: no chapters found for series 'NoSuch'" in result.output
    assert cli_env["clients"] == []  # no client built for a series with no chapters


def test_story_summarize_force_re_summarizes(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1")
    ready(cfg, "Chapter 1")
    with SummaryStore(SeriesPaths.from_config(cfg, SERIES).db) as store:
        store.set("Chapter 1", "Old.", "old-model")
    cli_env["replies"].append('{"summary": "New."}')
    result = runner.invoke(app, ["story", "summarize", SERIES])
    assert result.exit_code == 0
    assert "story: Chapter 1: skipped (already summarized, use --force)" in result.output
    result = runner.invoke(app, ["story", "summarize", SERIES, "--force"])
    assert result.exit_code == 0
    assert "story: Chapter 1: summarized" in result.output
    with SummaryStore(SeriesPaths.from_config(cfg, SERIES).db) as store:
        row = store.get("Chapter 1")
        assert row is not None and row.summary == "New."


def test_story_summarize_chapter_flag_narrows_and_orders(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 10", "Chapter 2")
    ready(cfg, "Chapter 2")
    ready(cfg, "Chapter 10")
    cli_env["replies"].extend(['{"summary": "Ten."}', '{"summary": "Two."}'])
    result = runner.invoke(app, ["story", "summarize", SERIES, "-c", "Chapter 10", "-c", "Chapter 2"])
    assert result.exit_code == 0
    assert result.output.index("Chapter 10: summarized") < result.output.index("Chapter 2: summarized")
    result = runner.invoke(app, ["story", "summarize", SERIES, "--json", "--force", "-c", "Chapter 2"])
    assert json.loads(result.output)["done"] == ["Chapter 2"]


def test_story_summarize_skips_no_final(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1")
    result = runner.invoke(app, ["story", "summarize", SERIES])
    assert result.exit_code == 0
    assert "story: Chapter 1: skipped (no final.json yet)" in result.output
    assert cli_env["clients"][-1].closed  # the client was built and closed even with nothing to do


def test_story_summarize_needs_no_gpu_even_on_a_cuda_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path, device="cuda")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "get_secrets", lambda: None)
    monkeypatch.setattr(omniscan.cli, "OllamaClient", FakeCliClient)
    monkeypatch.setattr("omniscan.gpu.lock.acquire_gpu_lock", _boom)
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", _boom)
    make_series(cfg, "Chapter 1")  # no chapters with final.json: nothing to chat about
    result = runner.invoke(app, ["story", "summarize", SERIES])
    assert result.exit_code == 0  # any lock/manager call would have raised AssertionError
