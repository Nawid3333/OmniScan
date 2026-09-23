"""Tests for `omniscan glossary propose` (card C5c): CLI surface, faked chat client, no GPU.

The GPU-freedom proof: with a cuda config and acquire_gpu_lock/build_vram_manager patched to blow
up, the command still exits 0 — it must never import or call either (card: "no GPU lock, no VRAM
manager"). No other test file covers glossary_app's commands, so this is a new file."""

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
from omniscan.core.schemas import BBox, Region, RegionsArtifact
from omniscan.glossary.store import GlossaryStore
from omniscan.llm.ollama import ChatResponse

SERIES = "S"
runner = CliRunner()

TERMS_REPLY = '{"terms": [{"source": "민준", "target": "Minjun", "type": "person"}]}'


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


def write_ocr(cfg: Config, chapter: str) -> None:
    paths: ChapterPaths = SeriesPaths.from_config(cfg, SERIES).chapter(chapter)
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=0, y0=0, x1=10, y1=10),
                text="민준이",
            )
        ]
    ).save(paths.artifact("ocr.json"))


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
    raise AssertionError("the glossary propose command must never touch the GPU lock or the VRAM manager")


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


def test_glossary_propose_happy_path(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 1")
    cli_env["replies"].append(TERMS_REPLY)
    result = runner.invoke(app, ["glossary", "propose", SERIES])
    assert result.exit_code == 0
    assert (
        "glossary: 1 chapter(s) scanned, 1 with OCR text, 1 line(s), 0 term(s) above --min-chapters"
        in result.output
    )
    assert "glossary: 0 term(s) proposed, 0 locked" in result.output  # one chapter: below --min-chapters 2
    assert cli_env["clients"][-1].closed


def test_glossary_propose_writes_proposed_entries_and_exports_yaml(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1", "Chapter 2")
    write_ocr(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 2")
    cli_env["replies"].extend([TERMS_REPLY, TERMS_REPLY])
    result = runner.invoke(app, ["glossary", "propose", SERIES])
    assert result.exit_code == 0
    assert "glossary: 1 term(s) proposed, 0 locked" in result.output
    with GlossaryStore(SeriesPaths.from_config(cfg, SERIES).db) as store:
        row = store.list()[0]
    assert (row.source, row.target, row.status, row.origin) == ("민준", "Minjun", "proposed", "llm")
    assert SeriesPaths.from_config(cfg, SERIES).glossary_yaml.is_file()


def test_glossary_propose_json_output_has_locked_zero(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1", "Chapter 2")
    write_ocr(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 2")
    cli_env["replies"].extend([TERMS_REPLY, TERMS_REPLY])
    result = runner.invoke(app, ["glossary", "propose", SERIES, "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert set(parsed) == {
        "chapters_scanned",
        "chapters_with_lines",
        "lines_scanned",
        "aggregated_above_threshold",
        "merge",
    }
    assert parsed["merge"]["locked"] == 0  # proposals never auto-lock
    assert parsed["merge"]["proposed"] == 1
    assert parsed["merge"]["conflicts"] == [] and parsed["merge"]["rejected"] == []


def test_glossary_propose_dry_run_writes_nothing(cli_env: dict[str, Any]) -> None:
    cfg = cli_env["cfg"]
    make_series(cfg, "Chapter 1")
    write_ocr(cfg, "Chapter 1")
    cli_env["replies"].append(TERMS_REPLY)
    result = runner.invoke(app, ["glossary", "propose", SERIES, "--dry-run", "--min-chapters", "1"])
    assert result.exit_code == 0
    assert "glossary: 1 term(s) proposed, 0 locked" in result.output
    assert "glossary: dry run — nothing written" in result.output
    sp = SeriesPaths.from_config(cfg, SERIES)
    assert not sp.db.exists() and not sp.glossary_yaml.exists()


def test_glossary_propose_no_chapters_exits_2(cli_env: dict[str, Any]) -> None:
    result = runner.invoke(app, ["glossary", "propose", "NoSuch"])
    assert result.exit_code == 2
    assert "glossary: no chapters found for series 'NoSuch'" in result.output
    assert cli_env["clients"] == []


def test_glossary_propose_needs_no_gpu_even_on_a_cuda_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path, device="cuda")
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "get_secrets", lambda: None)
    monkeypatch.setattr(omniscan.cli, "OllamaClient", FakeCliClient)
    monkeypatch.setattr("omniscan.gpu.lock.acquire_gpu_lock", _boom)
    monkeypatch.setattr("omniscan.gpu.groups.build_vram_manager", _boom)
    make_series(cfg, "Chapter 1")  # no ocr.json anywhere: nothing to chat about
    result = runner.invoke(app, ["glossary", "propose", SERIES])
    assert result.exit_code == 0  # any lock/manager call would have raised AssertionError
