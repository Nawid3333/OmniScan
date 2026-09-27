"""Tests for `omniscan edit` (src/omniscan/edits/cli.py) — a hand-built chapter, fake model, no torch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.edits.cli as edit_cli
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.edits.cli import edit_app
from omniscan.llm.ollama import ChatResponse

runner = CliRunner()
ARGS = ["S", "Chapter 1"]


def region(rid: str, y0: int, text: str, order: int) -> Region:
    box = BBox(x0=10, y0=y0, x1=110, y1=y0 + 50)
    return Region(id=rid, slice_index=0, kind="bubble_text", bbox=box, text=text, reading_order=order)


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChapterPaths:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    series = SeriesPaths.from_config(cfg, "S")
    (series.library_dir / "Chapter 1").mkdir(parents=True)
    paths = series.chapter("Chapter 1")
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(regions=[region("r0001", 10, "안녕", 0), region("r0002", 200, "반가워", 1)]).save(
        paths.artifact("ocr.json")
    )
    FinalArtifact(judge_model="judge", lines=[FinalLine(region_id="r0001", text="Hi", decision="pick")]).save(
        paths.artifact("final.json")
    )
    return paths


def edit(*args: str) -> Any:
    return runner.invoke(edit_app, list(args))


def ocr(paths: ChapterPaths) -> dict[str, Region]:
    return {r.id: r for r in store.current_regions(paths)}


def english(paths: ChapterPaths) -> dict[str, str]:
    return {line.region_id: line.text for line in FinalArtifact.load(paths.artifact("final.json")).lines}


def test_region_edits_from_the_command_line(paths: ChapterPaths) -> None:
    assert edit("text", *ARGS, "r0002", "반가워요").output == "edit: r0002 source text saved\n"
    assert edit("kind", *ARGS, "r0002", "free_text").exit_code == 0
    assert edit("box", *ARGS, "r0002", "20", "210", "150", "260").output == "edit: r0002 box 20,210,150,260\n"
    moved = ocr(paths)["r0002"]
    assert (moved.text, moved.kind, moved.bbox) == (
        "반가워요",
        "free_text",
        BBox(x0=20, y0=210, x1=150, y1=260),
    )
    assert (
        edit("add", *ARGS, "10", "400", "90", "450", "--kind", "sfx", "--text", "쾅").output
        == "edit: added m0001\n"
    )
    assert ocr(paths)["m0001"].kind == "sfx"
    assert edit("delete", *ARGS, "r0001").exit_code == 0 and "r0001" not in ocr(paths)

    shown = json.loads(edit("show", *ARGS, "--json").output)
    assert [(r["id"], r["edited"]) for r in shown["regions"]] == [("r0002", True), ("m0001", True)]
    assert shown["deleted"] == [{"id": "r0001", "text": "안녕"}]
    table = edit("show", *ARGS)
    assert (
        table.exit_code == 0
        and "반가워요" in table.output
        and "deleted by hand: r0001 (안녕)" in table.output
    )

    assert edit("revert", *ARGS, "r0001").output == "edit: r0001 reverted\n" and "r0001" in ocr(paths)
    assert edit("revert", *ARGS, "m0001").output == "edit: m0001 (drawn by hand) removed\n"


def test_english_lines_and_revert(paths: ChapterPaths) -> None:
    assert edit("english", *ARGS, "r0002", "Nice to see you").exit_code == 0
    assert edit("english", *ARGS, "r0001", "Hello!").exit_code == 0
    assert english(paths) == {"r0001": "Hello!", "r0002": "Nice to see you"}
    shown = json.loads(edit("show", *ARGS, "--json").output)
    assert [(r["english"], r["hand_translated"]) for r in shown["regions"]] == [
        ("Hello!", True),
        ("Nice to see you", True),
    ]
    assert (
        edit("revert", *ARGS, "r0001", "--english").output == "edit: r0001 English is the judge's again: Hi\n"
    )
    assert edit("revert", *ARGS, "r0002", "--english").output.endswith("again: (none)\n")


def test_output_cuts(paths: ChapterPaths) -> None:
    assert edit("cuts", *ARGS).output == "cuts: one image per slice\n"
    assert edit("cuts", *ARGS, "400", "150", "400").output == "cuts: 150, 400\n"
    assert edit("cuts", *ARGS).output == "cuts: 150, 400\n"
    assert edit("cuts", *ARGS, "--reset").output == "cuts: one image per slice\n"


def test_errors_exit_2_with_the_reason(paths: ChapterPaths) -> None:
    missing = edit("text", *ARGS, "r0009", "x")
    assert missing.exit_code == 2 and "region 'r0009' not found in ocr.json" in missing.output
    backwards = edit("box", *ARGS, "r0001", "100", "10", "20", "60")
    assert backwards.exit_code == 2 and "second corner" in backwards.output
    outside = edit("cuts", *ARGS, "900")
    assert outside.exit_code == 2 and "outside the strip" in outside.output
    assert edit("revert", *ARGS, "r0002").exit_code == 2  # nothing to revert
    assert edit("kind", *ARGS, "r0001", "poster").exit_code == 2  # not a kind (click usage error)


class FakeChat:
    """chat_json replies `EN(<source>)` per region; remembers being closed."""

    def __init__(self) -> None:
        self.closed = False

    def chat(self, model: str, messages: list[dict[str, Any]], **_: Any) -> ChatResponse:
        regions = json.loads(messages[-1]["content"].split("Regions (reading order):\n", 1)[1])
        answer = {"translations": [{"id": r["id"], "text": f"EN({r['text']})"} for r in regions]}
        return ChatResponse(json.dumps(answer, ensure_ascii=False), model, True, None, 1, 1, {})

    def close(self) -> None:
        self.closed = True


def test_translate_prints_suggestions_and_applies_them(
    paths: ChapterPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = tmp_path / "translation_profiles.toml"
    profiles.write_text(
        '[profiles.cloud]\nendpoint = "local"\nmodel = "cloud-model"\nstyle = "chat_json"\n', encoding="utf-8"
    )
    fake = FakeChat()
    monkeypatch.setattr(edit_cli, "default_profile_paths", lambda: [profiles])
    monkeypatch.setattr(edit_cli, "make_chat_client", lambda _cfg: fake)
    shown = edit("translate", *ARGS, "r0002")
    assert shown.output == "r0002 [cloud] EN(반가워)\n" and fake.closed
    assert "r0002" not in english(paths)  # nothing written without --apply
    kept = edit("translate", *ARGS, "r0002", "--apply")
    assert kept.output.endswith("edit: r0002 English kept\n") and english(paths)["r0002"] == "EN(반가워)"
    assert store.load_edits(paths).translations[0].suggested_by == "cloud"
    unknown = edit("translate", *ARGS, "r0002", "--profile", "nope")
    assert unknown.exit_code == 2 and "unknown profile 'nope'" in unknown.output
