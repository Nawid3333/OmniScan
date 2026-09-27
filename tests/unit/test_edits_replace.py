"""Tests for find & replace across English lines and source texts (edits/replace.py) through the CLI and the web API
(torch-free)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import omniscan.edits.cli as edit_cli
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    Region,
    RegionKind,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.edits.replace import Change, FindReplace, apply_changes, plan
from omniscan.web.app import create_app


def region(rid: str, y0: int, text: str, *, kind: RegionKind = "bubble_text") -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=BBox(x0=10, y0=y0, x1=110, y1=y0 + 50), text=text)


# ---------------------------------------------------------------- the rule


def test_literal_whole_word_and_case_rules() -> None:
    assert (
        FindReplace("Jinwoo", "Jin-Woo").replacer()("Jinwoo? JINWOO! Jinwoo's")
        == "Jin-Woo? JINWOO! Jin-Woo's"
    )
    assert (
        FindReplace("cat", "dog", whole_word=True).replacer()("cat catalog bobcat cat.")
        == "dog catalog bobcat dog."
    )
    any_case = FindReplace("jinwoo", "jin-woo", case_sensitive=False).replacer()
    assert any_case("Jinwoo, JINWOO, jinwoo") == "Jin-woo, JIN-WOO, jin-woo"  # each match keeps its case
    assert FindReplace("a.c", "x").replacer()("abc a.c") == "abc x"  # literal: the dot is a dot


def test_regex_rules_and_their_errors() -> None:
    swap = FindReplace(r"(\w+)-nim", r"\1-sama", regex=True).replacer()
    assert swap("Hyung-nim and Noona-nim") == "Hyung-sama and Noona-sama"
    assert FindReplace(r"\.{3}", "…", regex=True).replacer()("Wait... what...") == "Wait… what…"
    with pytest.raises(ValueError, match="nothing to find"):
        FindReplace("", "x").replacer()
    with pytest.raises(ValueError, match="bad regular expression"):
        FindReplace("(", "x", regex=True).replacer()
    with pytest.raises(ValueError, match="bad replacement"):
        FindReplace("a", r"\2", regex=True).replacer()("a")


# ---------------------------------------------------------------- a series of two chapters


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


def make_chapter(cfg: Config, name: str, lines: dict[str, str]) -> ChapterPaths:
    paths = SeriesPaths.from_config(cfg, "S").chapter(name)
    paths.raw_dir.mkdir(parents=True)
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    regions = [
        region("r0001", 10, "진우야"),
        region("r0002", 200, "진우 형"),
        region("r0003", 400, "site.com 진우", kind="watermark"),
    ]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[FinalLine(region_id=rid, text=text, decision="pick") for rid, text in lines.items()],
    ).save(paths.artifact("final.json"))
    return paths


@pytest.fixture
def chapters(cfg: Config) -> tuple[ChapterPaths, ChapterPaths]:
    one = make_chapter(cfg, "Chapter 1", {"r0001": "Jinwoo!", "r0002": "JINWOO, wait."})
    two = make_chapter(cfg, "Chapter 2", {"r0001": "Hey Jinwoo."})
    return one, two


def english(paths: ChapterPaths) -> dict[str, str]:
    return {line.region_id: line.text for line in FinalArtifact.load(paths.artifact("final.json")).lines}


def test_plan_and_apply_record_hand_edits_as_one_undo_step(
    chapters: tuple[ChapterPaths, ChapterPaths],
) -> None:
    one, _two = chapters
    rule = FindReplace("jinwoo", "Jin-Woo", case_sensitive=False)
    changes = plan(one, rule, "english")
    assert changes == [
        Change("Chapter 1", "r0001", "Jinwoo!", "Jin-Woo!"),
        Change("Chapter 1", "r0002", "JINWOO, wait.", "JIN-WOO, wait."),
    ]
    apply_changes(one, changes, "english", direction="ltr")
    lines = FinalArtifact.load(one.artifact("final.json")).lines
    assert [(line.text, line.decision) for line in lines] == [
        ("Jin-Woo!", "manual"),
        ("JIN-WOO, wait.", "manual"),
    ]
    assert store.history_steps(one) == (1, 0)
    store.undo(one, direction="ltr")
    assert english(one) == {"r0001": "Jinwoo!", "r0002": "JINWOO, wait."}
    sources = plan(one, FindReplace("진우", "성진우"), "source")
    assert [(c.region_id, c.after) for c in sources] == [
        ("r0001", "성진우야"),
        ("r0002", "성진우 형"),
    ]  # not the watermark
    apply_changes(one, sources, "source", direction="ltr")
    assert [r.text for r in store.current_regions(one)][:2] == ["성진우야", "성진우 형"]


def test_edit_replace_on_the_command_line(
    chapters: tuple[ChapterPaths, ChapterPaths], cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    one, two = chapters
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    runner = CliRunner()
    dry = runner.invoke(edit_cli.edit_app, ["replace", "S", "Jinwoo", "Jin-Woo", "-i", "--word", "--dry-run"])
    assert dry.output.splitlines() == [
        "Chapter 1 r0001: Jinwoo! → Jin-Woo!",
        "Chapter 1 r0002: JINWOO, wait. → JIN-WOO, wait.",
        "Chapter 2 r0001: Hey Jinwoo. → Hey Jin-Woo.",
        "replace: would replace 3 English line(s) in 2 chapter(s)",
    ]
    assert not one.artifact("edits.json").exists()
    only_two = runner.invoke(
        edit_cli.edit_app, ["replace", "S", "Jinwoo", "Jin-Woo", "--chapter", "Chapter 2"]
    )
    assert only_two.output.endswith("replace: replaced 1 English line(s) in 1 chapter(s)\n")
    assert english(two) == {"r0001": "Hey Jin-Woo."} and english(one)["r0001"] == "Jinwoo!"
    unknown = runner.invoke(edit_cli.edit_app, ["replace", "S", "a", "b", "-c", "Chapter 9"])
    assert unknown.exit_code == 2 and "no chapter 'Chapter 9' in S" in unknown.output
    bad = runner.invoke(edit_cli.edit_app, ["replace", "S", "(", "b", "--regex"])
    assert bad.exit_code == 2 and "bad regular expression" in bad.output


def test_find_and_replace_over_the_web_api(chapters: tuple[ChapterPaths, ChapterPaths], cfg: Config) -> None:
    one, two = chapters
    client = TestClient(create_app(cfg))
    body = {"find": "Jinwoo", "replace": "Jin-Woo", "case_sensitive": False}
    preview = client.post("/api/series/S/replace", json=body)
    assert preview.status_code == 200 and not preview.json()["applied"]
    assert [(c["chapter"], c["region_id"]) for c in preview.json()["changes"]] == [
        ("Chapter 1", "r0001"),
        ("Chapter 1", "r0002"),
        ("Chapter 2", "r0001"),
    ]
    assert not one.artifact("edits.json").exists()
    done = client.post("/api/series/S/replace", json={**body, "chapters": ["Chapter 1"], "dry_run": False})
    assert done.json()["applied"] and len(done.json()["changes"]) == 2
    assert english(one) == {"r0001": "Jin-Woo!", "r0002": "JIN-WOO, wait."} and english(two) == {
        "r0001": "Hey Jinwoo."
    }
    assert client.post("/api/series/S/replace", json={**body, "chapters": ["Chapter 9"]}).status_code == 404
    bad = client.post("/api/series/S/replace", json={**body, "find": "(", "regex": True})
    assert bad.status_code == 422 and "bad regular expression" in bad.json()["detail"]
