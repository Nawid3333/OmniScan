"""`omniscan qa`: runs the qa stage (faked here) and prints what qa.json lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import QaArtifact, QaIssue

runner = CliRunner()


@pytest.fixture
def series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    paths = SeriesPaths.from_config(cfg, "S")
    for name in ("Chapter 1", "Chapter 2"):
        (paths.library_dir / name).mkdir(parents=True)
    issue = QaIssue(
        region_id="r0002", kind="source_left", message="the original text is still readable", read="안녕"
    )
    QaArtifact(checked=4, issues=[issue]).save(paths.chapter("Chapter 1").artifact("qa.json"))
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(omniscan.cli, "_run_stages", lambda *args, **kwargs: calls.append((*args, kwargs)))
    return calls


def test_qa_prints_the_issues_of_every_chapter(series: list[tuple[Any, ...]]) -> None:
    result = runner.invoke(app, ["qa", "S"])
    assert result.exit_code == 0, result.output
    assert result.output == (
        "qa: Chapter 1: r0002 source_left: the original text is still readable\n"
        "qa: 1 issue(s) in 2 chapter(s)\n"
    )
    name, stages, _series, chapters, force, kwargs = series[0]
    assert (name, [type(s).__name__ for s in stages], chapters, force, kwargs) == (
        "qa",
        ["QaStage"],
        None,
        False,
        {"progress": True},
    )
    shown = json.loads(runner.invoke(app, ["qa", "S", "-c", "Chapter 1", "--json"]).output)
    assert shown == {
        "Chapter 1": [
            {
                "region_id": "r0002",
                "kind": "source_left",
                "message": "the original text is still readable",
                "read": "안녕",
            }
        ]
    }
