"""Translator Studio QA pass (the session's save/undo behaviour is covered by test_edits_session.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    LayoutArtifact,
    LayoutItem,
    Region,
    RegionsArtifact,
)
from omniscan.studio.qa import check_chapter


def _region(region_id: str, text: str, kind: str = "bubble_text", y: int = 0) -> Region:
    return Region(id=region_id, slice_index=0, kind=kind, bbox=BBox(x0=0, y0=y, x1=20, y1=y + 10), text=text)  # type: ignore[arg-type]


REGIONS = [
    _region("r0001", "안녕", y=0),
    _region("r0002", "쾅", kind="sfx", y=20),
    _region("r0003", "광고", y=40),
]


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    """A chapter work dir with ocr.json, final.json and layout.json."""
    chapter = ChapterPaths("S", "Ch 1", tmp_path / "raw", tmp_path / "work", tmp_path / "out", tmp_path / "f")
    RegionsArtifact(regions=REGIONS).save(chapter.artifact("ocr.json"))
    FinalArtifact(
        judge_model="fake",
        lines=[
            FinalLine(region_id="r0001", text="Hello", decision="pick", flags=["uncertain"]),
            FinalLine(region_id="r0002", text="BOOM", decision="pick"),
        ],
    ).save(chapter.artifact("final.json"))
    LayoutArtifact(
        items=[
            LayoutItem(
                region_id="r0002",
                font_role="sfx",
                font="f.ttf",
                size_px=10,
                lines=["BOOM"],
                box=BBox(x0=0, y0=20, x1=20, y1=30),
                overflow=True,
            )
        ]
    ).save(chapter.artifact("layout.json"))
    return chapter


def test_qa_finds_each_kind_of_issue(paths: ChapterPaths) -> None:
    regions = [*REGIONS, _region("r0004", "짧다", y=60), _region("r0005", "워터", kind="watermark", y=80)]
    translations = {"r0001": "Hello", "r0002": "BOOM", "r0004": "Hi 안녕 " + "and so on " * 8}
    final = FinalArtifact.load(paths.artifact("final.json")).lines
    layout = LayoutArtifact.load(paths.artifact("layout.json")).items

    kinds = [(issue.region_id, issue.kind) for issue in check_chapter(regions, translations, final, layout)]

    assert kinds == [
        ("r0001", "uncertain"),
        ("r0002", "overflow"),
        ("r0003", "untranslated"),
        ("r0004", "source_left"),
        ("r0004", "too_long"),
    ]


def test_qa_keeps_untranslated_sfx_quiet() -> None:
    assert check_chapter([_region("r1", "쾅", kind="sfx")], {}) == []
