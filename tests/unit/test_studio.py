"""Translator Studio services: session edits and save, the corrections log, the QA pass, OCR re-run safety."""

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
    StudioEdits,
)
from omniscan.studio.corrections import CORRECTIONS_NAME, Correction, append_corrections, load_corrections
from omniscan.studio.edits import apply_source_edits, load_edits
from omniscan.studio.qa import check_chapter
from omniscan.studio.session import StudioSession


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


def test_session_rows_show_machine_lines(paths: ChapterPaths) -> None:
    rows = StudioSession(paths).rows()
    assert [(row.region_id, row.source, row.english, row.edited) for row in rows] == [
        ("r0001", "안녕", "Hello", False),
        ("r0002", "쾅", "BOOM", False),
        ("r0003", "광고", "", False),
    ]


def test_session_save_writes_ocr_studio_and_corrections(paths: ChapterPaths) -> None:
    session = StudioSession(paths)
    session.set_source("r0001", "안녕하세요")
    session.set_translation("r0001", "Good day")
    session.remove_region("r0003")
    assert session.dirty

    assert session.save() == 3
    assert not session.dirty
    ocr = RegionsArtifact.load(paths.artifact("ocr.json")).regions
    assert [(r.id, r.text) for r in ocr] == [("r0001", "안녕하세요"), ("r0002", "쾅")]
    edits = load_edits(paths)
    assert edits.translations == {"r0001": "Good day"}
    assert edits.sources == {"r0001": "안녕하세요"} and edits.removed == ["r0003"]
    log = load_corrections(paths.artifact(CORRECTIONS_NAME))
    assert [(c.region_id, c.field, c.before, c.after) for c in log] == [
        ("r0001", "source", "안녕", "안녕하세요"),
        ("r0001", "translation", "Hello", "Good day"),
        ("r0003", "removed", "광고", ""),
    ]
    assert log[1].source_text == "안녕하세요" and log[1].series == "S"


def test_session_reopen_keeps_edits_and_logs_only_new_changes(paths: ChapterPaths) -> None:
    first = StudioSession(paths)
    first.set_translation("r0002", "BAM")
    first.save()

    again = StudioSession(paths)
    assert again.rows()[1].english == "BAM" and again.rows()[1].edited
    assert again.save() == 0  # nothing changed since the last save
    again.set_translation("r0002", "BOOM")  # back to the machine's line: the manual edit is dropped
    assert again.save() == 1
    assert load_edits(paths).translations == {}
    assert len(load_corrections(paths.artifact(CORRECTIONS_NAME))) == 2


def test_session_unknown_region_raises(paths: ChapterPaths) -> None:
    with pytest.raises(KeyError):
        StudioSession(paths).set_translation("r9999", "x")


def test_session_without_artifacts_has_no_regions(tmp_path: Path) -> None:
    chapter = ChapterPaths("S", "C", tmp_path, tmp_path / "w", tmp_path / "o", tmp_path / "f")
    session = StudioSession(chapter)
    assert not session.has_regions and session.rows() == []


def test_apply_source_edits_survives_an_ocr_rerun() -> None:
    edits = StudioEdits(sources={"r0001": "fixed"}, removed=["r0003"])
    assert [(r.id, r.text) for r in apply_source_edits(REGIONS, edits)] == [
        ("r0001", "fixed"),
        ("r0002", "쾅"),
    ]


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


def test_corrections_log_appends_and_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    item = Correction(
        "S", "C", "r1", "translation", "a", "b", "src", "bubble_text", "2026-09-27T00:00:00+00:00"
    )
    assert append_corrections(path, [item]) == 1
    assert append_corrections(path, []) == 0
    path.write_text(path.read_text(encoding="utf-8") + "not json\n", encoding="utf-8")
    assert load_corrections(path) == [item]
    assert load_corrections(tmp_path / "missing.jsonl") == []
