"""Series dialogs and their Studio buttons (offscreen): find and replace across chapters, the consistency report."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, FinalArtifact, FinalLine, QaArtifact, QaIssue, Region, RegionsArtifact
from omniscan.edits.store import history_steps
from omniscan.gui.series_dialogs import ConsistencyDialog, ReplaceDialog
from omniscan.gui.services.series_check import Consistency, consistency
from omniscan.gui.studio_view import StudioView
from omniscan.qa.consistency import Divergence, Rendering, TermMiss
from omniscan.qa.leftover import QA_FILE
from omniscan.translate.on_demand import english_lines
from tests.fixtures.gui_library import CHAPTERS, SERIES, build_library


def _lines(cfg: Config, chapter: str, english: str) -> None:
    """Two regions in `chapter`; r0001 has the English line `english`."""
    paths = SeriesPaths.from_config(cfg, SERIES).chapter(chapter)
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=5, y0=10, x1=35, y1=40),
                text="헌터",
            ),
            Region(
                id="r0002",
                slice_index=2,
                kind="bubble_text",
                bbox=BBox(x0=5, y0=220, x1=35, y1=260),
                text="뭐",
            ),
        ]
    ).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="fake", lines=[FinalLine(region_id="r0001", text=english, decision="pick")]
    ).save(paths.artifact("final.json"))


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """The fixture library: "Hunter" in chapter 1, "hunter guild" in chapter 2, no lines in chapter 4."""
    cfg = build_library(tmp_path / "lib")
    _lines(cfg, CHAPTERS[0], "The Hunter came.")
    _lines(cfg, CHAPTERS[1], "A hunter guild.")
    RegionsArtifact(regions=[]).save(
        SeriesPaths.from_config(cfg, SERIES).chapter(CHAPTERS[3]).artifact("ocr.json")
    )
    return cfg


def _english(cfg: Config, chapter: str) -> dict[str, str]:
    return english_lines(SeriesPaths.from_config(cfg, SERIES).chapter(chapter))


def test_replace_previews_every_line_and_writes_only_the_ticked_ones(qapp: QApplication, cfg: Config) -> None:
    dialog = ReplaceDialog(cfg, SERIES, CHAPTERS[0])
    dialog.find_edit.setText("hunter")
    dialog.replace_edit.setText("slayer")
    dialog.case_box.setChecked(False)  # any case; the replacement takes each match's case
    assert dialog.preview() == 1  # this chapter only
    dialog.scope_combo.setCurrentText("Whole series")
    assert dialog.preview() == 2 and dialog.status_label.text() == "2 line(s) in 2 chapter(s) would change"
    cells = [[dialog.table.item(row, c).text() for c in range(1, 5)] for row in range(2)]  # type: ignore[union-attr]
    assert cells == [
        [CHAPTERS[0], "r0001", "The Hunter came.", "The Slayer came."],
        [CHAPTERS[1], "r0001", "A hunter guild.", "A slayer guild."],
    ]
    dialog.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)  # type: ignore[union-attr]
    assert dialog.replace_ticked() == 1 and dialog.applied == 1
    assert _english(cfg, CHAPTERS[0])["r0001"] == "The Slayer came."
    assert _english(cfg, CHAPTERS[1])["r0001"] == "A hunter guild."  # unticked: left alone
    paths = SeriesPaths.from_config(cfg, SERIES).chapter(CHAPTERS[0])
    assert history_steps(paths) == (1, 0)  # one undo step
    assert dialog.table.rowCount() == 0 and not dialog.replace_button.isEnabled()


def test_a_bad_rule_is_shown_and_nothing_is_written(qapp: QApplication, cfg: Config) -> None:
    dialog = ReplaceDialog(cfg, SERIES, CHAPTERS[0])
    dialog.find_edit.setText("(unclosed")
    dialog.regex_box.setChecked(True)
    assert dialog.preview() == 0 and "bad regular expression" in dialog.status_label.text()
    dialog.regex_box.setChecked(False)
    dialog.find_edit.setText("dragon")
    assert dialog.preview() == 0 and dialog.status_label.text() == "no line would change"
    assert dialog.replace_ticked() == 0


def test_the_consistency_report_lists_both_kinds_and_opens_a_line(qapp: QApplication) -> None:
    report = Consistency(
        divergences=[
            Divergence(
                source="헌터",
                renderings=[
                    Rendering("Hunter", [("Episode 01", "r0001"), ("Episode 03", "r0004")]),
                    Rendering("Slayer", [("Episode 02", "r0002")]),
                ],
            )
        ],
        misses=[TermMiss("Episode 02", "r0007", "게이트", "Gate", "The portal opened.")],
    )
    dialog = ConsistencyDialog(SERIES, report)
    assert (
        dialog.tabs.tabText(0) == "Translated differently (1)"
        and dialog.tabs.tabText(1) == "Missed locked terms (1)"
    )
    assert [dialog.divergence_table.item(1, c).text() for c in range(4)] == [  # type: ignore[union-attr]
        "헌터",
        "Slayer",
        "1",
        "Episode 02 r0002",
    ]
    assert dialog.divergence_table.item(0, 3).text() == "Episode 01 r0001, Episode 03 r0004"  # type: ignore[union-attr]
    opened: list[tuple[str, str]] = []
    dialog.open_line.connect(lambda chapter, region: opened.append((chapter, region)))
    dialog.divergence_table.cellDoubleClicked.emit(0, 1)
    dialog.miss_table.cellDoubleClicked.emit(0, 4)
    assert opened == [("Episode 01", "r0001"), ("Episode 02", "r0007")]


def test_the_real_report_finds_one_source_translated_two_ways(qapp: QApplication, cfg: Config) -> None:
    found = consistency(cfg, SERIES)
    assert [(d.source, [r.english for r in d.renderings]) for d in found.divergences] == [
        ("헌터", ["The Hunter came.", "A hunter guild."])
    ]


def _wait(qapp: QApplication, done: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done()


def test_studio_saves_first_reloads_after_a_replace_and_opens_report_lines(
    qapp: QApplication, cfg: Config
) -> None:
    calls: list[tuple[str, str, bool]] = []

    def fake_replace(cfg: Config, series: str, chapter: str, parent: object) -> int:
        paths = SeriesPaths.from_config(cfg, series).chapter(chapter)
        calls.append((series, chapter, history_steps(paths)[0] == 1))  # the pending edit was saved first
        dialog = ReplaceDialog(cfg, series, chapter)
        dialog.find_edit.setText("Hunter")
        dialog.replace_edit.setText("Slayer")
        dialog.preview()
        return dialog.replace_ticked()

    view = StudioView(cfg, replace_fn=fake_replace)
    view.resize(1200, 600)
    view.show()
    assert view.open_chapter(SERIES, CHAPTERS[0])
    view.table.item(1, 4).setText("What?")  # type: ignore[union-attr]  # an unsaved edit
    assert view.find_replace() == 1
    assert calls == [(SERIES, CHAPTERS[0], True)]
    assert view.table.item(0, 4).text() == "The Slayer came."  # type: ignore[union-attr]  # reloaded
    assert view.status_label.text() == "find and replace changed 1 line(s)"

    assert view.check_consistency()
    _wait(qapp, lambda: view.consistency_dialog is not None)
    dialog = view.consistency_dialog
    assert dialog is not None and dialog.divergence_table.rowCount() == 2
    dialog.divergence_table.cellDoubleClicked.emit(1, 0)  # "A hunter guild." in chapter 2
    session = view.session()
    assert session is not None and session.paths.chapter == CHAPTERS[1]
    assert view.selected_ids() == ["r0001"]


def test_read_finished_pages_flags_the_lines_whose_source_text_still_shows(
    qapp: QApplication, cfg: Config
) -> None:
    def fake_qa(cfg: Config, paths: ChapterPaths) -> list[QaIssue]:
        issues = [
            QaIssue(region_id="r0001", kind="source_left", message="source text still readable", read="헌터")
        ]
        QaArtifact(checked=2, issues=issues).save(paths.artifact(QA_FILE))
        return issues

    view = StudioView(cfg, qa_fn=fake_qa)
    view.resize(1200, 600)
    view.show()
    assert view.open_chapter(SERIES, CHAPTERS[0])
    assert view.qa_button.isEnabled() and "finished page" not in view.table.item(0, 6).text()  # type: ignore[union-attr]
    assert view.read_finished()
    _wait(qapp, lambda: not view.is_busy())
    assert view.table.item(0, 6).text() == "finished page: source text still readable"  # type: ignore[union-attr]
    assert view.status_label.text() == "finished pages: 1 line(s) still show source text or a watermark"
    reopened = StudioView(cfg)  # a later session shows what the last re-read found, from qa.json
    assert reopened.open_chapter(SERIES, CHAPTERS[0])
    assert "finished page: source text still readable" in reopened.table.item(0, 6).text()  # type: ignore[union-attr]
