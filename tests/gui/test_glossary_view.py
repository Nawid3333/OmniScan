"""GlossaryView tests (offscreen): adding terms, filtering, status/type/remove actions and in-place edits."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import GlossaryEntry
from omniscan.glossary.edit import add_term, remove_term
from omniscan.glossary.store import GlossaryStore
from omniscan.gui.glossary_view import ALL_STATUSES, COLUMNS, GlossaryView
from tests.fixtures.gui_library import SERIES, build_library

_SOURCE, _TARGET, _ALIASES, _NOTES = (
    COLUMNS.index(name) for name in ("Source", "English", "Aliases", "Notes")
)


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config over the synthetic fixture library."""
    return build_library(tmp_path / "lib")


def _paths(cfg: Config) -> SeriesPaths:
    return SeriesPaths.from_config(cfg, SERIES)


def _seed(cfg: Config) -> list[GlossaryEntry]:
    """Three machine-proposed terms, as a translate run's proposal pass leaves them."""
    paths = _paths(cfg)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    with GlossaryStore(paths.db) as store:
        return [
            store.add(GlossaryEntry(source="성진우", target="Sung Jinwoo", type="person", count=12)),
            store.add(
                GlossaryEntry(source="게이트", target="Gate", type="other", aliases=["게이트들"], count=7)
            ),
            store.add(GlossaryEntry(source="헌터 협회", target="Hunter Association", type="org", count=3)),
        ]


def _stored(cfg: Config) -> dict[str, GlossaryEntry]:
    with GlossaryStore(_paths(cfg).db) as store:
        return {entry.source: entry for entry in store.list()}


def _view(qapp: QApplication, cfg: Config) -> GlossaryView:
    view = GlossaryView(cfg)
    view.resize(1000, 500)
    view.show()
    qapp.processEvents()
    return view


def test_an_empty_glossary_is_read_without_creating_it(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    assert view.series_combo.currentText() == SERIES
    assert view.entries() == [] and view.count_label.text() == "no terms yet"
    assert not _paths(cfg).db.exists()  # looking at the page never writes a database
    assert view.add_button.isEnabled()
    assert not any(b.isEnabled() for b in (view.lock_button, view.reject_button, view.remove_button))


def test_adding_a_term_locks_it_as_the_users_and_writes_the_yaml(qapp: QApplication, cfg: Config) -> None:
    view = _view(qapp, cfg)
    view.source_edit.setText("  차해인 ")
    view.target_edit.setText("Cha  Hae-in")
    view.new_type_combo.setCurrentText("person")
    entry = view.add()
    assert entry is not None
    assert (entry.source, entry.target, entry.type, entry.status, entry.origin) == (
        "차해인",
        "Cha Hae-in",
        "person",
        "locked",
        "user",
    )
    assert [e.source for e in view.entries()] == ["차해인"] and view.selected_ids() == [entry.id]
    assert view.source_edit.text() == "" and view.target_edit.text() == ""
    assert "Cha Hae-in" in _paths(cfg).glossary_yaml.read_text(encoding="utf-8")

    view.source_edit.setText("차해인")
    view.target_edit.setText("Hae-in")
    assert view.add() is None and "already in the glossary" in view.status_label.text()
    view.source_edit.setText("새 용어")
    view.target_edit.setText("   ")
    assert view.add() is None and view.status_label.text() == "the target is empty"
    assert len(_stored(cfg)) == 1


def test_a_term_added_under_a_filter_that_would_hide_it_is_shown(qapp: QApplication, cfg: Config) -> None:
    _seed(cfg)
    view = _view(qapp, cfg)
    view.status_combo.setCurrentText("rejected")
    view.search_edit.setText("Sung")
    view.source_edit.setText("마나")
    view.target_edit.setText("mana")
    entry = view.add()
    assert entry is not None
    assert view.status_combo.currentText() == ALL_STATUSES and view.search_edit.text() == ""
    assert view.selected_ids() == [entry.id]


def test_status_filter_and_search(qapp: QApplication, cfg: Config) -> None:
    seeded = _seed(cfg)
    view = _view(qapp, cfg)
    assert [e.source for e in view.entries()] == ["성진우", "게이트", "헌터 협회"]
    assert view.count_label.text() == "3 of 3 term(s), 0 locked"
    view.search_edit.setText("hunter")  # case-insensitive, in the English
    assert [e.source for e in view.entries()] == ["헌터 협회"]
    view.search_edit.setText("게이트들")  # an alias finds its term
    assert [e.source for e in view.entries()] == ["게이트"]
    view.search_edit.clear()
    view.select_ids([seeded[0].id])  # type: ignore[list-item]
    assert view.set_status("locked") == 1
    view.status_combo.setCurrentText("locked")
    assert [e.source for e in view.entries()] == ["성진우"]
    assert view.count_label.text() == "1 of 3 term(s), 1 locked"
    view.status_combo.setCurrentText("proposed")
    assert [e.source for e in view.entries()] == ["게이트", "헌터 협회"]


def test_lock_reject_propose_type_and_remove_apply_to_every_selected_row(
    qapp: QApplication, cfg: Config
) -> None:
    seeded = _seed(cfg)
    ids = [entry.id for entry in seeded]
    view = _view(qapp, cfg)
    view.select_ids(ids[:2])  # type: ignore[arg-type]
    assert view.lock_button.isEnabled() and view.remove_button.isEnabled()
    assert view.set_status("locked") == 2
    stored = _stored(cfg)
    assert (stored["성진우"].status, stored["게이트"].status, stored["헌터 협회"].status) == (
        "locked",
        "locked",
        "proposed",
    )
    assert stored["성진우"].origin == "llm"  # a status change alone keeps the origin
    assert view.selected_ids() == ids[:2]  # the selection survives the refresh

    assert view.set_status("rejected") == 2 and view.set_status("proposed") == 2
    assert {e.status for e in _stored(cfg).values()} == {"proposed"}

    view.type_combo.setCurrentText("place")
    view._on_type_picked(view.type_combo.currentIndex())
    assert view.type_combo.currentIndex() == 0  # back to the placeholder
    assert (_stored(cfg)["성진우"].type, _stored(cfg)["게이트"].type) == ("place", "place")
    assert _stored(cfg)["헌터 협회"].type == "org"

    view.select_ids([ids[2]])  # type: ignore[list-item]
    assert view.remove_selected() == 1
    assert sorted(_stored(cfg)) == ["게이트", "성진우"]
    assert [e.source for e in view.entries()] == ["성진우", "게이트"]


def test_editing_cells_stores_the_words_and_refusals_put_the_old_ones_back(
    qapp: QApplication, cfg: Config
) -> None:
    _seed(cfg)
    view = _view(qapp, cfg)
    view.table.item(0, _TARGET).setText("Seong Jin-woo")  # type: ignore[union-attr]
    stored = _stored(cfg)["성진우"]
    assert (stored.target, stored.origin) == ("Seong Jin-woo", "user")  # new words make it the user's
    assert view.status_label.text() == "saved 성진우 → Seong Jin-woo"

    view.table.item(1, _ALIASES).setText("게이트들, 문 ,")  # type: ignore[union-attr]
    assert _stored(cfg)["게이트"].aliases == ["게이트들", "문"]
    view.table.item(1, _NOTES).setText("the dungeon portals")  # type: ignore[union-attr]
    assert _stored(cfg)["게이트"].notes == "the dungeon portals"

    view.table.item(2, _SOURCE).setText("게이트")  # type: ignore[union-attr]
    assert "already in the glossary" in view.status_label.text()
    assert view.table.item(2, _SOURCE).text() == "헌터 협회"  # type: ignore[union-attr]
    view.table.item(2, _TARGET).setText("  ")  # type: ignore[union-attr]
    assert view.status_label.text() == "the target is empty"
    assert _stored(cfg)["헌터 협회"].target == "Hunter Association"


def test_a_term_removed_elsewhere_is_reported_and_the_table_refreshed(
    qapp: QApplication, cfg: Config
) -> None:
    seeded = _seed(cfg)
    view = _view(qapp, cfg)
    remove_term(_paths(cfg), seeded[0].id)  # type: ignore[arg-type]  # the CLI or the web Studio, meanwhile
    view.select_ids([seeded[0].id])  # type: ignore[list-item]
    assert view.set_status("locked") == 0
    assert "has no entry" in view.status_label.text()
    assert [e.source for e in view.entries()] == ["게이트", "헌터 협회"]


def test_the_main_window_page_shows_terms_added_meanwhile(
    qapp: QApplication, cfg: Config, tmp_path: Path
) -> None:
    from omniscan.gui.main_window import PAGES
    from tests.gui.test_main_window import _window  # every worker-backed service faked

    window = _window(cfg, QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    assert window.glossary_view.entries() == []
    add_term(_paths(cfg), "마나", "mana")  # e.g. `omniscan glossary add` while the app is open
    window.show_page(PAGES.index("Glossary"))
    assert window.stack.currentWidget() is window.glossary_view
    assert [e.source for e in window.glossary_view.entries()] == ["마나"]
