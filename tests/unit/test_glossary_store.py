"""Tests for omniscan.glossary.store (tmp_path sqlite files)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from omniscan.core.schemas import GlossaryEntry
from omniscan.glossary.store import GlossaryStore


def entry(**overrides: object) -> GlossaryEntry:
    """A GlossaryEntry with the given field overrides."""
    fields: dict[str, object] = {"source": "지훈", "target": "Jihoon"}
    fields.update(overrides)
    return GlossaryEntry(**fields)  # type: ignore[arg-type]


def test_add_returns_copy_with_id_and_leaves_original_untouched(tmp_path: Path) -> None:
    original = entry(aliases=["훈이형"])
    with GlossaryStore(tmp_path / "series.db") as store:
        added = store.add(original)
        assert isinstance(added.id, int) and added.id > 0
        assert original.id is None
        assert added.source == original.source and added.aliases == original.aliases


def test_add_with_id_set_raises(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store, pytest.raises(ValueError, match="id"):
        store.add(entry(id=1))


def test_get_round_trips_all_fields(tmp_path: Path) -> None:
    original = entry(
        aliases=["훈이형"],
        pronouns="he/him",
        gender="male",
        type="person",
        notes="protagonist",
        status="locked",
        origin="user",
        first_seen_chapter=1.5,
        count=3,
    )
    with GlossaryStore(tmp_path / "series.db") as store:
        added = store.add(original)
        assert store.get(added.id) == added  # type: ignore[arg-type]


def test_get_missing_id_returns_none(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store:
        store.add(entry())
        assert store.get(999) is None


def test_update_changes_fields(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store:
        added = store.add(entry())
        store.update(added.model_copy(update={"status": "locked", "target": "Ji-hoon"}))
        fetched = store.get(added.id)  # type: ignore[arg-type]
        assert fetched is not None
        assert fetched.status == "locked" and fetched.target == "Ji-hoon"


def test_update_requires_id(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store, pytest.raises(ValueError, match="id"):
        store.update(entry())


def test_update_unknown_id_raises_key_error(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store, pytest.raises(KeyError):
        store.update(entry(id=42))


def test_delete_removes_row(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store:
        added = store.add(entry())
        store.delete(added.id)  # type: ignore[arg-type]
        assert store.get(added.id) is None  # type: ignore[arg-type]


def test_delete_unknown_id_raises_key_error(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store, pytest.raises(KeyError):
        store.delete(999)


def test_list_ordered_and_status_filtered(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store:
        first = store.add(entry(source="지훈", target="Jihoon", status="locked"))
        second = store.add(entry(source="학교", target="school"))
        third = store.add(entry(source="하늘", target="sky", status="rejected"))
        assert [e.id for e in store.list()] == [first.id, second.id, third.id]
        assert [e.id for e in store.list(status="locked")] == [first.id]
        assert [e.id for e in store.list(status="proposed")] == [second.id]


def test_find_by_source_exact_only(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store:
        added = store.add(entry(source="지훈", target="Jihoon"))
        assert store.find_by_source("지훈") == added
        assert store.find_by_source("지훈이") is None
        assert store.find_by_source("지") is None


def test_same_source_different_type_allowed(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store:
        store.add(entry(source="하늘", target="Sky", type="person"))
        store.add(entry(source="하늘", target="sky", type="other"))
        assert len(store.list()) == 2


def test_close_twice_and_context_manager(tmp_path: Path) -> None:
    store = GlossaryStore(tmp_path / "series.db")
    store.close()
    store.close()
    with GlossaryStore(tmp_path / "series2.db") as ctx_store:
        ctx_store.add(entry())
        assert ctx_store.list()
    with pytest.raises(sqlite3.ProgrammingError):
        ctx_store.list()
