"""Tests for omniscan.story.store (tmp_path sqlite files)."""

from __future__ import annotations

from pathlib import Path

from omniscan.story.store import SummaryStore


def test_set_inserts_then_get_and_list_reflect_it(tmp_path: Path) -> None:
    with SummaryStore(tmp_path / "series.db") as store:
        stored = store.set("Chapter 001", "Minjun enters the gate.", "m")
        assert (stored.chapter, stored.summary, stored.model) == (
            "Chapter 001",
            "Minjun enters the gate.",
            "m",
        )
        assert stored.updated_at  # generated inside set, not passed in
        fetched = store.get("Chapter 001")
        assert fetched is not None
        assert (fetched.chapter, fetched.summary, fetched.model) == (
            "Chapter 001",
            "Minjun enters the gate.",
            "m",
        )
        assert fetched.updated_at
        assert store.list() == [fetched]


def test_second_set_updates_in_place(tmp_path: Path) -> None:
    with SummaryStore(tmp_path / "series.db") as store:
        store.set("Chapter 001", "First version.", "m1")
        updated = store.set("Chapter 001", "Second version.", "m2")
        fetched = store.get("Chapter 001")
        assert fetched is not None
        assert (fetched.summary, fetched.model) == ("Second version.", "m2")
        assert fetched.updated_at >= updated.updated_at
        assert len(store.list()) == 1  # no duplicate row


def test_get_unknown_chapter_returns_none(tmp_path: Path) -> None:
    with SummaryStore(tmp_path / "series.db") as store:
        store.set("Chapter 001", "Something.", "m")
        assert store.get("Chapter 002") is None


def test_delete_returns_true_iff_a_row_was_removed(tmp_path: Path) -> None:
    with SummaryStore(tmp_path / "series.db") as store:
        store.set("Chapter 001", "Something.", "m")
        assert store.delete("Chapter 001") is True
        assert store.delete("Chapter 001") is False
        assert store.delete("Chapter 002") is False
        assert store.list() == []


def test_table_survives_closing_and_reopening(tmp_path: Path) -> None:
    with SummaryStore(tmp_path / "series.db") as store:
        store.set("Chapter 001", "Something happened.", "m")
    with SummaryStore(tmp_path / "series.db") as reopened:
        fetched = reopened.get("Chapter 001")
        assert fetched is not None and fetched.summary == "Something happened."


def test_list_orders_by_chapter_name_ascending(tmp_path: Path) -> None:
    with SummaryStore(tmp_path / "series.db") as store:
        store.set("Chapter 010", "Ten.", "m")
        store.set("Chapter 002", "Two.", "m")
        store.set("Chapter 001", "One.", "m")
        assert [row.chapter for row in store.list()] == ["Chapter 001", "Chapter 002", "Chapter 010"]
