"""Tests for omniscan.watermark.store (tmp_path watermarks.json files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.watermark.store import WatermarkStore


def test_list_on_fresh_dir_returns_empty_and_creates_nothing(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "series")
    assert store.list() == []
    assert not (tmp_path / "series").exists()


def test_add_creates_dir_file_assigns_index0_and_persists(tmp_path: Path) -> None:
    work_dir = tmp_path / "series"
    store = WatermarkStore(work_dir)
    region = store.add(0.0, 0.0, 1.0, 0.05, note="bottom-right logo")
    assert region.index == 0
    assert (work_dir / "watermarks.json").is_file()
    assert WatermarkStore(work_dir).list() == [region]


def test_add_three_regions_assigns_sequential_indices(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "series")
    for i in range(3):
        store.add(0.0, i / 4, 1.0, (i + 1) / 4)
    assert [region.index for region in store.list()] == [0, 1, 2]


def test_remove_keeps_indices_and_next_add_skips_removed(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "series")
    for _ in range(3):
        store.add(0.0, 0.0, 1.0, 0.1)
    store.remove(1)
    indices = [region.index for region in store.list()]
    assert indices == [0, 2]
    assert store.add(0.0, 0.0, 1.0, 0.1).index == 3


def test_remove_unknown_index_raises_key_error(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "series")
    store.add(0.0, 0.0, 1.0, 0.1)
    with pytest.raises(KeyError):
        store.remove(7)


def test_add_equal_fractions_raises_value_error(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "series")
    with pytest.raises(ValueError, match="x0_frac"):
        store.add(0.5, 0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="y0_frac"):
        store.add(0.0, 0.9, 1.0, 0.8)


def test_add_fraction_outside_unit_range_raises_value_error(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "series")
    with pytest.raises(ValueError, match="x0_frac"):
        store.add(-0.1, 0.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="x1_frac"):
        store.add(0.0, 0.0, 1.5, 0.1)


def test_note_round_trips_including_none(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "series")
    store.add(0.0, 0.0, 1.0, 0.1, note="group logo")
    store.add(0.0, 0.0, 1.0, 0.1)
    regions = WatermarkStore(tmp_path / "series").list()
    assert regions[0].note == "group logo"
    assert regions[1].note is None
