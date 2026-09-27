"""Tests for omniscan.pipeline.eta: per-page stage rates, time left, history file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniscan.pipeline.eta import EtaEstimator, format_duration, load_history, save_history


def test_no_estimate_until_every_pending_stage_has_a_rate() -> None:
    eta = EtaEstimator([("c1", "ingest"), ("c1", "ocr"), ("c2", "ingest"), ("c2", "ocr")])
    assert eta.remaining_seconds() is None
    eta.record("c1", "ingest", "done", 10.0)
    assert eta.remaining_seconds() is None  # ocr never measured yet
    eta.record("c1", "ocr", "done", 30.0)
    assert eta.remaining_seconds() == pytest.approx(40.0)


def test_rates_are_per_page() -> None:
    eta = EtaEstimator([("small", "ocr"), ("big", "ocr")], pages={"small": 10, "big": 40})
    eta.record("small", "ocr", "done", 5.0)  # 0.5 s per page
    assert eta.remaining_seconds() == pytest.approx(20.0)


def test_history_gives_an_estimate_before_anything_ran() -> None:
    eta = EtaEstimator(
        [("c1", "ingest"), ("c1", "ocr")], pages={"c1": 20}, history={"ingest": 0.1, "ocr": 1.0}
    )
    assert eta.remaining_seconds() == pytest.approx(22.0)
    eta.record("c1", "ingest", "done", 4.0)  # this run's measurement beats the history
    assert eta.rate("ingest") == pytest.approx(0.2)


def test_skipped_costs_nothing_and_failed_drops_the_chapters_rest() -> None:
    eta = EtaEstimator([("c1", "a"), ("c1", "b"), ("c2", "a"), ("c2", "b")], history={"a": 1.0, "b": 1.0})
    eta.record("c1", "a", "skipped", 0.0)
    eta.record("c2", "a", "failed", 3.0)
    assert eta.remaining_seconds() == pytest.approx(1.0)  # only c1/b is left


def test_fraction_done_is_time_weighted() -> None:
    eta = EtaEstimator([("c1", "fast"), ("c1", "slow")], history={"fast": 1.0, "slow": 9.0})
    assert eta.fraction_done() == 0.0
    eta.record("c1", "fast", "done", 1.0)
    assert eta.fraction_done() == pytest.approx(0.1)
    eta.record("c1", "slow", "done", 9.0)
    assert eta.fraction_done() == 1.0


def test_fraction_done_counts_steps_without_rates() -> None:
    eta = EtaEstimator([("c1", "a"), ("c1", "b")])
    eta.record("c1", "a", "done", 2.0)
    assert eta.fraction_done() == 0.5


def test_format_duration() -> None:
    assert format_duration(44.6) == "45 s"
    assert format_duration(200) == "3 min 20 s"
    assert format_duration(3900) == "1 h 05 min"


def test_history_round_trip_and_moving_average(tmp_path: Path) -> None:
    path = tmp_path / "work" / "eta_history.json"
    assert load_history(path, "auto/full") == {}
    save_history(path, "auto/full", {"ocr": 1.0})
    assert load_history(path, "auto/full") == {"ocr": 1.0}
    save_history(path, "auto/full", {"ocr": 2.0})
    assert load_history(path, "auto/full")["ocr"] == pytest.approx(1.3)
    assert load_history(path, "cpu/background") == {}


def test_a_corrupt_history_file_is_ignored_and_replaced(tmp_path: Path) -> None:
    path = tmp_path / "eta_history.json"
    path.write_text("not json", encoding="utf-8")
    assert load_history(path, "auto/full") == {}
    save_history(path, "auto/full", {"ingest": 0.5})
    assert json.loads(path.read_text(encoding="utf-8")) == {"auto/full": {"ingest": 0.5}}
