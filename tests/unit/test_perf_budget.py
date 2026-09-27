"""Tests for tests/perf/budget.py — comparing stage timings with a device's baseline and recording them."""

from __future__ import annotations

import json
from pathlib import Path

from tests.perf.budget import load_baselines, record, report, slower_stages


def test_a_stage_is_slower_only_past_its_tolerance_and_slack() -> None:
    baseline = {"ocr": 10.0, "typeset": 0.1}
    assert slower_stages({"ocr": 13.2, "typeset": 0.3, "export": 99.0}, baseline) == []  # export: no baseline
    assert slower_stages({"ocr": 13.3, "typeset": 0.4}, baseline) == [
        "ocr: 13.30s > budget 13.25s (baseline 10.00s)",
        "typeset: 0.40s > budget 0.38s (baseline 0.10s)",
    ]


def test_record_keeps_other_devices_and_report_shows_both(tmp_path: Path) -> None:
    path = tmp_path / "baselines.json"
    assert load_baselines(path) == {}
    record("cuda:AMD Radeon RX 9070 XT", {"ocr": 1.23456}, path)
    record("cpu", {"ocr": 9.0}, path)
    assert json.loads(path.read_text()) == {"cpu": {"ocr": 9.0}, "cuda:AMD Radeon RX 9070 XT": {"ocr": 1.235}}
    text = report("cpu", {"ocr": 8.5, "export": 1.0}, load_baselines(path)["cpu"])
    assert "ocr               8.50s   (9.00s)" in text and "(no baseline)" in text
