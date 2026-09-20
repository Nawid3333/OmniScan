"""G3 timeline: marks record wall-clock offsets, the table formats them, exit writes the destination."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from omniscan.gpu import timeline


@pytest.fixture(autouse=True)
def _clean_timeline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No destination env var and an empty mark list before and after every test."""
    monkeypatch.delenv("OMNISCAN_TIMELINE", raising=False)
    timeline.clear()
    yield
    timeline.clear()


def test_mark_is_a_noop_without_the_env_var() -> None:
    timeline.mark("never recorded")
    assert timeline._marks == []


def test_marks_record_monotonic_times_and_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNISCAN_TIMELINE", "1")
    timeline.mark("first")
    timeline.mark("second")
    timeline.mark("third")
    times = [t for _name, t, _gap in timeline._marks]
    gaps = [gap for _name, _t, gap in timeline._marks]
    assert times == sorted(times)
    assert gaps[0] == times[0]  # the first mark's gap starts from the module import
    assert gaps[1] == pytest.approx(times[1] - times[0], abs=1e-6)
    assert [name for name, _t, _gap in timeline._marks] == ["first", "second", "third"]


def test_mark_records_from_every_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNISCAN_TIMELINE", "1")

    def worker() -> None:
        timeline.mark("from-thread")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [name for name, _t, _gap in timeline._marks].count("from-thread") == 4


def test_format_table_lists_marks_and_the_gaps_above_one_second() -> None:
    text = timeline.format_table(
        [("imported", 0.5, 0.5), ("config", 0.9, 0.4), ("acquire vision end", 3.4, 2.5)]
    )
    assert "t=   0.50  gap=   0.50  imported" in text
    assert "t=   3.40  gap=   2.50  acquire vision end" in text
    assert "gaps >= 1 s: acquire vision end 2.50s" in text
    assert "last at 3.40 s" in text


def test_format_table_without_marks_or_small_gaps() -> None:
    assert timeline.format_table([]).startswith("timeline: 0 marks")
    small = timeline.format_table([("a", 0.4, 0.4), ("b", 0.9, 0.5)])
    assert "gaps >= 1 s" not in small


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, None, id="unset"),
        pytest.param("", None, id="empty"),
        pytest.param("1", "stderr", id="one"),
        pytest.param("YES", "stderr", id="case-insensitive-on"),
        pytest.param(r"V:\tmp\timeline.txt", r"V:\tmp\timeline.txt", id="path"),
    ],
)
def test_destination(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str | None) -> None:
    if value is None:
        monkeypatch.delenv("OMNISCAN_TIMELINE", raising=False)
    else:
        monkeypatch.setenv("OMNISCAN_TIMELINE", value)
    assert timeline._destination() == expected


def test_print_at_exit_writes_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    destination = tmp_path / "timeline.txt"
    monkeypatch.setenv("OMNISCAN_TIMELINE", str(destination))
    timeline.mark("hello")
    timeline._print_at_exit()
    assert "hello" in destination.read_text(encoding="utf-8")
