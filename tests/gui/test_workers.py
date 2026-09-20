"""run_task/TaskWorker tests (offscreen): finished/progress/failed delivery and concurrency."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from omniscan.gui.workers import run_task

WAIT_S = 5.0


def _pump(qapp: QApplication, predicate: Any, timeout_s: float = WAIT_S) -> None:
    """Process events until `predicate()` holds; worker signals arrive as queued events."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("timed out waiting for the worker's signals")
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def test_finished_delivers_the_return_value(qapp: QApplication) -> None:
    release = threading.Event()

    def fn(progress: Any) -> int:
        release.wait(WAIT_S)
        return 7

    signals = run_task(fn)  # the worker is blocked inside fn: connect before it can emit
    finished: list[Any] = []
    failed: list[str] = []
    signals.finished.connect(finished.append)
    signals.failed.connect(failed.append)
    release.set()
    _pump(qapp, lambda: bool(finished) or bool(failed))
    assert finished == [7]
    assert failed == []


def test_progress_arrives_in_order_with_zero_total_for_none(qapp: QApplication) -> None:
    release = threading.Event()

    def fn(progress: Any) -> str:
        release.wait(WAIT_S)
        progress(1, None)
        progress(2, 100)
        progress(3, 0)
        return "done"

    signals = run_task(fn)
    seen: list[tuple[int, int]] = []
    signals.progress.connect(lambda done, total: seen.append((done, total)))
    finished: list[Any] = []
    signals.finished.connect(finished.append)
    release.set()
    _pump(qapp, lambda: bool(finished))
    assert seen == [(1, 0), (2, 100), (3, 0)]
    assert finished == ["done"]


def test_exception_becomes_failed_without_crashing(qapp: QApplication) -> None:
    release = threading.Event()

    def fn(progress: Any) -> None:
        release.wait(WAIT_S)
        raise ValueError("boom")

    signals = run_task(fn)
    failed: list[str] = []
    finished: list[Any] = []
    signals.failed.connect(failed.append)
    signals.finished.connect(finished.append)
    release.set()
    _pump(qapp, lambda: bool(failed))
    assert failed == ["ValueError: boom"]
    assert finished == []  # finished is never emitted for a failed task


def test_two_tasks_run_concurrently_and_both_finish(qapp: QApplication) -> None:
    pool = QThreadPool()
    pool.setMaxThreadCount(2)
    gates = {tag: threading.Event() for tag in (0, 1)}
    threads: dict[int, int] = {}

    def make(tag: int) -> Any:
        def fn(progress: Any) -> str:
            threads[tag] = threading.get_ident()
            gates[tag].wait(WAIT_S)
            return f"done-{tag}"

        return fn

    signals = [run_task(make(tag), pool=pool) for tag in (0, 1)]
    finished: list[str] = []
    for item in signals:
        item.finished.connect(finished.append)
    gates[0].set()
    gates[1].set()
    _pump(qapp, lambda: len(finished) == 2)
    assert sorted(finished) == ["done-0", "done-1"]
    assert len(set(threads.values())) == 2  # both ran at the same time, on different threads


def test_explicit_pool_is_used(qapp: QApplication) -> None:
    pool = QThreadPool()
    pool.setMaxThreadCount(1)
    release = threading.Event()

    def fn(progress: Any) -> str:
        release.wait(WAIT_S)
        return "ran"

    signals = run_task(fn, pool=pool)
    finished: list[Any] = []
    signals.finished.connect(finished.append)
    release.set()
    _pump(qapp, lambda: bool(finished))
    assert finished == ["ran"]
