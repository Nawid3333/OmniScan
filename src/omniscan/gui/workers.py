"""Task workers: run a service call on a thread-pool thread and report back via signals.

`run_task(fn)` starts `fn(progress)` on the global `QThreadPool` and returns the worker's
`WorkerSignals`; the caller must keep that reference until `finished`/`failed` arrives (queued
connections deliver it on the GUI thread). Any exception becomes `failed` — a worker never
crashes the thread.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

Progress = Callable[[int, int | None], None]
TaskFn = Callable[[Progress], Any]


class WorkerSignals(QObject):
    """Signals of one TaskWorker (queued connections deliver them on the receiving thread)."""

    progress = Signal(int, int)  # done, total (0 when the total is unknown)
    finished = Signal(object)  # the task's return value
    failed = Signal(str)  # "<ExceptionType>: <message>"


class TaskWorker(QRunnable):
    """Runs `fn(progress)` on a pool thread; emits `finished`/`failed`/`progress` via `signals`."""

    signals: WorkerSignals

    def __init__(self, fn: TaskFn) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._fn = fn

    def run(self) -> None:
        """Call `fn` with a progress callback; emit `finished` or `failed` (never raises)."""

        def progress(done: int, total: int | None) -> None:
            self.signals.progress.emit(int(done), int(total or 0))

        try:
            result = self._fn(progress)
        except Exception as exc:  # any failure must reach the GUI as `failed`, not kill the thread
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(result)


def run_task(fn: TaskFn, *, pool: QThreadPool | None = None) -> WorkerSignals:
    """Start `fn(progress)` on `pool` (default: the global one) and return its WorkerSignals.

    Keep the returned signals object referenced until `finished`/`failed` arrives: its
    connection to the keep-alive slot below (a closure over the worker) is what holds the
    queued worker in memory.
    """
    worker = TaskWorker(fn)
    signals = worker.signals

    def _keep_alive() -> None:
        """No-op slot; the connection holds this closure, and with it the worker."""

    signals.finished.connect(_keep_alive)
    signals.failed.connect(_keep_alive)
    (pool or QThreadPool.globalInstance()).start(worker)
    return signals
