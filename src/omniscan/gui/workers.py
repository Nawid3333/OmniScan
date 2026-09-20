"""Task workers: run a service call on a thread-pool thread and report back via signals.

`run_task(fn)` starts `fn(progress)` on the global `QThreadPool` and returns the worker's
`WorkerSignals`; the caller must keep that reference until `finished`/`failed` arrives (queued
connections deliver it on the GUI thread). Any exception becomes `failed` — a worker never
crashes the thread.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

Progress = Callable[[int, int | None], None]
TaskFn = Callable[[Progress], Any]

_alive: list["WorkerSignals"] = []  # anchors in-flight signals objects against the GC


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

    The task starts on the next event-loop pass (a zero-delay timer), so connections the
    caller makes right after `run_task` returns are always in place before `fn` runs — a fast
    task can never emit into the void. The module-level `_alive` registry additionally holds
    every in-flight signals object (signals -> connection -> closure -> worker -> signals is a
    cycle the GC could otherwise collect mid-run, dropping the queued signals); `_release`
    removes it when `finished`/`failed` arrives. Keeping the returned reference until then is
    still good practice.
    """
    worker = TaskWorker(fn)
    signals = worker.signals
    _alive.append(signals)
    target = pool or QThreadPool.globalInstance()

    def _release() -> None:
        """Drop the registry anchor; the task is done, its signals can be collected."""
        try:
            _alive.remove(signals)
        except ValueError:  # already released (finished and failed never both fire)
            pass

    signals.finished.connect(_release)
    signals.failed.connect(_release)
    QTimer.singleShot(0, lambda: target.start(worker))
    return signals
