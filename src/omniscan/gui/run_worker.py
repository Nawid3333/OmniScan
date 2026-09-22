"""RunWorker: the run controller on a QThread, so the GUI never blocks on `run_pipeline`.

Signals carry plain dataclasses (`gui.services.runs`) and are delivered on the receiving thread.
The worker owns the cancel event and the gate channel: `cancel()` stops the run after the current
stage (the runner's `after_stage` hook) and answers a pending step-mode gate with abort, and
`respond_preview` answers Continue/Abort while the worker sits in the gate. Created on the GUI
thread, `run()` executes on the worker thread — the controller builds and releases the chat
client/GPU scheduler there, as the CLI does.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal

from omniscan.gui.services.runs import GateChannel, RunController


class RunWorker(QThread):
    """Drives one `RunController` on its own thread; one run at a time."""

    stage_reported = Signal(object)  # StageUpdate: one finished stage outcome (queued to the GUI)
    preview_ready = Signal(object)  # StepPreview: the worker then blocks until the gate is answered
    run_finished = Signal(object)  # RunOutcome
    run_failed = Signal(str)  # "<ExceptionType>: <message>" (e.g. a config/profile error)

    def __init__(self, controller: RunController, parent: QObject | None = None) -> None:
        """Own the controller plus the handshake objects; `start()` runs it."""
        super().__init__(parent)
        self._controller = controller
        self.cancel_event = threading.Event()
        self.channel = GateChannel()

    def run(self) -> None:
        """Run the controller to completion; every failure becomes `run_failed` (never raises)."""
        try:
            outcome = self._controller.run(
                on_stage=self.stage_reported.emit,
                on_preview=self.preview_ready.emit,
                cancel_event=self.cancel_event,
                channel=self.channel,
            )
        except Exception as exc:  # surface instead of killing the thread
            self.run_failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.run_finished.emit(outcome)

    def cancel(self) -> None:
        """Stop after the current stage; answers a pending gate with abort so the worker unblocks."""
        self.cancel_event.set()
        self.channel.respond("abort")

    def respond_preview(self, answer: str) -> None:
        """Release the blocked step-mode gate with "continue" or "abort"."""
        if answer in ("continue", "abort"):
            self.channel.respond(answer)  # pyright: ignore[reportArgumentType]
