"""Process timeline: cheap wall-clock marks to find the idle gaps around the pipeline stages.

`mark(name)` records seconds since this module was first imported; with the `OMNISCAN_TIMELINE` env
var unset it is one dict lookup. `OMNISCAN_TIMELINE=1` (also true/yes/on) prints the table to stderr
at process exit, any other value writes it to that path. Marks arrive from every thread; the table
lists each mark's time since start, its gap to the previous mark and, at the end, the gaps >= 1 s.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from pathlib import Path

_START = time.perf_counter()
_lock = threading.Lock()
_marks: list[tuple[str, float, float]] = []  # (name, seconds since start, gap since the previous mark)
_ON = ("1", "true", "yes", "on")


def _destination() -> str | None:
    """Where the table goes: None when disabled, "stderr" for on/1/true/yes, else the value as a path."""
    raw = os.environ.get("OMNISCAN_TIMELINE")
    if raw is None or raw.strip() == "":
        return None
    stripped = raw.strip()
    return "stderr" if stripped.lower() in _ON else stripped


def mark(name: str) -> None:
    """Record a wall-clock mark (no-op unless OMNISCAN_TIMELINE is set)."""
    if _destination() is None:
        return
    now = time.perf_counter() - _START
    with _lock:
        gap = now - (_marks[-1][1] if _marks else 0.0)
        _marks.append((name, now, gap))


def clear() -> None:
    """Forget every mark recorded so far (tests)."""
    with _lock:
        _marks.clear()


def format_table(marks: list[tuple[str, float, float]]) -> str:
    """The timeline table for `marks` (the `(name, since start, gap)` tuples `mark` records)."""
    total = marks[-1][1] if marks else 0.0
    lines = [f"timeline: {len(marks)} marks, last at {total:.2f} s"]
    lines += [f"  t={t:7.2f}  gap={gap:7.2f}  {name}" for name, t, gap in marks]
    big = sorted(((name, gap) for name, _t, gap in marks if gap >= 1.0), key=lambda item: -item[1])
    if big:
        lines.append("  gaps >= 1 s: " + ", ".join(f"{name} {gap:.2f}s" for name, gap in big))
    return "\n".join(lines) + "\n"


def _print_at_exit() -> None:
    """Write the recorded marks to their destination (no-op when OMNISCAN_TIMELINE is unset)."""
    destination = _destination()
    if destination is None:
        return
    with _lock:
        snapshot = list(_marks)
    text = format_table(snapshot)
    if destination == "stderr":
        sys.stderr.write(text)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


atexit.register(_print_at_exit)
