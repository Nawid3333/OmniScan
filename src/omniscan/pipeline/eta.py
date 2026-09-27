"""Time-remaining estimates for a pipeline run, learned from the stages that already finished.

Every stage's cost is measured in seconds per page: a stage that took 30 s on a 60-page chapter predicts 15 s for a
30-page one. Rates come from this run first (they reflect the machine's current load and usage level) and fall back
to the rates remembered from earlier runs on the same device, so even the first chapter gets an estimate once
OmniScan has run on this machine before. A stage with neither has no estimate yet (`remaining_seconds` is None).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

log = logging.getLogger(__name__)

HISTORY_NAME = "eta_history.json"
_HISTORY_WEIGHT = 0.3  # how much one finished run moves the remembered rate (exponential moving average)


class EtaEstimator:
    """Tracks the (chapter, stage) pairs of one run and predicts how long the rest will take."""

    def __init__(
        self,
        work: Iterable[tuple[str, str]],
        pages: Mapping[str, int] | None = None,
        history: Mapping[str, float] | None = None,
    ) -> None:
        """`work`: every (chapter, stage) the run will do; `pages`: chapter sizes; `history`: seconds per page."""
        self._pending = set(work)
        self._total = len(self._pending)
        self._pages = dict(pages or {})
        self._history = dict(history or {})
        self._seconds: dict[str, float] = {}
        self._units: dict[str, float] = {}

    def _size(self, chapter: str) -> float:
        return float(max(1, self._pages.get(chapter, 1)))

    def record(self, chapter: str, stage: str, status: str, seconds: float) -> None:
        """One finished (chapter, stage); a failed stage removes the chapter's later work from the estimate."""
        self._pending.discard((chapter, stage))
        if status == "done":
            self._seconds[stage] = self._seconds.get(stage, 0.0) + seconds
            self._units[stage] = self._units.get(stage, 0.0) + self._size(chapter)
        elif status == "failed":  # the runner skips a failed chapter's later passes
            self._pending = {item for item in self._pending if item[0] != chapter}

    def rate(self, stage: str) -> float | None:
        """Seconds per page for `stage`: this run's measurement, else the remembered one, else None."""
        units = self._units.get(stage, 0.0)
        if units > 0:
            return self._seconds[stage] / units
        return self._history.get(stage)

    def remaining_seconds(self) -> float | None:
        """Predicted seconds until the run ends; None while some pending stage has never been measured."""
        total = 0.0
        for chapter, stage in self._pending:
            rate = self.rate(stage)
            if rate is None:
                return None
            total += rate * self._size(chapter)
        return total

    def fraction_done(self) -> float:
        """How much of the run is finished, 0..1, weighted by predicted time when every stage has a rate."""
        if self._total == 0:
            return 1.0
        remaining = self.remaining_seconds()
        spent = sum(self._seconds.values())
        if remaining is not None and spent + remaining > 0:
            return spent / (spent + remaining)
        return (self._total - len(self._pending)) / self._total

    def learned_rates(self) -> dict[str, float]:
        """The seconds-per-page rates this run measured (to fold into the history)."""
        return {stage: self._seconds[stage] / units for stage, units in self._units.items() if units > 0}


def format_duration(seconds: float) -> str:
    """A short human duration: "45 s", "3 min 20 s", "1 h 05 min"."""
    whole = round(seconds)
    if whole < 60:
        return f"{whole} s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def load_history(path: Path, device: str) -> dict[str, float]:
    """The remembered seconds-per-page rates for `device`; {} when the file is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rates = data.get(device, {})
        return {str(stage): float(rate) for stage, rate in rates.items()}
    except OSError, ValueError, AttributeError, TypeError:
        return {}


def save_history(path: Path, device: str, learned: Mapping[str, float]) -> None:
    """Fold one run's measured rates into the history file (moving average per stage); never raises."""
    if not learned:
        return
    try:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except OSError, ValueError:
            data = {}
        stored = data.get(device)
        rates: dict[str, float] = stored if isinstance(stored, dict) else {}
        for stage, rate in learned.items():
            old = rates.get(stage)
            rates[stage] = (
                rate if old is None else (1 - _HISTORY_WEIGHT) * float(old) + _HISTORY_WEIGHT * rate
            )
        data[device] = rates
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not save the time-estimate history: %s", exc)
