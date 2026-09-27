"""Per-device stage baselines for the perf tests (tests/perf/baselines.json): compare and record timings.

A stage fails when it takes longer than its recorded baseline by more than TOLERANCE (a share of the baseline)
plus SLACK seconds (timer noise on very short stages). A device without a baseline only reports its timings;
`pytest --perf --perf-update` records them as the new baseline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

BASELINES = Path(__file__).with_name("baselines.json")
TOLERANCE = 0.30
SLACK = 0.25  # seconds


def load_baselines(path: Path = BASELINES) -> dict[str, dict[str, float]]:
    """device key -> stage -> recorded seconds ({} when nothing was recorded yet)."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def slower_stages(measured: Mapping[str, float], baseline: Mapping[str, float]) -> list[str]:
    """One line per stage that is over its budget (stages without a baseline are never over)."""
    return [
        f"{stage}: {seconds:.2f}s > budget {baseline[stage] * (1 + TOLERANCE) + SLACK:.2f}s "
        f"(baseline {baseline[stage]:.2f}s)"
        for stage, seconds in measured.items()
        if stage in baseline and seconds > baseline[stage] * (1 + TOLERANCE) + SLACK
    ]


def record(device: str, measured: Mapping[str, float], path: Path = BASELINES) -> None:
    """Store `measured` (rounded to milliseconds) as the baseline of `device`, keeping the other devices."""
    baselines = load_baselines(path)
    baselines[device] = {stage: round(seconds, 3) for stage, seconds in measured.items()}
    path.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def report(device: str, measured: Mapping[str, float], baseline: Mapping[str, float]) -> str:
    """A table of the measured seconds per stage next to the baseline."""
    lines = [f"perf on {device}:"]
    for stage, seconds in measured.items():
        known = f"{baseline[stage]:.2f}s" if stage in baseline else "no baseline"
        lines.append(f"  {stage:<14} {seconds:7.2f}s   ({known})")
    return "\n".join(lines)
