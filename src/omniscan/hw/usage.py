"""How hard OmniScan may use the machine: `gpu.usage` = "full" | "balanced" | "background".

"full" leaves every limit where torch and the config put it. "balanced" keeps a quarter of the CPU and of the GPU's
memory for other programs. "background" is for running while the user browses or works: a quarter of the CPU
threads, half the GPU memory and a lower process priority, so the operating system serves the user's apps first.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Literal

import psutil

log = logging.getLogger(__name__)

UsageLevel = Literal["full", "balanced", "background"]
USAGE_LEVELS: tuple[UsageLevel, ...] = ("full", "balanced", "background")

# level -> (share of the logical CPU cores, share of the GPU's total memory, lower the process priority)
_PROFILES: dict[UsageLevel, tuple[float, float, bool]] = {
    "full": (1.0, 1.0, False),
    "balanced": (0.75, 0.75, False),
    "background": (0.25, 0.5, True),
}


@dataclass(frozen=True, slots=True)
class UsageLimits:
    """The concrete limits one usage level means on this machine."""

    level: UsageLevel
    cpu_threads: int  # worker threads for torch's CPU ops and the JPEG codec
    vram_fraction: float  # the most of the GPU's total memory the models may take
    low_priority: bool  # run below normal process priority


def usage_limits(level: UsageLevel, logical_cores: int | None = None) -> UsageLimits:
    """The limits for `level` on a machine with `logical_cores` (default: this one's); never below one thread."""
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 1)
    cpu_share, vram_fraction, low_priority = _PROFILES[level]
    return UsageLimits(level, max(1, int(cores * cpu_share)), vram_fraction, low_priority)


def vram_budget_gib(limits: UsageLimits, configured_gib: float, total_gib: float | None) -> float:
    """The VRAM budget to enforce: the configured one, capped at the level's share of the GPU (when known)."""
    if total_gib is None:
        return configured_gib
    return min(configured_gib, round(total_gib * limits.vram_fraction, 2))


def apply_process_limits(limits: UsageLimits) -> None:
    """Apply the CPU side of `limits` to this process: torch/OpenCV thread counts and, for background, priority.

    "full" changes nothing. The priority is only ever lowered (raising it again needs admin rights on most systems),
    so a process that once ran in the background stays there until it exits.
    """
    if limits.level == "full":
        return
    try:
        import torch

        torch.set_num_threads(limits.cpu_threads)
    except Exception as exc:  # torch missing or broken: the GUI and CLI must still start
        log.debug("could not limit torch threads: %s", exc)
    try:
        import cv2

        cv2.setNumThreads(limits.cpu_threads)
    except Exception as exc:
        log.debug("could not limit OpenCV threads: %s", exc)
    if limits.low_priority:
        _lower_priority()


def _lower_priority() -> None:
    process = psutil.Process()
    try:
        if sys.platform == "win32":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif process.nice() < 10:
            process.nice(10)
    except (psutil.Error, OSError) as exc:
        log.warning("could not lower the process priority: %s", exc)
