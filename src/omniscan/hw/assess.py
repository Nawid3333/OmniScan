"""Per-model compatibility: a pure function from (catalog entry, hardware snapshot) to level/device/messages.

Levels: `ok` (runs on a GPU with enough VRAM, or fast on the CPU), `slow`, `warn` (CPU only and
slow), `incompatible` (no supported backend, or the RAM/VRAM it needs is not there). Cloud models
are always `ok`; Ollama models report the device string `ollama` because the daemon serves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omniscan.hw.detect import GpuInfo, HardwareInfo
from omniscan.models.catalog import ModelEntry

Level = Literal["ok", "slow", "warn", "incompatible"]

_LEVEL_ORDER: dict[Level, int] = {"ok": 0, "slow": 1, "warn": 2, "incompatible": 3}
_CPU_LEVEL: dict[str, Level] = {
    "fast": "ok",
    "ok": "slow",
    "slow": "warn",
    "unusable": "incompatible",
}


@dataclass(frozen=True, slots=True)
class Compatibility:
    """How well one catalog entry would run on a detected machine, with human-readable reasons."""

    level: Level
    device: str | None  # where it would run: "cuda:1", "mps", "ollama", "cpu", None when incompatible
    messages: tuple[str, ...]  # human readable reasons (empty for a clean "ok")


def assess(entry: ModelEntry, hw: HardwareInfo) -> Compatibility:
    """Level, device and reasons for running `entry` on `hw` (card H1 part 3)."""
    return _with_disk_warning(entry, hw, _base_assessment(entry, hw))


def _base_assessment(entry: ModelEntry, hw: HardwareInfo) -> Compatibility:
    """The GPU/CPU decision; the disk-space warning is layered on top for every entry."""
    messages: list[str] = []
    if entry.format == "cloud":  # served by Ollama Cloud, never bound to this machine
        if entry.notes:
            messages.append(entry.notes)
        return Compatibility("ok", None, tuple(messages))
    candidate, integrated_only = _candidate_gpu(entry, hw)
    if integrated_only and candidate is not None:
        messages.append(f"only an integrated GPU ({candidate.name}) is available")
    if candidate is not None and (entry.min_vram_gb is None or candidate.vram_gb >= entry.min_vram_gb):
        device = "ollama" if entry.format == "ollama" else candidate.device
        return Compatibility("ok", device, tuple(messages))
    if candidate is not None:  # the GPU is there but too small
        messages.append(
            f"needs {entry.min_vram_gb:g} GB of GPU memory, your {candidate.name} has {candidate.vram_gb:g} GB"
        )
    if (entry.backends and "cpu" not in entry.backends) or not entry.cpu_ok:
        if candidate is None:
            messages.append("needs a supported GPU (cuda/rocm/mps); none found")
        return Compatibility("incompatible", None, tuple(messages))
    if entry.min_ram_gb is not None and hw.ram_gb < entry.min_ram_gb:
        messages.append(f"needs {entry.min_ram_gb:g} GB of RAM, you have {hw.ram_gb:g} GB")
        return Compatibility("incompatible", None, tuple(messages))
    level = _CPU_LEVEL[entry.cpu_speed]
    message = "runs on the CPU" + (" (slow)" if level in ("slow", "warn") else "")
    device = "ollama" if entry.format == "ollama" else "cpu"
    if level == "incompatible":
        return Compatibility(level, None, (*messages, message))
    return Compatibility(level, device, (*messages, message))


def _candidate_gpu(entry: ModelEntry, hw: HardwareInfo) -> tuple[GpuInfo | None, bool]:
    """First GPU whose backend is allowed (discrete before integrated); True when only integrated."""
    allowed = set(entry.backends) or None
    for gpu in hw.gpus:
        if not gpu.integrated and (allowed is None or gpu.backend in allowed):
            return gpu, False
    for gpu in hw.gpus:
        if gpu.integrated and (allowed is None or gpu.backend in allowed):
            return gpu, True
    return None, False


def _with_disk_warning(entry: ModelEntry, hw: HardwareInfo, compat: Compatibility) -> Compatibility:
    """Add a free-disk warning when the download would not fit; raises the level to at least `warn`."""
    needed_gb = entry.size_mb / 1024 * 1.2
    if hw.disk_free_gb >= needed_gb:
        return compat
    message = f"needs about {needed_gb:.1f} GB of free disk space, {hw.disk_free_gb:g} GB free"
    level = compat.level if _LEVEL_ORDER[compat.level] >= _LEVEL_ORDER["warn"] else "warn"
    return Compatibility(level, compat.device, (*compat.messages, message))
