"""Hardware detection: one plain snapshot of OS, CPU, RAM, GPUs, torch build and free disk.

`detect_hardware` imports torch lazily and never fails when torch is missing or broken (then there
are no GPUs and `best_device` is "cpu"), so the command and the future settings screen always get
something to render. What a model would run like on that snapshot lives in `hw.assess`.
"""

from __future__ import annotations

import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import psutil

from omniscan.core.config import PathsConfig

Backend = Literal["cuda", "rocm", "mps", "xpu", "cpu"]
Vendor = Literal["nvidia", "amd", "apple", "intel", "other"]
TorchBuild = Literal["cuda", "rocm", "mps", "cpu"]

GIB = float(2**30)

_VENDOR_PATTERNS: tuple[tuple[Vendor, re.Pattern[str]], ...] = (
    ("nvidia", re.compile(r"nvidia|geforce|rtx|gtx|quadro|tesla", re.IGNORECASE)),
    ("amd", re.compile(r"amd|radeon|instinct", re.IGNORECASE)),
    ("apple", re.compile(r"apple|\bm[1-5]\b", re.IGNORECASE)),
    ("intel", re.compile(r"intel|arc|iris|uhd", re.IGNORECASE)),
)
_AMD_INTEGRATED = re.compile(r"radeon\s*\d{3}m")
_INTEL_INTEGRATED = re.compile(r"uhd|iris")


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """One torch-visible accelerator (or the Apple unified GPU): name, backend, memory, device."""

    index: int
    name: str
    vendor: Vendor
    backend: Backend  # "cuda" (NVIDIA), "rocm" (AMD via HIP), "mps", "xpu", else "cpu"
    vram_gb: float  # total memory in GiB; for MPS the unified memory available to the GPU
    integrated: bool
    device: str  # the torch device string to use, e.g. "cuda:1", "mps"


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """The machine snapshot behind `omniscan hardware` and the per-model compatibility check."""

    os: str  # "windows" | "linux" | "macos" | other sys.platform
    arch: str  # "x64" | "arm64" | platform.machine() lower-cased
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_gb: float
    gpus: tuple[GpuInfo, ...]  # torch-visible accelerators, best first
    torch_build: TorchBuild  # what THIS torch was built for
    best_device: str  # what resolve_device("auto") would pick, "cpu" when no usable GPU
    onnxruntime_providers: tuple[str, ...]  # empty when onnxruntime is not installed
    disk_free_gb: float  # free space where the models live (nearest existing parent)


def classify_vendor(name: str) -> Vendor:
    """Vendor for a GPU marketing name (case-insensitive substring rules, first match wins)."""
    for vendor, pattern in _VENDOR_PATTERNS:
        if pattern.search(name):
            return vendor
    return "other"


def is_integrated(name: str, vram_gb: float, vendor: str) -> bool:
    """Heuristic: Apple has one unified GPU; Intel UHD/Iris and small AMD `Graphics` parts are integrated."""
    lower = name.lower()
    if vendor == "apple":
        return False
    if vendor == "intel":
        return bool(_INTEL_INTEGRATED.search(lower))  # Arc is discrete
    if vendor == "amd":
        return bool(
            "radeon(tm) graphics" in lower or "radeon graphics" in lower or _AMD_INTEGRATED.search(lower)
        )
    return False


def detect_hardware(models_dir: Path | None = None) -> HardwareInfo:
    """Snapshot this machine: OS, CPU, RAM, torch-visible GPUs (best first) and free disk at models_dir."""
    ram_gb = round(psutil.virtual_memory().total / GIB, 1)
    gpus, torch_build = _detect_gpus(ram_gb)
    return HardwareInfo(
        os=_os_name(),
        arch=_arch(),
        cpu_name=_cpu_name(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_gb=ram_gb,
        gpus=tuple(sorted(gpus, key=lambda gpu: (gpu.integrated, -gpu.vram_gb, gpu.index))),
        torch_build=torch_build,
        best_device=_best_device(),
        onnxruntime_providers=_onnxruntime_providers(),
        disk_free_gb=_disk_free_gb(models_dir),
    )


def _os_name() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine or "unknown"


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "unknown"


def _detect_gpus(ram_gb: float) -> tuple[list[GpuInfo], TorchBuild]:
    """All torch-visible accelerators plus the build this torch was made for; [] when torch is unusable."""
    try:
        import torch
    except Exception:  # missing or broken: the app stays usable on the CPU paths
        return [], "cpu"
    build: TorchBuild = "cpu"
    if getattr(torch.version, "hip", None):
        build = "rocm"
    elif getattr(torch.version, "cuda", None):
        build = "cuda"
    gpus: list[GpuInfo] = []
    try:
        if torch.cuda.is_available():
            backend: Backend = "rocm" if build == "rocm" else "cuda"
            for index in range(torch.cuda.device_count()):
                gpus.append(_cuda_gpu(torch, index, backend))
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            gpus.append(
                GpuInfo(
                    index=0,
                    name="Apple GPU",
                    vendor="apple",
                    backend="mps",
                    vram_gb=ram_gb,
                    integrated=False,
                    device="mps",
                )
            )
            if build == "cpu":
                build = "mps"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for index in range(xpu.device_count()):
                gpus.append(_xpu_gpu(xpu, index))
    except Exception:  # a half-broken torch build must not take the whole snapshot down
        pass
    return gpus, build


def _cuda_gpu(torch: ModuleType, index: int, backend: Backend) -> GpuInfo:
    props = torch.cuda.get_device_properties(index)
    name = str(props.name)
    vram_gb = round(props.total_memory / GIB, 1)
    vendor = classify_vendor(name)
    return GpuInfo(
        index=index,
        name=name,
        vendor=vendor,
        backend=backend,
        vram_gb=vram_gb,
        integrated=is_integrated(name, vram_gb, vendor),
        device=f"cuda:{index}",
    )


def _xpu_gpu(xpu: ModuleType, index: int) -> GpuInfo:
    props = xpu.get_device_properties(index)
    name = str(props.name)
    vram_gb = round(props.total_memory / GIB, 1)
    vendor = classify_vendor(name)
    return GpuInfo(
        index=index,
        name=name,
        vendor=vendor,
        backend="xpu",
        vram_gb=vram_gb,
        integrated=is_integrated(name, vram_gb, vendor),
        device=f"xpu:{index}",
    )


def _best_device() -> str:
    try:
        from omniscan.gpu.device import resolve_device

        return str(resolve_device("auto"))
    except Exception:
        return "cpu"


def _onnxruntime_providers() -> tuple[str, ...]:
    try:
        import onnxruntime  # pyright: ignore[reportMissingImports]  # optional, not a dependency
    except Exception:
        return ()
    try:
        return tuple(onnxruntime.get_available_providers())
    except Exception:
        return ()


def _disk_free_gb(models_dir: Path | None) -> float:
    path = models_dir or PathsConfig().models_dir
    probe = next((parent for parent in (path, *path.parents) if parent.exists()), Path.home())
    try:
        return round(psutil.disk_usage(str(probe)).free / GIB, 1)
    except Exception:
        return 0.0
