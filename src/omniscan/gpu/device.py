"""Compute-device selection: an explicit device wins; `auto` picks the strongest discrete GPU, then Apple MPS, then CPU.

One code path for every vendor: NVIDIA (CUDA) and AMD (ROCm/HIP) both appear as torch "cuda" devices, Intel Arc /
Core Ultra graphics as "xpu", Apple Silicon as "mps". Which of them exist depends on the torch build the machine
installed (the `cuda` / `rocm-gfx1201` / `xpu` / `mps` / `cpu` extra in `pyproject.toml`).

Why not simply `cuda:0`: on Windows the integrated GPU is enumerated first and crashes on the first kernel of a
discrete-GPU wheel (verified on Ryzen 5 7600X + RX 9070 XT), while WSL only exposes the discrete card.
"""

from __future__ import annotations

import re
from types import ModuleType

import torch

# Intel integrated graphics (UHD / Iris / the Core Ultra "Graphics" iGPU); Arc A/B-series cards are discrete
_INTEL_INTEGRATED = re.compile(r"uhd|iris|^intel\(r\) graphics$|^intel graphics$", re.IGNORECASE)


def resolve_device(spec: str | torch.device = "auto") -> torch.device:
    """The torch device for `spec` ("auto", "cpu", "mps", "cuda[:N]", "xpu[:N]" or a `torch.device`).

    A named CUDA/HIP, XPU or MPS device that this torch build cannot reach falls back to CPU. "auto" chooses the
    discrete CUDA/HIP GPU with the most multiprocessors (ties: most memory), then the discrete Intel XPU with the most
    memory, skipping integrated GPUs; then Apple MPS; then CPU (integrated GPUs are slow and may lack kernels).
    """
    if isinstance(spec, torch.device):
        device = spec
    elif spec != "auto":
        device = torch.device(spec)
    else:
        return _auto_device()
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "xpu" and not _xpu_available():
        return torch.device("cpu")
    if device.type == "mps" and not _mps_available():
        return torch.device("cpu")
    return device


def accelerator(device: torch.device) -> ModuleType | None:
    """The torch device module (`torch.cuda` / `torch.xpu`) with memory and sync APIs for `device`; None otherwise."""
    if device.type == "cuda":
        return torch.cuda
    if device.type == "xpu":
        return getattr(torch, "xpu", None)
    return None


def select_device(device: torch.device) -> None:
    """Make `device` the current device of its backend (a no-op on CPU/MPS).

    MIOpen (AMD) and oneDNN (Intel) launch kernels against the *current* device, not the tensors' device, so every
    model loader calls this first — otherwise the integrated GPU enumerated first on Windows gets the work.
    """
    module = accelerator(device)
    if module is not None:
        module.set_device(device)


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and xpu.is_available())


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        best: tuple[tuple[int, int], int] | None = None
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            if getattr(props, "is_integrated", False):
                continue
            score = (int(props.multi_processor_count), int(props.total_memory))
            if best is None or score > best[0]:
                best = (score, index)
        if best is not None:
            return torch.device("cuda", best[1])
    if _xpu_available():
        best_xpu: tuple[int, int] | None = None
        for index in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(index)
            if _INTEL_INTEGRATED.search(str(props.name).strip()):
                continue
            if best_xpu is None or int(props.total_memory) > best_xpu[0]:
                best_xpu = (int(props.total_memory), index)
        if best_xpu is not None:
            return torch.device("xpu", best_xpu[1])
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")
