"""Compute-device selection: an explicit device wins; `auto` picks the strongest discrete GPU, then Apple MPS, then CPU.

Why not simply `cuda:0`: on Windows the integrated GPU is enumerated first and crashes on the first kernel of a
discrete-GPU wheel (verified on Ryzen 5 7600X + RX 9070 XT), while WSL only exposes the discrete card.
"""

from __future__ import annotations

import torch


def resolve_device(spec: str | torch.device = "auto") -> torch.device:
    """The torch device for `spec` ("auto", "cpu", "mps", "cuda", "cuda:N" or a `torch.device`).

    A named CUDA/HIP or MPS device that this torch build cannot reach falls back to CPU. "auto" chooses the discrete
    GPU with the most multiprocessors (ties: most memory), skipping integrated GPUs; with only integrated GPUs it
    returns CPU (they are slow and may lack kernels), and without any GPU it tries Apple MPS before CPU.
    """
    if isinstance(spec, torch.device):
        device = spec
    elif spec != "auto":
        device = torch.device(spec)
    else:
        return _auto_device()
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not _mps_available():
        return torch.device("cpu")
    return device


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


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
        return torch.device("cuda", best[1]) if best is not None else torch.device("cpu")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")
