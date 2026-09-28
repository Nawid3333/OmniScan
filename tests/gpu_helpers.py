"""Device-neutral sync and memory helpers for the `gpu` tests: CUDA/HIP, Intel XPU and Apple MPS alike."""

from __future__ import annotations

import torch

from omniscan.gpu.device import accelerator


def synchronize(device: torch.device) -> None:
    """Wait for every kernel queued on `device` (a no-op on CPU)."""
    if device.type == "mps":
        torch.mps.synchronize()
        return
    module = accelerator(device)
    if module is not None:
        module.synchronize(device)


def empty_cache(device: torch.device) -> None:
    """Hand the allocator's cached blocks on `device` back to the driver, so later GPU tests start clean."""
    if device.type == "mps":
        torch.mps.empty_cache()
        return
    module = accelerator(device)
    if module is not None:
        module.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    """Restart the peak-allocation counter of `device` (MPS keeps none)."""
    module = accelerator(device)
    if module is not None:
        module.reset_peak_memory_stats(device)


def max_memory_allocated(device: torch.device) -> int | None:
    """Peak bytes allocated on `device` since the last reset; None where the backend keeps no peak (MPS, CPU)."""
    module = accelerator(device)
    return None if module is None else int(module.max_memory_allocated(device))


def device_name(device: torch.device) -> str:
    """The card's name for CUDA/HIP and XPU devices, the device type otherwise (MPS, CPU)."""
    module = accelerator(device)
    return device.type if module is None else str(module.get_device_name(device))
