"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _skip_gpu_when_unavailable(request: pytest.FixtureRequest) -> None:
    """Skip tests marked `gpu` when no CUDA device is reachable (torch imported lazily)."""
    if "gpu" not in request.keywords:
        return
    try:
        import torch
    except Exception:
        pytest.skip("torch not importable")
    from omniscan.gpu.device import resolve_device

    device = resolve_device("auto")
    if device.type != "cuda":
        pytest.skip("no usable discrete GPU available")
    torch.cuda.set_device(device)  # bare "cuda" and torch.cuda.synchronize() then mean the right card
