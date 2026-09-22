"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _skip_gpu_when_unavailable(request: pytest.FixtureRequest) -> Iterator[None]:
    """Skip tests marked `gpu` when no CUDA device is reachable (torch imported lazily); otherwise
    hold the cross-process real-GPU lock for the test's duration (gpu/lock.py) so two processes
    (two GLM builders, or a builder and the director) never run GPU code at the same instant —
    concurrent access on this ROCm-Windows stack has produced a corrupted export and a hard process
    crash. `pytest -m "not gpu"` never touches this lock and stays fast/hermetic."""
    if "gpu" not in request.keywords:
        yield
        return
    try:
        import torch
    except Exception:
        pytest.skip("torch not importable")
    from omniscan.gpu.device import resolve_device

    device = resolve_device("auto")
    if device.type != "cuda":
        pytest.skip("no usable discrete GPU available")
    from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock

    handle = acquire_gpu_lock(
        on_wait=lambda: print(f"\n{request.node.nodeid}: waiting for exclusive GPU access...")
    )
    try:
        torch.cuda.set_device(device)  # bare "cuda" and torch.cuda.synchronize() then mean the right card
        yield
    finally:
        release_gpu_lock(handle)
