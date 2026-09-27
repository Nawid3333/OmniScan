"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """--perf runs the performance tests (tests/perf); --perf-update records their timings as the baseline."""
    parser.addoption("--perf", action="store_true", help="run the `perf` tests (stage timings vs baselines)")
    parser.addoption(
        "--perf-update", action="store_true", help="record the perf timings as this device's baseline"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `perf` marker."""
    config.addinivalue_line("markers", "perf: stage-timing regression test, runs only with --perf")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the `perf` tests unless --perf (or --perf-update) is given: they are slow and need the GPU."""
    if config.getoption("--perf") or config.getoption("--perf-update"):
        return
    skip = pytest.mark.skip(reason="perf test: run with --perf")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip)


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
