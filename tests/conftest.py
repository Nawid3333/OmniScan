"""Shared pytest fixtures."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    """--perf runs the performance tests (tests/perf); --perf-update records their timings as the baseline."""
    parser.addoption("--perf", action="store_true", help="run the `perf` tests (stage timings vs baselines)")
    parser.addoption(
        "--perf-update", action="store_true", help="record the perf timings as this device's baseline"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `perf` and `gpu_backend` markers."""
    config.addinivalue_line("markers", "perf: stage-timing regression test, runs only with --perf")
    config.addinivalue_line(
        "markers",
        "gpu_backend(*types): a `gpu` test that needs one of these device types (cuda, xpu, mps); "
        "skipped on the others",
    )


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
    """Skip tests marked `gpu` when no GPU is reachable (torch imported lazily); otherwise make the
    `resolve_device("auto")` GPU current, whatever its vendor (cuda for NVIDIA/AMD, xpu, mps), and
    hold the cross-process real-GPU lock for the test's duration (gpu/lock.py) so two processes
    (two GLM builders, or a builder and the director) never run GPU code at the same instant —
    concurrent access on this ROCm-Windows stack has produced a corrupted export and a hard process
    crash. A `gpu_backend(...)` marker further limits the test to those device types.
    `pytest -m "not gpu"` never touches this lock and stays fast/hermetic."""
    if "gpu" not in request.keywords:
        yield
        return
    try:
        importlib.import_module("torch")
    except Exception:
        pytest.skip("torch not importable")
    from omniscan.gpu.device import resolve_device, select_device

    device = resolve_device("auto")
    if device.type == "cpu":
        pytest.skip("no usable discrete GPU available")
    backend = request.node.get_closest_marker("gpu_backend")
    if backend is not None and device.type not in backend.args:
        pytest.skip(f"needs a {' or '.join(backend.args)} device, this machine has {device.type}")
    from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock

    handle = acquire_gpu_lock(
        on_wait=lambda: print(f"\n{request.node.nodeid}: waiting for exclusive GPU access...")
    )
    try:
        select_device(device)  # bare "cuda"/"xpu" and a device-less synchronize() then mean the right card
        yield
    finally:
        release_gpu_lock(handle)


@pytest.fixture
def gpu_device() -> torch.device:
    """The GPU the `gpu` tests run on: `resolve_device("auto")`, whatever its vendor."""
    from omniscan.gpu.device import resolve_device

    return resolve_device("auto")
