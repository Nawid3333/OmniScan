"""Cross-process real-GPU lock (gpu/lock.py): CPU-only, no real GPU needed — the lock is plain
file/OS mechanics, exercised here by opening the same lock path more than once within one process."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from omniscan.gpu import lock as lock_module
from omniscan.gpu.lock import acquire_gpu_lock, gpu_lock, release_gpu_lock


@pytest.fixture(autouse=True)
def _isolated_lock_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets its own lock file so tests never contend with each other or a real run."""
    monkeypatch.setattr(lock_module, "LOCK_PATH", tmp_path / "gpu.lock")


def test_second_acquire_is_blocked_while_the_first_is_held() -> None:
    handle1 = acquire_gpu_lock()
    probe = lock_module.LOCK_PATH.open("a+b")
    try:
        assert not lock_module._try_lock(probe)
    finally:
        probe.close()
        release_gpu_lock(handle1)
    handle2 = acquire_gpu_lock()  # free again after release
    release_gpu_lock(handle2)


def test_acquire_blocks_until_release_and_calls_on_wait_exactly_once() -> None:
    holder = acquire_gpu_lock()
    waits: list[int] = []
    released_at = time.monotonic()

    def release_after_delay() -> None:
        nonlocal released_at
        time.sleep(0.2)
        release_gpu_lock(holder)
        released_at = time.monotonic()

    threading.Thread(target=release_after_delay).start()
    start = time.monotonic()
    handle = acquire_gpu_lock(poll_seconds=0.05, on_wait=lambda: waits.append(1))
    acquired_at = time.monotonic()
    release_gpu_lock(handle)
    assert waits == [1]  # called once, not once per poll
    assert acquired_at >= start + 0.15  # really waited, not a race
    assert acquired_at >= released_at - 0.05  # only after the holder actually released


def test_acquire_without_on_wait_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    holder = acquire_gpu_lock()
    threading.Timer(0.1, lambda: release_gpu_lock(holder)).start()
    handle = acquire_gpu_lock(poll_seconds=0.05)  # default on_wait: logs, never raises
    release_gpu_lock(handle)


def test_gpu_lock_context_manager_releases_on_exception() -> None:
    with pytest.raises(ValueError, match="boom"), gpu_lock():
        raise ValueError("boom")
    # released: a fresh acquire does not block
    handle = acquire_gpu_lock()
    release_gpu_lock(handle)


def test_gpu_lock_context_manager_excludes_concurrent_use() -> None:
    order: list[str] = []

    def worker(name: str, hold: float) -> None:
        with gpu_lock(poll_seconds=0.02):
            order.append(f"{name}-start")
            time.sleep(hold)
            order.append(f"{name}-end")

    t1 = threading.Thread(target=worker, args=("a", 0.15))
    t2 = threading.Thread(target=worker, args=("b", 0.0))
    t1.start()
    time.sleep(0.03)  # let t1 acquire first
    t2.start()
    t1.join()
    t2.join()
    # b never starts until a has fully finished (no interleaving)
    assert order == ["a-start", "a-end", "b-start", "b-end"]


def test_release_closes_the_handle() -> None:
    handle = acquire_gpu_lock()
    release_gpu_lock(handle)
    assert handle.closed
