"""Cross-process exclusive access to the real GPU.

Every OmniScan process lives in its own git worktree (`V:\\OmniScan-wt\\<card>`) but there is only
one physical GPU. Two processes doing real GPU work at the same instant — two GLM builders, or the
director live-checking a card while a builder is mid-run — has already produced a corrupted export
once and a hard process crash once on this ROCm-Windows stack (see `docs/GPU_NOTES.md`); the owner
also heard the crash directly (2026-09-22). `gpu_lock()` is an OS-level advisory lock on a file in
the shared temp directory (the same path regardless of which worktree the caller runs from), so at
most one process anywhere holds it at a time. It is released automatically if the holder exits or
crashes — no stale-lock cleanup is needed.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

LOCK_PATH = Path(tempfile.gettempdir()) / "omniscan-gpu.lock"  # shared by every worktree/process


def _try_lock(handle: IO[bytes]) -> bool:
    """Non-blocking exclusive lock on the first byte of `handle`."""
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN)


def acquire_gpu_lock(*, poll_seconds: float = 2.0, on_wait: Callable[[], None] | None = None) -> IO[bytes]:
    """Block until this process holds the exclusive real-GPU lock; return the handle that holds it.

    `on_wait` is called once, the first time this call has to wait (another process holds the lock).
    Release with `release_gpu_lock(handle)`; the lock is also released automatically if the process
    exits or crashes while holding it (an OS advisory lock, not a sentinel file)."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+b")
    waited = False
    while not _try_lock(handle):
        if not waited:
            if on_wait is not None:
                on_wait()
            else:
                log.info("waiting for exclusive GPU access (another process is using it)")
            waited = True
        time.sleep(poll_seconds)
    if waited:
        log.info("acquired exclusive GPU access")
    return handle


def release_gpu_lock(handle: IO[bytes]) -> None:
    """Release a handle returned by `acquire_gpu_lock` and close it."""
    try:
        _unlock(handle)
    finally:
        handle.close()


@contextmanager
def gpu_lock(*, poll_seconds: float = 2.0, on_wait: Callable[[], None] | None = None) -> Iterator[None]:
    """`with gpu_lock(): ...` around any code that touches the real GPU (see module docstring)."""
    handle = acquire_gpu_lock(poll_seconds=poll_seconds, on_wait=on_wait)
    try:
        yield
    finally:
        release_gpu_lock(handle)
