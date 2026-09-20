"""Background GPU library warm-up: hides the once-per-process MIOpen/rocFFT/rocBLAS stalls.

The first convolution of a process loads MIOpen's Composable-Kernel library (9-14 s) and the first
FFT, GEMM and transposed convolution cost 1-3 s each, all while the GPU idles (docs/GPU_NOTES.md).
Running them once on a daemon thread at pipeline start overlaps the stalls with ingest, JPEG
decoding and model loading; the main thread's own first convolution then costs ~0.01 s.
"""

from __future__ import annotations

import logging
import threading
import time

import torch
import torch.nn.functional as F  # noqa: N812 — torch's standard alias

from omniscan.gpu.timeline import mark

log = logging.getLogger(__name__)


class GpuWarmup:
    """Initialises the GPU libraries once per process on a daemon thread; never raises."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._started = False
        self._start_lock = threading.Lock()
        self._seconds: float | None = None
        self._error: str | None = None
        self._done = threading.Event()
        if device.type != "cuda":  # cpu/mps: nothing to warm up
            self._seconds = 0.0
            self._done.set()

    def start(self) -> None:
        """Spawn the warm-up thread; a no-op once started or when the device is not CUDA."""
        with self._start_lock:
            if self._started or self._device.type != "cuda":
                return
            self._started = True
            threading.Thread(target=self._run, name="gpu-warmup", daemon=True).start()

    def wait(self, timeout: float | None = None) -> bool:
        """True when the warm-up has finished (at once when there is nothing to do)."""
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        """Whether the warm-up thread has finished."""
        return self._done.is_set()

    @property
    def seconds(self) -> float | None:
        """Wall time of the warm-up, None until done."""
        return self._seconds

    @property
    def error(self) -> str | None:
        """\"<ExceptionType>: <message>\" of the first failing step, None when every step succeeded."""
        return self._error

    def _indexed(self) -> torch.device:
        """The device with an explicit index: set_device needs one; an index-less "cuda" means the current one."""
        device = self._device
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def _run(self) -> None:
        started = time.perf_counter()
        mark("gpu warmup thread begin")
        try:
            device = self._indexed()
            torch.cuda.set_device(device)  # MIOpen launches against the current device
            with torch.inference_mode():
                self._run_steps(device)
        except Exception as exc:  # set_device itself failed; the thread must never raise
            self._record("set_device", exc)
        self._seconds = time.perf_counter() - started
        log.info("GPU warm-up finished in %.1f s", self._seconds)
        mark("gpu warmup thread end")
        self._done.set()  # after the log, so wait() returning implies the finish was logged

    def _run_steps(self, device: torch.device) -> None:
        """The warm-up ops (docs/GPU_NOTES.md first-call stalls), each with its own error handling."""
        box: list[torch.Tensor] = []

        def conv3x3() -> None:
            image = torch.empty((8, 32, 320, 320), device=device, dtype=torch.float32)
            box.append(image)  # the 1x1 conv and the interpolation reuse the same input
            F.conv2d(image, torch.empty((64, 32, 3, 3), device=device, dtype=torch.float32), padding=1)

        def conv1x1() -> None:
            F.conv2d(box[0], torch.empty((64, 32, 1, 1), device=device, dtype=torch.float32))

        def conv_transpose() -> None:
            F.conv_transpose2d(
                torch.empty((1, 64, 64, 64), device=device, dtype=torch.float32),
                torch.empty((64, 32, 4, 4), device=device, dtype=torch.float32),
                stride=2,
                padding=1,
            )

        def fft() -> None:
            for side in (256, 512):  # LaMa's rfft2/irfft2 window sizes
                spectrum = torch.fft.rfft2(
                    torch.empty((1, 64, side, side), device=device, dtype=torch.float32)
                )
                torch.fft.irfft2(spectrum, s=(512, 512))

        def bmm() -> None:
            torch.bmm(
                torch.empty((4, 128, 128), device=device, dtype=torch.float32),
                torch.empty((4, 128, 128), device=device, dtype=torch.float32),
            )

        def interpolate() -> None:
            F.interpolate(box[0], size=(160, 160), mode="bilinear", antialias=True)

        for name, step in (
            ("conv3x3", conv3x3),
            ("conv1x1", conv1x1),
            ("conv_transpose2d", conv_transpose),
            ("rfft2/irfft2", fft),
            ("bmm", bmm),
            ("interpolate", interpolate),
        ):
            try:
                step()
            except Exception as exc:
                self._record(name, exc)
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            self._record("synchronize", exc)

    def _record(self, step: str, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        log.warning("GPU warm-up step %s failed: %s", step, message)
        if self._error is None:
            self._error = message


_instances: dict[str, GpuWarmup] = {}
_lock = threading.Lock()


def start_warmup(device: torch.device) -> GpuWarmup:
    """The process-wide GpuWarmup for `device`: created and started on the first call, shared after."""
    with _lock:
        warmup = _instances.get(str(device))
        if warmup is None:
            warmup = GpuWarmup(device)
            _instances[str(device)] = warmup
    warmup.start()
    return warmup
