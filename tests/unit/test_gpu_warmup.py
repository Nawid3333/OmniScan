"""G2 GPU warm-up: the exact op sequence on a background thread, recorded without touching a GPU."""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
import torch

from omniscan.gpu import warmup as warmup_module
from omniscan.gpu.warmup import GpuWarmup, start_warmup

CUDA = torch.device("cuda", 0)


class Recorder:
    """Record of the patched torch ops: (name, args, kwargs) tuples plus the calling threads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.threads: set[int] = set()

    def names(self) -> list[str]:
        return [name for name, _args, _kwargs in self.calls]


def patch_ops(
    monkeypatch: pytest.MonkeyPatch, recorder: Recorder, fail_at: frozenset[str] = frozenset()
) -> None:
    """Replace every op the warm-up thread uses with a recorder; ops in `fail_at` raise every time."""
    counters: dict[str, int] = {}

    def recorded(op: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> None:
            counters[op] = counters.get(op, 0) + 1
            recorder.calls.append((op, args, kwargs))
            recorder.threads.add(threading.get_ident())
            if op in fail_at:
                raise RuntimeError(f"{op} failed #{counters[op]}")

        return fn

    for module, attr, op in (
        (torch, "empty", "empty"),
        (torch.nn.functional, "conv2d", "conv2d"),
        (torch.nn.functional, "conv_transpose2d", "conv_transpose2d"),
        (torch.fft, "rfft2", "rfft2"),
        (torch.fft, "irfft2", "irfft2"),
        (torch, "bmm", "bmm"),
        (torch.nn.functional, "interpolate", "interpolate"),
        (torch.cuda, "set_device", "set_device"),
        (torch.cuda, "synchronize", "synchronize"),
    ):
        monkeypatch.setattr(module, attr, recorded(op))


EXPECTED_SEQUENCE = [
    "set_device",
    "empty",  # conv3x3 input [8, 32, 320, 320]
    "empty",
    "conv2d",  # 3x3 weight + convolution
    "empty",
    "conv2d",  # 1x1 weight + convolution on the same input
    "empty",
    "empty",
    "conv_transpose2d",
    "empty",
    "rfft2",
    "irfft2",  # 256 window
    "empty",
    "rfft2",
    "irfft2",  # 512 window
    "empty",
    "empty",
    "bmm",
    "interpolate",
    "synchronize",
]


def test_sequence_runs_exactly_once_on_another_thread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder)
    main_thread = threading.get_ident()
    with caplog.at_level(logging.INFO, logger="omniscan.gpu.warmup"):
        warmup = GpuWarmup(CUDA)
        warmup.start()
        assert warmup.wait(10.0)

    assert recorder.names() == EXPECTED_SEQUENCE
    assert recorder.threads and main_thread not in recorder.threads and len(recorder.threads) == 1
    assert recorder.calls[1][1] == ((8, 32, 320, 320),)
    assert recorder.calls[1][2] == {"device": CUDA, "dtype": torch.float32}
    assert recorder.calls[3][2] == {"padding": 1}  # the 3x3 convolution
    assert recorder.calls[8][2] == {"stride": 2, "padding": 1}  # conv_transpose2d
    assert recorder.calls[11][2] == {"s": (512, 512)}  # irfft2 of the 256 window
    assert recorder.calls[12][1] == ((1, 64, 512, 512),)
    assert recorder.calls[18][2] == {"size": (160, 160), "mode": "bilinear", "antialias": True}
    assert warmup.done and warmup.wait() is True
    assert warmup.seconds is not None and warmup.seconds >= 0.0
    assert warmup.error is None
    assert "GPU warm-up finished in" in caplog.text


def test_failing_step_is_logged_recorded_once_and_does_not_stop_the_rest(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder, fail_at=frozenset({"rfft2"}))
    with caplog.at_level(logging.INFO, logger="omniscan.gpu.warmup"):
        warmup = GpuWarmup(CUDA)
        warmup.start()
        assert warmup.wait(10.0)

    assert warmup.error == "RuntimeError: rfft2 failed #1"
    names = recorder.names()
    assert names.count("rfft2") == 1 and "irfft2" not in names  # the fft step aborted at its first op
    for later in ("bmm", "interpolate", "synchronize"):
        assert later in names  # the remaining steps still ran
    assert "GPU warm-up step rfft2/irfft2 failed: RuntimeError: rfft2 failed #1" in caplog.text


def test_first_failure_is_kept_when_several_steps_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder, fail_at=frozenset({"conv2d", "bmm"}))
    warmup = GpuWarmup(CUDA)
    warmup.start()
    assert warmup.wait(10.0)

    assert warmup.error == "RuntimeError: conv2d failed #1"  # the first failure only
    assert recorder.names().count("conv2d") == 2  # both convolutions were attempted


def test_start_twice_runs_the_sequence_once(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder)
    warmup = GpuWarmup(CUDA)
    warmup.start()
    assert warmup.wait(10.0)
    calls_after_first = len(recorder.calls)

    warmup.start()
    assert warmup.wait(1.0)
    assert len(recorder.calls) == calls_after_first


def test_non_cuda_devices_do_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder)
    for device in (torch.device("cpu"), torch.device("mps")):
        warmup = GpuWarmup(device)
        warmup.start()
        warmup.start()
        assert warmup.done
        assert warmup.wait() is True
        assert warmup.seconds == 0.0
        assert warmup.error is None
    assert recorder.calls == []


def test_index_less_cuda_device_targets_the_current_one(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    warmup = GpuWarmup(torch.device("cuda"))  # no index: warm up the device the pipeline's ops use
    warmup.start()
    assert warmup.wait(10.0)

    set_devices = [args[0] for name, args, _kwargs in recorder.calls if name == "set_device"]
    assert set_devices == [torch.device("cuda", 2)]
    assert recorder.calls[1][2] == {"device": torch.device("cuda", 2), "dtype": torch.float32}
    assert warmup.error is None


def test_start_warmup_shares_one_instance_per_device(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder)
    monkeypatch.setattr(warmup_module, "_instances", {})

    first = start_warmup(torch.device("cuda", 0))
    assert start_warmup(torch.device("cuda", 0)) is first
    other = start_warmup(torch.device("cuda", 1))
    assert other is not first
    assert first.wait(10.0) and other.wait(10.0)

    devices = [args[0] for name, args, _kwargs in recorder.calls if name == "set_device"]
    assert len(devices) == 2 and {str(device) for device in devices} == {"cuda:0", "cuda:1"}
    assert len(recorder.threads) == 2  # one warm-up thread per device


def test_wait_times_out_while_a_step_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_ops(monkeypatch, Recorder())
    blocked, release = threading.Event(), threading.Event()

    def blocking_bmm(*args: Any, **kwargs: Any) -> None:
        blocked.set()
        release.wait(10.0)

    monkeypatch.setattr(torch, "bmm", blocking_bmm)
    warmup = GpuWarmup(CUDA)
    warmup.start()
    assert blocked.wait(10.0)

    assert warmup.wait(0.01) is False  # the warm-up is still inside the blocked step
    release.set()
    assert warmup.wait(10.0) is True
    assert warmup.done and warmup.error is None


def test_gate_holds_the_steps_until_it_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """G3: the gate keeps the MIOpen find off the pipeline's startup decode (first acquire)."""
    recorder = Recorder()
    patch_ops(monkeypatch, recorder)
    gate = threading.Event()

    warmup = GpuWarmup(CUDA, gate=gate)
    warmup.start()
    assert warmup.wait(0.05) is False  # still waiting for the gate
    assert recorder.names() == ["set_device"]  # set_device runs before the gate, nothing after

    gate.set()
    assert warmup.wait(10.0)
    assert recorder.names() == EXPECTED_SEQUENCE
    assert warmup.error is None


def test_start_warmup_forwards_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    patch_ops(monkeypatch, recorder)
    monkeypatch.setattr(warmup_module, "_instances", {})
    gate = threading.Event()

    warmup = start_warmup(torch.device("cuda", 0), gate=gate)
    assert warmup._gate is gate
