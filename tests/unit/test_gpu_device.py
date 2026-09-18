"""Tests for omniscan.gpu.device.resolve_device (torch's device queries are faked; CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from omniscan.gpu.device import resolve_device


def fake_cuda(monkeypatch: pytest.MonkeyPatch, devices: list[SimpleNamespace] | None) -> None:
    """Make torch see `devices` (None = no CUDA/HIP at all)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: devices is not None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(devices or []))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: (devices or [])[i])


def gpu(*, integrated: bool, sms: int, gib: float) -> SimpleNamespace:
    return SimpleNamespace(is_integrated=integrated, multi_processor_count=sms, total_memory=int(gib * 2**30))


def fake_mps(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)


def test_auto_skips_the_integrated_gpu_listed_first(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, [gpu(integrated=True, sms=1, gib=24.2), gpu(integrated=False, sms=32, gib=15.9)])
    assert resolve_device("auto") == torch.device("cuda", 1)
    assert resolve_device() == torch.device("cuda", 1)


def test_auto_prefers_more_multiprocessors_then_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(
        monkeypatch,
        [
            gpu(integrated=False, sms=40, gib=8),
            gpu(integrated=False, sms=96, gib=16),
            gpu(integrated=False, sms=96, gib=24),
        ],
    )
    assert resolve_device("auto") == torch.device("cuda", 2)


def test_auto_with_only_an_integrated_gpu_uses_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, [gpu(integrated=True, sms=2, gib=24)])
    assert resolve_device("auto") == torch.device("cpu")


def test_auto_without_cuda_tries_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, None)
    fake_mps(monkeypatch, True)
    assert resolve_device("auto") == torch.device("mps")
    fake_mps(monkeypatch, False)
    assert resolve_device("auto") == torch.device("cpu")


def test_explicit_devices_win_over_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, [gpu(integrated=True, sms=1, gib=24), gpu(integrated=False, sms=32, gib=16)])
    assert resolve_device("cuda:0") == torch.device("cuda", 0)
    assert resolve_device("cuda:1") == torch.device("cuda", 1)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cuda", 1)) == torch.device("cuda", 1)


def test_unreachable_named_devices_fall_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, None)
    fake_mps(monkeypatch, False)
    assert resolve_device("cuda:0") == torch.device("cpu")
    assert resolve_device("cuda") == torch.device("cpu")
    assert resolve_device("mps") == torch.device("cpu")
    assert resolve_device(torch.device("cuda", 0)) == torch.device("cpu")


def test_invalid_spec_raises() -> None:
    with pytest.raises(RuntimeError):
        resolve_device("not-a-device")
