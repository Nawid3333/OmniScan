"""Tests for omniscan.gpu.device.resolve_device (torch's device queries are faked; CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from omniscan.gpu.device import accelerator, resolve_device, select_device


def fake_cuda(monkeypatch: pytest.MonkeyPatch, devices: list[SimpleNamespace] | None) -> None:
    """Make torch see `devices` (None = no CUDA/HIP at all)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: devices is not None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(devices or []))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: (devices or [])[i])


def gpu(*, integrated: bool, sms: int, gib: float) -> SimpleNamespace:
    return SimpleNamespace(is_integrated=integrated, multi_processor_count=sms, total_memory=int(gib * 2**30))


def fake_mps(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)


def fake_xpu(monkeypatch: pytest.MonkeyPatch, names_gib: list[tuple[str, float]] | None) -> None:
    """Make torch see Intel XPU devices `names_gib` (None = no XPU backend at all)."""
    props = [SimpleNamespace(name=name, total_memory=int(gib * 2**30)) for name, gib in names_gib or []]
    xpu = SimpleNamespace(
        is_available=lambda: names_gib is not None,
        device_count=lambda: len(props),
        get_device_properties=lambda i: props[i],
    )
    monkeypatch.setattr(torch, "xpu", xpu, raising=False)


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


def test_auto_uses_a_discrete_intel_xpu_when_there_is_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, None)
    fake_mps(monkeypatch, False)
    fake_xpu(monkeypatch, [("Intel(R) Graphics", 16), ("Intel(R) Arc(TM) B580 Graphics", 12)])
    assert resolve_device("auto") == torch.device("xpu", 1)


def test_auto_skips_intel_integrated_graphics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, None)
    fake_mps(monkeypatch, False)
    fake_xpu(monkeypatch, [("Intel(R) Iris(R) Xe Graphics", 8), ("Intel(R) UHD Graphics 770", 8)])
    assert resolve_device("auto") == torch.device("cpu")


def test_auto_prefers_a_discrete_cuda_gpu_over_xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, [gpu(integrated=False, sms=32, gib=16)])
    fake_xpu(monkeypatch, [("Intel(R) Arc(TM) A770 Graphics", 16)])
    assert resolve_device("auto") == torch.device("cuda", 0)


def test_named_xpu_needs_an_xpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_xpu(monkeypatch, [("Intel(R) Arc(TM) A770 Graphics", 16)])
    assert resolve_device("xpu:0") == torch.device("xpu", 0)
    fake_xpu(monkeypatch, None)
    assert resolve_device("xpu") == torch.device("cpu")


def test_accelerator_maps_each_backend_to_its_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_xpu(monkeypatch, [])
    assert accelerator(torch.device("cuda", 0)) is torch.cuda
    assert accelerator(torch.device("xpu", 0)) is torch.xpu
    assert accelerator(torch.device("cpu")) is None
    assert accelerator(torch.device("mps")) is None


def test_select_device_sets_the_current_device_of_its_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, torch.device]] = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(("cuda", d)))
    xpu = SimpleNamespace(set_device=lambda d: calls.append(("xpu", d)))
    monkeypatch.setattr(torch, "xpu", xpu, raising=False)
    select_device(torch.device("cuda", 1))
    select_device(torch.device("xpu", 0))
    select_device(torch.device("cpu"))
    select_device(torch.device("mps"))
    assert calls == [("cuda", torch.device("cuda", 1)), ("xpu", torch.device("xpu", 0))]
