"""Tests for omniscan.hw.detect (torch, the device picker and psutil are faked; CPU only)."""

from __future__ import annotations

import platform
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from omniscan.hw.detect import classify_vendor, detect_hardware, is_integrated


class FakeDevice:
    """Stand-in for torch.device that str()s like one ("cuda:1", "mps", "cpu")."""

    def __init__(self, *parts: object) -> None:
        self.text = ":".join(str(part) for part in parts)

    def __str__(self) -> str:
        return self.text


def gpu_props(name: str, gib: float, sms: int = 32) -> SimpleNamespace:
    """A torch.cuda.get_device_properties(i) result as detect_hardware and resolve_device read it."""
    return SimpleNamespace(name=name, total_memory=int(gib * 2**30), multi_processor_count=sms)


def fake_torch(
    *,
    devices: list[SimpleNamespace] | None = None,
    hip: str | None = None,
    cuda: str | None = None,
    mps: bool = False,
    xpu_devices: list[SimpleNamespace] | None = None,
) -> types.ModuleType:
    """A torch module exposing only the surface detect_hardware queries."""
    module = types.ModuleType("torch")
    module.version = SimpleNamespace(hip=hip, cuda=cuda)  # type: ignore[attr-defined]
    module.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: devices is not None,
        device_count=lambda: len(devices or []),
        get_device_properties=lambda i: (devices or [])[i],
    )
    module.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))  # type: ignore[attr-defined]
    module.device = lambda *parts: FakeDevice(*parts)  # type: ignore[attr-defined]
    if xpu_devices is not None:
        module.xpu = SimpleNamespace(  # type: ignore[attr-defined]
            is_available=lambda: True,
            device_count=lambda: len(xpu_devices),
            get_device_properties=lambda i: xpu_devices[i],
        )
    return module


def install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    module: types.ModuleType | None,
    *,
    best: str = "cpu",
    resolve_fails: bool = False,
) -> None:
    """Put `module` into sys.modules as torch and stub the auto device picker (None = import fails)."""
    monkeypatch.setitem(sys.modules, "torch", module)

    def resolve_device(spec: object = "auto") -> FakeDevice:
        if resolve_fails:
            raise RuntimeError("no usable device")
        return FakeDevice(best)

    picker = types.ModuleType("omniscan.gpu.device")
    picker.resolve_device = resolve_device  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omniscan.gpu.device", picker)


def fake_psutil(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ram_gb: float = 31.3,
    physical: int = 8,
    logical_count: int = 16,
    disk_free_gb: float = 100.0,
) -> list[Path]:
    """Patch psutil's queries; returns the list that collects every path disk_usage was called with."""

    def cpu_count(logical: bool = True) -> int:
        return logical_count if logical else physical

    probed: list[Path] = []

    def disk_usage(path: object) -> SimpleNamespace:
        probed.append(Path(str(path)))
        return SimpleNamespace(free=int(disk_free_gb * 2**30))

    monkeypatch.setattr(psutil, "cpu_count", cpu_count)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=int(ram_gb * 2**30)))
    monkeypatch.setattr(psutil, "disk_usage", disk_usage)
    return probed


# ---------------------------------------------------------------- vendor and integrated heuristics


@pytest.mark.parametrize(
    ("name", "vendor"),
    [
        ("NVIDIA GeForce RTX 4070", "nvidia"),
        ("AMD Radeon RX 9070 XT", "amd"),
        ("AMD Radeon(TM) Graphics", "amd"),
        ("AMD Radeon 780M", "amd"),
        ("Intel(R) UHD Graphics 770", "intel"),
        ("Intel(R) Arc(TM) A770", "intel"),
        ("Apple M3 Max", "apple"),
        ("ASPEED Graphics", "other"),
    ],
)
def test_classify_vendor(name: str, vendor: str) -> None:
    assert classify_vendor(name) == vendor


@pytest.mark.parametrize(
    ("name", "vram_gb", "vendor", "integrated"),
    [
        ("Apple M3 Max", 36.0, "apple", False),
        ("Intel(R) UHD Graphics 770", 0.1, "intel", True),
        ("Intel(R) Iris(R) Xe Graphics", 0.1, "intel", True),
        ("Intel(R) Arc(TM) A770", 16.0, "intel", False),
        ("AMD Radeon(TM) Graphics", 0.5, "amd", True),
        ("AMD Radeon Graphics", 0.5, "amd", True),
        ("AMD Radeon 780M", 0.5, "amd", True),
        ("AMD Radeon RX 9070 XT", 16.0, "amd", False),
        ("NVIDIA GeForce RTX 4070", 12.0, "nvidia", False),
    ],
)
def test_is_integrated(name: str, vram_gb: float, vendor: str, integrated: bool) -> None:
    assert is_integrated(name, vram_gb, vendor) is integrated


# ---------------------------------------------------------------- detect_hardware


def test_detect_nvidia_cuda_build_sorts_discrete_first(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(
        monkeypatch,
        fake_torch(
            cuda="12.8",
            devices=[
                gpu_props("AMD Radeon(TM) Graphics", 24.0, sms=2),
                gpu_props("NVIDIA GeForce RTX 4070", 12.0),
            ],
        ),
        best="cuda:1",
    )
    fake_psutil(monkeypatch, ram_gb=32.0)
    hw = detect_hardware()
    assert hw.os == "windows"
    assert hw.arch == (
        "x64" if platform.machine().lower() in ("amd64", "x86_64") else platform.machine().lower()
    )
    assert hw.torch_build == "cuda"
    assert hw.best_device == "cuda:1"
    assert [(g.name, g.vendor, g.backend, g.vram_gb, g.integrated, g.device) for g in hw.gpus] == [
        ("NVIDIA GeForce RTX 4070", "nvidia", "cuda", 12.0, False, "cuda:1"),
        (
            "AMD Radeon(TM) Graphics",
            "amd",
            "cuda",
            24.0,
            True,
            "cuda:0",
        ),  # integrated last, 24 GB notwithstanding
    ]
    assert hw.ram_gb == 32.0
    assert (hw.cpu_cores_physical, hw.cpu_cores_logical) == (8, 16)
    assert hw.disk_free_gb == 100.0


def test_detect_rocm_build_with_amd_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(
        monkeypatch,
        fake_torch(
            hip="6.4.0",
            devices=[
                gpu_props("AMD Radeon(TM) Graphics", 1.0, sms=1),
                gpu_props("AMD Radeon RX 9070 XT", 16.0),
            ],
        ),
        best="cuda:1",
    )
    fake_psutil(monkeypatch)
    hw = detect_hardware()
    assert hw.torch_build == "rocm"
    assert [g.name for g in hw.gpus] == ["AMD Radeon RX 9070 XT", "AMD Radeon(TM) Graphics"]  # discrete first
    assert all(g.backend == "rocm" and g.vendor == "amd" for g in hw.gpus)
    assert [g.integrated for g in hw.gpus] == [False, True]


def test_detect_mps_only(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(monkeypatch, fake_torch(mps=True), best="mps")
    fake_psutil(monkeypatch, ram_gb=36.0, physical=10, logical_count=10)
    hw = detect_hardware()
    assert hw.torch_build == "mps"
    assert hw.best_device == "mps"
    assert [(g.name, g.vendor, g.backend, g.vram_gb, g.integrated, g.device) for g in hw.gpus] == [
        ("Apple GPU", "apple", "mps", 36.0, False, "mps")
    ]
    assert hw.cpu_cores_physical == hw.cpu_cores_logical == 10


def test_detect_xpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(
        monkeypatch,
        fake_torch(xpu_devices=[gpu_props("Intel(R) Arc(TM) A770", 16.0)]),
        best="cpu",
    )
    fake_psutil(monkeypatch)
    hw = detect_hardware()
    assert hw.torch_build == "cpu"  # not one of cuda/rocm/mps
    assert [(g.name, g.backend, g.device) for g in hw.gpus] == [("Intel(R) Arc(TM) A770", "xpu", "xpu:0")]


def test_detect_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(monkeypatch, fake_torch())
    fake_psutil(monkeypatch)
    hw = detect_hardware()
    assert hw.torch_build == "cpu"
    assert hw.gpus == ()
    assert hw.best_device == "cpu"


def test_detect_broken_torch_and_device_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    monkeypatch.setitem(sys.modules, "omniscan.gpu.device", None)  # the picker cannot import either
    fake_psutil(monkeypatch)
    hw = detect_hardware()
    assert hw.gpus == ()
    assert hw.torch_build == "cpu"
    assert hw.best_device == "cpu"


def test_best_device_failure_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(
        monkeypatch, fake_torch(cuda="12.8", devices=[gpu_props("NVIDIA A100", 40.0)]), resolve_fails=True
    )
    fake_psutil(monkeypatch)
    hw = detect_hardware()
    assert hw.best_device == "cpu"
    assert hw.gpus != ()  # the GPUs are still listed even though auto-picking failed


def test_disk_probes_the_nearest_existing_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_fake_torch(monkeypatch, fake_torch())
    probed = fake_psutil(monkeypatch)
    detect_hardware(tmp_path / "models")  # the models dir does not exist yet
    assert probed == [tmp_path]
    probed.clear()
    detect_hardware(tmp_path)  # an existing models_dir is probed directly
    assert probed == [tmp_path]
