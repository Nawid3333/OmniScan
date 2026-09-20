"""Tests for omniscan.hw.assess (card H1 part 3): every rule of the compatibility check."""

from __future__ import annotations

from typing import Any

import pytest

from omniscan.hw.assess import assess
from omniscan.hw.detect import GpuInfo, HardwareInfo, classify_vendor
from omniscan.models.catalog import ModelEntry


def gpu(
    name: str = "NVIDIA GeForce RTX 4070",
    gib: float = 12.0,
    index: int = 0,
    integrated: bool = False,
    backend: str = "cuda",
    device: str | None = None,
) -> GpuInfo:
    vendor = classify_vendor(name)
    return GpuInfo(
        index=index,
        name=name,
        vendor=vendor,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        vram_gb=gib,
        integrated=integrated,
        device=device or f"{backend}:{index}",
    )


def hw(
    gpus: tuple[GpuInfo, ...] = (),
    ram_gb: float = 32.0,
    disk_free_gb: float = 500.0,
    torch_build: str = "cuda",
) -> HardwareInfo:
    return HardwareInfo(
        os="windows",
        arch="x64",
        cpu_name="AMD Ryzen 9",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_gb=ram_gb,
        gpus=gpus,
        torch_build=torch_build,  # type: ignore[arg-type]
        best_device=gpus[0].device if gpus else "cpu",
        onnxruntime_providers=(),
        disk_free_gb=disk_free_gb,
    )


def entry(**kwargs: Any) -> ModelEntry:
    base: dict[str, Any] = dict(
        id="m",
        name="M",
        kind="vision",
        format="zip",
        size_mb=100,
        license="Apache-2.0",
        description="d",
    )
    base.update(kwargs)
    return ModelEntry(**base)


# ---------------------------------------------------------------- rules


def test_cloud_is_always_ok_with_its_notes() -> None:
    result = assess(entry(format="cloud", ollama_name="m:1b-cloud", notes="needs internet"), hw())
    assert (result.level, result.device) == ("ok", None)
    assert result.messages == ("needs internet",)
    assert assess(entry(format="cloud", ollama_name="m:1b-cloud", notes=""), hw()).messages == ()


def test_gpu_with_enough_vram_is_ok_on_its_device() -> None:
    machine = hw(gpus=(gpu("NVIDIA GeForce RTX 4070", 12.0, index=1),))
    result = assess(entry(min_vram_gb=9.0), machine)
    assert (result.level, result.device, result.messages) == ("ok", "cuda:1", ())


def test_vram_too_small_falls_back_to_cpu_with_both_messages() -> None:
    machine = hw(gpus=(gpu("NVIDIA GeForce RTX 4070", 4.0, index=1),))
    result = assess(entry(min_vram_gb=9.0, cpu_speed="slow"), machine)
    assert result.level == "warn"
    assert result.device == "cpu"
    assert result.messages == (
        "needs 9 GB of GPU memory, your NVIDIA GeForce RTX 4070 has 4 GB",
        "runs on the CPU (slow)",
    )


def test_backend_not_allowed_is_incompatible() -> None:
    machine = hw(gpus=(gpu("Apple GPU", 36.0, backend="mps", device="mps"),), torch_build="mps")
    result = assess(entry(backends=["cuda"]), machine)
    assert (result.level, result.device) == ("incompatible", None)
    assert result.messages == ("needs a supported GPU (cuda/rocm/mps); none found",)


def test_cpu_not_ok_without_gpu_is_incompatible() -> None:
    result = assess(entry(cpu_ok=False), hw())
    assert (result.level, result.device) == ("incompatible", None)
    assert result.messages == ("needs a supported GPU (cuda/rocm/mps); none found",)


def test_vram_message_is_kept_when_the_cpu_path_refuses() -> None:
    machine = hw(gpus=(gpu("NVIDIA GeForce RTX 4070", 4.0, index=1),))
    result = assess(entry(min_vram_gb=9.0, cpu_ok=False), machine)
    assert result.level == "incompatible"
    assert result.messages == ("needs 9 GB of GPU memory, your NVIDIA GeForce RTX 4070 has 4 GB",)


@pytest.mark.parametrize(
    ("cpu_speed", "level", "message"),
    [
        ("fast", "ok", "runs on the CPU"),
        ("ok", "slow", "runs on the CPU (slow)"),
        ("slow", "warn", "runs on the CPU (slow)"),
        ("unusable", "incompatible", "runs on the CPU"),
    ],
)
def test_cpu_speed_mapping(cpu_speed: str, level: str, message: str) -> None:
    result = assess(entry(cpu_speed=cpu_speed), hw())
    assert result.level == level
    assert result.device == (None if level == "incompatible" else "cpu")
    assert result.messages == (message,)


def test_ram_too_small_is_incompatible() -> None:
    result = assess(entry(min_ram_gb=16.0, cpu_speed="fast"), hw(ram_gb=8.0))
    assert (result.level, result.device) == ("incompatible", None)
    assert result.messages == ("needs 16 GB of RAM, you have 8 GB",)


def test_only_an_integrated_gpu_is_used_with_a_message() -> None:
    machine = hw(gpus=(gpu("AMD Radeon(TM) Graphics", 1.0, index=0, integrated=True),))
    result = assess(entry(min_vram_gb=1.0), machine)
    assert result.level == "ok"
    assert result.device == "cuda:0"
    assert result.messages == ("only an integrated GPU (AMD Radeon(TM) Graphics) is available",)


def test_integrated_gpu_skipped_when_a_discrete_one_exists() -> None:
    machine = hw(
        gpus=(
            gpu("AMD Radeon(TM) Graphics", 1.0, index=0, integrated=True),
            gpu("AMD Radeon RX 9070 XT", 16.0, index=1),
        )
    )
    result = assess(entry(min_vram_gb=1.0), machine)
    assert (result.level, result.device, result.messages) == ("ok", "cuda:1", ())


def test_backend_filter_picks_a_matching_gpu() -> None:
    machine = hw(
        gpus=(
            gpu("AMD Radeon(TM) Graphics", 1.0, index=0, integrated=True, backend="rocm"),
            gpu("Apple GPU", 36.0, index=0, backend="mps", device="mps"),
        )
    )
    result = assess(entry(backends=["mps"]), machine)
    assert (result.level, result.device, result.messages) == ("ok", "mps", ())


def test_ollama_model_reports_the_ollama_device() -> None:
    machine = hw(gpus=(gpu("AMD Radeon RX 9070 XT", 16.0, index=1),))
    result = assess(entry(format="ollama", ollama_name="gemma4:12b", min_vram_gb=8.5), machine)
    assert (result.level, result.device, result.messages) == ("ok", "ollama", ())


# ---------------------------------------------------------------- disk warning


def test_disk_warning_raises_ok_to_warn() -> None:
    result = assess(entry(size_mb=20000, min_vram_gb=0.0), hw(disk_free_gb=10.0, gpus=(gpu(),)))
    assert result.level == "warn"
    assert result.device == "cuda:0"
    assert result.messages == ("needs about 23.4 GB of free disk space, 10 GB free",)


def test_disk_warning_keeps_incompatible_and_warn_levels() -> None:
    machine = hw(disk_free_gb=10.0)
    assert assess(entry(size_mb=20000, cpu_speed="unusable"), machine).level == "incompatible"
    assert assess(entry(size_mb=20000, cpu_speed="slow"), machine).level == "warn"
    assert assess(entry(size_mb=100, format="cloud", ollama_name="c:1b-cloud"), machine).messages == ()
