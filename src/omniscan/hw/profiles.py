"""Named machine profiles: the hardware snapshots OmniScan is tuned and tested against without owning the machines.

Each profile is a `HardwareInfo` as `hw.detect` would produce it on that PC — the same object the tuning plan,
the model fit check, the doctor and the settings screen read. `omniscan hardware --simulate <name>` and the
`OMNISCAN_SIMULATE_HARDWARE` environment variable swap the real snapshot for a profile, so the planning side of
the program (never the tensor side: torch still runs on whatever device is really there) can be exercised on
any PC and in CI for every GPU vendor.
"""

from __future__ import annotations

import os

from omniscan.hw.detect import GpuInfo, HardwareInfo

SIMULATE_ENV = "OMNISCAN_SIMULATE_HARDWARE"


def _gpu(
    index: int, name: str, vendor: str, backend: str, vram_gb: float, *, integrated: bool = False
) -> GpuInfo:
    """A GPU entry with the torch device string the backend uses."""
    device = "mps" if backend == "mps" else f"{'xpu' if backend == 'xpu' else 'cuda'}:{index}"
    return GpuInfo(
        index=index,
        name=name,
        vendor=vendor,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        vram_gb=vram_gb,
        integrated=integrated,
        device=device,
    )


def _machine(
    name: str,
    *,
    os_name: str,
    arch: str = "x64",
    cpu: str,
    cores: tuple[int, int],
    ram_gb: float,
    gpus: tuple[GpuInfo, ...] = (),
    torch_build: str,
    best_device: str,
    disk_free_gb: float = 200.0,
) -> tuple[str, HardwareInfo]:
    return name, HardwareInfo(
        os=os_name,
        arch=arch,
        cpu_name=cpu,
        cpu_cores_physical=cores[0],
        cpu_cores_logical=cores[1],
        ram_gb=ram_gb,
        gpus=gpus,
        torch_build=torch_build,  # type: ignore[arg-type]
        best_device=best_device,
        onnxruntime_providers=(),
        disk_free_gb=disk_free_gb,
    )


PROFILES: dict[str, HardwareInfo] = dict(
    (
        _machine(
            "rtx4090",
            os_name="windows",
            cpu="AMD Ryzen 9 7950X 16-Core Processor",
            cores=(16, 32),
            ram_gb=64.0,
            gpus=(_gpu(0, "NVIDIA GeForce RTX 4090", "nvidia", "cuda", 24.0),),
            torch_build="cuda",
            best_device="cuda:0",
        ),
        _machine(
            "rtx3060",
            os_name="windows",
            cpu="Intel(R) Core(TM) i5-12400F",
            cores=(6, 12),
            ram_gb=32.0,
            gpus=(_gpu(0, "NVIDIA GeForce RTX 3060", "nvidia", "cuda", 12.0),),
            torch_build="cuda",
            best_device="cuda:0",
        ),
        _machine(
            "gtx1650-laptop",
            os_name="windows",
            cpu="Intel(R) Core(TM) i5-10300H",
            cores=(4, 8),
            ram_gb=16.0,
            gpus=(_gpu(0, "NVIDIA GeForce GTX 1650", "nvidia", "cuda", 4.0),),
            torch_build="cuda",
            best_device="cuda:0",
            disk_free_gb=40.0,
        ),
        _machine(
            "rx9070xt",
            os_name="windows",
            cpu="AMD Ryzen 5 7600X 6-Core Processor",
            cores=(6, 12),
            ram_gb=64.0,
            gpus=(
                _gpu(1, "AMD Radeon RX 9070 XT", "amd", "rocm", 16.0),
                _gpu(0, "AMD Radeon(TM) Graphics", "amd", "rocm", 1.0, integrated=True),
            ),
            torch_build="rocm",
            best_device="cuda:1",
        ),
        _machine(
            "rx7800xt-linux",
            os_name="linux",
            cpu="AMD Ryzen 7 7700X 8-Core Processor",
            cores=(8, 16),
            ram_gb=32.0,
            gpus=(_gpu(0, "AMD Radeon RX 7800 XT", "amd", "rocm", 16.0),),
            torch_build="rocm",
            best_device="cuda:0",
        ),
        _machine(
            "arc-b580",
            os_name="windows",
            cpu="Intel(R) Core(TM) i5-14600K",
            cores=(14, 20),
            ram_gb=32.0,
            gpus=(
                _gpu(1, "Intel(R) Arc(TM) B580 Graphics", "intel", "xpu", 12.0),
                _gpu(0, "Intel(R) UHD Graphics 770", "intel", "xpu", 8.0, integrated=True),
            ),
            torch_build="cpu",
            best_device="xpu:1",
        ),
        _machine(
            "m2-air",
            os_name="macos",
            arch="arm64",
            cpu="Apple M2",
            cores=(8, 8),
            ram_gb=16.0,
            gpus=(_gpu(0, "Apple GPU", "apple", "mps", 16.0),),
            torch_build="mps",
            best_device="mps",
        ),
        _machine(
            "m4-max",
            os_name="macos",
            arch="arm64",
            cpu="Apple M4 Max",
            cores=(16, 16),
            ram_gb=64.0,
            gpus=(_gpu(0, "Apple GPU", "apple", "mps", 64.0),),
            torch_build="mps",
            best_device="mps",
        ),
        _machine(
            "cpu-laptop",
            os_name="windows",
            cpu="Intel(R) Core(TM) i3-1115G4",
            cores=(2, 4),
            ram_gb=8.0,
            torch_build="cpu",
            best_device="cpu",
            disk_free_gb=25.0,
        ),
        _machine(
            "igpu-only",
            os_name="windows",
            cpu="AMD Ryzen 5 5600G with Radeon Graphics",
            cores=(6, 12),
            ram_gb=16.0,
            gpus=(_gpu(0, "AMD Radeon(TM) Graphics", "amd", "rocm", 2.0, integrated=True),),
            torch_build="rocm",
            best_device="cpu",
        ),
        _machine(
            "cpu-server-linux",
            os_name="linux",
            cpu="AMD EPYC 7543 32-Core Processor",
            cores=(32, 64),
            ram_gb=128.0,
            torch_build="cpu",
            best_device="cpu",
            disk_free_gb=800.0,
        ),
    )
)


def profile_names() -> list[str]:
    """Every profile name, in catalogue order."""
    return list(PROFILES)


def profile(name: str) -> HardwareInfo:
    """The named profile; ValueError (naming the known ones) for an unknown name."""
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown hardware profile {name!r} (known: {', '.join(PROFILES)})") from None


def simulated_profile() -> str | None:
    """The profile named by `OMNISCAN_SIMULATE_HARDWARE`, None when the variable is unset or empty."""
    return os.environ.get(SIMULATE_ENV) or None
