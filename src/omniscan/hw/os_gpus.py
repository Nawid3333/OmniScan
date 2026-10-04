"""The graphics cards the operating system reports, without torch (#44).

The packaged app starts with the CPU build of torch, which cannot see any GPU, so `hw.detect` finds none. To pick
the GPU runtime to download (`omniscan runtime install`), the cards are read from the OS instead:

- NVIDIA: `nvidia-smi` (name and memory), on Windows and Linux;
- Windows: the display drivers' entries in the registry (names and the real memory size,
  `HardwareInformation.qwMemorySize`), else the adapters (`Win32_VideoController`), through PowerShell;
- Linux: `lspci` (names only; memory unknown);
- macOS on Apple Silicon: the unified GPU (Metal), with the machine's memory.

Each command has a short timeout and any failure only means fewer cards: this never stops the app.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable

import psutil

from omniscan.hw.detect import GIB, Backend, GpuInfo, classify_vendor, is_integrated

Runner = Callable[[list[str]], str]  # runs a command, returns its stdout ("" when it fails)

_TIMEOUT_S = 10
_BACKENDS: dict[str, Backend] = {"nvidia": "cuda", "amd": "rocm", "intel": "xpu", "apple": "mps"}
# The display adapters' driver entries: their names and the real memory size (Win32_VideoController.AdapterRAM
# stops at 4 GB); the adapters themselves when no driver entry carries a name.
_WINDOWS_QUERY = (
    r"$cls = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'; "
    "$found = @(Get-ChildItem $cls -ErrorAction SilentlyContinue | ForEach-Object { "
    "$p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; "
    "if ($p.DriverDesc) { [pscustomobject]@{ name = $p.DriverDesc; ram = $p.'HardwareInformation.qwMemorySize' } } }); "
    "if (-not $found) { $found = @(Get-CimInstance Win32_VideoController | ForEach-Object { "
    "[pscustomobject]@{ name = $_.Name; ram = $_.AdapterRAM } }) }; "
    "$found | ConvertTo-Json -Compress"
)


def _run(command: list[str]) -> str:
    """stdout of `command`, or "" when it is missing, fails or takes too long."""
    if shutil.which(command[0]) is None:
        return ""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False)
    except OSError, subprocess.SubprocessError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def os_gpus(run: Runner = _run, *, system: str | None = None, machine: str | None = None) -> list[GpuInfo]:
    """The graphics cards the OS reports, discrete ones first (`run`, `system` and `machine` are for tests)."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            ram = psutil.virtual_memory().total / GIB
            return [_gpu(0, "Apple GPU", round(ram, 1), "mps")]
        return []
    found: dict[str, float] = {}  # name -> memory in GiB (0.0 when unknown)
    for line in run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    ).splitlines():
        name, _, mib = line.partition(",")
        if name.strip():
            found[name.strip()] = round(float(mib or 0) / 1024, 1) if mib.strip().isdigit() else 0.0
    if system == "win32":
        for name, ram in _windows_adapters(run(["powershell", "-NoProfile", "-Command", _WINDOWS_QUERY])):
            found.setdefault(name, ram)
    elif system.startswith("linux"):
        for name in _lspci_names(run(["lspci"])):
            if not any(name in known or known in name for known in found):
                found[name] = 0.0
    gpus = [
        _gpu(index, name, ram, None)
        for index, (name, ram) in enumerate(found.items())
        if classify_vendor(name) != "other"
    ]
    return sorted(gpus, key=lambda gpu: (gpu.integrated, -gpu.vram_gb))


def _gpu(index: int, name: str, vram_gb: float, device: str | None) -> GpuInfo:
    """A GpuInfo for an OS-reported card (its torch device is unknown until the matching runtime runs)."""
    vendor = classify_vendor(name)
    return GpuInfo(
        index=index,
        name=name,
        vendor=vendor,
        backend=_BACKENDS.get(vendor, "cpu"),
        vram_gb=vram_gb,
        integrated=is_integrated(name, vram_gb, vendor),
        device=device or "cpu",
    )


def _windows_adapters(text: str) -> list[tuple[str, float]]:
    """(name, GiB) of every display adapter in the PowerShell JSON (one object or a list)."""
    try:
        data = json.loads(text) if text.strip() else []
    except ValueError:
        return []
    items = data if isinstance(data, list) else [data]
    adapters: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            ram = item.get("ram")
            adapters.append(
                (item["name"].strip(), round(ram / GIB, 1) if isinstance(ram, int) and ram > 0 else 0.0)
            )
    return adapters


def _lspci_names(text: str) -> list[str]:
    """The device names of the VGA, 3D and display controllers in `lspci` output."""
    names: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if "vga compatible controller" in lower or "3d controller" in lower or "display controller" in lower:
            names.append(line.split(": ", 1)[-1].strip())
    return names
