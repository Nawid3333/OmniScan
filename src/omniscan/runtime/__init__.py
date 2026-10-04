"""The GPU runtime of the packaged app (#44): the PyTorch build for this PC's graphics card, downloaded on the first
start instead of shipped in the installer.

The installer carries the CPU build of torch, so the app runs on every PC from the first start, and stays small.
`install` uses the `uv` that ships next to the programs to download the build the hardware asks for (cuda for
NVIDIA, rocm-gfx1201 for the RX 9070 series on Windows, xpu for Intel Arc / Core Ultra, mps on Apple Silicon) into a
per-user folder, at the torch version `uv.lock` pins for that backend, and marks it active. `activate` — the first
thing the packaged programs do — then makes `torch` and `torchvision` load from that folder instead of the bundled
CPU copy, and puts the folder on the import path for the packages they brought along (ROCm's SDK, Intel's
runtime). A checkout picks its torch with `uv sync --extra <backend>` and never activates a runtime.
"""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the launchers import this module before anything may import torch: keep it light
    from omniscan.hw.detect import HardwareInfo
    from omniscan.hw.tune import TorchExtra

ACTIVE_FILE = "active"  # in the runtime root: the folder name of the runtime in use
OVERRIDDEN = ("torch", "torchvision", "torchgen", "functorch")  # served from the runtime even though bundled
PYPI = "https://pypi.org/simple"
# where each backend's torch comes from (pyproject.toml's [tool.uv.index] entries); None: PyPI itself
INDEXES: dict[TorchExtra, str | None] = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu129",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "mps": None,
    "rocm-gfx1201": "https://stable.repo.amd.com/rocm/whl-next/",
}
# where an OS needs another index than uv.lock's: cu129 has no Windows wheels of the locked torch, cu130 has them
# (same version; it needs an NVIDIA driver from the 580 series on)
OS_INDEXES: dict[tuple[str, TorchExtra], str] = {("win32", "cuda"): "https://download.pytorch.org/whl/cu130"}
# the torch and torchvision each backend installs: the versions uv.lock pins for its extra (a test keeps them
# equal), so a runtime is the build the project tests with. The mps wheels are PyPI's macOS wheels, the same
# release as the CPU index's.
VERSIONS: dict[TorchExtra, tuple[str, str]] = {
    "cpu": ("2.14.0", "0.29.0"),
    "cuda": ("2.13.0", "0.28.0"),
    "xpu": ("2.14.0", "0.29.0"),
    "mps": ("2.14.0", "0.29.0"),
    "rocm-gfx1201": ("2.13.0", "0.28.0"),
}
_PLATFORMS = {  # uv's --python-platform per (sys.platform, machine)
    ("win32", "amd64"): "x86_64-pc-windows-msvc",
    ("linux", "x86_64"): "x86_64-manylinux_2_28",
    ("linux", "aarch64"): "aarch64-manylinux_2_28",
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
}


class RuntimeSetupError(RuntimeError):
    """A runtime cannot be installed or used here (no uv, an unknown backend, a failed download)."""


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The torch build this PC should run and why."""

    backend: TorchExtra
    gpu: str | None  # the card it is for, None on a CPU-only PC
    note: str | None  # why a card present cannot be used, when it cannot


def runtime_root() -> Path:
    """Where runtimes live: `$OMNISCAN_RUNTIME_DIR`, else the per-user data folder of this OS."""
    override = os.environ.get("OMNISCAN_RUNTIME_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "OmniScan"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "OmniScan"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "omniscan"
    return base / "runtime"


def recommend(hw: HardwareInfo) -> Recommendation:
    """The backend for this PC: the GPU torch already sees, else the cards the OS reports (the packaged app's CPU
    torch sees none)."""
    from omniscan.hw.os_gpus import os_gpus
    from omniscan.hw.tune import main_gpu, torch_extra_for

    gpu = main_gpu(hw)
    if gpu is None:
        listed = [card for card in os_gpus() if not card.integrated]
        gpu = listed[0] if listed else None
    backend, note = torch_extra_for(hw, gpu)
    return Recommendation(backend=backend, gpu=gpu.name if gpu is not None else None, note=note)


def requirements(backend: TorchExtra) -> list[str]:
    """What to install for `backend`: torch and torchvision at the versions uv.lock pins for it (without the local
    `+cu129` tag, which the backend's index adds)."""
    if backend not in VERSIONS:
        raise RuntimeSetupError(f"unknown backend {backend!r} (one of {', '.join(VERSIONS)})")
    torch, torchvision = VERSIONS[backend]
    extra = "[device-gfx1201]" if backend == "rocm-gfx1201" else ""
    return [f"torch{extra}=={torch}", f"torchvision{extra}=={torchvision}"]


def find_uv() -> Path | None:
    """The uv to download with: `$OMNISCAN_UV`, the one next to the packaged programs, else uv on PATH."""
    override = os.environ.get("OMNISCAN_UV")
    if override:
        return Path(override)
    name = "uv.exe" if sys.platform == "win32" else "uv"
    bundled = Path(sys.executable).parent / name
    if getattr(sys, "frozen", False) and bundled.is_file():
        return bundled
    found = shutil.which("uv")
    return Path(found) if found else None


def install_command(uv: Path, backend: TorchExtra, target: Path) -> list[str]:
    """The uv command that downloads `backend`'s torch into `target` for this app's Python and platform."""
    key = (sys.platform, platform.machine().lower())
    if key not in _PLATFORMS:
        raise RuntimeSetupError(f"no runtime downloads for {key[0]} on {key[1]}")
    command = [
        str(uv),
        "pip",
        "install",
        "--target",
        str(target),
        "--no-config",  # no uv.toml / pyproject.toml of the current folder or the user may change the indexes
        "--python-version",
        f"{sys.version_info.major}.{sys.version_info.minor}",
        "--python-platform",
        _PLATFORMS[key],
    ]
    index = OS_INDEXES.get((sys.platform, backend), INDEXES[backend])
    if index is not None:  # an extra index comes first: torch from the backend's index, the rest from PyPI
        command += ["--index-url", PYPI, "--extra-index-url", index]
    return [*command, *requirements(backend)]


def install_env(*, system: str | None = None, mac_version: str | None = None) -> dict[str, str]:
    """The environment for the uv command. On macOS it targets this Mac's version: uv assumes macOS 13 otherwise,
    and the torch wheels need 14."""
    env = dict(os.environ)
    if (system or sys.platform) == "darwin":
        release = (mac_version if mac_version is not None else platform.mac_ver()[0]).split(".")
        if release[0]:
            env["MACOSX_DEPLOYMENT_TARGET"] = ".".join([*release, "0"][:2])
    return env


def folder_name(backend: TorchExtra) -> str:
    """The runtime's folder name: backend and torch version (an app pinning another version brings its own)."""
    if backend not in VERSIONS:
        raise RuntimeSetupError(f"unknown backend {backend!r} (one of {', '.join(VERSIONS)})")
    return f"{backend}-torch{VERSIONS[backend][0]}"


Runner = Callable[
    [list[str], Mapping[str, str]], int
]  # runs a command in an environment, returns its exit code


def install(backend: TorchExtra, *, root: Path | None = None, run: Runner | None = None) -> Path:
    """Download `backend`'s torch into the runtime folder and make it the active runtime; returns the folder.
    `run` runs the uv command and returns its exit code (default: show its output). RuntimeSetupError when uv is
    missing or the download fails (nothing is left half-installed)."""
    uv = find_uv()
    if uv is None:
        raise RuntimeSetupError("uv was not found: the packaged app ships it next to its programs")
    root = root or runtime_root()
    target = root / folder_name(backend)
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    command = install_command(uv, backend, partial)
    code = (run or _show)(command, install_env())
    if code != 0:
        shutil.rmtree(partial, ignore_errors=True)
        raise RuntimeSetupError(f"downloading the {backend} runtime failed (uv exit code {code})")
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    (root / ACTIVE_FILE).write_text(target.name, encoding="utf-8")
    return target


def _show(command: list[str], env: Mapping[str, str]) -> int:
    """Run `command` in `env` with its output on this console; its exit code."""
    return subprocess.run(command, env=env, check=False).returncode


def use(name: str | None, *, root: Path | None = None) -> None:
    """Make the runtime folder `name` the active one (None: the bundled CPU torch). FileNotFoundError when it is
    not installed."""
    root = root or runtime_root()
    marker = root / ACTIVE_FILE
    if name is None:
        marker.unlink(missing_ok=True)
        return
    if name not in installed(root):
        raise FileNotFoundError(f"no runtime {name!r} in {root}")
    marker.write_text(name, encoding="utf-8")


def remove(name: str, *, root: Path | None = None) -> None:
    """Delete an installed runtime; the active one first falls back to the bundled torch. FileNotFoundError when it
    is not installed."""
    root = root or runtime_root()
    if name not in installed(root):  # only a folder the listing shows: never "..", a path or anything else
        raise FileNotFoundError(f"no runtime {name!r} in {root}")
    current = active(root)
    if current is not None and current.name == name:
        use(None, root=root)
    shutil.rmtree(root / name)


def installed(root: Path | None = None) -> list[str]:
    """The runtime folders installed, sorted."""
    root = root or runtime_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.endswith(".partial"))


def active(root: Path | None = None) -> Path | None:
    """The active runtime's folder, or None (the bundled torch)."""
    root = root or runtime_root()
    marker = root / ACTIVE_FILE
    if not marker.is_file():
        return None
    name = marker.read_text(encoding="utf-8").strip()
    return root / name if name in installed(root) else None


class _RuntimeFinder(importlib.metadata.DistributionFinder):
    """Serves the overridden packages (torch, torchvision, …) and their submodules from the runtime folder, ahead
    of the bundled copies, and their metadata too: `importlib.metadata.version("torch")` (which transformers
    checks) reports the torch that runs, not the bundled one."""

    def __init__(self, folder: Path, names: frozenset[str]) -> None:
        self.folder = folder
        self.names = names

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        """The runtime's spec for an overridden module, else None (the other finders decide)."""
        if fullname.partition(".")[0] not in self.names:
            return None
        search = [str(self.folder)] if "." not in fullname or path is None else list(path)
        return importlib.machinery.PathFinder.find_spec(fullname, search, target)

    def find_distributions(
        self, context: importlib.metadata.DistributionFinder.Context | None = None
    ) -> Iterable[importlib.metadata.Distribution]:
        """The runtime's metadata for an overridden package, else none (the other finders decide)."""
        name = context.name if context is not None else None
        if name is None or name.replace("-", "_").lower() not in self.names:
            return ()
        here = importlib.metadata.DistributionFinder.Context(name=name, path=[str(self.folder)])
        return importlib.metadata.MetadataPathFinder.find_distributions(here)


def activate(root: Path | None = None) -> Path | None:
    """Load torch and torchvision from the active runtime from now on (call it before anything imports torch);
    returns its folder, or None when there is none or torch is already imported."""
    folder = active(root)
    if folder is None or "torch" in sys.modules:
        return None
    names = frozenset(name for name in OVERRIDDEN if (folder / name).is_dir())
    sys.meta_path.insert(0, _RuntimeFinder(folder, names))
    sys.path.append(str(folder))  # the packages torch brought along, and the dist-info of all of them
    return folder
