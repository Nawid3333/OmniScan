"""The packaged app's GPU runtime (runtime/) and the torch-free GPU detection it relies on (hw/os_gpus.py), #44."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omniscan import runtime
from omniscan.hw.detect import GpuInfo, HardwareInfo
from omniscan.hw.os_gpus import os_gpus
from omniscan.runtime.cli import runtime_app


def _hw(os_name: str = "windows", gpus: tuple[GpuInfo, ...] = ()) -> HardwareInfo:
    return HardwareInfo(
        os=os_name,
        arch="x64",
        cpu_name="CPU",
        cpu_cores_physical=6,
        cpu_cores_logical=12,
        ram_gb=32.0,
        gpus=gpus,
        torch_build="cpu",
        best_device="cpu",
        onnxruntime_providers=(),
        disk_free_gb=100.0,
    )


# ---------------------------------------------------------------- the OS's graphics cards


def test_cards_come_from_nvidia_smi_the_windows_drivers_or_lspci() -> None:
    def windows(command: list[str]) -> str:
        if command[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 4070, 12282\n"
        rows = [
            {"name": "AMD Radeon(TM) Graphics", "ram": 536870912},
            {"name": "NVIDIA GeForce RTX 4070", "ram": 0},
        ]
        return json.dumps(rows)

    gpus = os_gpus(windows, system="win32", machine="AMD64")
    assert [(g.name, g.backend, g.vram_gb, g.integrated) for g in gpus] == [
        ("NVIDIA GeForce RTX 4070", "cuda", 12.0, False),  # nvidia-smi's memory wins over the driver's 0
        ("AMD Radeon(TM) Graphics", "rocm", 0.5, True),
    ]
    one = os_gpus(
        lambda c: (
            "" if c[0] == "nvidia-smi" else json.dumps({"name": "AMD Radeon RX 9070 XT", "ram": 17095983104})
        ),
        system="win32",
    )
    assert [(g.name, g.vram_gb) for g in one] == [("AMD Radeon RX 9070 XT", 15.9)]

    lspci = "00:02.0 VGA compatible controller: Intel Corporation Arc A770 Graphics\n00:1f.3 Audio device: Intel\n"
    assert [
        (g.name, g.backend) for g in os_gpus(lambda c: lspci if c[0] == "lspci" else "", system="linux")
    ] == [("Intel Corporation Arc A770 Graphics", "xpu")]
    mac = os_gpus(lambda c: "", system="darwin", machine="arm64")
    assert [(g.name, g.backend, g.device) for g in mac] == [("Apple GPU", "mps", "mps")]
    assert os_gpus(lambda c: "", system="darwin", machine="x86_64") == []
    assert os_gpus(lambda c: "not json", system="win32") == []


def test_the_recommendation_uses_the_os_cards_when_torch_sees_none(monkeypatch: pytest.MonkeyPatch) -> None:
    card = GpuInfo(
        index=0,
        name="AMD Radeon RX 9070 XT",
        vendor="amd",
        backend="rocm",
        vram_gb=15.9,
        integrated=False,
        device="cpu",
    )
    monkeypatch.setattr("omniscan.hw.os_gpus.os_gpus", lambda: [card])
    assert runtime.recommend(_hw("windows")) == runtime.Recommendation(
        "rocm-gfx1201", "AMD Radeon RX 9070 XT", None
    )
    linux = runtime.recommend(_hw("linux"))
    assert linux.backend == "cpu" and linux.note is not None and "no extra" in linux.note
    monkeypatch.setattr("omniscan.hw.os_gpus.os_gpus", lambda: [])
    assert runtime.recommend(_hw()) == runtime.Recommendation("cpu", None, None)


# ---------------------------------------------------------------- installing and choosing runtimes


def test_the_download_asks_for_this_apps_torch_from_the_backends_index(tmp_path: Path) -> None:
    import importlib.metadata

    torch_version = importlib.metadata.version("torch").split("+")[0]
    assert runtime.requirements("cuda")[0] == f"torch=={torch_version}"
    assert runtime.requirements("rocm-gfx1201")[0] == f"torch[device-gfx1201]=={torch_version}"
    with pytest.raises(runtime.RuntimeSetupError, match="unknown backend"):
        runtime.requirements("tpu")  # type: ignore[arg-type]
    command = runtime.install_command(Path("uv"), "cuda", tmp_path / "t")
    assert command[:5] == ["uv", "pip", "install", "--target", str(tmp_path / "t")]
    assert (
        command[command.index("--python-version") + 1] == f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    assert command[command.index("--index-url") + 1] == "https://download.pytorch.org/whl/cu129"
    assert command[command.index("--extra-index-url") + 1] == runtime.PYPI
    assert "--index-url" not in runtime.install_command(Path("uv"), "mps", tmp_path / "t")  # PyPI itself


def test_install_use_remove(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "find_uv", lambda: Path("uv"))
    commands: list[list[str]] = []

    def download(command: list[str]) -> int:
        commands.append(command)
        target = Path(command[command.index("--target") + 1])
        (target / "torch").mkdir(parents=True)
        return 0

    folder = runtime.install("cpu", root=tmp_path, run=download)
    assert folder.name == runtime.folder_name("cpu") and (folder / "torch").is_dir()
    assert runtime.active(tmp_path) == folder and runtime.installed(tmp_path) == [folder.name]
    assert commands[0][commands[0].index("--target") + 1].endswith(".partial")

    with pytest.raises(runtime.RuntimeSetupError, match="exit code 3"):
        runtime.install("cuda", root=tmp_path, run=lambda command: 3)
    assert runtime.installed(tmp_path) == [folder.name]  # nothing half-installed

    runtime.use(None, root=tmp_path)
    assert runtime.active(tmp_path) is None
    runtime.use(folder.name, root=tmp_path)
    with pytest.raises(FileNotFoundError):
        runtime.use("cuda-torch9", root=tmp_path)
    runtime.remove(folder.name, root=tmp_path)
    assert runtime.active(tmp_path) is None and runtime.installed(tmp_path) == []
    for outside in ("..", ".", "", str(tmp_path), "../runtime"):  # never anything but an installed runtime
        with pytest.raises(FileNotFoundError):
            runtime.remove(outside, root=tmp_path)
        with pytest.raises(FileNotFoundError):
            runtime.use(outside, root=tmp_path)
    (tmp_path / runtime.ACTIVE_FILE).write_text(
        "..", encoding="utf-8"
    )  # a marker pointing outside is ignored
    assert runtime.active(tmp_path) is None and tmp_path.is_dir()

    monkeypatch.setattr(runtime, "find_uv", lambda: None)
    with pytest.raises(runtime.RuntimeSetupError, match="uv was not found"):
        runtime.install("cpu", root=tmp_path, run=download)


def test_an_active_runtime_wins_over_the_bundled_torch(tmp_path: Path) -> None:
    """In a fresh interpreter: a "bundled" torch first on the path, the runtime's own one served instead."""
    bundled = tmp_path / "bundle" / "torch"
    bundled.mkdir(parents=True)
    (bundled / "__init__.py").write_text('WHERE = "bundled"\n', encoding="utf-8")
    root = tmp_path / "runtime"
    served = root / "cuda-torch2" / "torch"
    (served / "cuda").mkdir(parents=True)
    (served / "__init__.py").write_text('WHERE = "runtime"\n', encoding="utf-8")
    (served / "cuda" / "__init__.py").write_text('WHERE = "runtime cuda"\n', encoding="utf-8")
    (root / "cuda-torch2" / "helper_pkg.py").write_text('WHERE = "brought along"\n', encoding="utf-8")
    (root / runtime.ACTIVE_FILE).write_text("cuda-torch2", encoding="utf-8")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path / "bundle")!r})
        from omniscan.runtime import activate
        from pathlib import Path
        print(activate(Path({str(root)!r})).name)
        import torch, torch.cuda, helper_pkg
        print(torch.WHERE, "|", torch.cuda.WHERE, "|", helper_pkg.WHERE)
        """
    )
    src = str(Path(runtime.__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": src, "SYSTEMROOT": "C:\\Windows"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["cuda-torch2", "runtime | runtime cuda | brought along"]


def test_the_install_command_is_for_the_packaged_app(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(runtime_app, ["install", "--backend", "cuda"])
    assert result.exit_code == 2 and "uv sync --extra" in result.output
    monkeypatch.setattr(runtime, "install", lambda backend: Path("/runtimes") / f"{backend}-torch2")
    forced = CliRunner().invoke(runtime_app, ["install", "--backend", "xpu", "--force"])
    assert forced.exit_code == 0 and "installed xpu-torch2" in forced.output
    bad = CliRunner().invoke(runtime_app, ["install", "--backend", "tpu", "--force"])
    assert bad.exit_code == 2 and "unknown backend" in bad.output
