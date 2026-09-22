"""HardwareService tests (Qt-free): the report filters the models rows into per-model warnings."""

from __future__ import annotations

from omniscan.core.config import Config
from omniscan.gui.services.hardware import HardwareService
from omniscan.hw.detect import GpuInfo, HardwareInfo
from omniscan.models.rows import ModelRow


def _hw() -> HardwareInfo:
    """A small static snapshot: one AMD GPU via ROCm."""
    return HardwareInfo(
        os="windows",
        arch="x64",
        cpu_name="Test CPU",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_gb=32.0,
        gpus=(
            GpuInfo(
                index=0,
                name="Test GPU",
                vendor="amd",
                backend="rocm",
                vram_gb=16.0,
                integrated=False,
                device="cuda:0",
            ),
        ),
        torch_build="rocm",
        best_device="cuda:0",
        onnxruntime_providers=(),
        disk_free_gb=100.0,
    )


def _row(name: str, level: str, device: str | None, messages: tuple[str, ...]) -> ModelRow:
    """One catalog row with the fit fields that matter for the warning list."""
    return ModelRow(
        id=name,
        name=name,
        kind="weights",
        role=None,
        family=None,
        size_class=None,
        size_mb=1,
        required=False,
        format="safetensors",
        license="apache-2.0",
        description="",
        langs=("ko",),
        recommended_for=(),
        notes="",
        used_by=(),
        status="installed",
        installed_path=None,
        fit_level=level,
        fit_device=device,
        fit_messages=messages,
    )


def test_report_lists_only_models_that_do_not_fit() -> None:
    """Rows with fit level `ok` are left out; the others keep level, device and messages."""
    rows = [
        _row("good-model", "ok", "cuda:0", ()),
        _row("slow-model", "slow", "cuda:0", ("needs 24 GB",)),
        _row("broken-model", "incompatible", None, ("no ROCm build",)),
    ]
    service = HardwareService(Config(), rows=lambda: (rows, _hw()))

    report = service.report()
    assert report.info.best_device == "cuda:0"
    assert [(w.name, w.level, w.device, w.messages) for w in report.warnings] == [
        ("slow-model", "slow", "cuda:0", ("needs 24 GB",)),
        ("broken-model", "incompatible", None, ("no ROCm build",)),
    ]


def test_report_without_warnings_is_empty() -> None:
    """An all-ok machine reports no warnings."""
    service = HardwareService(Config(), rows=lambda: ([_row("good-model", "ok", "cuda:0", ())], _hw()))

    assert service.report().warnings == ()
