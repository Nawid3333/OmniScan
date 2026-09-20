"""Tests closing the gaps found by the mutation review (card Q5).

Each test kills a mutant that survived the existing suite; see `tests/mutants/hw_hf/`
and `docs/reports/Q5.md`. The tests are self-contained on purpose: they pin one behaviour each.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from omniscan.core.config import Config, PathsConfig, series_config
from omniscan.hw.assess import assess
from omniscan.hw.detect import (
    GpuInfo,
    HardwareInfo,
    _arch,
    _os_name,
    classify_vendor,
    detect_hardware,
)
from omniscan.models.catalog import ModelEntry
from omniscan.models.download import ModelDownloadError, download_model, verify_models
from omniscan.models.store import MARKER_NAME, model_status

HF_REVISION = "d" * 40


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


def gpu(name: str = "NVIDIA GeForce RTX 4070", gib: float = 12.0, index: int = 0) -> GpuInfo:
    return GpuInfo(
        index=index,
        name=name,
        vendor=classify_vendor(name),
        backend="cuda",
        vram_gb=gib,
        integrated=False,
        device=f"cuda:{index}",
    )


def hw(**kwargs: Any) -> HardwareInfo:
    base: dict[str, Any] = dict(
        os="windows",
        arch="x64",
        cpu_name="AMD Ryzen 9",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_gb=32.0,
        gpus=(),
        torch_build="cuda",
        best_device="cpu",
        onnxruntime_providers=(),
        disk_free_gb=500.0,
    )
    base.update(kwargs)
    return HardwareInfo(**base)


def hf_entry(**overrides: Any) -> ModelEntry:
    """A valid hf entry whose one catalog file hashes the b"weights" payload."""
    data: dict[str, Any] = {
        "id": "ocr-rec-x",
        "name": "X",
        "kind": "ocr",
        "format": "hf",
        "size_mb": 1,
        "license": "Apache-2.0",
        "description": "d",
        "upstream_repo": "org/rec",
        "upstream_revision": HF_REVISION,
        "files": {"model.safetensors": hashlib.sha256(b"weights").hexdigest()},
    }
    data.update(overrides)
    return ModelEntry(**data)


def _fake_psutil(monkeypatch: pytest.MonkeyPatch, *, ram_gb: float = 32.0) -> None:
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=int(ram_gb * 2**30)))
    monkeypatch.setattr(psutil, "cpu_count", lambda logical: 16)
    monkeypatch.setattr(psutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))


def _fake_torch_cuda(monkeypatch: pytest.MonkeyPatch, props: list[SimpleNamespace]) -> None:
    module = types.ModuleType("torch")
    module.version = SimpleNamespace(hip=None, cuda="12.8")  # type: ignore[attr-defined]
    module.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        device_count=lambda: len(props),
        get_device_properties=lambda i: props[i],
    )
    module.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)


def _fake_picker(monkeypatch: pytest.MonkeyPatch, best: str) -> None:
    def resolve_device(spec: object = "auto") -> str:
        return best

    picker = types.ModuleType("omniscan.gpu.device")
    picker.resolve_device = resolve_device  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omniscan.gpu.device", picker)


# ---------------------------------------------------------------- hw.assess boundaries


def test_ram_equal_to_min_is_not_incompatible() -> None:
    result = assess(entry(min_ram_gb=32.0, cpu_speed="fast"), hw(ram_gb=32.0))
    assert (result.level, result.device, result.messages) == ("ok", "cpu", ("runs on the CPU",))


def test_disk_free_exactly_the_needed_size_is_no_warning() -> None:
    machine = hw(disk_free_gb=1.2, gpus=(gpu(),))
    result = assess(entry(size_mb=1024, min_vram_gb=0.0), machine)
    assert (result.level, result.device, result.messages) == ("ok", "cuda:0", ())


def test_disk_warning_raises_a_slow_cpu_level_to_warn() -> None:
    result = assess(entry(size_mb=20000, cpu_speed="ok"), hw(disk_free_gb=10.0))
    assert result.level == "warn"
    assert result.messages == (
        "runs on the CPU (slow)",
        "needs about 23.4 GB of free disk space, 10 GB free",
    )


# ---------------------------------------------------------------- hw.detect details


def test_vendor_word_nvidia_alone_classifies_nvidia() -> None:
    assert classify_vendor("NVIDIA A100 80GB PCIe") == "nvidia"
    assert classify_vendor("NVIDIA Tesla T4") == "nvidia"


def test_vendor_word_amd_alone_classifies_amd() -> None:
    assert classify_vendor("AMD FirePro W7100") == "amd"


def test_ram_is_rounded_to_one_decimal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # no torch: no GPUs
    monkeypatch.setitem(sys.modules, "omniscan.gpu.device", None)
    _fake_psutil(monkeypatch, ram_gb=31.3)
    assert detect_hardware().ram_gb == 31.3


def test_gpu_vram_is_rounded_to_one_decimal(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch_cuda(
        monkeypatch, [SimpleNamespace(name="NVIDIA GeForce RTX 5080", total_memory=int(15.7 * 2**30))]
    )
    _fake_psutil(monkeypatch)
    _fake_picker(monkeypatch, "cuda:0")
    assert detect_hardware().gpus[0].vram_gb == 15.7


def test_discrete_gpus_are_sorted_largest_vram_first(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch_cuda(
        monkeypatch,
        [
            SimpleNamespace(name="NVIDIA GeForce RTX 4070", total_memory=12 * 2**30),
            SimpleNamespace(name="NVIDIA GeForce RTX 4090", total_memory=24 * 2**30),
        ],
    )
    _fake_psutil(monkeypatch)
    _fake_picker(monkeypatch, "cuda:1")
    info = detect_hardware()
    assert [(g.name, g.device) for g in info.gpus] == [
        ("NVIDIA GeForce RTX 4090", "cuda:1"),
        ("NVIDIA GeForce RTX 4070", "cuda:0"),
    ]


def test_os_name_maps_darwin_to_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _os_name() == "macos"
    monkeypatch.setattr(sys, "platform", "freebsd14")
    assert _os_name() == "freebsd14"  # unknown platforms pass through


def test_arch_maps_aarch64_to_arm64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert _arch() == "arm64"
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    assert _arch() == "riscv64"  # unknown machines pass through


# ---------------------------------------------------------------- store: hf marker shapes


def test_hf_marker_without_file_sizes_is_corrupt(tmp_path: Path) -> None:
    entry = hf_entry()
    folder = tmp_path / "ocr-rec-x"
    folder.mkdir()
    (folder / "model.safetensors").write_text("weights", encoding="utf-8")
    (folder / MARKER_NAME).write_text(
        json.dumps({"id": entry.id, "revision": entry.upstream_revision}), encoding="utf-8"
    )
    assert model_status(entry, tmp_path, ollama_names=None) == "corrupt"
    (folder / MARKER_NAME).write_text(
        json.dumps({"id": entry.id, "revision": entry.upstream_revision, "file_sizes": ["x"]}),
        encoding="utf-8",
    )
    assert model_status(entry, tmp_path, ollama_names=None) == "corrupt"


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="a marker that is valid JSON but not an object crashes model_status "
    "(AttributeError) instead of returning corrupt",
)
def test_hf_marker_that_is_a_json_list_is_corrupt(tmp_path: Path) -> None:
    folder = tmp_path / "ocr-rec-x"
    folder.mkdir()
    (folder / "model.safetensors").write_text("weights", encoding="utf-8")
    (folder / MARKER_NAME).write_text("[1, 2]", encoding="utf-8")
    assert model_status(hf_entry(), tmp_path, ollama_names=None) == "corrupt"


# ---------------------------------------------------------------- download: hf guards


def test_download_hf_without_revision_raises_and_downloads_nothing(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_download(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        folder = Path(kwargs["local_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"weights")
        return str(folder)

    with pytest.raises(ModelDownloadError, match="no upstream_repo/upstream_revision"):
        download_model(hf_entry(upstream_revision=None), tmp_path, hf_download=fake_download)
    assert calls == []
    assert not (tmp_path / "ocr-rec-x").exists()


def test_verify_models_checks_the_ollama_names(tmp_path: Path) -> None:
    entry = ModelEntry(
        id="llm-x",
        name="X",
        kind="llm",
        format="ollama",
        size_mb=1,
        license="Gemma terms",
        description="d",
        ollama_name="x:1b",
    )
    assert verify_models([entry], tmp_path, ollama_names={"x:1b"}) == {"llm-x": "installed"}
    assert verify_models([entry], tmp_path, ollama_names={"y:2b"}) == {"llm-x": "missing"}


# ---------------------------------------------------------------- series_config


def test_export_section_is_allowed_per_series(tmp_path: Path) -> None:
    cfg = Config(paths=PathsConfig(models_dir=tmp_path / "models"))
    series_dir = tmp_path / "lib" / "S"
    series_dir.mkdir(parents=True)
    (series_dir / "series.toml").write_text("[export]\njpeg_quality = 80\n", encoding="utf-8")
    merged = series_config(cfg, series_dir)
    assert merged.export.jpeg_quality == 80
    assert merged.export.subsampling == cfg.export.subsampling  # sibling keys survive the merge
