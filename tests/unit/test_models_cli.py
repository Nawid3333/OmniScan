"""Tests for the `omniscan models` command group (cards U2a and O1a)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.hw.detect import GpuInfo, HardwareInfo
from omniscan.models.store import MARKER_NAME

runner = CliRunner()

CATALOG_TOML = """\
[[model]]
id = "det"
name = "Detector"
kind = "vision"
required = true
format = "zip"
size_mb = 10
license = "Apache-2.0"
description = "detector"
used_by = []
role = "detector"
mirror_url = "https://mirror/det.zip"
sha256 = "{sha}"
bytes = 100
upstream_repo = "org/det"
upstream_revision = "rev1"

[[model]]
id = "lama"
name = "LaMa"
kind = "inpaint"
format = "file"
size_mb = 5
license = "Apache-2.0"
description = "inpainter"
used_by = []
role = "inpaint"
mirror_url = "https://mirror/big-lama.pt"
sha256 = "{sha}"
bytes = 4
upstream_url = "https://upstream/big-lama.pt"
install_path = "lama/big-lama.pt"

[[model]]
id = "llm-x"
name = "X"
kind = "llm"
format = "ollama"
size_mb = 7
license = "Gemma terms"
description = "llm"
used_by = []
role = "llm"
ollama_name = "x:1b"

[[model]]
id = "llm-c"
name = "C"
kind = "llm"
format = "cloud"
size_mb = 0
license = "Gemma terms"
description = "cloud llm"
used_by = []
role = "llm"
ollama_name = "c:1b-cloud"

[[model]]
id = "ocr-rec-x"
name = "X rec"
kind = "ocr"
format = "hf"
size_mb = 3
license = "Apache-2.0"
description = "hf rec"
used_by = []
role = "recognizer"
family = "ppocrv6"
size_class = "tiny"
langs = ["ko"]
upstream_repo = "org/rec"
upstream_revision = "{revision}"

[model.files]
"model.safetensors" = "{hf_sha}"
"""

COMPAT_TOML = """\
[[model]]
id = "det"
name = "Detector"
kind = "vision"
required = true
format = "zip"
size_mb = 10
license = "Apache-2.0"
description = "detector"
used_by = []
mirror_url = "https://mirror/det.zip"
sha256 = "{sha}"
bytes = 100
upstream_repo = "org/det"
upstream_revision = "rev1"

[[model]]
id = "llm-big"
name = "Big"
kind = "llm"
format = "ollama"
size_mb = 20000
license = "Gemma terms"
description = "big llm"
used_by = []
ollama_name = "big:31b"
min_vram_gb = 21.0
cpu_speed = "unusable"
"""

SHA = "a" * 64
REVISION = "c" * 40
HF_SHA = hashlib.sha256(b"weights").hexdigest()
JSON_KEYS = (
    "id",
    "name",
    "kind",
    "required",
    "format",
    "size_mb",
    "license",
    "description",
    "used_by",
    "role",
    "family",
    "size_class",
    "langs",
    "recommended_for",
    "notes",
    "status",
    "installed_path",
    "compatibility",
)


def fake_hardware(gpus: tuple[GpuInfo, ...] = ()) -> HardwareInfo:
    """A CPU-only machine by default: every GPU-less assessment is decided by the cpu_speed rules."""
    return HardwareInfo(
        os="windows",
        arch="x64",
        cpu_name="AMD Ryzen 9",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_gb=32.0,
        gpus=gpus,
        torch_build="cpu",
        best_device=gpus[0].device if gpus else "cpu",
        onnxruntime_providers=("CPUExecutionProvider",),
        disk_free_gb=500.0,
    )


def install_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Config:
    """The shared fixture body: temp models_dir, fake catalog, unreachable Ollama, no GPUs."""
    config = Config(paths=PathsConfig(models_dir=tmp_path / "models"))
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: config)
    monkeypatch.setattr(omniscan.cli, "_ollama_model_names", lambda *args: None)
    monkeypatch.setattr("omniscan.hw.detect.detect_hardware", lambda *args, **kwargs: fake_hardware())
    catalog_file = tmp_path / "catalog.toml"
    catalog_file.write_text(text.format(sha=SHA, revision=REVISION, hf_sha=HF_SHA), encoding="utf-8")
    from omniscan.models import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "default_catalog_path", lambda: catalog_file)
    monkeypatch.setattr(catalog_module, "machine_catalog_path", lambda: tmp_path / "machine.toml")
    return config


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Temp models_dir + a small fake catalog; Ollama is treated as unreachable."""
    return install_catalog(tmp_path, monkeypatch, CATALOG_TOML)


@pytest.fixture
def compat_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """The same, but the catalog carries hardware requirements (a warn and an incompatible model)."""
    return install_catalog(tmp_path, monkeypatch, COMPAT_TOML)


def install_det(cfg: Config, sha: str = SHA) -> None:
    folder = cfg.paths.models_dir / "det"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_NAME).write_text(json.dumps({"id": "det", "sha256": sha}), encoding="utf-8")


def install_hf(cfg: Config, content: bytes = b"weights") -> None:
    """A fake hf install: one catalog file plus a marker that records its (correct) size."""
    folder = cfg.paths.models_dir / "ocr-rec-x"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(content)
    (folder / MARKER_NAME).write_text(
        json.dumps(
            {
                "id": "ocr-rec-x",
                "source": "upstream",
                "revision": REVISION,
                "files_verified": 1,
                "file_sizes": {"model.safetensors": len(content)},
            }
        ),
        encoding="utf-8",
    )


def fake_download(entry: Any, models_dir: Any, **kwargs: Any) -> str:
    """Stand-in for download_model: installs the folder + marker without network."""
    folder = models_dir / entry.id
    marker = folder / MARKER_NAME
    if marker.is_file():
        return "already installed"
    folder.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"id": entry.id, "sha256": entry.sha256, "source": "mirror"}), encoding="utf-8"
    )
    return "mirror"


# ---------------------------------------------------------------- list


def test_list_text_rows_and_footer(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "det  vision  detector  10 MB  required  missing  -  detector" in result.output
    assert "lama  inpaint  inpaint  5 MB  optional  missing  -  inpainter" in result.output
    assert "llm-x  llm  llm  7 MB  optional  unknown  -  llm" in result.output  # Ollama unreachable
    assert "llm-c  llm  llm  0 MB  optional  cloud  -  cloud llm" in result.output
    assert "ocr-rec-x  ocr  recognizer  3 MB  optional  missing  ko  hf rec" in result.output
    assert "1 required model(s) missing, 10 MB to download" in result.output


def test_list_json_keys_and_statuses(cfg: Config) -> None:
    install_det(cfg)
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["models_dir"] == str(cfg.paths.models_dir)
    assert [m["id"] for m in payload["models"]] == ["det", "lama", "llm-x", "llm-c", "ocr-rec-x"]
    assert list(payload["models"][0]) == list(JSON_KEYS)
    by_id = {m["id"]: m for m in payload["models"]}
    assert by_id["det"]["status"] == "installed"
    assert by_id["det"]["installed_path"] == str(cfg.paths.models_dir / "det")
    assert by_id["lama"]["status"] == "missing" and by_id["lama"]["installed_path"] is None
    assert by_id["llm-x"]["status"] == "unknown" and by_id["llm-x"]["installed_path"] is None
    assert by_id["llm-c"]["status"] == "cloud"
    assert by_id["det"]["required"] is True and by_id["lama"]["required"] is False


def test_list_json_descriptive_fields(cfg: Config) -> None:
    by_id = {
        m["id"]: m for m in json.loads(runner.invoke(app, ["models", "list", "--json"]).output)["models"]
    }
    assert by_id["ocr-rec-x"]["role"] == "recognizer"
    assert by_id["ocr-rec-x"]["family"] == "ppocrv6"
    assert by_id["ocr-rec-x"]["size_class"] == "tiny"
    assert by_id["ocr-rec-x"]["langs"] == ["ko"]
    assert by_id["ocr-rec-x"]["recommended_for"] == []
    assert by_id["ocr-rec-x"]["notes"] == ""
    assert by_id["det"]["role"] == "detector"
    assert by_id["lama"]["family"] == ""


def test_list_role_filter(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "list", "--role", "recognizer"])
    assert result.exit_code == 0
    assert "ocr-rec-x  ocr  recognizer  3 MB  optional  missing  ko  hf rec" in result.output
    for absent in ("det", "lama", "llm-x", "llm-c"):
        assert f"\n{absent}  " not in result.output


def test_list_lang_filter(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "list", "--lang", "ko"])
    assert result.exit_code == 0
    assert "ocr-rec-x  ocr  recognizer" in result.output
    assert "\ndet  " not in result.output and "\nlama  " not in result.output


def test_list_with_reachable_ollama(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "_ollama_model_names", lambda *args: {"x:1b"})
    result = runner.invoke(app, ["models", "list"])
    assert "llm-x  llm  llm  7 MB  optional  installed  -  llm" in result.output


# ---------------------------------------------------------------- download


def test_download_installs_and_is_listed_installed(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniscan.models.download.download_model", fake_download)
    result = runner.invoke(app, ["models", "download", "det"])
    assert result.exit_code == 0
    assert "det: installed from mirror" in result.output
    result = runner.invoke(app, ["models", "list", "--json"])
    by_id = {m["id"]: m for m in json.loads(result.output)["models"]}
    assert by_id["det"]["status"] == "installed"
    result = runner.invoke(app, ["models", "download", "det"])
    assert "det: already installed" in result.output


def test_download_unknown_id_exits_2(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "download", "nope"])
    assert result.exit_code == 2
    assert "unknown model id(s) nope" in result.output
    assert "known: det, lama, llm-x, llm-c, ocr-rec-x" in result.output


def test_download_required_adds_missing_required(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    downloaded: list[str] = []
    original = fake_download

    def spy(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        downloaded.append(entry.id)
        return original(entry, models_dir, **kwargs)

    monkeypatch.setattr("omniscan.models.download.download_model", spy)
    result = runner.invoke(app, ["models", "download", "--required"])
    assert result.exit_code == 0
    assert downloaded == ["det"]  # the only required model, and it was missing
    assert "det: installed from mirror" in result.output


def test_download_required_skips_installed(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    install_det(cfg)
    downloaded: list[str] = []

    def spy(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        downloaded.append(entry.id)
        return fake_download(entry, models_dir, **kwargs)

    monkeypatch.setattr("omniscan.models.download.download_model", spy)
    result = runner.invoke(app, ["models", "download", "--required"])
    assert result.exit_code == 0
    assert downloaded == []
    assert "det: already installed" in result.output


def test_download_failure_continues_and_exits_1(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from omniscan.models.download import ModelDownloadError

    def failing(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        raise ModelDownloadError(f"{entry.id}: mirror down, upstream down")

    monkeypatch.setattr("omniscan.models.download.download_model", failing)
    result = runner.invoke(app, ["models", "download", "det", "lama"])
    assert result.exit_code == 1
    assert "models: det: mirror down, upstream down" in result.output


def test_download_ollama_installed_skips_the_pull(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omniscan.cli, "_ollama_model_names", lambda *args: {"x:1b"})

    def spying(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        raise AssertionError("download_model must not be called for an installed ollama model")

    monkeypatch.setattr("omniscan.models.download.download_model", spying)
    result = runner.invoke(app, ["models", "download", "llm-x"])
    assert result.exit_code == 0
    assert "llm-x: already installed" in result.output


# ---------------------------------------------------------------- remove and verify


def test_remove(cfg: Config) -> None:
    install_det(cfg)
    result = runner.invoke(app, ["models", "remove", "det"])
    assert result.exit_code == 0
    assert "det: removed" in result.output
    assert not (cfg.paths.models_dir / "det").exists()
    result = runner.invoke(app, ["models", "remove", "det"])
    assert result.exit_code == 0
    assert "det: nothing to remove" in result.output


def test_remove_unknown_id_exits_2(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "remove", "nope"])
    assert result.exit_code == 2


def test_verify_corrupt_file_exits_1(cfg: Config) -> None:
    target = cfg.paths.models_dir / "lama" / "big-lama.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"too long")  # wrong size -> corrupt
    result = runner.invoke(app, ["models", "verify", "lama"])
    assert result.exit_code == 1
    assert "lama: corrupt" in result.output


def test_verify_installed_exits_0(cfg: Config) -> None:
    install_det(cfg)
    result = runner.invoke(app, ["models", "verify", "det"])
    assert result.exit_code == 0
    assert "det: installed" in result.output


def test_verify_default_covers_non_llm(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 0
    assert "det: missing" in result.output
    assert "lama: missing" in result.output
    assert "ocr-rec-x: missing" in result.output
    assert "llm-x" not in result.output  # llm models are excluded by default
    assert "llm-c" not in result.output


def test_verify_hf_deep_rehashes_a_same_size_modified_file(cfg: Config) -> None:
    install_hf(cfg)
    shallow = runner.invoke(app, ["models", "verify", "ocr-rec-x"])  # sizes match: no rehash
    assert shallow.exit_code == 0
    assert "ocr-rec-x: installed" in shallow.output
    install_hf(cfg, content=b"weightx")  # same size, different bytes
    deep = runner.invoke(app, ["models", "verify", "ocr-rec-x", "--deep"])
    assert deep.exit_code == 1
    assert "ocr-rec-x: corrupt" in deep.output


# ---------------------------------------------------------------- progress printer


def test_progress_prints_once_per_five_percent_step(capsys: pytest.CaptureFixture[str]) -> None:
    from omniscan.cli import _models_progress

    on_progress = _models_progress({})
    total = 100_000_000
    for done in (0, 1, 4_999_999, 5_000_000, 9_999_999, total):
        on_progress("det", done, total)
    out = capsys.readouterr().out
    assert out == "det: 0% (0/100 MB)\ndet: 5% (5/100 MB)\ndet: 100% (100/100 MB)\n"
    on_progress("det", 50_000_000, None)  # unknown total: no line at all
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------- hardware command (H1)


def test_hardware_text_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    machine = HardwareInfo(
        os="windows",
        arch="x64",
        cpu_name="AMD Ryzen 9",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_gb=31.3,
        gpus=(
            GpuInfo(
                index=1,
                name="AMD Radeon RX 9070 XT",
                vendor="amd",
                backend="rocm",
                vram_gb=16.0,
                integrated=False,
                device="cuda:1",
            ),
            GpuInfo(
                index=0,
                name="AMD Radeon(TM) Graphics",
                vendor="amd",
                backend="rocm",
                vram_gb=0.5,
                integrated=True,
                device="cuda:0",
            ),
        ),
        torch_build="rocm",
        best_device="cuda:1",
        onnxruntime_providers=("CPUExecutionProvider",),
        disk_free_gb=123.4,
    )
    monkeypatch.setattr("omniscan.hw.detect.detect_hardware", lambda *args, **kwargs: machine)
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "OS: windows / x64" in result.output
    assert "CPU: AMD Ryzen 9 (8 physical, 16 logical)" in result.output
    assert "RAM: 31.3 GB" in result.output
    assert "torch build: rocm" in result.output
    assert "best device: cuda:1" in result.output
    assert "#1 AMD Radeon RX 9070 XT  16.0 GB  rocm  device cuda:1" in result.output
    assert "#0 AMD Radeon(TM) Graphics  0.5 GB  rocm  device cuda:0  (integrated)" in result.output
    assert "onnxruntime providers: CPUExecutionProvider" in result.output
    assert "free disk at the models folder: 123.4 GB" in result.output


def test_hardware_json(cfg: Config) -> None:
    result = runner.invoke(app, ["hardware", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["os"] == "windows" and payload["arch"] == "x64"
    assert payload["best_device"] == "cpu"  # the cfg fixture fakes a CPU-only machine
    assert payload["onnxruntime_providers"] == ["CPUExecutionProvider"]
    assert payload["disk_free_gb"] == 500.0


# ---------------------------------------------------------------- models list fit (H1)


def test_list_fit_column_and_explanations(cfg: Config) -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "det  vision  detector  10 MB  required  missing  -  detector  slow" in result.output
    assert "llm-c  llm  llm  0 MB  optional  cloud  -  cloud llm  ok" in result.output
    assert "  det: runs on the CPU (slow)" in result.output
    assert "  lama: runs on the CPU (slow)" in result.output
    assert "llm-c: " not in result.output  # cloud is always ok


def test_list_json_has_compatibility_and_hardware(cfg: Config) -> None:
    install_det(cfg)
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    by_id = {m["id"]: m for m in payload["models"]}
    assert by_id["det"]["compatibility"] == {
        "level": "slow",
        "device": "cpu",
        "messages": ["runs on the CPU (slow)"],
    }
    assert by_id["llm-c"]["compatibility"] == {"level": "ok", "device": None, "messages": []}
    assert payload["hardware"]["os"] == "windows"  # the HardwareInfo dict, not just ids
    assert payload["hardware"]["gpus"] == []


def test_list_json_fit_on_a_gpu_machine(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = GpuInfo(
        index=1,
        name="AMD Radeon RX 9070 XT",
        vendor="amd",
        backend="rocm",
        vram_gb=16.0,
        integrated=False,
        device="cuda:1",
    )
    monkeypatch.setattr(
        "omniscan.hw.detect.detect_hardware", lambda *args, **kwargs: fake_hardware(gpus=(gpu,))
    )
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "det  vision  detector  10 MB  required  missing  -  detector  ok" in result.output
    assert "runs on the CPU" not in result.output


# ---------------------------------------------------------------- download compatibility gate (H1)


def test_download_warns_for_a_warn_model_and_proceeds(
    compat_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("omniscan.models.download.download_model", fake_download)
    result = runner.invoke(app, ["models", "download", "det"])
    assert result.exit_code == 0
    assert "det: slow: runs on the CPU (slow)" in result.output
    assert "det: installed from mirror" in result.output


def test_download_refuses_incompatible_without_force(
    compat_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded: list[str] = []

    def spy(entry: Any, models_dir: Any, **kwargs: Any) -> str:
        downloaded.append(entry.id)
        return fake_download(entry, models_dir, **kwargs)

    monkeypatch.setattr("omniscan.models.download.download_model", spy)
    result = runner.invoke(app, ["models", "download", "llm-big"])
    assert result.exit_code == 1
    assert "llm-big: incompatible: runs on the CPU" in result.output
    assert downloaded == []  # refused: nothing downloaded
    assert not (compat_cfg.paths.models_dir / "llm-big").exists()


def test_download_incompatible_proceeds_with_force(
    compat_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("omniscan.models.download.download_model", fake_download)
    result = runner.invoke(app, ["models", "download", "llm-big", "--force"])
    assert result.exit_code == 0
    assert "llm-big: incompatible: runs on the CPU" in result.output
    assert "llm-big: installed from mirror" in result.output
