"""Tests for the shared models-list row builder (`models.rows`, card U3a part 1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.hw.detect import GpuInfo, HardwareInfo
from omniscan.models import rows as rows_module
from omniscan.models.catalog import ModelEntry
from omniscan.models.rows import ModelRow, build_rows, row_to_json
from omniscan.models.store import MARKER_NAME

runner = CliRunner()

SHA = "a" * 64
HF_SHA = hashlib.sha256(b"weights").hexdigest()

GPU = GpuInfo(
    index=1,
    name="AMD Radeon RX 9070 XT",
    vendor="amd",
    backend="rocm",
    vram_gb=16.0,
    integrated=False,
    device="cuda:1",
)


def fake_hardware(gpus: tuple[GpuInfo, ...] = (), disk_free_gb: float = 500.0) -> Any:
    """A CPU-only machine by default; fit is decided by the cpu_speed rules."""
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
        disk_free_gb=disk_free_gb,
    )


def catalog_entries() -> list[ModelEntry]:
    """Four entries: required zip, hf recognizer (ko), an ollama llm and a cloud llm."""
    return [
        ModelEntry(
            id="det",
            name="Detector",
            kind="vision",
            required=True,
            format="zip",
            size_mb=10,
            license="Apache-2.0",
            description="detector",
            used_by=["detect"],
            role="detector",
            mirror_url="https://mirror/det.zip",
            sha256=SHA,
            bytes=100,
            upstream_repo="org/det",
            upstream_revision="rev1",
        ),
        ModelEntry(
            id="rec",
            name="Recognizer",
            kind="ocr",
            format="hf",
            size_mb=3,
            license="Apache-2.0",
            description="hf rec",
            role="recognizer",
            family="ppocrv6",
            size_class="tiny",
            langs=["ko"],
            recommended_for=["ko"],
            notes="v6 tiny",
            upstream_repo="org/rec",
            upstream_revision="c" * 40,
            files={"model.safetensors": HF_SHA},
        ),
        ModelEntry(
            id="llm-x",
            name="X",
            kind="llm",
            format="ollama",
            size_mb=7,
            license="Gemma terms",
            description="llm",
            role="llm",
            ollama_name="x:1b",
        ),
        ModelEntry(
            id="llm-c",
            name="C",
            kind="llm",
            format="cloud",
            size_mb=0,
            license="Gemma terms",
            description="cloud llm",
            role="llm",
            ollama_name="c:1b-cloud",
        ),
    ]


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Temp models_dir; the catalog is injected per test via monkeypatched load_catalog."""
    return Config(paths=PathsConfig(models_dir=tmp_path / "models"))


def install_zip(cfg: Config, model_id: str = "det", sha: str = SHA) -> None:
    """A fake zip install: folder + install marker with the catalog sha."""
    folder = cfg.paths.models_dir / model_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_NAME).write_text(json.dumps({"id": model_id, "sha256": sha}), encoding="utf-8")


def rows_for(cfg: Config, ollama_names: set[str] | None = None) -> tuple[list[ModelRow], Any]:
    """build_rows over the hand-made catalog with the fixture's fake CPU-only hardware."""
    return build_rows(cfg, catalog=catalog_entries(), hardware=fake_hardware(), ollama_names=ollama_names)


# ---------------------------------------------------------------- fields and statuses


def test_row_copies_every_catalog_field(cfg: Config) -> None:
    rows, _ = rows_for(cfg)
    row = rows[1]
    assert (row.id, row.name, row.kind, row.role) == ("rec", "Recognizer", "ocr", "recognizer")
    assert (row.family, row.size_class) == ("ppocrv6", "tiny")
    assert (row.size_mb, row.required, row.format) == (3, False, "hf")
    assert (row.license, row.description, row.notes) == ("Apache-2.0", "hf rec", "v6 tiny")
    assert row.langs == ("ko",) and row.recommended_for == ("ko",) and row.used_by == ()
    assert row.installed_path is None and row.status == "missing"


def test_status_zip_installed_hf_missing_cloud(cfg: Config) -> None:
    install_zip(cfg)
    rows, _ = rows_for(cfg)
    by_id = {row.id: row for row in rows}
    assert by_id["det"].status == "installed"
    assert by_id["det"].installed_path == str(cfg.paths.models_dir / "det")
    assert by_id["rec"].status == "missing" and by_id["rec"].installed_path is None
    assert by_id["llm-c"].status == "cloud" and by_id["llm-c"].installed_path is None


def test_status_ollama_with_and_without_daemon_names(cfg: Config) -> None:
    rows, _ = rows_for(cfg, ollama_names={"x:1b"})
    assert {row.id: row.status for row in rows}["llm-x"] == "installed"
    rows, _ = rows_for(cfg, ollama_names=None)  # None = Ollama unreachable
    assert {row.id: row.status for row in rows}["llm-x"] == "unknown"


def test_status_zip_corrupt_marker(cfg: Config) -> None:
    install_zip(cfg, sha="b" * 64)
    rows, _ = rows_for(cfg)
    assert {row.id: row.status for row in rows}["det"] == "corrupt"


# ---------------------------------------------------------------- hardware fit


def test_fit_on_cpu_only_machine(cfg: Config) -> None:
    rows, _ = rows_for(cfg)
    by_id = {row.id: row for row in rows}
    assert by_id["det"].fit_level == "slow"
    assert by_id["det"].fit_device == "cpu"
    assert by_id["det"].fit_messages == ("runs on the CPU (slow)",)
    assert by_id["llm-c"].fit_level == "ok"  # cloud is always ok
    assert by_id["llm-c"].fit_device is None and by_id["llm-c"].fit_messages == ()


def test_fit_on_a_gpu_machine(cfg: Config) -> None:
    rows, _ = build_rows(
        cfg, catalog=catalog_entries(), hardware=fake_hardware(gpus=(GPU,)), ollama_names=None
    )
    by_id = {row.id: row for row in rows}
    assert by_id["det"].fit_level == "ok" and by_id["det"].fit_messages == ()
    assert by_id["det"].fit_device == "cuda:1"
    assert by_id["llm-x"].fit_device == "ollama"  # the daemon serves it


def test_fit_incompatible_cpu_speed(cfg: Config) -> None:
    big = ModelEntry(
        id="llm-big",
        name="Big",
        kind="llm",
        format="ollama",
        size_mb=20000,
        license="Gemma terms",
        description="big llm",
        ollama_name="big:31b",
        min_vram_gb=21.0,
        cpu_speed="unusable",
    )
    (row,) = build_rows(cfg, catalog=[big], hardware=fake_hardware())[0]
    assert row.fit_level == "incompatible"
    assert row.fit_device is None
    assert row.fit_messages == ("runs on the CPU",)


def test_fit_disk_warning_raises_level_to_warn(cfg: Config) -> None:
    (row,) = build_rows(
        cfg, catalog=catalog_entries()[:1], hardware=fake_hardware(gpus=(GPU,), disk_free_gb=0.0)
    )[0]
    assert row.fit_level == "warn"
    assert "free disk space" in row.fit_messages[-1]


def test_explicit_hardware_is_reused_not_detected(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_detect(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not re-detect")

    monkeypatch.setattr("omniscan.hw.detect.detect_hardware", no_detect)
    rows, hw = build_rows(cfg, catalog=catalog_entries(), hardware=fake_hardware(gpus=(GPU,)))
    assert rows and hw.best_device == "cuda:1"


def test_without_hardware_detect_hardware_is_called(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "omniscan.hw.detect.detect_hardware", lambda *args, **kwargs: fake_hardware(gpus=(GPU,))
    )
    _rows, hw = build_rows(cfg, catalog=catalog_entries(), ollama_names=None)
    assert hw.best_device == "cuda:1"


# ---------------------------------------------------------------- filters and order


def test_filters_and_catalog_order(cfg: Config) -> None:
    rows, _ = rows_for(cfg)
    assert [row.id for row in rows] == ["det", "rec", "llm-x", "llm-c"]  # catalog order kept
    only_rec, _ = build_rows(cfg, catalog=catalog_entries(), role="recognizer", hardware=fake_hardware())
    assert [row.id for row in only_rec] == ["rec"]
    by_lang, _ = build_rows(cfg, catalog=catalog_entries(), lang="ko", hardware=fake_hardware())
    assert [row.id for row in by_lang] == ["rec"]
    both, _ = build_rows(cfg, catalog=catalog_entries(), role="llm", lang="ko", hardware=fake_hardware())
    assert both == []


# ---------------------------------------------------------------- JSON parity with the CLI


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


def test_row_to_json_matches_cli_payload(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """`omniscan models list --json` and row_to_json produce identical dicts for the same catalog."""
    install_zip(cfg)
    entries = catalog_entries()
    monkeypatch.setattr(rows_module, "load_catalog", lambda: entries)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    monkeypatch.setattr(omniscan.cli, "_ollama_model_names", lambda *args: None)
    monkeypatch.setattr("omniscan.hw.detect.detect_hardware", lambda *args, **kwargs: fake_hardware())
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)

    cli_rows, _hw = build_rows(cfg, catalog=entries, hardware=fake_hardware(), ollama_names=None)
    expected = {
        "models_dir": str(cfg.paths.models_dir),
        "models": [row_to_json(row) for row in cli_rows],
        "hardware": asdict(fake_hardware()),
    }
    assert payload == json.loads(json.dumps(expected))  # json round-trip: tuples become lists
    for model in payload["models"]:
        assert list(model) == list(JSON_KEYS)
    by_id = {model["id"]: model for model in payload["models"]}
    assert by_id["det"]["installed_path"] == str(cfg.paths.models_dir / "det")
    assert by_id["det"]["compatibility"] == {
        "level": "slow",
        "device": "cpu",
        "messages": ["runs on the CPU (slow)"],
    }
    assert by_id["rec"]["family"] == "ppocrv6" and by_id["rec"]["langs"] == ["ko"]
    assert by_id["llm-c"]["compatibility"] == {"level": "ok", "device": None, "messages": []}


def test_row_to_json_family_defaults_to_empty_string(cfg: Config) -> None:
    (row,) = build_rows(cfg, catalog=catalog_entries()[:1], hardware=fake_hardware())[0]
    assert row.family is None and row.size_class is None
    assert row_to_json(row)["family"] == "" and row_to_json(row)["size_class"] == ""


def test_ollama_model_names_returns_none_on_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("no daemon")

    monkeypatch.setattr(httpx, "get", refuse)
    assert rows_module.ollama_model_names("http://localhost:11434") is None


def test_ollama_model_names_parses_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "x:1b"}, {"name": "y:7b"}]}

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response())
    assert rows_module.ollama_model_names("http://localhost:11434") == {"x:1b", "y:7b"}
