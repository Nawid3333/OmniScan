"""Tests for omniscan.doctor."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from omniscan import doctor
from omniscan.core.config import Config, PathsConfig, Secrets
from omniscan.doctor import CheckResult, run_all_checks
from omniscan.models.catalog import ModelEntry
from omniscan.models.store import MARKER_NAME

REQUIRED_NAMES = (
    "python",
    "torch_gpu",
    "rocm",
    "rocjpeg",
    "ollama_local",
    "ollama_models",
    "ollama_cloud",
    "secrets",
    "paths",
    "models",
    "codec",
)

_CATALOG_SHA = "b" * 64


def _zip_entry(model_id: str, size_mb: int = 100, *, required: bool = True) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name="M",
        kind="vision",
        required=required,
        format="zip",
        size_mb=size_mb,
        license="Apache-2.0",
        description="d",
        mirror_url=f"https://mirror/{model_id}.zip",
        sha256=_CATALOG_SHA,
        bytes=10,
        upstream_repo=f"up/{model_id}",
        upstream_revision="rev1",
    )


def _install(models_dir: Path, model_id: str, sha: str) -> None:
    folder = models_dir / model_id
    folder.mkdir(parents=True)
    (folder / "weights.bin").write_text("hello", encoding="utf-8")
    (folder / MARKER_NAME).write_text(json.dumps({"id": model_id, "sha256": sha}), encoding="utf-8")


def _models_cfg(tmp_path: Path) -> Config:
    return Config(paths=PathsConfig(models_dir=tmp_path / "models"))


def make_secrets(**kwargs: str | None) -> Secrets:
    """Build Secrets without touching ~/.config/omniscan/secrets.env."""
    aliases = {k.upper(): v for k, v in kwargs.items()}
    return Secrets(_env_file=None, **aliases)  # type: ignore[call-arg]


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def test_python_ok() -> None:
    result = doctor.check_python()
    assert result.status == "OK"
    assert result.detail.startswith("3.14")


def test_ollama_local_ok() -> None:
    cfg = Config()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{cfg.ollama.local_url}/api/version"
        return httpx.Response(200, json={"version": "0.34.2"})

    result = doctor.check_ollama_local(cfg, mock_client(handler))
    assert result.status == "OK"
    assert "0.34.2" in result.detail


def test_ollama_local_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    result = doctor.check_ollama_local(Config(), mock_client(handler))
    assert result.status == "FAIL"
    assert "unreachable" in result.detail


def test_ollama_models_missing() -> None:
    models = [{"name": n} for n in doctor.REQUIRED_OLLAMA_MODELS if n != "gemma4:12b"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": models})

    result = doctor.check_ollama_models(Config(), mock_client(handler))
    assert result.status == "WARN"
    assert "gemma4:12b" in result.detail


def test_ollama_models_all_present() -> None:
    models = [{"name": n} for n in doctor.REQUIRED_OLLAMA_MODELS]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": models})

    result = doctor.check_ollama_models(Config(), mock_client(handler))
    assert result.status == "OK"


def test_ollama_cloud_no_key_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request must be made without a key")

    result = doctor.check_ollama_cloud(Config(), make_secrets(), mock_client(handler))
    assert result.status == "WARN"


def test_ollama_cloud_key_rejected() -> None:
    key = "sk-super-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {key}"
        return httpx.Response(401)

    result = doctor.check_ollama_cloud(Config(), make_secrets(ollama_api_key=key), mock_client(handler))
    assert result.status == "FAIL"
    assert key not in result.detail


def test_secrets_missing_listed() -> None:
    result = doctor.check_secrets(make_secrets())
    assert result.status == "WARN"
    assert result.detail == "not set: OLLAMA_API_KEY"


def test_secrets_all_set() -> None:
    result = doctor.check_secrets(make_secrets(ollama_api_key="test-key"))
    assert result.status == "OK"
    assert result.detail == "all optional secrets set"


def test_rocm_not_applicable_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    result = doctor.check_rocm()
    assert result.status == "OK"
    assert "Windows" in result.detail


def test_rocm_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("rocminfo")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = doctor.check_rocm()
    assert result.status == "FAIL"


def test_rocm_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["rocminfo"], 0, stdout="Name:                    gfx1201\n", stderr=""
        )

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = doctor.check_rocm()
    assert result.status == "OK"
    assert result.detail == "gfx1201 agent found"


def test_paths_missing_warns(tmp_path: Any) -> None:
    paths = PathsConfig(
        library_root=tmp_path / "library",
        work_root=tmp_path / "work",
        output_root=tmp_path / "output",
    )
    result = doctor.check_paths(Config(paths=paths))
    assert result.status == "WARN"
    assert (tmp_path / "library").exists() is False
    assert (tmp_path / "work").exists() is False
    assert (tmp_path / "output").exists() is False


# ---------------------------------------------------------------- models (card U2b)


def test_models_all_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_zip_entry("a"), _zip_entry("b"), _zip_entry("c")]
    cfg = _models_cfg(tmp_path)
    for entry in entries:
        _install(cfg.paths.models_dir, entry.id, _CATALOG_SHA)
    monkeypatch.setattr(doctor, "load_catalog", lambda: entries)
    result = doctor.check_models(cfg)
    assert result.status == "OK"
    assert result.detail == "3 required models installed"


def test_models_missing_warns_with_ids_size_and_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [_zip_entry("a"), _zip_entry("b", 159), _zip_entry("c", 23), _zip_entry("opt", required=False)]
    cfg = _models_cfg(tmp_path)
    _install(cfg.paths.models_dir, "a", _CATALOG_SHA)
    monkeypatch.setattr(doctor, "load_catalog", lambda: entries)
    result = doctor.check_models(cfg)
    assert result.status == "WARN"
    assert "2 required model(s) not installed (182 MB): b, c" in result.detail
    assert 'run "omniscan models download --required"' in result.detail
    assert "opt" not in result.detail  # optional models are never required


def test_models_corrupt_counts_as_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_zip_entry("a")]
    cfg = _models_cfg(tmp_path)
    _install(cfg.paths.models_dir, "a", "c" * 64)  # wrong marker hash: corrupt
    monkeypatch.setattr(doctor, "load_catalog", lambda: entries)
    result = doctor.check_models(cfg)
    assert result.status == "WARN"
    assert "1 required model(s) not installed (100 MB): a" in result.detail


def test_models_broken_catalog_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> list[ModelEntry]:
        raise ValueError("bad catalog")

    monkeypatch.setattr(doctor, "load_catalog", broken)
    result = doctor.check_models(_models_cfg(tmp_path))
    assert result.status == "WARN"
    assert "bad catalog" in result.detail


def test_run_all_checks_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> CheckResult:
        raise RuntimeError("boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.34.2"})
        return httpx.Response(200, json={"models": [{"name": n} for n in doctor.REQUIRED_OLLAMA_MODELS]})

    monkeypatch.setattr(doctor, "check_rocm", boom)
    results = run_all_checks(Config(), make_secrets(), http=mock_client(handler))
    assert [r.name for r in results] == list(REQUIRED_NAMES)
    rocm = results[REQUIRED_NAMES.index("rocm")]
    assert rocm.status == "FAIL"
    assert rocm.detail == "RuntimeError: boom"


@pytest.mark.gpu
def test_torch_gpu_real() -> None:
    result = doctor.check_torch_gpu()
    assert result.status == "OK"
    assert "gfx1201" in result.detail
