"""Tests for omniscan.doctor."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from omniscan import doctor
from omniscan.core.config import Config, PathsConfig, Secrets
from omniscan.doctor import CheckResult, run_all_checks

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
    "codec",
)


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
    assert "EXTRACTPICS_API_KEY" in result.detail
    assert "OMNISCAN_RELAY_CLIENT_TOKEN" in result.detail


def test_rocm_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("rocminfo")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = doctor.check_rocm()
    assert result.status == "FAIL"


def test_rocm_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["rocminfo"], 0, stdout="Name:                    gfx1201\n", stderr=""
        )

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
