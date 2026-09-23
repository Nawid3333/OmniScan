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


# --------------------------------------------------------- ollama_models (card B32)


# Mirrors the shipped config/translation_profiles.toml shape: two disabled profiles (one unreferenced,
# one a fallback target) plus one enabled profile whose fallback is the disabled one.
_MAIN_PROFILES_TOML = """\
[profiles.unused-local]
enabled = false
endpoint = "local"
model = "unused:1b"
style = "chat_json"

[profiles.fallback-local]
enabled = false
endpoint = "local"
model = "fallback:12b"
style = "translategemma"

[profiles.main-cloud]
endpoint = "local"
model = "main:31b-cloud"
style = "chat_json"
think = false
fallback = "fallback-local"
"""

_MAIN_JUDGE_TOML = '[judge]\nmodel = "main:31b-cloud"\n'


def _use_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profiles_toml: str = _MAIN_PROFILES_TOML,
    judge_toml: str = _MAIN_JUDGE_TOML,
) -> None:
    """Point doctor's no-argument config loading at temp files (the real user dir is left alone)."""
    (tmp_path / "translation_profiles.toml").write_text(profiles_toml, encoding="utf-8")
    (tmp_path / "judge.toml").write_text(judge_toml, encoding="utf-8")
    monkeypatch.setattr(doctor, "default_profile_paths", lambda: [tmp_path / "translation_profiles.toml"])
    monkeypatch.setattr(doctor, "default_judge_paths", lambda: [tmp_path / "judge.toml"])


def test_required_models_shipped_files_pinned() -> None:
    """Regression against the real shipped config (no mocking): enabled profile, then its fallback."""
    assert doctor.required_ollama_models() == ["gemma4:31b-cloud", "translategemma:12b"]


def test_required_models_constant_is_shipped_defaults() -> None:
    """The module constant is derived from the shipped files only, so it is machine-independent."""
    assert doctor.REQUIRED_OLLAMA_MODELS == ("gemma4:31b-cloud", "translategemma:12b")


def test_required_models_disabled_profile_vs_disabled_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.toml"
    first.write_text(
        '[profiles.alone-local]\nenabled = false\nendpoint = "local"\nmodel = "alone:1b"\nstyle = "chat_json"\n',
        encoding="utf-8",
    )
    second = tmp_path / "second.toml"
    second.write_text(
        '[profiles.fallback-local]\nenabled = false\nendpoint = "local"\nmodel = "fallback:12b"\n'
        'style = "translategemma"\n\n[profiles.main-cloud]\nendpoint = "local"\nmodel = "main:31b"\n'
        'style = "chat_json"\nfallback = "fallback-local"\n',
        encoding="utf-8",
    )
    judge = tmp_path / "judge.toml"
    judge.write_text('[judge]\nmodel = "judge:1b"\n', encoding="utf-8")
    monkeypatch.setattr(doctor, "default_profile_paths", lambda: [first, second])
    monkeypatch.setattr(doctor, "default_judge_paths", lambda: [judge])
    models = doctor.required_ollama_models()
    assert models == ["main:31b", "fallback:12b", "judge:1b"]
    assert "alone:1b" not in models


def test_required_models_judge_model_not_duplicated(tmp_path: Path) -> None:
    profiles = tmp_path / "translation_profiles.toml"
    profiles.write_text(
        '[profiles.only]\nendpoint = "local"\nmodel = "shared:1b"\nstyle = "chat_json"\n', encoding="utf-8"
    )
    judge = tmp_path / "judge.toml"
    judge.write_text('[judge]\nmodel = "shared:1b"\n', encoding="utf-8")
    assert doctor.required_ollama_models([profiles], [judge]) == ["shared:1b"]


def test_ollama_models_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "fallback:12b"}]})

    result = doctor.check_ollama_models(Config(), mock_client(handler))
    assert result.status == "WARN"
    assert result.detail == "missing: main:31b-cloud"


def test_ollama_models_all_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(tmp_path, monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "main:31b-cloud"}, {"name": "fallback:12b"}]})

    result = doctor.check_ollama_models(Config(), mock_client(handler))
    assert result.status == "OK"
    assert result.detail == "2 models, all required present"


def test_ollama_models_broken_config_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(
        tmp_path,
        monkeypatch,
        profiles_toml=(
            '[profiles.main]\nendpoint = "local"\nmodel = "main:1b"\nstyle = "chat_json"\n'
            'fallback = "missing"\n'
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    result = doctor.check_ollama_models(Config(), mock_client(handler))
    assert result.status == "WARN"
    assert result.detail.startswith("ValueError: ")
    assert "unknown fallback profile" in result.detail


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
