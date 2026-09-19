"""Tests for the `omniscan update` command group (card U6)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.update.github import DEFAULT_REPO, platform_key

runner = CliRunner()

TAG = "v1.0.0"
CURRENT = "0.1.0"
ASSET_BYTES = b"omniscan release build"
ASSET_SHA256 = hashlib.sha256(ASSET_BYTES).hexdigest()
ASSET_NAME = f"omniscan-{platform_key()}.zip"
DOWNLOAD_BASE = f"https://github.com/owner/repo/releases/download/{TAG}"
SUMS = f"{ASSET_SHA256}  {ASSET_NAME}\n"
NOTES = "New OCR pass.\nBugfixes."


def release_payload(tag: str, **overrides: Any) -> dict[str, Any]:
    """One GitHub release payload entry; `overrides` replaces top-level keys."""
    payload: dict[str, Any] = {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "body": NOTES if tag == TAG else "",
        "published_at": "2026-01-02T03:04:05Z",
        "assets": [
            {
                "name": ASSET_NAME,
                "browser_download_url": f"{DOWNLOAD_BASE}/{ASSET_NAME}",
                "size": len(ASSET_BYTES),
            },
            {"name": "SHA256SUMS", "browser_download_url": f"{DOWNLOAD_BASE}/SHA256SUMS", "size": len(SUMS)},
        ],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A config with every path under tmp_path, patched into the CLI."""
    config = Config(
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: config)
    monkeypatch.setattr("omniscan.update.version._installed_version", lambda name: CURRENT)
    return config


def patch_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Request]:
    """Route the CLI's HTTP through a mock transport and record every request."""
    seen: list[httpx.Request] = []

    def fake_client() -> httpx.Client:
        def recording(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        return httpx.Client(transport=httpx.MockTransport(recording), follow_redirects=True, timeout=5.0)

    monkeypatch.setattr(omniscan.cli, "_http_client", fake_client)
    return seen


def serving_releases(request: httpx.Request) -> httpx.Response:
    assert str(request.url).startswith(f"https://api.github.com/repos/{DEFAULT_REPO}/releases")
    return httpx.Response(200, json=[release_payload(TAG), release_payload("v0.9.0")])


# ---------------------------------------------------------------- check


def test_check_reports_a_newer_release(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, serving_releases)
    result = runner.invoke(app, ["update", "check"])
    assert result.exit_code == 0
    assert f"update available: v1.0.0 (current v{CURRENT})" in result.output
    assert "New OCR pass." in result.output
    assert "Bugfixes." in result.output


def test_check_json(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, serving_releases)
    result = runner.invoke(app, ["update", "check", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "current": CURRENT,
        "latest": "1.0.0",
        "update_available": True,
        "tag": TAG,
        "notes": NOTES,
        "asset": ASSET_NAME,
    }


def test_check_up_to_date(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(
        monkeypatch,
        lambda request: httpx.Response(200, json=[release_payload(CURRENT)]),
    )
    result = runner.invoke(app, ["update", "check"])
    assert result.exit_code == 0
    assert f"up to date (v{CURRENT})" in result.output

    result = runner.invoke(app, ["update", "check", "--json"])
    payload = json.loads(result.output)
    assert payload["update_available"] is False
    assert payload["latest"] is None
    assert payload["tag"] is None
    assert payload["asset"] is None


def test_check_github_error_exits_1(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, lambda request: httpx.Response(404))
    result = runner.invoke(app, ["update", "check"])
    assert result.exit_code == 1
    assert f"no releases visible at github.com/{DEFAULT_REPO}" in result.output


def test_check_sends_the_token_and_never_prints_it(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "sekrit-token")
    seen = patch_http(
        monkeypatch,
        lambda request: httpx.Response(200, json=[release_payload(TAG)]),
    )
    result = runner.invoke(app, ["update", "check"])
    assert result.exit_code == 0
    assert seen[0].headers["Authorization"] == "Bearer sekrit-token"
    assert "sekrit-token" not in result.output


def test_check_without_a_token_sends_no_authorization_header(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    seen = patch_http(monkeypatch, serving_releases)
    result = runner.invoke(app, ["update", "check"])
    assert result.exit_code == 0
    assert "Authorization" not in seen[0].headers


# ---------------------------------------------------------------- download


def download_handler(request: httpx.Request) -> httpx.Response:
    if str(request.url).startswith("https://api.github.com/"):
        return serving_releases(request)
    name = request.url.path.rsplit("/", 1)[-1]
    if name == "SHA256SUMS":
        return httpx.Response(200, text=SUMS)
    if name == ASSET_NAME:
        return httpx.Response(200, content=ASSET_BYTES)
    raise AssertionError(f"unexpected URL {request.url}")


def test_download_stages_the_verified_build(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, download_handler)
    result = runner.invoke(app, ["update", "download"])
    assert result.exit_code == 0, result.output
    updates_dir = cfg.paths.work_root.parent / "updates"
    staged = updates_dir / TAG / ASSET_NAME
    assert staged.read_bytes() == ASSET_BYTES
    assert str(staged) in result.output
    meta = json.loads((updates_dir / TAG / "staged.json").read_text(encoding="utf-8"))
    assert meta["tag"] == TAG
    assert meta["asset"] == ASSET_NAME
    assert meta["sha256"] == ASSET_SHA256


def test_download_when_up_to_date(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(
        monkeypatch,
        lambda request: httpx.Response(200, json=[release_payload(CURRENT)]),
    )
    result = runner.invoke(app, ["update", "download"])
    assert result.exit_code == 0
    assert "already up to date" in result.output
    assert not (cfg.paths.work_root.parent / "updates").exists()


def test_download_github_error_exits_1(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, headers={"X-RateLimit-Remaining": "0"})

    patch_http(monkeypatch, handler)
    result = runner.invoke(app, ["update", "download"])
    assert result.exit_code == 1
    assert "rate limit" in result.output
