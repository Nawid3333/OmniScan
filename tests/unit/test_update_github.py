"""Tests for omniscan.update.github (card U6)."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from omniscan.update.github import (
    DEFAULT_REPO,
    Asset,
    ReleaseInfo,
    UpdateError,
    checksum_asset,
    latest_release,
    list_app_releases,
    platform_key,
    select_asset,
)
from omniscan.update.version import parse_version


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def asset_payload(name: str, tag: str, size: int = 100) -> dict[str, Any]:
    return {
        "name": name,
        "browser_download_url": f"https://github.com/owner/repo/releases/download/{tag}/{name}",
        "size": size,
    }


def release_payload(
    tag: str,
    *,
    draft: bool = False,
    prerelease: bool = False,
    body: str | None = "notes",
    assets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "body": body,
        "published_at": "2026-01-02T03:04:05Z",
        "assets": assets or [],
    }


PAYLOAD = [
    release_payload("models-v1", body="model mirror"),
    release_payload("v9.9.9", draft=True),
    release_payload(
        "v1.0.0",
        assets=[asset_payload("omniscan-windows-x64.zip", "v1.0.0"), asset_payload("SHA256SUMS", "v1.0.0")],
    ),
    release_payload("v1.1.0-beta.1", prerelease=True, body=None),
    release_payload("v0.9.0"),
]


def serving(
    payload: Any, seen: list[httpx.Request] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return handler


# ---------------------------------------------------------------- listing


def test_default_repo() -> None:
    assert DEFAULT_REPO == "Nawid3333/OmniScan"


def test_list_app_releases_filters_and_sorts() -> None:
    seen: list[httpx.Request] = []
    releases = list_app_releases(mock_client(serving(PAYLOAD, seen)), "owner/repo")

    assert [r.tag for r in releases] == ["v1.1.0-beta.1", "v1.0.0", "v0.9.0"]
    request = seen[0]
    assert str(request.url) == "https://api.github.com/repos/owner/repo/releases?per_page=100"
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert request.headers["User-Agent"] == "OmniScan-updater"
    assert "Authorization" not in request.headers


def test_list_app_releases_sends_token_only_when_given() -> None:
    seen: list[httpx.Request] = []
    list_app_releases(mock_client(serving(PAYLOAD, seen)), "owner/repo", token="sekrit")
    assert seen[0].headers["Authorization"] == "Bearer sekrit"


def test_release_info_fields() -> None:
    releases = list_app_releases(mock_client(serving(PAYLOAD)), "owner/repo")

    beta, stable = releases[0], releases[1]
    assert beta.version == parse_version("v1.1.0-beta.1")
    assert beta.prerelease is True
    assert beta.notes == ""
    assert beta.published_at == "2026-01-02T03:04:05Z"
    assert stable.prerelease is False
    assert stable.notes == "notes"
    assert stable.assets == (
        Asset(
            "omniscan-windows-x64.zip",
            "https://github.com/owner/repo/releases/download/v1.0.0/omniscan-windows-x64.zip",
            100,
        ),
        Asset("SHA256SUMS", "https://github.com/owner/repo/releases/download/v1.0.0/SHA256SUMS", 100),
    )


def test_latest_release_channels() -> None:
    releases = list_app_releases(mock_client(serving(PAYLOAD)), "owner/repo")
    assert latest_release(releases) is releases[1]  # v1.0.0
    assert latest_release(releases, channel="beta") is releases[0]  # v1.1.0-beta.1
    assert latest_release([]) is None


# ---------------------------------------------------------------- errors


def raise_from_list(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> str:
    with pytest.raises(UpdateError) as excinfo:
        list_app_releases(mock_client(handler), "owner/repo", **kwargs)
    return str(excinfo.value)


def test_404_is_reported_as_private_or_wrong_name() -> None:
    message = raise_from_list(lambda request: httpx.Response(404))
    assert message == (
        "no releases visible at github.com/owner/repo (the repository may be private or the name wrong)"
    )


def test_403_rate_limit_message_has_utc_reset_time() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1893456000"})

    assert raise_from_list(handler) == "GitHub rate limit reached; try again after 00:00 UTC"


def test_429_rate_limit_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1893456000"})

    assert "rate limit" in raise_from_list(handler)


def test_rate_limit_without_reset_time_says_a_while() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, headers={"X-RateLimit-Remaining": "0"})

    assert "a while" in raise_from_list(handler)


def test_403_without_rate_limit_headers_is_a_status_error() -> None:
    assert raise_from_list(lambda request: httpx.Response(403)) == "GitHub answered HTTP 403"


def test_500_is_reported_with_the_status() -> None:
    assert raise_from_list(lambda request: httpx.Response(500)) == "GitHub answered HTTP 500"


def test_connect_error_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert raise_from_list(handler) == "cannot reach GitHub: connection refused"


def test_malformed_json_is_reported() -> None:
    assert "malformed JSON" in raise_from_list(lambda request: httpx.Response(200, text="<html>nope</html>"))


# ---------------------------------------------------------------- platform_key


@pytest.mark.parametrize(
    ("platform_value", "machine_value", "expected"),
    [
        ("win32", "AMD64", "windows-x64"),
        ("linux", "x86_64", "linux-x64"),
        ("darwin", "arm64", "macos-arm64"),
        ("linux", "aarch64", "linux-arm64"),
    ],
)
def test_platform_key(
    platform_value: str, machine_value: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform_value)
    monkeypatch.setattr(platform, "machine", lambda: machine_value)
    assert platform_key() == expected


@pytest.mark.parametrize(
    ("platform_value", "machine_value", "message"),
    [
        ("win32", "riscv64", "unsupported architecture: riscv64"),
        ("freebsd14", "amd64", "unsupported platform: freebsd14"),
    ],
)
def test_platform_key_unsupported(
    platform_value: str, machine_value: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform_value)
    monkeypatch.setattr(platform, "machine", lambda: machine_value)
    with pytest.raises(UpdateError, match=message):
        platform_key()


# ---------------------------------------------------------------- assets


def make_release(tag: str, names: tuple[str, ...]) -> ReleaseInfo:
    return ReleaseInfo(
        tag=tag,
        version=parse_version(tag),
        prerelease=False,
        notes="",
        published_at=None,
        assets=tuple(
            Asset(name, f"https://github.com/o/r/releases/download/{tag}/{name}", 1) for name in names
        ),
    )


def test_select_asset_matches_the_platform_zip() -> None:
    picked = select_asset(
        make_release("v1.0.0", ("omniscan-windows-x64.zip", "omniscan-macos-arm64.zip", "SHA256SUMS")),
        "windows-x64",
    )
    assert picked.name == "omniscan-windows-x64.zip"
    assert picked.url == "https://github.com/o/r/releases/download/v1.0.0/omniscan-windows-x64.zip"
    assert picked.size == 1


def test_select_asset_missing_build_message() -> None:
    with pytest.raises(UpdateError, match=r"release v1\.0\.0 has no build for macos-arm64"):
        select_asset(make_release("v1.0.0", ("omniscan-windows-x64.zip",)), "macos-arm64")


def test_checksum_asset_found() -> None:
    assert checksum_asset(make_release("v1.0.0", ("SHA256SUMS",))).name == "SHA256SUMS"


def test_checksum_asset_missing_message() -> None:
    with pytest.raises(UpdateError, match=r"release v0\.9\.0 has no SHA256SUMS"):
        checksum_asset(make_release("v0.9.0", ()))
