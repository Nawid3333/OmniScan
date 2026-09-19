"""Listing app releases on GitHub Releases and picking assets (card U6)."""

from __future__ import annotations

import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cmp_to_key
from typing import Any, Literal

import httpx

from omniscan.update.version import Version, compare_versions, is_app_tag, parse_version

DEFAULT_REPO = "Nawid3333/OmniScan"

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "OmniScan-updater",
}
_OS_NAMES = {"win32": "windows", "darwin": "macos", "linux": "linux"}
_ARCH_NAMES = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}


class UpdateError(RuntimeError):
    """An updater failure whose message is safe to show to the user."""


@dataclass(frozen=True, slots=True)
class Asset:
    """One file attached to a release."""

    name: str
    url: str  # browser_download_url
    size: int


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    """The part of a GitHub release the updater needs."""

    tag: str
    version: Version
    prerelease: bool
    notes: str
    published_at: str | None
    assets: tuple[Asset, ...]


def platform_key() -> str:
    """This machine as f"{os}-{arch}" (e.g. "windows-x64"); UpdateError when unsupported."""
    os_name = _OS_NAMES.get(sys.platform)
    if os_name is None:
        raise UpdateError(f"unsupported platform: {sys.platform}")
    machine = platform.machine().lower()
    arch = _ARCH_NAMES.get(machine)
    if arch is None:
        raise UpdateError(f"unsupported architecture: {platform.machine()}")
    return f"{os_name}-{arch}"


def list_app_releases(
    client: httpx.Client, repo: str = DEFAULT_REPO, *, token: str | None = None
) -> list[ReleaseInfo]:
    """The app releases of `repo`, newest version first (drafts and non-app tags are skipped)."""
    headers = dict(_HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = client.get(
            f"https://api.github.com/repos/{repo}/releases",
            params={"per_page": "100"},
            headers=headers,
            follow_redirects=True,
        )
    except httpx.TransportError as exc:
        raise UpdateError(f"cannot reach GitHub: {exc}") from exc
    if response.status_code == 404:
        raise UpdateError(
            f"no releases visible at github.com/{repo} (the repository may be private or the name wrong)"
        )
    if response.status_code in (403, 429) and response.headers.get("X-RateLimit-Remaining") == "0":
        raise UpdateError(f"GitHub rate limit reached; try again after {_rate_limit_retry(response)}")
    if response.status_code >= 400:
        raise UpdateError(f"GitHub answered HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise UpdateError(f"GitHub answered with malformed JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise UpdateError("GitHub answered with malformed JSON")
    releases = [_release_info(entry) for entry in payload if _is_app_release(entry)]
    releases.sort(key=cmp_to_key(lambda a, b: compare_versions(a.version, b.version)), reverse=True)
    return releases


def _is_app_release(entry: Any) -> bool:
    """A payload entry this updater should keep: a non-draft release whose tag is an app tag."""
    if not isinstance(entry, dict) or entry.get("draft"):
        return False
    tag = entry.get("tag_name")
    return isinstance(tag, str) and is_app_tag(tag)


def _release_info(entry: dict[str, Any]) -> ReleaseInfo:
    """Build a ReleaseInfo from one GitHub release payload entry."""
    tag = entry["tag_name"]
    assets = tuple(
        Asset(name=asset["name"], url=asset["browser_download_url"], size=int(asset["size"]))
        for asset in entry.get("assets", [])
    )
    return ReleaseInfo(
        tag=tag,
        version=parse_version(tag),
        prerelease=bool(entry.get("prerelease")),
        notes=entry.get("body") or "",
        published_at=entry.get("published_at"),
        assets=assets,
    )


def _rate_limit_retry(response: httpx.Response) -> str:
    """A UTC wall-clock time from X-RateLimit-Reset, or "a while" when it is missing."""
    reset = response.headers.get("X-RateLimit-Reset")
    if reset is None or not reset.isdigit():
        return "a while"
    return datetime.fromtimestamp(int(reset), tz=UTC).strftime("%H:%M UTC")


def latest_release(
    releases: Sequence[ReleaseInfo], *, channel: Literal["stable", "beta"] = "stable"
) -> ReleaseInfo | None:
    """The newest release for the channel; stable ignores prereleases. None when there is none."""
    candidates = [r for r in releases if channel == "beta" or (not r.prerelease and not r.version.pre)]
    if not candidates:
        return None
    return max(candidates, key=cmp_to_key(lambda a, b: compare_versions(a.version, b.version)))


def select_asset(release: ReleaseInfo, key: str) -> Asset:
    """The release's omniscan-<key>.zip build."""
    for asset in release.assets:
        if asset.name == f"omniscan-{key}.zip":
            return asset
    raise UpdateError(f"release {release.tag} has no build for {key}")


def checksum_asset(release: ReleaseInfo) -> Asset:
    """The release's SHA256SUMS asset."""
    for asset in release.assets:
        if asset.name == "SHA256SUMS":
            return asset
    raise UpdateError(f"release {release.tag} has no SHA256SUMS")
