"""Checking for, downloading, checksumming and staging an app release (card U6)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import httpx

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
from omniscan.update.version import Version, compare_versions, current_version

_CHECKSUM_LINE_RE = re.compile(r"^([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class StagedUpdate:
    """A verified release build staged under the updates directory."""

    version: Version
    tag: str
    path: Path
    sha256: str
    asset: str


def parse_checksums(text: str) -> dict[str, str]:
    """Parse sha256sum output into {name: digest}; digest values are lower-cased hex."""
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _CHECKSUM_LINE_RE.fullmatch(stripped)
        if match is None:
            raise UpdateError(f"bad line in SHA256SUMS: {line!r}")
        checksums[match.group(2)] = match.group(1).lower()
    return checksums


def check_for_update(
    client: httpx.Client,
    *,
    repo: str = DEFAULT_REPO,
    channel: str = "stable",
    current: Version | None = None,
    token: str | None = None,
) -> ReleaseInfo | None:
    """The newest release strictly newer than `current` (default: the installed one), else None."""
    current = current_version() if current is None else current
    release = latest_release(
        list_app_releases(client, repo, token=token),
        channel=cast("Literal['stable', 'beta']", channel),
    )
    if release is None or compare_versions(release.version, current) <= 0:
        return None
    return release


def download_update(
    client: httpx.Client,
    release: ReleaseInfo,
    updates_dir: Path,
    *,
    key: str | None = None,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> StagedUpdate:
    """Stage the release's verified build under <updates_dir>/<tag>/ and describe it."""
    key = platform_key() if key is None else key
    asset = select_asset(release, key)
    checksums = parse_checksums(_fetch_text(client, checksum_asset(release)))
    if asset.name not in checksums:
        raise UpdateError(f"SHA256SUMS has no entry for {asset.name}")
    sha256 = checksums[asset.name]
    tag_dir = updates_dir / release.tag
    target = tag_dir / asset.name
    if not (target.is_file() and _file_sha256(target) == sha256):
        _download_and_verify(client, asset, tag_dir, sha256, on_progress)
    _write_staged_json(tag_dir / "staged.json", release, asset.name, sha256)
    return StagedUpdate(
        version=release.version, tag=release.tag, path=target, sha256=sha256, asset=asset.name
    )


def _fetch_text(client: httpx.Client, asset: Asset) -> str:
    """GET an asset's body as text; HTTP and network errors become UpdateError."""
    try:
        response = client.get(asset.url, follow_redirects=True)
    except httpx.TransportError as exc:
        raise UpdateError(f"cannot reach GitHub: {exc}") from exc
    if response.status_code >= 400:
        raise UpdateError(f"HTTP {response.status_code} while downloading {asset.name}")
    return response.text


def _download_and_verify(
    client: httpx.Client,
    asset: Asset,
    tag_dir: Path,
    expected: str,
    on_progress: Callable[[int, int | None], None] | None,
) -> None:
    """Stream the asset to <name>.part, hash it, and os.replace it to <name> on a match."""
    tag_dir.mkdir(parents=True, exist_ok=True)
    part = tag_dir / f"{asset.name}.part"
    digest = hashlib.sha256()
    done = 0
    try:
        with client.stream("GET", asset.url, follow_redirects=True) as response:
            if response.status_code >= 400:
                raise UpdateError(f"HTTP {response.status_code} while downloading {asset.name}")
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None and length.isdigit() else None
            with part.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
        if digest.hexdigest() != expected:
            raise UpdateError(f"checksum mismatch for {asset.name}")
        part.replace(tag_dir / asset.name)  # os.replace: atomic on the same volume
    except BaseException:
        part.unlink(missing_ok=True)  # .part files never remain
        raise


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_staged_json(path: Path, release: ReleaseInfo, asset: str, sha256: str) -> None:
    """Record the staged build next to it (its parent directory exists by then)."""
    payload = {
        "tag": release.tag,
        "version": str(release.version),
        "asset": asset,
        "sha256": sha256,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
