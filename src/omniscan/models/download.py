"""Download, remove and verify catalog models: GitHub-release mirror with automatic upstream fallback.

Downloads stream into a `.part` file that is hashed while written, so a bad download never reaches
its install path; the `.part` is always cleaned up, also on failure. The mirror (a GitHub release of
this repo) is private today, so every mirror failure falls back to the upstream source. `hf` entries
skip the mirror: they come from the upstream Hugging Face repo only, verified per file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx

from omniscan.core.config import get_config
from omniscan.models.catalog import ModelEntry
from omniscan.models.store import MARKER_NAME, Status, file_sha256, install_path, model_status

ModelSource = Literal["mirror", "upstream", "ollama", "already installed"]

_HfDownload = Callable[..., str]  # huggingface_hub.snapshot_download or a test fake
_Progress = Callable[[str, int, int | None], None]  # (model id, done bytes, total bytes or None)

_CHUNK = 1 << 20  # 1 MiB: streamed download and hashing


class ModelDownloadError(RuntimeError):
    """A model download failed (all sources tried, or a request that cannot be served)."""


class _SourceError(Exception):
    """One download source failed (HTTP status, network error, hash or size mismatch)."""


def download_model(
    entry: ModelEntry,
    models_dir: Path,
    *,
    client: httpx.Client | None = None,
    hf_download: _HfDownload | None = None,
    ollama_url: str | None = None,
    on_progress: _Progress | None = None,
) -> str:
    """Install one catalog entry into `models_dir`.

    Returns "mirror", "upstream", "ollama" or "already installed". An already-installed zip/file/hf
    entry is left untouched without any request; a cloud entry is an error (nothing to download).
    """
    if entry.format == "cloud":
        raise ModelDownloadError(f"{entry.id} runs on Ollama Cloud, nothing to download")
    models_dir = Path(models_dir)
    if entry.format == "ollama":
        return _pull_ollama(entry, ollama_url, on_progress, client)
    path = install_path(entry, models_dir)
    if path is None:  # unreachable for a validated catalog entry
        raise ModelDownloadError(f"{entry.id}: no install path")
    if model_status(entry, models_dir, ollama_names=None) == "installed":
        return "already installed"
    if entry.format == "zip":
        return _download_zip(entry, models_dir, client, hf_download, on_progress)
    if entry.format == "hf":
        return _download_hf(entry, models_dir, hf_download)
    return _download_file(entry, path, client, on_progress)


def remove_model(
    entry: ModelEntry,
    models_dir: Path,
    *,
    client: httpx.Client | None = None,
    ollama_url: str | None = None,
) -> bool:
    """Remove an installed model (folder, file, or the Ollama daemon's copy); True when removed."""
    if entry.format == "cloud":
        return False
    models_dir = Path(models_dir)
    if entry.format == "ollama":
        return _delete_ollama(entry, ollama_url, client)
    path = install_path(entry, models_dir)
    if path is None or not path.exists():
        return False
    if entry.format in ("zip", "hf"):
        shutil.rmtree(path)
    else:
        path.unlink()
        parent = path.parent
        if parent.resolve() != models_dir.resolve() and not any(parent.iterdir()):
            parent.rmdir()
    return True


def verify_models(
    entries: Iterable[ModelEntry],
    models_dir: Path,
    *,
    ollama_names: set[str] | None = None,
    deep: bool = False,
) -> dict[str, Status]:
    """Recompute the status of every entry (`omniscan models verify`; `--deep` rehashes hf files)."""
    return {
        entry.id: model_status(entry, models_dir, ollama_names=ollama_names, deep=deep) for entry in entries
    }


# ---------------------------------------------------------------- zip and file sources


def _download_zip(
    entry: ModelEntry,
    models_dir: Path,
    client: httpx.Client | None,
    hf_download: _HfDownload | None,
    on_progress: _Progress | None,
) -> str:
    """Mirror zip first (hash-verified), then the upstream Hugging Face repo."""
    http = client if client is not None else httpx.Client(follow_redirects=True, timeout=None)
    part = models_dir / f"{entry.id}.zip.part"
    try:
        errors: list[str] = []
        for source in ("mirror", "upstream"):
            try:
                if source == "mirror":
                    if entry.mirror_url is None:  # unreachable for a validated entry
                        raise _SourceError("no mirror_url")
                    _stream_to_part(entry, http, entry.mirror_url, part, on_progress)
                    _extract_verified_zip(entry, models_dir, part, source="mirror")
                    return "mirror"
                return _install_zip_upstream(entry, models_dir, hf_download)
            except _SourceError as exc:
                errors.append(f"{source}: {exc}")
                part.unlink(missing_ok=True)
        raise ModelDownloadError(f"{entry.id}: download failed ({'; '.join(errors)})")
    finally:
        part.unlink(missing_ok=True)  # never leave a .part behind, also on failure
        if client is None:
            http.close()


def _download_file(
    entry: ModelEntry, path: Path, client: httpx.Client | None, on_progress: _Progress | None
) -> str:
    """Mirror file (hash-verified), then the upstream URL (same check, moved into place atomically)."""
    http = client if client is not None else httpx.Client(follow_redirects=True, timeout=None)
    part = path.parent / (path.name + ".part")
    try:
        errors: list[str] = []
        for source in ("mirror", "upstream"):
            url = entry.mirror_url if source == "mirror" else entry.upstream_url
            try:
                if url is None:  # unreachable for a validated catalog entry
                    raise _SourceError("no URL")
                _stream_to_part(entry, http, url, part, on_progress)
                part.replace(path)  # atomic move into place (same as os.replace)
                return source
            except _SourceError as exc:
                # A wrong hash on the upstream URL is a hard error: the download never reaches the
                # target — the `.part` is deleted here and `path` stays as it was.
                errors.append(f"{source}: {exc}")
                part.unlink(missing_ok=True)
        raise ModelDownloadError(f"{entry.id}: download failed ({'; '.join(errors)})")
    finally:
        part.unlink(missing_ok=True)
        if client is None:
            http.close()


def _stream_to_part(
    entry: ModelEntry, http: httpx.Client, url: str, part: Path, on_progress: _Progress | None
) -> None:
    """GET `url` into `part`, hashing while writing; raise _SourceError on any mismatch or error."""
    part.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    done = 0
    try:
        with http.stream("GET", url) as response:
            if response.status_code != 200:
                raise _SourceError(f"HTTP {response.status_code}")
            total: int | None = None
            length = response.headers.get("content-length")
            if length is not None and length.isdigit():
                total = int(length)
            with part.open("wb") as fh:
                for chunk in response.iter_bytes(_CHUNK):
                    digest.update(chunk)
                    fh.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(entry.id, done, total)
    except httpx.HTTPError as exc:
        raise _SourceError(f"{type(exc).__name__}: {exc}") from exc
    if entry.sha256 is not None and digest.hexdigest() != entry.sha256:
        raise _SourceError(f"sha256 mismatch (expected {entry.sha256}, got {digest.hexdigest()})")
    if entry.bytes is not None and done != entry.bytes:
        raise _SourceError(f"size mismatch (expected {entry.bytes} bytes, got {done})")


def _extract_verified_zip(entry: ModelEntry, models_dir: Path, part: Path, *, source: str) -> None:
    """Extract a hash-verified zip into `models_dir` with zip-slip protection, then mark it installed."""
    root = (models_dir / entry.id).resolve()
    try:
        with zipfile.ZipFile(part) as zf:
            for member in zf.infolist():  # validate everything before extracting anything
                _check_zip_member(entry, member, models_dir)
            zf.extractall(models_dir)
    except zipfile.BadZipFile as exc:
        raise _SourceError(f"invalid zip: {exc}") from exc
    marker = {
        "id": entry.id,
        "sha256": entry.sha256,
        "source": source,
        "revision": entry.upstream_revision,
        "installed_at": datetime.now(tz=UTC).isoformat(),
    }
    (root / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    part.unlink(missing_ok=True)  # the zip is deleted after extraction


def _check_zip_member(entry: ModelEntry, member: zipfile.ZipInfo, models_dir: Path) -> None:
    """Reject a member that would land outside `<models_dir>/<entry.id>/` (zip-slip)."""
    name = member.filename or ""
    if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
        raise ModelDownloadError(f"{entry.id}: zip-slip: absolute member path {name!r}")
    dest = (models_dir / name).resolve()
    root = (models_dir / entry.id).resolve()
    if dest != root and root not in dest.parents:
        raise ModelDownloadError(f"{entry.id}: zip-slip: member {name!r} escapes <models_dir>/{entry.id}/")


def _install_zip_upstream(entry: ModelEntry, models_dir: Path, hf_download: _HfDownload | None) -> str:
    """snapshot_download the upstream Hugging Face repo into `<models_dir>/<entry.id>`."""
    if entry.upstream_repo is None:  # unreachable for a validated catalog entry
        raise _SourceError("no upstream_repo")
    if hf_download is None:
        from huggingface_hub import snapshot_download  # lazy: only needed on the fallback path

        hf_download = snapshot_download
    folder = models_dir / entry.id
    try:
        hf_download(entry.upstream_repo, revision=entry.upstream_revision, local_dir=str(folder))
    except Exception as exc:
        raise _SourceError(f"huggingface {entry.upstream_repo}: {type(exc).__name__}: {exc}") from exc
    # Upstream files cannot be compared with the mirror hash (different packaging), so the marker
    # records the CATALOG hash as the model's identity, not a verified checksum of these files.
    marker = {
        "id": entry.id,
        "sha256": entry.sha256,
        "source": "upstream",
        "revision": entry.upstream_revision,
        "installed_at": datetime.now(tz=UTC).isoformat(),
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    return "upstream"


def _download_hf(entry: ModelEntry, models_dir: Path, hf_download: _HfDownload | None) -> str:
    """Fetch an `hf` entry from its upstream Hugging Face repo and verify every catalog file.

    `snapshot_download` fetches only the files the catalog lists (as `allow_patterns`); each is
    verified against its recorded sha256 — one bad or missing file deletes the whole folder.
    """
    if hf_download is None:
        from huggingface_hub import snapshot_download  # lazy: only needed on the download path

        hf_download = snapshot_download
    repo, revision = entry.upstream_repo, entry.upstream_revision
    if repo is None or revision is None:  # unreachable for a validated catalog entry
        raise ModelDownloadError(f"{entry.id}: no upstream_repo/upstream_revision")
    folder = models_dir / entry.id
    try:
        hf_download(repo, revision=revision, local_dir=str(folder), allow_patterns=list(entry.files))
    except Exception as exc:
        raise ModelDownloadError(
            f"{entry.id}: huggingface {entry.upstream_repo}: {type(exc).__name__}: {exc}"
        ) from exc
    bad = []
    for name, sha256 in entry.files.items():
        file_path = folder / name
        if not file_path.is_file():
            bad.append(f"{name} (missing)")
        elif file_sha256(file_path) != sha256:
            bad.append(f"{name} (sha256 mismatch)")
    if bad:
        shutil.rmtree(folder, ignore_errors=True)  # never keep a half-verified install
        raise ModelDownloadError(f"{entry.id}: verification failed, folder removed: {'; '.join(bad)}")
    marker = {
        "id": entry.id,
        "source": "upstream",
        "revision": entry.upstream_revision,
        "installed_at": datetime.now(tz=UTC).isoformat(),
        "files_verified": len(entry.files),
        "file_sizes": {name: (folder / name).stat().st_size for name in entry.files},
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    return "upstream"


# ---------------------------------------------------------------- ollama


def _pull_ollama(
    entry: ModelEntry, ollama_url: str | None, on_progress: _Progress | None, client: httpx.Client | None
) -> str:
    """Pull the entry's tag through the local daemon's /api/pull (NDJSON progress lines)."""
    base = ollama_url or get_config().ollama.local_url
    tag = entry.ollama_name
    if tag is None:  # unreachable for a validated catalog entry
        raise ModelDownloadError(f"{entry.id}: no ollama_name")
    http = client if client is not None else httpx.Client(timeout=None)
    try:
        with http.stream("POST", f"{base}/api/pull", json={"model": tag, "stream": True}) as response:
            if response.status_code != 200:
                raise ModelDownloadError(f"ollama pull {tag}: HTTP {response.status_code}")
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError as exc:
                    raise ModelDownloadError(f"ollama pull {tag}: bad progress line: {line!r}") from exc
                if "error" in data:
                    raise ModelDownloadError(f"ollama pull {tag}: {data['error']}")
                completed = data.get("completed")
                if on_progress is not None and completed is not None:
                    on_progress(entry.id, int(completed), data.get("total"))
                if data.get("status") == "success":
                    return "ollama"
    except httpx.HTTPError as exc:
        raise ModelDownloadError(f"ollama pull {tag}: {type(exc).__name__}: {exc}") from exc
    finally:
        if client is None:
            http.close()
    return "ollama"


def _delete_ollama(entry: ModelEntry, ollama_url: str | None, client: httpx.Client | None) -> bool:
    """DELETE the tag on the daemon; True when the daemon accepted the delete."""
    base = ollama_url or get_config().ollama.local_url
    tag = entry.ollama_name
    if tag is None:
        return False
    http = client if client is not None else httpx.Client(timeout=30.0)
    try:
        response = http.request("DELETE", f"{base}/api/delete", json={"model": tag})
    except httpx.HTTPError as exc:
        raise ModelDownloadError(f"ollama delete {tag}: {type(exc).__name__}: {exc}") from exc
    finally:
        if client is None:
            http.close()
    return response.status_code == 200
