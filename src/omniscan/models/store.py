"""Model install state on disk: where a model lives and whether it is intact."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from omniscan.core.config import Config
from omniscan.models.catalog import ModelEntry, load_catalog

Status = Literal["installed", "missing", "corrupt", "cloud", "unknown"]

MARKER_NAME = ".installed.json"

_CHUNK = 1 << 20  # 1 MiB: streamed hashing


def file_sha256(path: Path) -> str:
    """Lowercase hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_path(entry: ModelEntry, models_dir: Path) -> Path | None:
    """Where the model lives on disk, or None for ollama/cloud models (the daemon stores them)."""
    if entry.format == "zip":
        return models_dir / entry.id
    if entry.format == "file" and entry.install_path is not None:
        return models_dir / entry.install_path
    return None


def model_status(entry: ModelEntry, models_dir: Path, *, ollama_names: set[str] | None) -> Status:
    """`installed` / `missing` / `corrupt` / `cloud` / `unknown` (ollama tag check impossible).

    zip: the install marker's sha256 must equal the catalog's. file: the size and sha256 must both
    match. ollama: checked against `ollama_names` (`None` = Ollama unreachable). cloud: nothing on
    disk by definition.
    """
    if entry.format == "cloud":
        return "cloud"
    if entry.format == "ollama":
        if ollama_names is None:
            return "unknown"
        tag = entry.ollama_name
        return "installed" if tag is not None and tag in ollama_names else "missing"
    path = install_path(entry, models_dir)
    if path is None:  # unreachable for a validated catalog entry
        return "corrupt"
    if entry.format == "zip":
        if not path.is_dir():
            return "missing"
        marker = path / MARKER_NAME
        if not marker.is_file():
            return "corrupt"
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return "corrupt"
        return "installed" if data.get("sha256") == entry.sha256 else "corrupt"
    # file
    if not path.is_file():
        return "missing"
    if entry.sha256 is None or entry.bytes is None or path.stat().st_size != entry.bytes:
        return "corrupt"
    return "installed" if file_sha256(path) == entry.sha256 else "corrupt"


def resolve_model_path(
    entry_id: str, cfg: Config, catalog: Sequence[ModelEntry] | None = None
) -> Path | None:
    """The install path of `entry_id` when it is installed, else None (U2b's loader hook)."""
    entries = catalog if catalog is not None else load_catalog()
    entry = next((e for e in entries if e.id == entry_id), None)
    if entry is None:
        return None
    path = install_path(entry, cfg.paths.models_dir)
    if path is None or model_status(entry, cfg.paths.models_dir, ollama_names=None) != "installed":
        return None
    return path
