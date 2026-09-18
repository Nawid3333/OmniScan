"""Resumability contract: content hashes + per-chapter manifest deciding whether a stage must re-run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from omniscan.core.schemas import Manifest, StageRecord

_CHUNK = 1 << 20


def hash_file(path: Path) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(obj: Any) -> str:
    """sha256 of a canonical JSON encoding (sorted keys, no whitespace); paths/sets are stringified."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(f"cannot hash {type(obj).__name__}")


def hash_inputs(files: Iterable[Path], extra: Mapping[str, Any] | None = None) -> str:
    """Combined hash of input files (name + content) and optional extra JSON-able data.

    Missing files hash as '<missing>' so that creating them later invalidates the stage.
    """
    parts: list[tuple[str, str]] = []
    for path in sorted(files, key=lambda p: p.as_posix()):
        parts.append((path.name, hash_file(path) if path.is_file() else "<missing>"))
    return hash_json({"files": parts, "extra": dict(extra or {})})


def load_manifest(path: Path, series: str, chapter: str) -> Manifest:
    """Load the chapter manifest, or return an empty one if it doesn't exist yet."""
    if path.is_file():
        return Manifest.load(path)
    return Manifest(series=series, chapter=chapter)


def is_up_to_date(
    manifest: Manifest,
    stage: str,
    *,
    version: int,
    input_hash: str,
    config_hash: str,
    work_dir: Path,
) -> bool:
    """True if the stage last finished successfully with identical version/inputs/config and its outputs exist."""
    record: StageRecord | None = manifest.stages.get(stage)
    if record is None or record.status != "done":
        return False
    if (record.version, record.input_hash, record.config_hash) != (version, input_hash, config_hash):
        return False
    return all((work_dir / name).exists() for name in record.outputs)
