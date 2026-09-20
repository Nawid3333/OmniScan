"""One row per catalog model: the entry's fields plus install status and hardware fit.

Qt-free and shared by `omniscan models list` and the GUI's models service, so both render the
same data. The row builder never talks to Ollama itself; the caller passes the daemon's pulled
tags (or `None` = unreachable) and a hardware snapshot if it already has one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from omniscan.core.config import Config
from omniscan.hw.detect import HardwareInfo
from omniscan.models.catalog import ModelEntry, load_catalog


@dataclass(frozen=True, slots=True)
class ModelRow:
    """Everything the models list needs about one model: catalog fields, status, hardware fit."""

    id: str
    name: str
    kind: str
    role: str | None
    family: str | None
    size_class: str | None
    size_mb: int
    required: bool
    format: str
    license: str
    description: str
    langs: tuple[str, ...]
    recommended_for: tuple[str, ...]
    notes: str
    used_by: tuple[str, ...]
    status: str  # missing | installed | corrupt | cloud | unknown
    installed_path: str | None
    fit_level: str  # ok | slow | warn | incompatible
    fit_device: str | None
    fit_messages: tuple[str, ...]


def ollama_model_names(base_url: str, timeout_s: float = 2.0) -> set[str] | None:
    """The daemon's pulled model tags, or None when Ollama is unreachable within `timeout_s`."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=timeout_s)
        response.raise_for_status()
        return {model["name"] for model in response.json().get("models", [])}
    except httpx.HTTPError, ValueError, KeyError, TypeError:
        return None


def build_rows(
    cfg: Config,
    *,
    role: str | None = None,
    lang: str | None = None,
    catalog: Sequence[ModelEntry] | None = None,
    hardware: HardwareInfo | None = None,
    ollama_names: set[str] | None = None,
) -> tuple[list[ModelRow], HardwareInfo]:
    """Rows for the catalog (role/lang filtered, catalog order kept) plus the hardware snapshot."""
    from omniscan.hw.assess import assess
    from omniscan.hw.detect import detect_hardware
    from omniscan.models.store import install_path, model_status

    entries = list(catalog) if catalog is not None else load_catalog()
    if role is not None:
        entries = [entry for entry in entries if entry.role == role]
    if lang is not None:
        entries = [entry for entry in entries if lang in entry.langs]
    hw = hardware if hardware is not None else detect_hardware(cfg.paths.models_dir)
    rows: list[ModelRow] = []
    for entry in entries:
        status = model_status(entry, cfg.paths.models_dir, ollama_names=ollama_names)
        path = install_path(entry, cfg.paths.models_dir)
        compat = assess(entry, hw)
        rows.append(
            ModelRow(
                id=entry.id,
                name=entry.name,
                kind=entry.kind,
                role=entry.role,
                family=entry.family or None,
                size_class=entry.size_class or None,
                size_mb=entry.size_mb,
                required=entry.required,
                format=entry.format,
                license=entry.license,
                description=entry.description,
                langs=tuple(entry.langs),
                recommended_for=tuple(entry.recommended_for),
                notes=entry.notes,
                used_by=tuple(entry.used_by),
                status=status,
                installed_path=str(path) if path is not None and status == "installed" else None,
                fit_level=compat.level,
                fit_device=compat.device,
                fit_messages=compat.messages,
            )
        )
    return rows, hw


def row_to_json(row: ModelRow) -> dict[str, Any]:
    """The per-model object `omniscan models list --json` prints (same keys, same order)."""
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "required": row.required,
        "format": row.format,
        "size_mb": row.size_mb,
        "license": row.license,
        "description": row.description,
        "used_by": list(row.used_by),
        "role": row.role,
        "family": row.family or "",
        "size_class": row.size_class or "",
        "langs": list(row.langs),
        "recommended_for": list(row.recommended_for),
        "notes": row.notes,
        "status": row.status,
        "installed_path": row.installed_path,
        "compatibility": {
            "level": row.fit_level,
            "device": row.fit_device,
            "messages": list(row.fit_messages),
        },
    }
