"""YAML import/export of the glossary (`SeriesPaths.glossary_yaml`) for human review and hand-editing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import yaml

from omniscan.core.schemas import GlossaryEntry
from omniscan.glossary.store import GlossaryStore


def export_yaml(store: GlossaryStore, path: Path) -> None:
    """Write every entry (all statuses, id ascending) as a plain-dict YAML list, atomically."""
    data = [entry.model_dump(mode="json") for entry in store.list()]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def import_yaml(store: GlossaryStore, path: Path, *, mode: Literal["merge", "replace"] = "merge") -> int:
    """Read a YAML list of glossary entries into the store; returns the number of entries written."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = [GlossaryEntry.model_validate(item) for item in raw]
    written = 0
    if mode == "replace":
        for entry in store.list():
            store.delete(cast(int, entry.id))
        for entry in entries:
            store.add(entry.model_copy(update={"id": None}))
            written += 1
    else:
        for entry in entries:
            existing = store.find_by_source(entry.source)
            if existing is None:
                store.add(entry.model_copy(update={"id": None}))
            else:
                store.update(entry.model_copy(update={"id": cast(int, existing.id)}))
            written += 1
    return written
