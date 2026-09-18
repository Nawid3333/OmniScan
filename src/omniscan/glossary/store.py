"""SQLite working copy of a series glossary (`SeriesPaths.db`), one row per `GlossaryEntry`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal

from omniscan.core.schemas import GlossaryEntry

# One column per GlossaryEntry field except id; `aliases` is a JSON-encoded TEXT column.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("source", "TEXT NOT NULL"),
    ("target", "TEXT NOT NULL"),
    ("type", "TEXT NOT NULL"),
    ("gender", "TEXT NOT NULL"),
    ("pronouns", "TEXT"),
    ("aliases", "TEXT NOT NULL"),
    ("notes", "TEXT"),
    ("status", "TEXT NOT NULL"),
    ("origin", "TEXT NOT NULL"),
    ("first_seen_chapter", "REAL"),
    ("count", "INTEGER NOT NULL"),
)
_COLUMN_NAMES: tuple[str, ...] = tuple(name for name, _ in _COLUMNS)
_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS glossary ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    + ", ".join(f"{name} {sql_type}" for name, sql_type in _COLUMNS)
    + ")"
)
_INSERT = f"INSERT INTO glossary ({', '.join(_COLUMN_NAMES)}) VALUES ({', '.join('?' * len(_COLUMN_NAMES))})"
_UPDATE = f"UPDATE glossary SET {', '.join(f'{name} = ?' for name in _COLUMN_NAMES)} WHERE id = ?"
_SELECT = f"SELECT id, {', '.join(_COLUMN_NAMES)} FROM glossary"


def _entry_from_row(row: tuple[object, ...]) -> GlossaryEntry:
    values = dict(zip(("id", *_COLUMN_NAMES), row, strict=True))
    values["aliases"] = json.loads(str(values["aliases"]))
    return GlossaryEntry.model_validate(values)


def _params(entry: GlossaryEntry) -> tuple[object, ...]:
    return tuple(
        json.dumps(getattr(entry, name)) if name == "aliases" else getattr(entry, name)
        for name in _COLUMN_NAMES
    )


class GlossaryStore:
    """CRUD access to the `glossary` table of a series sqlite db."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._closed = False
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def add(self, entry: GlossaryEntry) -> GlossaryEntry:
        """Insert a new row (entry.id must be None) and return a copy with the new row id."""
        if entry.id is not None:
            raise ValueError(f"entry.id must be None to add, got {entry.id}")
        cur = self._conn.execute(_INSERT, _params(entry))
        self._conn.commit()
        return entry.model_copy(update={"id": cur.lastrowid})

    def get(self, entry_id: int) -> GlossaryEntry | None:
        """The entry with this id, or None."""
        row = self._conn.execute(f"{_SELECT} WHERE id = ?", (entry_id,)).fetchone()
        return _entry_from_row(row) if row else None

    def update(self, entry: GlossaryEntry) -> None:
        """Overwrite every column of the row with this id (must exist)."""
        if entry.id is None:
            raise ValueError("entry.id must be set to update")
        if not self._exists(entry.id):
            raise KeyError(entry.id)
        self._conn.execute(_UPDATE, (*_params(entry), entry.id))
        self._conn.commit()

    def delete(self, entry_id: int) -> None:
        """Remove the row with this id (must exist)."""
        if not self._exists(entry_id):
            raise KeyError(entry_id)
        self._conn.execute("DELETE FROM glossary WHERE id = ?", (entry_id,))
        self._conn.commit()

    def list(self, *, status: Literal["proposed", "locked", "rejected"] | None = None) -> list[GlossaryEntry]:
        """All entries ordered by id ascending, filtered by status when given."""
        sql = f"{_SELECT} WHERE status = ? ORDER BY id" if status else f"{_SELECT} ORDER BY id"
        rows = self._conn.execute(sql, (status,) if status else ()).fetchall()
        return [_entry_from_row(row) for row in rows]

    def find_by_source(self, source: str) -> GlossaryEntry | None:
        """Exact (case-sensitive) match on the source column; the lowest id wins."""
        row = self._conn.execute(f"{_SELECT} WHERE source = ? ORDER BY id LIMIT 1", (source,)).fetchone()
        return _entry_from_row(row) if row else None

    def _exists(self, entry_id: int) -> bool:
        return self._conn.execute("SELECT 1 FROM glossary WHERE id = ?", (entry_id,)).fetchone() is not None

    def close(self) -> None:
        """Close the connection; safe to call twice."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> GlossaryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
