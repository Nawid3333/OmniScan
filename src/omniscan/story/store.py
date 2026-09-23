"""SQLite store of per-chapter story summaries (the `chapter_summary` table of `SeriesPaths.db`)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from omniscan.core.schemas import utcnow

_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS chapter_summary ("
    "chapter TEXT PRIMARY KEY, summary TEXT NOT NULL, model TEXT NOT NULL, updated_at TEXT NOT NULL)"
)
_SELECT = "SELECT chapter, summary, model, updated_at FROM chapter_summary"
_UPSERT = (
    "INSERT INTO chapter_summary (chapter, summary, model, updated_at) VALUES (?, ?, ?, ?) "
    "ON CONFLICT (chapter) DO UPDATE SET summary=excluded.summary, model=excluded.model, "
    "updated_at=excluded.updated_at"
)


@dataclass(frozen=True, slots=True)
class ChapterSummary:
    """One chapter's stored English summary."""

    chapter: str
    summary: str
    model: str
    updated_at: str  # ISO 8601 UTC, from omniscan.core.schemas.utcnow().isoformat()


class SummaryStore:
    """CRUD access to the `chapter_summary` table of a series sqlite db (alongside `GlossaryStore`)."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._closed = False
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def set(self, chapter: str, summary: str, model: str) -> ChapterSummary:
        """Insert or overwrite this chapter's summary (`updated_at` is generated here, not passed in)."""
        row = ChapterSummary(chapter=chapter, summary=summary, model=model, updated_at=utcnow().isoformat())
        self._conn.execute(_UPSERT, (row.chapter, row.summary, row.model, row.updated_at))
        self._conn.commit()
        return row

    def get(self, chapter: str) -> ChapterSummary | None:
        """This chapter's summary, or None."""
        row = self._conn.execute(f"{_SELECT} WHERE chapter = ?", (chapter,)).fetchone()
        return ChapterSummary(*row) if row else None

    def list(self) -> list[ChapterSummary]:
        """All rows ordered by chapter name ascending (reading order is the caller's concern)."""
        rows = self._conn.execute(f"{_SELECT} ORDER BY chapter").fetchall()
        return [ChapterSummary(*row) for row in rows]

    def delete(self, chapter: str) -> bool:
        """Remove this chapter's summary; True iff a row was removed."""
        cursor = self._conn.execute("DELETE FROM chapter_summary WHERE chapter = ?", (chapter,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the connection; safe to call twice."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> SummaryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
