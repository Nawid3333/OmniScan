"""Persistent job queue store: one `jobs` sqlite table, one row per queued pipeline run over a series."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from omniscan.core.config import Config
from omniscan.core.schemas import utcnow

JobStatus = Literal["queued", "running", "paused", "done", "failed", "cancelled"]
STATUSES: tuple[str, ...] = ("queued", "running", "paused", "done", "failed", "cancelled")
KNOWN_STAGES: tuple[str, ...] = (
    "ingest",
    "slice",
    "detect",
    "ocr",
    "translate",
    "judge",
    "inpaint",
    "typeset",
    "export",
)

_COLUMNS = (
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "series TEXT NOT NULL",
    "chapters TEXT",  # JSON array or NULL (= every chapter)
    "stages TEXT NOT NULL",  # JSON array, in execution order
    "force INTEGER NOT NULL",
    "priority INTEGER NOT NULL",
    "status TEXT NOT NULL",
    "attempts INTEGER NOT NULL",
    "max_attempts INTEGER NOT NULL",
    "error TEXT",
    "created_at TEXT NOT NULL",
    "started_at TEXT",
    "finished_at TEXT",
)
_COLUMN_NAMES: tuple[str, ...] = tuple(column.split()[0] for column in _COLUMNS)
_CREATE_TABLE = f"CREATE TABLE IF NOT EXISTS jobs ({', '.join(_COLUMNS)})"
_SELECT = f"SELECT {', '.join(_COLUMN_NAMES)} FROM jobs"


@dataclass(frozen=True, slots=True)
class Job:
    """One queued pipeline run: stages over a series (optionally a subset of chapters)."""

    id: int
    series: str
    chapters: tuple[str, ...] | None  # None = every chapter of the series
    stages: tuple[str, ...]  # non-empty, executed in this order
    force: bool
    priority: int  # higher runs first
    status: JobStatus
    attempts: int  # number of times the job has been claimed
    max_attempts: int
    error: str | None  # last error message
    created_at: datetime  # timezone-aware UTC
    started_at: datetime | None  # set on every claim
    finished_at: datetime | None  # set when status becomes done / failed / cancelled


def queue_db_path(cfg: Config) -> Path:
    """The job queue's sqlite db under the work root."""
    return cfg.paths.work_root / "queue.db"


def _fmt(dt: datetime) -> str:
    return dt.isoformat()


def _parse(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text is not None else None


def _job_from_row(row: tuple[Any, ...]) -> Job:
    (
        job_id,
        series,
        chapters,
        stages,
        force,
        priority,
        status,
        attempts,
        max_attempts,
        error,
        created_at,
        started_at,
        finished_at,
    ) = row
    return Job(
        id=int(job_id),
        series=str(series),
        chapters=tuple(json.loads(chapters)) if chapters is not None else None,
        stages=tuple(json.loads(stages)),
        force=bool(force),
        priority=int(priority),
        status=cast("JobStatus", status),
        attempts=int(attempts),
        max_attempts=int(max_attempts),
        error=None if error is None else str(error),
        created_at=_parse(created_at) or utcnow(),
        started_at=_parse(started_at),
        finished_at=_parse(finished_at),
    )


def _job_params(job: Job) -> tuple[object, ...]:
    return (
        job.series,
        None if job.chapters is None else json.dumps(list(job.chapters)),
        json.dumps(list(job.stages)),
        int(job.force),
        job.priority,
        job.status,
        job.attempts,
        job.max_attempts,
        job.error,
        _fmt(job.created_at),
        None if job.started_at is None else _fmt(job.started_at),
        None if job.finished_at is None else _fmt(job.finished_at),
    )


class QueueStore:
    """CRUD + state machine for the `jobs` table of a queue sqlite db (exactly one worker per db)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, isolation_level=None)  # claim_next uses BEGIN IMMEDIATE
        self._closed = False
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)

    def add(
        self,
        series: str,
        stages: Sequence[str],
        *,
        chapters: Sequence[str] | None = None,
        priority: int = 0,
        max_attempts: int = 2,
        force: bool = False,
    ) -> Job:
        """Insert a new queued job (stages kept exactly as given) and return it with its id."""
        if not series.strip():
            raise ValueError("series must not be empty")
        if not stages:
            raise ValueError("stages must not be empty")
        unknown = next((name for name in stages if name not in KNOWN_STAGES), None)
        if unknown is not None:
            raise ValueError(f"unknown stage {unknown!r} (known: {', '.join(KNOWN_STAGES)})")
        if chapters is not None and not chapters:
            raise ValueError("chapters must not be empty when given")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        job = Job(
            id=0,
            series=series,
            chapters=None if chapters is None else tuple(chapters),
            stages=tuple(stages),
            force=force,
            priority=priority,
            status="queued",
            attempts=0,
            max_attempts=max_attempts,
            error=None,
            created_at=utcnow(),
            started_at=None,
            finished_at=None,
        )
        columns = _COLUMN_NAMES[1:]
        cur = self._conn.execute(
            f"INSERT INTO jobs ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            _job_params(job),
        )
        return replace(job, id=cast("int", cur.lastrowid))

    def get(self, job_id: int) -> Job | None:
        """The job with this id, or None."""
        row = self._conn.execute(f"{_SELECT} WHERE id = ?", (job_id,)).fetchone()
        return _job_from_row(row) if row else None

    def list(self, status: JobStatus | None = None) -> list[Job]:
        """All jobs ordered by id ascending, filtered by status when given."""
        sql = f"{_SELECT} WHERE status = ? ORDER BY id" if status else f"{_SELECT} ORDER BY id"
        rows = self._conn.execute(sql, (status,) if status else ()).fetchall()
        return [_job_from_row(row) for row in rows]

    def claim_next(self) -> Job | None:
        """Atomically start the highest-priority queued job (ties: lowest id); None when nothing is queued."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                f"{_SELECT} WHERE status = 'queued' ORDER BY priority DESC, id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            job = _job_from_row(row)
            started_at = utcnow()
            self._conn.execute(
                "UPDATE jobs SET status = 'running', attempts = ?, started_at = ?, finished_at = NULL"
                " WHERE id = ?",
                (job.attempts + 1, _fmt(started_at), job.id),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return replace(
            job, status="running", attempts=job.attempts + 1, started_at=started_at, finished_at=None
        )

    def complete(self, job_id: int) -> Job:
        """Mark a running job done (clears the error, sets finished_at)."""
        job = self._expect_status(job_id, "complete", ("running",))
        return self._write(
            job,
            status="done",
            error=None,
            started_at=job.started_at,
            attempts=job.attempts,
            finished_at=utcnow(),
        )

    def fail(self, job_id: int, error: str, *, retry: bool = True) -> Job:
        """Record the error on a running job; requeue while attempts remain, else mark failed."""
        job = self._expect_status(job_id, "fail", ("running",))
        will_retry = retry and job.attempts < job.max_attempts
        return self._write(
            job,
            status="queued" if will_retry else "failed",
            error=error,
            started_at=job.started_at,
            attempts=job.attempts,
            finished_at=None if will_retry else utcnow(),
        )

    def pause(self, job_id: int) -> Job:
        """Move a queued job to paused (it will not be claimed)."""
        job = self._expect_status(job_id, "pause", ("queued",))
        return self._write(
            job,
            status="paused",
            error=job.error,
            started_at=job.started_at,
            attempts=job.attempts,
            finished_at=None,
        )

    def resume(self, job_id: int) -> Job:
        """Move a paused job back to queued."""
        job = self._expect_status(job_id, "resume", ("paused",))
        return self._write(
            job,
            status="queued",
            error=job.error,
            started_at=job.started_at,
            attempts=job.attempts,
            finished_at=None,
        )

    def cancel(self, job_id: int) -> Job:
        """Cancel a queued or paused job (a running job cannot be cancelled)."""
        job = self._expect_status(job_id, "cancel", ("queued", "paused"))
        return self._write(
            job,
            status="cancelled",
            error=job.error,
            started_at=job.started_at,
            attempts=job.attempts,
            finished_at=utcnow(),
        )

    def retry(self, job_id: int) -> Job:
        """Reset a failed or cancelled job to queued with a fresh attempt counter."""
        job = self._expect_status(job_id, "retry", ("failed", "cancelled"))
        return self._write(job, status="queued", error=None, started_at=None, attempts=0, finished_at=None)

    def requeue_running(self) -> int:
        """Move every running job back to queued with its interrupted attempt refunded; return how many."""
        self._conn.execute(
            "UPDATE jobs SET status = 'queued', attempts = MAX(attempts - 1, 0) WHERE status = 'running'"
        )
        return int(self._conn.execute("SELECT changes()").fetchone()[0])

    def clear_finished(self) -> int:
        """Delete done and cancelled jobs (failed jobs stay for retry) and return how many were removed."""
        self._conn.execute("DELETE FROM jobs WHERE status IN ('done', 'cancelled')")
        return int(self._conn.execute("SELECT changes()").fetchone()[0])

    def close(self) -> None:
        """Close the connection; safe to call twice."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> QueueStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require(self, job_id: int) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _expect_status(self, job_id: int, action: str, allowed: tuple[str, ...]) -> Job:
        """The job with this id, checked to be in one of the states `action` may run from."""
        job = self._require(job_id)
        if job.status not in allowed:
            raise ValueError(f"job {job_id} is {job.status}, cannot {action}")
        return job

    def _write(
        self,
        job: Job,
        *,
        status: JobStatus,
        error: str | None,
        started_at: datetime | None,
        attempts: int,
        finished_at: datetime | None,
    ) -> Job:
        """Persist one state change of `job` and return the updated copy."""
        self._conn.execute(
            "UPDATE jobs SET status = ?, error = ?, started_at = ?, attempts = ?, finished_at = ? WHERE id = ?",
            (
                status,
                error,
                None if started_at is None else _fmt(started_at),
                attempts,
                None if finished_at is None else _fmt(finished_at),
                job.id,
            ),
        )
        return replace(
            job, status=status, error=error, started_at=started_at, attempts=attempts, finished_at=finished_at
        )
