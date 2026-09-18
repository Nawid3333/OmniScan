"""Tests for the queue worker drain loop (card B23)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from omniscan.queue.store import Job, QueueStore
from omniscan.queue.worker import PermanentJobError, QueueSummary, run_queue


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QueueStore]:
    """A QueueStore on a fresh tmp file."""
    with QueueStore(tmp_path / "queue.db") as store:
        yield store


def events_recorder() -> tuple[list[str], Callable[[str, Job | None], None]]:
    """A notifier collecting the event names it was called with."""
    events: list[str] = []

    def notifier(event: str, job: Job | None) -> None:
        events.append(event)

    return events, notifier


def test_two_jobs_run_in_queue_order(store: QueueStore) -> None:
    store.add("S1", ("slice",))
    store.add("S2", ("slice",))
    ran: list[int] = []
    events, notifier = events_recorder()

    summary = run_queue(store, lambda job: ran.append(job.id), notifier)

    assert ran == [1, 2]
    assert summary == QueueSummary(done=2, failed=0, retried=0, finished=summary.finished)
    assert all(job.status == "done" for job in summary.finished)
    assert events == ["job_done", "job_done", "queue_empty"]


def test_retry_then_success(store: QueueStore) -> None:
    store.add("S", ("slice",))
    calls: list[int] = []

    def flaky(job: Job) -> None:
        calls.append(job.id)
        if len(calls) == 1:
            raise RuntimeError("boom")

    events, notifier = events_recorder()
    summary = run_queue(store, flaky, notifier)

    assert calls == [1, 1]
    assert summary.done == 1 and summary.failed == 0 and summary.retried == 1
    assert [job.status for job in summary.finished] == ["queued", "done"]  # each execution, in order
    final = summary.finished[-1]
    assert final.status == "done"
    assert final.attempts == 2
    assert events == ["job_retry", "job_done", "queue_empty"]


def test_exhausted_attempts_fail(store: QueueStore) -> None:
    store.add("S", ("slice",), max_attempts=2)
    events, notifier = events_recorder()

    def boom(job: Job) -> None:
        raise RuntimeError("boom")

    summary = run_queue(store, boom, notifier)

    assert [job.status for job in summary.finished] == ["queued", "failed"]  # each execution, in order
    final = summary.finished[-1]
    assert final.status == "failed"
    assert final.error == "RuntimeError: boom"
    assert summary == QueueSummary(done=0, failed=1, retried=1, finished=summary.finished)
    assert events == ["job_retry", "job_failed", "queue_empty"]


def test_permanent_job_error_does_not_retry(store: QueueStore) -> None:
    store.add("S", ("slice",), max_attempts=5)
    events, notifier = events_recorder()
    ran: list[int] = []

    def broken(job: Job) -> None:
        ran.append(job.id)
        raise PermanentJobError("stage 'detect' is not implemented yet")

    summary = run_queue(store, broken, notifier)

    assert ran == [1]
    final = summary.finished[0]
    assert final.status == "failed"
    assert final.attempts == 1
    assert final.error == "PermanentJobError: stage 'detect' is not implemented yet"
    assert summary.done == 0 and summary.failed == 1 and summary.retried == 0
    assert events == ["job_failed", "queue_empty"]


def test_max_jobs_stops_early(store: QueueStore) -> None:
    for name in ("S1", "S2", "S3"):
        store.add(name, ("slice",))
    events, notifier = events_recorder()
    ran: list[int] = []

    summary = run_queue(store, lambda job: ran.append(job.id), notifier, max_jobs=1)

    assert ran == [1]
    assert summary.done == 1
    remaining = store.list(status="queued")
    assert [job.series for job in remaining] == ["S2", "S3"]
    assert events == ["job_done"]  # no queue_empty event when max_jobs stopped the loop


def test_crash_recovery_refunds_the_interrupted_attempt(store: QueueStore) -> None:
    store.add("S", ("slice",))
    crashed = store.claim_next()  # a worker died right after claiming
    assert crashed is not None
    ran: list[int] = []
    events, notifier = events_recorder()

    summary = run_queue(store, lambda job: ran.append(job.id), notifier)

    assert ran == [1]
    final = summary.finished[0]
    assert final.status == "done"
    assert final.attempts == 1  # the interrupted attempt was refunded by requeue_running
    assert events == ["job_done", "queue_empty"]


def test_keyboard_interrupt_propagates_and_refunds(store: QueueStore) -> None:
    store.add("S", ("slice",))

    def interrupted(job: Job) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_queue(store, interrupted)

    state = store.get(1)
    assert state is not None
    assert state.status == "queued"
    assert state.attempts == 0


def test_broken_notifier_never_fails_the_run(store: QueueStore) -> None:
    store.add("S", ("slice",))

    def broken_notifier(event: str, job: Job | None) -> None:
        raise RuntimeError("no notifications today")

    summary = run_queue(store, lambda job: None, broken_notifier)

    assert summary.done == 1
    final = summary.finished[0]
    assert final.status == "done"


def test_empty_queue_runs_nothing(store: QueueStore) -> None:
    events, notifier = events_recorder()
    ran: list[int] = []

    summary = run_queue(store, lambda job: ran.append(job.id), notifier)  # type: ignore[arg-type]

    assert summary == QueueSummary(done=0, failed=0, retried=0, finished=())
    assert ran == []
    assert events == []  # no queue_empty event when nothing ran


def test_notifier_receives_the_job_in_each_state(store: QueueStore) -> None:
    store.add("S", ("slice",))
    seen: list[tuple[str, Any]] = []

    def notifier(event: str, job: Job | None) -> None:
        seen.append((event, None if job is None else (job.id, job.status)))

    run_queue(store, lambda job: None, notifier)

    assert ("job_done", (1, "done")) in seen
    assert ("queue_empty", None) in seen
