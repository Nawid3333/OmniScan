"""Tests for the persistent job queue store (card B23)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC
from pathlib import Path

import pytest

from omniscan.core.config import Config, PathsConfig
from omniscan.queue.store import QueueStore, queue_db_path


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QueueStore]:
    """A QueueStore on a fresh tmp file."""
    with QueueStore(tmp_path / "queue.db") as store:
        yield store


# ---------------------------------------------------------------- add / get


def test_add_get_round_trip_defaults(store: QueueStore) -> None:
    job = store.add("S", ("slice",))
    assert job.id == 1
    assert job.chapters is None
    assert job.stages == ("slice",)
    assert job.force is False
    assert job.priority == 0
    assert job.status == "queued"
    assert job.attempts == 0
    assert job.max_attempts == 2
    assert job.error is None
    assert job.created_at.tzinfo is UTC
    assert job.started_at is None
    assert job.finished_at is None
    assert store.get(1) == job
    assert store.get(2) is None


def test_add_get_round_trip_options(store: QueueStore) -> None:
    job = store.add(
        "시리즈",
        ("ingest", "slice"),
        chapters=["Chapter 1", "Chapter 2"],
        priority=3,
        max_attempts=5,
        force=True,
    )
    fetched = store.get(1)
    assert fetched == job
    assert fetched is not None
    assert fetched.series == "시리즈"
    assert fetched.chapters == ("Chapter 1", "Chapter 2")
    assert fetched.stages == ("ingest", "slice")
    assert fetched.force is True
    assert fetched.priority == 3
    assert fetched.max_attempts == 5


def test_add_validation(store: QueueStore) -> None:
    for series, stages, chapters, max_attempts in (
        ("  ", ("slice",), None, 2),
        ("S", (), None, 2),
        ("S", ("slice", "nope"), None, 2),
        ("S", ("slice",), [], 2),
        ("S", ("slice",), None, 0),
    ):
        with pytest.raises(ValueError):
            store.add(series, stages, chapters=chapters, max_attempts=max_attempts)
    assert store.list() == []


def test_queue_db_path(tmp_path: Path) -> None:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        )
    )
    assert queue_db_path(cfg) == tmp_path / "work" / "queue.db"


# ---------------------------------------------------------------- claim order


def test_claim_order_priorities_then_id(store: QueueStore) -> None:
    first = store.add("S1", ("slice",), priority=0)
    second = store.add("S2", ("slice",), priority=5)
    third = store.add("S3", ("slice",), priority=5)
    fourth = store.add("S4", ("slice",), priority=-1)

    for expected in (second, third, first, fourth):
        job = store.claim_next()
        assert job is not None and job.id == expected.id
        assert job.status == "running"
        assert job.attempts == 1
        assert job.started_at is not None
    assert store.claim_next() is None


def test_claim_next_on_empty_queue_returns_none(store: QueueStore) -> None:
    assert store.claim_next() is None


def test_claim_next_never_claims_non_queued(store: QueueStore) -> None:
    paused = store.add("paused", ("slice",))
    store.pause(paused.id)
    assert store.claim_next() is None  # paused is never claimed

    running = store.add("running", ("slice",))
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == running.id  # the only still-queued job
    assert store.claim_next() is None  # running is never claimed

    done = store.add("done", ("slice",))
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == done.id
    store.complete(claimed.id)
    assert store.claim_next() is None  # done is never claimed

    failed = store.add("failed", ("slice",), max_attempts=1)
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == failed.id
    store.fail(claimed.id, "boom", retry=False)
    assert store.claim_next() is None  # failed is never claimed


# ---------------------------------------------------------------- fail semantics


def test_fail_requeues_then_fails(store: QueueStore) -> None:
    job = store.add("S", ("slice",), max_attempts=2)

    first = store.claim_next()
    assert first is not None and first.id == job.id
    requeued = store.fail(first.id, "RuntimeError: boom")
    assert requeued.status == "queued"
    assert requeued.attempts == 1
    assert requeued.error == "RuntimeError: boom"
    assert requeued.finished_at is None

    second = store.claim_next()
    assert second is not None and second.id == job.id and second.attempts == 2
    final = store.fail(second.id, "RuntimeError: boom again")
    assert final.status == "failed"
    assert final.finished_at is not None
    assert final.error == "RuntimeError: boom again"
    assert store.claim_next() is None


def test_fail_without_retry_fails_immediately(store: QueueStore) -> None:
    store.add("S", ("slice",), max_attempts=5)
    claimed = store.claim_next()
    assert claimed is not None
    final = store.fail(claimed.id, "PermanentJobError: nope", retry=False)
    assert final.status == "failed"
    assert final.attempts == 1
    assert final.error == "PermanentJobError: nope"
    assert final.finished_at is not None
    assert store.claim_next() is None


def test_complete_clears_the_error(store: QueueStore) -> None:
    store.add("S", ("slice",), max_attempts=2)
    claimed = store.claim_next()
    assert claimed is not None
    requeued = store.fail(claimed.id, "boom")
    assert requeued.error == "boom"
    second = store.claim_next()
    assert second is not None
    completed = store.complete(second.id)
    assert completed.status == "done"
    assert completed.error is None
    assert completed.finished_at is not None
    assert store.claim_next() is None


# ---------------------------------------------------------------- transitions


def test_pause_resume_round_trip(store: QueueStore) -> None:
    job = store.add("S", ("slice",))
    paused = store.pause(job.id)
    assert paused.status == "paused"
    resumed = store.resume(job.id)
    assert resumed.status == "queued"
    assert resumed.started_at is None
    assert resumed.finished_at is None
    claimed = store.claim_next()  # a resumed job is claimable again
    assert claimed is not None and claimed.id == job.id


def test_cancel_and_retry_round_trip(store: QueueStore) -> None:
    job = store.add("S", ("slice",))
    cancelled = store.cancel(job.id)
    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    retried = store.retry(job.id)
    assert retried.status == "queued"
    assert retried.attempts == 0
    assert retried.error is None
    assert retried.started_at is None
    assert retried.finished_at is None


def test_failed_job_can_be_retried(store: QueueStore) -> None:
    store.add("S", ("slice",), max_attempts=1)
    claimed = store.claim_next()
    assert claimed is not None
    failed = store.fail(claimed.id, "boom")
    assert failed.status == "failed"
    retried = store.retry(failed.id)
    assert retried.status == "queued"
    assert retried.attempts == 0
    assert retried.error is None


def test_illegal_transitions_raise_value_error(store: QueueStore) -> None:
    done_job = store.add("done", ("slice",))
    queued_job = store.add("queued", ("slice",))
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == done_job.id
    done = store.complete(claimed.id)

    cases = [
        ("pause", done.id, "job 1 is done, cannot pause"),
        ("resume", done.id, "job 1 is done, cannot resume"),
        ("cancel", done.id, "job 1 is done, cannot cancel"),
        ("retry", done.id, "job 1 is done, cannot retry"),
        ("complete", queued_job.id, "job 2 is queued, cannot complete"),
        ("fail", queued_job.id, "job 2 is queued, cannot fail"),
    ]
    for method, job_id, message in cases:
        with pytest.raises(ValueError) as excinfo:
            getattr(store, method)(job_id) if method != "fail" else store.fail(job_id, "boom")
        assert str(excinfo.value) == message


def test_cancel_of_running_job_raises(store: QueueStore) -> None:
    store.add("S", ("slice",))
    claimed = store.claim_next()
    assert claimed is not None
    with pytest.raises(ValueError) as excinfo:
        store.cancel(claimed.id)
    assert str(excinfo.value) == f"job {claimed.id} is running, cannot cancel"


def test_unknown_id_raises_key_error(store: QueueStore) -> None:
    for method in ("complete", "pause", "resume", "cancel", "retry"):
        with pytest.raises(KeyError):
            getattr(store, method)(99)
    with pytest.raises(KeyError):
        store.fail(99, "boom")


# ---------------------------------------------------------------- requeue / clear


def test_requeue_running_refunds_attempts(store: QueueStore) -> None:
    first = store.add("S1", ("slice",))
    second = store.add("S2", ("slice",))
    claim = store.claim_next()
    assert claim is not None and claim.id == first.id

    assert store.requeue_running() == 1
    state = store.get(first.id)
    assert state is not None
    assert state.status == "queued"
    assert state.attempts == 0  # the interrupted attempt does not count
    assert store.requeue_running() == 0
    del second


def test_requeue_running_keeps_prior_attempts(store: QueueStore) -> None:
    store.add("S", ("slice",), max_attempts=3)
    first = store.claim_next()
    assert first is not None
    store.fail(first.id, "boom")  # attempts = 1, queued again
    second = store.claim_next()
    assert second is not None and second.attempts == 2

    assert store.requeue_running() == 1
    state = store.get(1)
    assert state is not None
    assert state.status == "queued"
    assert state.attempts == 1  # only the interrupted attempt is refunded


def test_clear_finished_removes_done_and_cancelled_only(store: QueueStore) -> None:
    done = store.add("S1", ("slice",))
    failed = store.add("S2", ("slice",), max_attempts=1)
    queued = store.add("S3", ("slice",))
    claim = store.claim_next()
    assert claim is not None and claim.id == done.id
    store.complete(claim.id)
    claim = store.claim_next()
    assert claim is not None and claim.id == failed.id
    store.fail(claim.id, "boom", retry=False)

    assert store.clear_finished() == 1  # the done job
    assert store.get(done.id) is None
    assert store.get(failed.id) is not None  # failed stays for retry
    assert store.get(queued.id) is not None

    store.cancel(queued.id)
    assert store.clear_finished() == 1  # now the cancelled job
    assert store.get(queued.id) is None
    assert store.get(failed.id) is not None
    assert store.clear_finished() == 0


# ---------------------------------------------------------------- persistence / concurrency


def test_persistence_across_reopen(tmp_path: Path) -> None:
    with QueueStore(tmp_path / "queue.db") as store:
        job = store.add("S", ("ingest", "slice"), chapters=["Chapter 1"], priority=2, max_attempts=3)
        claim = store.claim_next()
        assert claim is not None
        store.fail(claim.id, "boom")
    with QueueStore(tmp_path / "queue.db") as store:
        fetched = store.get(job.id)
        assert fetched is not None
        assert fetched.status == "queued"
        assert fetched.attempts == 1
        assert fetched.error == "boom"
        assert fetched.series == "S"
        assert fetched.stages == ("ingest", "slice")
        assert fetched.chapters == ("Chapter 1",)
        assert fetched.priority == 2
        assert fetched.max_attempts == 3
        assert fetched.created_at.tzinfo is UTC


def test_two_stores_never_claim_the_same_job(tmp_path: Path) -> None:
    with QueueStore(tmp_path / "queue.db") as store:
        for i in range(10):
            store.add(f"S{i}", ("slice",))
    with QueueStore(tmp_path / "queue.db") as a, QueueStore(tmp_path / "queue.db") as b:
        claimed: list[int] = []
        for i in range(10):  # alternate the two stores
            source = a if i % 2 == 0 else b
            job = source.claim_next()
            assert job is not None
            claimed.append(job.id)
        assert sorted(claimed) == list(range(1, 11))
