"""Single-worker queue drain loop: claim jobs, run them through an executor, notify on the outcome.

Exactly one worker per `queue.db` is supported — the store has no second-worker lease. Recovering a crashed
worker is part of every drain: `run_queue` first moves jobs a dead worker left `running` back to `queued`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from omniscan.queue.store import Job, QueueStore

log = logging.getLogger("omniscan.queue")


class PermanentJobError(Exception):
    """Raised by an executor: the job can never succeed, do not retry it."""


Executor = Callable[[Job], None]  # raises on failure, returns None on success
Notifier = Callable[
    [str, "Job | None"], None
]  # (event, job); event is a job_done / job_failed / job_retry / queue_empty


@dataclass(frozen=True, slots=True)
class QueueSummary:
    """What one drain executed: outcome counts plus every executed job in its post-execution state."""

    done: int
    failed: int
    retried: int
    finished: tuple[Job, ...]


def run_queue(
    store: QueueStore,
    executor: Executor,
    notifier: Notifier | None = None,
    *,
    max_jobs: int | None = None,
) -> QueueSummary:
    """Drain the queue one job at a time until it is empty (or `max_jobs` have been executed).

    Recovering jobs a crashed worker left `running` is part of the drain. `max_jobs` stops after that
    many executed jobs; when the queue ran empty after at least one job, a `queue_empty` event fires.
    """
    store.requeue_running()
    done = failed = retried = 0
    finished: list[Job] = []
    drained = False  # the loop ended because claim_next() found nothing queued
    while max_jobs is None or len(finished) < max_jobs:
        job = store.claim_next()
        if job is None:
            drained = True
            break
        try:
            executor(job)
        except PermanentJobError as exc:
            final = store.fail(job.id, f"{type(exc).__name__}: {exc}", retry=False)
            finished.append(final)
            failed += 1
            _notify(notifier, "job_failed", final)
        except Exception as exc:  # the executor decides; the job records what happened
            final = store.fail(job.id, f"{type(exc).__name__}: {exc}")
            finished.append(final)
            if final.status == "queued":
                retried += 1
                _notify(notifier, "job_retry", final)
            else:
                failed += 1
                _notify(notifier, "job_failed", final)
        except BaseException:
            store.requeue_running()  # the interrupted job goes back to queued, attempt refunded
            raise
        else:
            final = store.complete(job.id)
            finished.append(final)
            done += 1
            _notify(notifier, "job_done", final)
    if drained and finished:
        _notify(notifier, "queue_empty", None)
    return QueueSummary(done=done, failed=failed, retried=retried, finished=tuple(finished))


def _notify(notifier: Notifier | None, event: str, job: Job | None) -> None:
    """Call the notifier; a broken notifier must never fail a job or the run."""
    if notifier is None:
        return
    try:
        notifier(event, job)
    except Exception as exc:
        log.warning("notifier failed for event %s: %s", event, exc)
