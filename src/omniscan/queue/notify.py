"""Queue completion notifications: log records, JSON webhooks and combined notifier stacks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from omniscan.queue.worker import Notifier

if TYPE_CHECKING:
    from omniscan.queue.store import Job

log = logging.getLogger("omniscan.queue")


def log_notifier(event: str, job: Job | None) -> None:
    """Emit one INFO line describing a queue event (event, job id/series/status/error when present)."""
    if job is None:
        log.info("queue event %s (no job)", event)
        return
    message = f"queue event {event}: job {job.id} {job.series!r} -> {job.status}"
    if job.error is not None:
        message += f": {job.error}"
    log.info(message)


def redacted_url(url: str) -> str:
    """`url` with only its scheme and host kept: webhook URLs carry their secret in the path or query
    (`https://discord.com/api/webhooks/<id>/<token>`), so a log line must never show the rest."""
    parsed = httpx.URL(url)
    return f"{parsed.scheme}://{parsed.host}/…" if parsed.host else "<webhook>"


def _failure(exc: httpx.HTTPError) -> str:
    """What went wrong, without the request URL httpx puts in its own messages."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"


def webhook_notifier(url: str, *, timeout: float = 5.0, client: httpx.Client | None = None) -> Notifier:
    """POST each event as JSON to `url`; HTTP and transport errors are logged at WARNING and swallowed.

    The log names only the webhook's host (`redacted_url`): the rest of a webhook URL is its secret.
    """

    def _notify(event: str, job: Job | None) -> None:
        payload = {"event": event, "job": None if job is None else _job_payload(job)}
        try:
            if client is None:
                with httpx.Client(timeout=timeout) as fresh:
                    response = fresh.post(url, json=payload)
            else:
                response = client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("webhook %s failed for event %s: %s", redacted_url(url), event, _failure(exc))

    return _notify


def combine(*notifiers: Notifier) -> Notifier:
    """Call every notifier in order; one raising is logged at WARNING and does not stop the rest."""

    def _notify(event: str, job: Job | None) -> None:
        for notifier in notifiers:
            try:
                notifier(event, job)
            except Exception as exc:
                log.warning(
                    "notifier %s failed for event %s: %s", getattr(notifier, "__name__", notifier), event, exc
                )

    return _notify


def _job_payload(job: Job) -> dict[str, object]:
    return {
        "id": job.id,
        "series": job.series,
        "chapters": None if job.chapters is None else list(job.chapters),
        "stages": list(job.stages),
        "status": job.status,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "error": job.error,
    }
