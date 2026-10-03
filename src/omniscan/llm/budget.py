"""A cap on the cloud requests of one pipeline run (`[translate] cloud_request_budget`, #41).

Ollama Cloud's usage limit is weekly and account-wide, so one big `omniscan run` could spend it. Inside
`cloud_budget(limit)` every request to a cloud model — an Ollama `*-cloud` / `:cloud` model or the cloud
endpoint, the OpenAI or Anthropic API — first calls `spend_cloud_request`; once `limit` requests were sent, the
next one raises `CloudBudgetError`, a rate-limit error, so the run carries on the way it does when the cloud
answers 429: a profile switches to its (local) fallback and the judge keeps its deterministic picks. Outside a
budget (the Studio's on-demand Translate, `omniscan translate`) nothing is counted.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from omniscan.llm.errors import OllamaRateLimitError


class CloudBudgetError(OllamaRateLimitError):
    """The run's cloud request budget is spent; status_code is 429 so callers treat it as the rate limit."""


@dataclass(slots=True)
class _Budget:
    limit: int
    spent: int = 0


_ACTIVE: ContextVar[_Budget | None] = ContextVar("omniscan_cloud_budget", default=None)


@contextmanager
def cloud_budget(limit: int) -> Iterator[None]:
    """Count the cloud requests made inside this block and refuse the ones past `limit` (0: no limit)."""
    token = _ACTIVE.set(_Budget(limit) if limit > 0 else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def spend_cloud_request(model: str) -> None:
    """Count one cloud request to `model`; CloudBudgetError when the active budget is already spent."""
    budget = _ACTIVE.get()
    if budget is None:
        return
    if budget.spent >= budget.limit:
        raise CloudBudgetError(
            f"cloud request budget spent: {budget.limit} request(s) this run ([translate] cloud_request_budget); "
            f"not asking {model} — profiles with a fallback switch to it",
            status_code=429,
        )
    budget.spent += 1


def cloud_requests_spent() -> int | None:
    """How many cloud requests the active budget counted, or None outside a budget."""
    budget = _ACTIVE.get()
    return None if budget is None else budget.spent
