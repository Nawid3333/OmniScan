"""The errors every chat client raises (Ollama, the OpenAI and Anthropic APIs, the cloud request budget).

`OllamaRateLimitError` is the "stop asking this model" signal the translate stage, the judge and the Studio
act on (a profile switches to its fallback, the judge keeps its deterministic picks); the API clients and the
per-run cloud budget raise subclasses of it so that machinery covers them unchanged.
"""

from __future__ import annotations


class OllamaError(RuntimeError):
    """A model request failed (non-retryable status, or retryable failures exhausted)."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OllamaRateLimitError(OllamaError):
    """Raised only when retries are exhausted on an HTTP 429. status_code is always 429."""
