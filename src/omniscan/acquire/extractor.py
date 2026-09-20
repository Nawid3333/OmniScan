"""Contract between the extraction back-ends and the acquire orchestration (owned by the director)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

type Mode = Literal["basic", "advanced"]

CREDITS_PER_EXTRACTION: dict[Mode, int] = {"basic": 1, "advanced": 2}


class ExtractError(RuntimeError):
    """An extraction failed (network error, failed job, unexpected response)."""


class AuthError(ExtractError):
    """The API key is missing or was rejected (HTTP 401)."""


class QuotaError(ExtractError):
    """Credits are exhausted or the rate limit persisted after retries (HTTP 402/429)."""


@dataclass(frozen=True, slots=True)
class ExtractedImage:
    """One image URL found on a chapter page, in document order."""

    url: str


@dataclass(frozen=True, slots=True)
class Extraction:
    """A finished extraction: every image URL of the page, unfiltered, in document order."""

    id: str
    page_url: str
    images: tuple[ExtractedImage, ...]
    credits: int


class Extractor(Protocol):
    """Anything that turns a chapter page URL into its image URLs (extract.pics, or a fake in tests)."""

    def extract(self, url: str, *, mode: Mode = "basic") -> Extraction:
        """Run one extraction to completion; raises ExtractError (AuthError, QuotaError) on failure."""
        ...


def credit_cost(count: int, mode: Mode = "basic") -> int:
    """Credits needed for `count` extractions in `mode`."""
    return count * CREDITS_PER_EXTRACTION[mode]
