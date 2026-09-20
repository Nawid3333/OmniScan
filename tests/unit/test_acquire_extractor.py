"""Contract tests for acquire.extractor."""

from __future__ import annotations

from omniscan.acquire.extractor import AuthError, ExtractError, QuotaError, credit_cost


def test_credit_cost_per_mode() -> None:
    assert credit_cost(0) == 0
    assert credit_cost(7) == 7
    assert credit_cost(7, "advanced") == 14


def test_errors_share_a_base() -> None:
    assert issubclass(AuthError, ExtractError)
    assert issubclass(QuotaError, ExtractError)
