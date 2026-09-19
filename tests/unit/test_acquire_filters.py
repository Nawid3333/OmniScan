"""Tests for omniscan.acquire.filters."""

from __future__ import annotations

from omniscan.acquire.filters import image_looks_like_page, url_looks_like_page

IMAGE_CASES = [
    ((300, 1000, 50000), (True, "")),
    ((299, 1000, 50000), (False, "too narrow")),
    # (800, 100, 50000) is "banner shape" (ratio 8.0), not ok: the card's acceptance table
    # lists it as ok, which contradicts its own rule width/height > 6.0 — see report B33a.
    ((800, 100, 50000), (False, "banner shape")),
    ((500, 100, 50000), (True, "")),
    ((800, 99, 50000), (False, "too short")),
    ((600, 100, 50000), (True, "")),
    ((601, 100, 50000), (False, "banner shape")),
    ((728, 90, 50000), (False, "too short")),
    ((800, 1200, 8000), (True, "")),
    ((800, 1200, 7999), (False, "tiny file")),
]


def test_boundary_values() -> None:
    for (width, height, size_bytes), expected in IMAGE_CASES:
        assert image_looks_like_page(width, height, size_bytes) == expected, (width, height, size_bytes)


def test_url_without_tokens() -> None:
    assert url_looks_like_page("https://cdn.example.org/ch1/001.jpg") == (True, "")


def test_url_tokens() -> None:
    assert url_looks_like_page("https://cdn.example.org/ch1/Logo.png") == (False, "url contains 'logo'")
    assert url_looks_like_page("https://cdn.example.org/thumbs/1.jpg") == (False, "url contains 'thumb'")
    assert url_looks_like_page("https://cdn.example.org/ads/x.jpg") == (False, "url contains '/ads/'")


def test_token_in_query_is_ignored() -> None:
    assert url_looks_like_page("https://cdn.example.org/ch1/1.jpg?next=logo.png") == (True, "")


def test_first_matching_token_in_tuple_order_wins() -> None:
    assert url_looks_like_page("https://x.example.org/logo_banner.png") == (False, "url contains 'logo'")


def test_non_http_scheme() -> None:
    assert url_looks_like_page("ftp://x/1.jpg") == (False, "not an http(s) url")
