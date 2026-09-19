"""Heuristics that tell chapter pages apart from logos, banners and other UI images."""

from __future__ import annotations

from urllib.parse import urlsplit

NON_PAGE_URL_TOKENS: tuple[str, ...] = (
    "logo",
    "banner",
    "avatar",
    "icon",
    "sprite",
    "emoji",
    "favicon",
    "watermark",
    "thumb",
    "/ads/",
    "/ad/",
    "advert",
)


def url_looks_like_page(url: str) -> tuple[bool, str]:
    """(True, "") when the URL looks like a chapter page; otherwise (False, reason)."""
    if urlsplit(url).scheme not in ("http", "https"):
        return (False, "not an http(s) url")
    path = urlsplit(url).path.lower()
    for token in NON_PAGE_URL_TOKENS:
        if token in path:
            return (False, f"url contains {token!r}")
    return (True, "")


def image_looks_like_page(width: int, height: int, size_bytes: int) -> tuple[bool, str]:
    """(True, "") when a decoded image looks like a chapter page; otherwise (False, reason)."""
    if width < 300:
        return (False, "too narrow")
    if height < 100:
        return (False, "too short")
    if width / height > 6.0:
        return (False, "banner shape")
    if size_bytes < 8000:
        return (False, "tiny file")
    return (True, "")
