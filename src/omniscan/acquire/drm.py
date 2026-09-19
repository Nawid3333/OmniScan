"""Refusal to acquire from paid platforms that protect their pages with DRM."""

from __future__ import annotations

from urllib.parse import urlsplit

NOTICE = "You are responsible for having the right to download and process this content."

DRM_PLATFORMS: dict[str, str] = {
    "comic.naver.com": "Naver Webtoon",
    "page.kakao.com": "KakaoPage",
    "webtoon.kakao.com": "Kakao Webtoon",
    "lezhin.com": "Lezhin",
    "lezhinus.com": "Lezhin",
    "ridibooks.com": "Ridibooks",
    "bomtoon.com": "Bomtoon",
    "kuaikanmanhua.com": "Kuaikan",
}


class DrmPlatformError(ValueError):
    """The URL belongs to a platform whose pages OmniScan does not acquire."""


def _host(url: str) -> str | None:
    """Lower-cased hostname of a URL with one leading 'www.' removed; None when it has no host."""
    host = urlsplit(url).hostname
    if host is None:
        return None
    host = host.lower()
    return host.removeprefix("www.")


def drm_platform(url: str) -> str | None:
    """Display name of the DRM platform a URL belongs to, or None for other hosts."""
    host = _host(url)
    if host is None:
        return None
    for domain, name in DRM_PLATFORMS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None


def check_allowed(url: str) -> None:
    """Raise DrmPlatformError when the URL belongs to a DRM-protected platform."""
    name = drm_platform(url)
    if name is not None:
        raise DrmPlatformError(
            f"{name} protects its pages against extraction; OmniScan does not acquire from paid"
            " DRM platforms — import files you own with `omniscan import` instead"
        )
