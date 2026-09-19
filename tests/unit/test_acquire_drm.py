"""Tests for omniscan.acquire.drm."""

from __future__ import annotations

import pytest

from omniscan.acquire.drm import DRM_PLATFORMS, NOTICE, DrmPlatformError, check_allowed, drm_platform


@pytest.mark.parametrize("domain", list(DRM_PLATFORMS))
def test_every_platform_is_recognised(domain: str) -> None:
    name = DRM_PLATFORMS[domain]
    assert drm_platform(f"https://{domain}/a/1.jpg") == name
    assert drm_platform(f"https://www.{domain.upper()}/a/1.jpg") == name


def test_subdomains_and_port() -> None:
    assert drm_platform("https://m.page.kakao.com/read/1") == "KakaoPage"
    assert drm_platform("https://comic.naver.com:8080/main") == "Naver Webtoon"


def test_lookalike_hosts_are_not_platforms() -> None:
    assert drm_platform("https://example.org/a/1.jpg") is None
    assert drm_platform("https://notlezhin.com/a") is None
    assert drm_platform("https://lezhin.com.evil.org/a") is None


def test_url_without_host() -> None:
    assert drm_platform("not-a-url") is None


def test_check_allowed_raises_with_platform_and_import_hint() -> None:
    with pytest.raises(DrmPlatformError) as excinfo:
        check_allowed("https://www.lezhin.com/episode/1")
    assert "Lezhin" in str(excinfo.value)
    assert "omniscan import" in str(excinfo.value)


def test_check_allowed_passes_other_hosts() -> None:
    assert check_allowed("https://example.org/a/1.jpg") is None


def test_notice() -> None:
    assert NOTICE == "You are responsible for having the right to download and process this content."
