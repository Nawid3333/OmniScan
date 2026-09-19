"""Gap tests from the Q3 mutation review: each one kills a mutant no existing test could see."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omniscan.acquire.drm import drm_platform
from omniscan.acquire.filters import url_looks_like_page
from omniscan.acquire.sources import ChapterSource, load_sources


def write_sources(tmp_path: Path, text: str) -> Path:
    """Write a sources.toml into tmp_path and return its path."""
    path = tmp_path / "sources.toml"
    path.write_text(text, encoding="utf-8")
    return path


# drm.py


def test_every_drm_platform_is_pinned_by_name() -> None:
    """Each platform is asserted directly, so dropping any dict entry is visible."""
    assert drm_platform("https://comic.naver.com/webtoon/1") == "Naver Webtoon"
    assert drm_platform("https://page.kakao.com/read/1") == "KakaoPage"
    assert drm_platform("https://webtoon.kakao.com/read/1") == "Kakao Webtoon"
    assert drm_platform("https://lezhin.com/episode/1") == "Lezhin"
    assert drm_platform("https://lezhinus.com/episode/1") == "Lezhin"
    assert drm_platform("https://ridibooks.com/book/1") == "Ridibooks"
    assert drm_platform("https://bomtoon.com/episode/1") == "Bomtoon"
    assert drm_platform("https://kuaikanmanhua.com/webtoon/1") == "Kuaikan"


def test_www_without_dot_is_a_different_host() -> None:
    """Only 'www.' is stripped, so a host literally called wwwlezhin.com is not Lezhin."""
    assert drm_platform("https://wwwlezhin.com/episode/1") is None


# filters.py


def test_http_urls_look_like_pages() -> None:
    """Plain http is a chapter-page scheme too, not only https."""
    assert url_looks_like_page("http://cdn.example.org/ch1/001.jpg") == (True, "")


# sources.py


def test_template_with_first_equal_to_last(tmp_path: Path) -> None:
    """A one-chapter template range is valid."""
    path = write_sources(
        tmp_path,
        '[template]\nurl = "https://example.org/a/{n}"\nfirst = 3\nlast = 3\n',
    )
    assert load_sources(path) == [ChapterSource("Chapter 3", "https://example.org/a/3")]


def test_template_bounds_must_be_integers(tmp_path: Path) -> None:
    """A string bound is rejected even when the other bound is a valid integer."""
    path = write_sources(
        tmp_path,
        '[template]\nurl = "https://example.org/a/{n}"\nfirst = "2"\nlast = 3\n',
    )
    with pytest.raises(ValueError, match=re.escape("template: 'first' and 'last' must be integers")):
        load_sources(path)


def test_template_names_may_not_repeat_without_placeholder(tmp_path: Path) -> None:
    """A template without {n} that spans two chapters produces duplicate names."""
    path = write_sources(
        tmp_path,
        '[template]\nurl = "https://example.org/a/{n}"\nfirst = 1\nlast = 2\nname = "Fixed"\n',
    )
    with pytest.raises(ValueError, match=re.escape("duplicate chapter name 'Fixed'")):
        load_sources(path)


def test_empty_template_table_is_ignored(tmp_path: Path) -> None:
    """An empty [template] table contributes no chapters and is not an error."""
    assert load_sources(write_sources(tmp_path, "[template]\n")) == []


def test_template_name_must_be_a_string(tmp_path: Path) -> None:
    """A non-string template name is rejected with the unsafe-name error."""
    path = write_sources(
        tmp_path,
        '[template]\nurl = "https://example.org/a/{n}"\nfirst = 1\nlast = 2\nname = 5\n',
    )
    with pytest.raises(ValueError, match=re.escape("unsafe chapter name 5")):
        load_sources(path)
