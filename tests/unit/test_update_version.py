"""Tests for omniscan.update.version (card U6)."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from itertools import pairwise

import pytest

from omniscan.update import version as version_module
from omniscan.update.version import Version, compare_versions, current_version, is_app_tag, parse_version

# The SemVer 2.0 spec's precedence example, ascending.
CHAIN = (
    "1.0.0-alpha",
    "1.0.0-alpha.1",
    "1.0.0-alpha.beta",
    "1.0.0-beta",
    "1.0.0-beta.2",
    "1.0.0-beta.11",
    "1.0.0-rc.1",
    "1.0.0",
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("v1.2.3", Version(1, 2, 3, ())),
        ("1.2.3", Version(1, 2, 3, ())),
        ("01.2.3", Version(1, 2, 3, ())),
        ("1.2.3-beta.2", Version(1, 2, 3, ("beta", 2))),
        ("v0.1.0-rc.1", Version(0, 1, 0, ("rc", 1))),
        ("v1.2.3-alpha.1.2", Version(1, 2, 3, ("alpha", 1, 2))),
        ("v1.2.3-1", Version(1, 2, 3, (1,))),
        ("v1.2.3-alpha-beta.1", Version(1, 2, 3, ("alpha-beta", 1))),
    ],
)
def test_parse_version_valid(text: str, expected: Version) -> None:
    assert parse_version(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "1.2",
        "1.2.3+meta",
        "1.2.3-beta.2+build",
        "latest",
        "",
        "v1.2.3-",
        "1.2.3-",
        "v1.2.3.4",
        "1.2.3.4",
        "models-v1",
        "v",
        "v1",
        "1.2.3 beta",
    ],
)
def test_parse_version_invalid(text: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f"not a version: {text!r}")):
        parse_version(text)


def test_version_str() -> None:
    assert str(Version(1, 2, 3, ())) == "1.2.3"
    assert str(parse_version("v1.2.3-beta.2")) == "1.2.3-beta.2"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v1.2.3", True),
        ("v0.1.0-rc.1", True),
        ("v10.0.0", True),
        ("models-v1", False),
        ("v1", False),
        ("v1.2", False),
        ("1.2.3", False),
        ("v1.2.3.4", False),
        ("v1.2.3-", False),
        ("v-1.2.3", False),
        ("", False),
    ],
)
def test_is_app_tag(tag: str, expected: bool) -> None:
    assert is_app_tag(tag) is expected


def test_compare_versions_semver_chain() -> None:
    parsed = [parse_version(tag) for tag in CHAIN]
    for lower, higher in pairwise(parsed):
        assert compare_versions(lower, higher) == -1
        assert compare_versions(higher, lower) == 1


def test_compare_versions_numeric_and_equality() -> None:
    assert compare_versions(parse_version("1.2.10"), parse_version("1.2.9")) == 1
    assert compare_versions(parse_version("2.0.0"), parse_version("1.99.99")) == 1
    assert compare_versions(parse_version("1.2.3"), parse_version("v1.2.3")) == 0
    assert compare_versions(parse_version("1.0.0-rc.1"), parse_version("v1.0.0-rc.1")) == 0


def test_compare_versions_prerelease_is_lower_than_its_core() -> None:
    assert compare_versions(parse_version("1.0.0-rc.1"), parse_version("1.0.0")) == -1
    assert compare_versions(parse_version("1.0.0"), parse_version("1.0.0-rc.1")) == 1


def test_current_version_parses_the_installed_metadata() -> None:
    assert isinstance(current_version(), Version)


def test_current_version_falls_back_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version_module, "_installed_version", missing)
    assert current_version() == Version(0, 0, 0, ())
