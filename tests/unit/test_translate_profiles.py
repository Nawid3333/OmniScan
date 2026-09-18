"""Tests for omniscan.translate.profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import DEFAULT_TOML, USER_TOML
from omniscan.translate.profiles import TranslationProfile, default_profile_paths, load_profiles

SHIPPED = Path(DEFAULT_TOML.parent / "translation_profiles.toml")


def write_profiles(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_file_loads_the_five_documented_profiles() -> None:
    profiles = load_profiles([SHIPPED])
    assert set(profiles) == {
        "translategemma-12b-local",
        "gemma4-12b-local",
        "gemma4-31b-cloud",
        "glm-5-3-flash-cloud",
        "kimi-k3-cloud",
    }
    tg = profiles["translategemma-12b-local"]
    assert (tg.endpoint, tg.model, tg.style, tg.think, tg.enabled) == (
        "local",
        "translategemma:12b",
        "translategemma",
        None,
        True,
    )
    g12 = profiles["gemma4-12b-local"]
    assert (g12.endpoint, g12.model, g12.style, g12.think, g12.enabled) == (
        "local",
        "gemma4:12b",
        "chat_json",
        False,
        True,
    )
    g31 = profiles["gemma4-31b-cloud"]
    assert (g31.endpoint, g31.model, g31.style, g31.think) == (
        "local",
        "gemma4:31b-cloud",
        "chat_json",
        False,
    )
    glm = profiles["glm-5-3-flash-cloud"]
    assert (glm.endpoint, glm.model, glm.style, glm.think, glm.enabled) == (
        "local",
        "glm-5.3-flash:cloud",
        "chat_json",
        False,
        False,
    )
    kimi = profiles["kimi-k3-cloud"]
    assert (kimi.endpoint, kimi.model, kimi.style, kimi.think, kimi.enabled) == (
        "local",
        "kimi-k3:cloud",
        "chat_json",
        False,
        False,
    )
    for profile in profiles.values():
        assert profile.temperature == 0.3
        assert profile.chunk_regions == 30


def test_later_file_replaces_profile_and_can_add_one(tmp_path: Path) -> None:
    first = write_profiles(
        tmp_path / "a.toml",
        """
[profiles.p1]
endpoint = "local"
model = "m1"
style = "chat_json"

[profiles.p2]
endpoint = "cloud"
model = "m2"
style = "chat_json"
""",
    )
    second = write_profiles(
        tmp_path / "b.toml",
        """
[profiles.p1]
endpoint = "cloud"
model = "m1b"
style = "translategemma"

[profiles.p3]
endpoint = "local"
model = "m3"
style = "translategemma"
""",
    )
    profiles = load_profiles([first, tmp_path / "missing.toml", second])
    assert set(profiles) == {"p1", "p2", "p3"}
    assert profiles["p1"].model == "m1b"
    assert profiles["p1"].style == "translategemma"
    assert profiles["p2"].endpoint == "cloud"


def test_default_profile_paths_order() -> None:
    paths = default_profile_paths()
    assert paths == [
        Path(DEFAULT_TOML.parent / "translation_profiles.toml"),
        Path(USER_TOML.parent / "translation_profiles.toml"),
    ]


@pytest.mark.parametrize(
    "body",
    [
        """
[profiles."../x"]
endpoint = "local"
model = "m"
style = "chat_json"
""",
        """
[profiles.".hidden"]
endpoint = "local"
model = "m"
style = "chat_json"
""",
        """
[profiles.""]
endpoint = "local"
model = "m"
style = "chat_json"
""",
        """
[profiles.p]
endpoint = "local"
model = "m"
style = "chat_json"
chunk_regions = 0
""",
        """
[profiles.p]
endpoint = "local"
model = "m"
style = "chat_json"
temperature = 3.0
""",
        """
[profiles.p]
endpoint = "local"
model = "m"
style = "greedy"
""",
        """
[profiles.p]
endpoint = "local"
model = "m"
style = "chat_json"
name = "other"
""",
    ],
    ids=["bad-name", "dot-name", "empty-name", "zero-chunk", "high-temperature", "unknown-style", "name-key"],
)
def test_invalid_profiles_raise_value_error_naming_the_file(tmp_path: Path, body: str) -> None:
    path = write_profiles(tmp_path / "profiles.toml", body)
    with pytest.raises(ValueError, match=r"profiles\.toml"):
        load_profiles([path])


def test_invalid_toml_raises_value_error_naming_the_file(tmp_path: Path) -> None:
    path = write_profiles(tmp_path / "broken.toml", "[profiles.p\nendpoint = ")
    with pytest.raises(ValueError, match=r"broken\.toml"):
        load_profiles([path])


def test_profile_defaults_and_name_validation_direct() -> None:
    profile = TranslationProfile(endpoint="local", model="m", style="chat_json", name="ok-name.1")
    assert profile.enabled is True
    assert profile.think is None
    assert profile.temperature == 0.3
    assert profile.chunk_regions == 30
    with pytest.raises(ValueError, match="invalid profile name"):
        TranslationProfile(endpoint="local", model="m", style="chat_json", name="-bad")
