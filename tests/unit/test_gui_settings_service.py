"""SettingsService tests (Qt-free): validated writes with the TOML asserted on disk."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from omniscan.core.config import Config, SettingError
from omniscan.gui.services import settings

# ---------------------------------------------------------------------- global settings


def test_set_global_writes_the_user_toml(tmp_path: Path) -> None:
    """A valid value lands in the TOML at the given path."""
    toml = tmp_path / "config.toml"
    settings.set_global("gpu", "device", "cuda:1", path=toml)
    settings.set_global("gpu", "warmup", True, path=toml)
    data = tomllib.loads(toml.read_text(encoding="utf-8"))
    assert data == {"gpu": {"device": "cuda:1", "warmup": True}}


def test_set_global_rejects_invalid_values(tmp_path: Path) -> None:
    """Bad values raise SettingError and write nothing."""
    toml = tmp_path / "config.toml"
    with pytest.raises(SettingError, match="not auto"):
        settings.set_global("gpu", "device", "not-a-device", path=toml)
    with pytest.raises(SettingError):
        settings.set_global("ocr", "engine", "does-not-exist", path=toml)
    assert not toml.exists()


def test_set_global_rejects_a_bad_gpu_device_pattern(tmp_path: Path) -> None:
    """gpu.device is pre-checked (auto | cpu | mps | xpu[:N] | cuda[:N]) before the config model."""
    with pytest.raises(SettingError, match=r"gpu\.device"):
        settings.set_global("gpu", "device", "cuda:9:9", path=tmp_path / "config.toml")


def test_clear_global_removes_the_key_and_the_empty_table(tmp_path: Path) -> None:
    """clear_global deletes the key; a table left empty disappears; a missing key reports False."""
    toml = tmp_path / "config.toml"
    settings.set_global("gpu", "device", "cuda:1", path=toml)
    settings.set_global("filter", "enabled", False, path=toml)
    assert settings.clear_global("gpu", "device", path=toml)
    data = tomllib.loads(toml.read_text(encoding="utf-8"))
    assert data == {"filter": {"enabled": False}}
    assert settings.clear_global("gpu", "device", path=toml) is False
    assert settings.clear_global("filter", "enabled", path=toml)
    assert tomllib.loads(toml.read_text(encoding="utf-8")) == {}


# ---------------------------------------------------------------------- per-series overrides


def test_series_override_round_trip(tmp_path: Path) -> None:
    """Set, list and remove one override; removing the last key drops the file."""
    series_dir = tmp_path / "Series"
    series_dir.mkdir()
    settings.set_series_override(series_dir, "slicer", "strategy", "fixed")
    data = tomllib.loads((series_dir / "series.toml").read_text(encoding="utf-8"))
    assert data == {"slicer": {"strategy": "fixed"}}
    assert settings.series_overrides(series_dir) == {"slicer": {"strategy": "fixed"}}

    assert settings.remove_series_override(series_dir, "slicer", "strategy") is True
    assert not (series_dir / "series.toml").exists()  # nothing left: an empty file is worse than none
    assert settings.series_overrides(series_dir) == {}
    assert settings.remove_series_override(series_dir, "slicer", "strategy") is False


def test_series_override_rejects_unknown_sections_and_values(tmp_path: Path) -> None:
    """Only SERIES_SECTIONS sections and real keys/values are accepted."""
    series_dir = tmp_path / "Series"
    series_dir.mkdir()
    with pytest.raises(SettingError):
        settings.set_series_override(series_dir, "gpu", "device", "auto")  # gpu is not per-series
    with pytest.raises(SettingError):
        settings.set_series_override(series_dir, "slicer", "not-a-key", 1)
    with pytest.raises(SettingError):
        settings.set_series_override(series_dir, "slicer", "strategy", "bogus")


# ---------------------------------------------------------------------- values, models, profiles


def test_parse_value_json_or_plain_text() -> None:
    """Numbers/bools/lists parse as JSON, other text stays a string, empty is refused."""
    assert settings.parse_value("1.5") == 1.5
    assert settings.parse_value("true") is True
    assert settings.parse_value("[1, 2]") == [1, 2]
    assert settings.parse_value("fixed") == "fixed"
    with pytest.raises(ValueError, match="enter a value"):
        settings.parse_value("   ")


def test_current_value_and_section_keys() -> None:
    """Values come from the config model; section keys come from the defaults."""
    cfg = Config()
    assert settings.current_value(cfg, "gpu", "device") == cfg.gpu.device
    assert "strategy" in settings.section_keys("slicer")


def test_model_ids_are_the_sorted_catalog() -> None:
    """The model combos get every catalog id, sorted (never empty)."""
    ids = settings.model_ids()
    assert ids and ids == tuple(sorted(ids))


def test_set_profile_enabled_round_trip(tmp_path: Path) -> None:
    """Toggling writes the user file: an existing entry flips, a new one copies the repo entry."""
    user = tmp_path / "translation_profiles.toml"
    settings.set_profile_enabled("glm-5-3-flash-cloud", True, user_path=user)
    data = tomllib.loads(user.read_text(encoding="utf-8"))["profiles"]
    assert data["glm-5-3-flash-cloud"]["enabled"] is True

    settings.set_profile_enabled("glm-5-3-flash-cloud", False, user_path=user)
    data = tomllib.loads(user.read_text(encoding="utf-8"))["profiles"]
    assert data["glm-5-3-flash-cloud"]["enabled"] is False

    settings.set_profile_enabled("kimi-k3-cloud", True, user_path=user)  # not in the user file yet
    data = tomllib.loads(user.read_text(encoding="utf-8"))["profiles"]
    assert data["kimi-k3-cloud"]["enabled"] is True
    assert data["kimi-k3-cloud"]["endpoint"]  # the repo entry's other keys were copied
    assert data["glm-5-3-flash-cloud"]["enabled"] is False  # the other toggled entry is untouched


def test_translation_profiles_lists_the_catalog(tmp_path: Path) -> None:
    """The effective profile list comes from the repo + user files (names unique)."""
    profiles = settings.translation_profiles()
    names = [profile.name for profile in profiles]
    assert len(names) == len(set(names)) and names
    assert settings.translation_profiles([tmp_path / "missing.toml"]) == []


def test_set_profile_enabled_rejects_unknown_names(tmp_path: Path) -> None:
    with pytest.raises(SettingError, match="unknown translation profile"):
        settings.set_profile_enabled("nope", True, user_path=tmp_path / "profiles.toml")
    assert not (tmp_path / "profiles.toml").exists()
