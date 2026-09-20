"""Per-series overrides: `<series>/series.toml` merged over the machine config."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import Config, PathsConfig, SeriesConfigError, series_config
from omniscan.core.stage import make_context


def _cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib",
            work_root=tmp_path / "work",
            output_root=tmp_path / "out",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        )
    )


def _write(series_dir: Path, text: str) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    (series_dir / "series.toml").write_text(text, encoding="utf-8")


def test_no_file_returns_the_same_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert series_config(cfg, tmp_path / "lib" / "S") is cfg


def test_overrides_merge_over_defaults(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    series_dir = tmp_path / "lib" / "S"
    _write(series_dir, '[slicer]\nstrategy = "page"\ntarget_height = 2000\n\n[ocr]\nlang = "ja"\n')
    merged = series_config(cfg, series_dir)
    assert merged.slicer.strategy == "page"
    assert merged.slicer.target_height == 2000
    assert merged.slicer.max_height == cfg.slicer.max_height  # untouched keys keep the machine value
    assert merged.ocr.lang == "ja"
    assert merged.paths == cfg.paths
    assert cfg.slicer.strategy == "smart"  # the input is not modified


def test_machine_sections_are_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    series_dir = tmp_path / "lib" / "S"
    _write(series_dir, '[paths]\nlibrary_root = "elsewhere"\n\n[gpu]\ndevice = "cpu"\n')
    with pytest.raises(SeriesConfigError, match=r"sections not allowed per series: gpu, paths"):
        series_config(cfg, series_dir)


def test_invalid_toml_and_invalid_values_raise(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    series_dir = tmp_path / "lib" / "S"
    _write(series_dir, "[slicer\n")
    with pytest.raises(SeriesConfigError, match=r"series\.toml"):
        series_config(cfg, series_dir)
    _write(series_dir, '[slicer]\nstrategy = "wobbly"\n')
    with pytest.raises(SeriesConfigError, match="strategy"):
        series_config(cfg, series_dir)


def test_make_context_applies_the_overrides(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(tmp_path / "lib" / "S", '[slicer]\nstrategy = "fixed"\n')
    ctx = make_context(cfg, "S", "Chapter 1")
    assert ctx.cfg.slicer.strategy == "fixed"
    other = make_context(cfg, "Other", "Chapter 1")
    assert other.cfg.slicer.strategy == "smart"


def test_dumps_toml_round_trips_scalars_lists_and_tables() -> None:
    import tomllib

    from omniscan.core.config import dumps_toml

    data = {
        "top": r'q"uote \ back',
        "paths": {"library_root": "C:/Users/me/лайбрари", "n": 3, "ratio": 0.25, "flag": True},
        "ocr": {"langs": ["ko", "ja"], "empty": []},
    }
    assert tomllib.loads(dumps_toml(data)) == data


def test_dumps_toml_rejects_unsupported_values() -> None:
    from omniscan.core.config import dumps_toml

    with pytest.raises(ValueError, match="cannot write"):
        dumps_toml({"a": {"b": None}})
    with pytest.raises(ValueError, match="nested table"):
        dumps_toml({"a": {"b": {"c": 1}}})


def test_set_series_setting_validates_and_writes(tmp_path: Path) -> None:
    from omniscan.core.config import SettingError, set_series_setting

    series_dir = tmp_path / "lib" / "S"
    path = set_series_setting(series_dir, "slicer", "strategy", "page")
    assert path == series_dir / "series.toml"
    set_series_setting(series_dir, "ocr", "lang", "ja")
    merged = series_config(_cfg(tmp_path), series_dir)
    assert (merged.slicer.strategy, merged.ocr.lang) == ("page", "ja")
    set_series_setting(series_dir, "slicer", "strategy", "fixed")  # replaces, keeps the other section
    merged = series_config(_cfg(tmp_path), series_dir)
    assert (merged.slicer.strategy, merged.ocr.lang) == ("fixed", "ja")
    before = path.read_text(encoding="utf-8")
    for section, key, value, match in (
        ("paths", "library_root", "x", "not allowed"),
        ("nope", "a", 1, "not allowed"),
        ("slicer", "nope", 1, "unknown key"),
        ("slicer", "strategy", "wobbly", "strategy"),
        ("slicer", "target_height", "tall", "target_height"),
    ):
        with pytest.raises(SettingError, match=match):
            set_series_setting(series_dir, section, key, value)
    assert path.read_text(encoding="utf-8") == before  # a refused write leaves the file untouched
    assert not list(series_dir.glob("*.tmp"))


def test_set_user_setting_writes_to_the_given_file(tmp_path: Path) -> None:
    import tomllib

    from omniscan.core.config import set_user_setting

    target = tmp_path / "cfg" / "config.toml"
    set_user_setting("gpu", "device", "cpu", path=target)
    set_user_setting("paths", "library_root", "D:/raws", path=target)
    assert tomllib.loads(target.read_text(encoding="utf-8")) == {
        "gpu": {"device": "cpu"},
        "paths": {"library_root": "D:/raws"},
    }
