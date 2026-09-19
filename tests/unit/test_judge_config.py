"""Tests for omniscan.translate.judge_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import DEFAULT_TOML, USER_TOML
from omniscan.translate.judge_config import JudgeConfig, default_judge_paths, load_judge_config

SHIPPED = Path(DEFAULT_TOML.parent / "judge.toml")


def write_judge(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_file_loads_the_documented_defaults() -> None:
    cfg = load_judge_config([SHIPPED])
    assert cfg.model == "gemma4:31b-cloud"
    assert cfg.endpoint == "local"
    assert cfg.think is False
    assert cfg.temperature == 0.2
    assert cfg.chunk_regions == 20
    assert cfg.max_repair_rounds == 1
    assert cfg.agree_threshold == 0.9
    assert cfg.always_judge is False
    assert cfg.prefer == ["gemma4-31b-cloud", "gemma4-12b-local", "translategemma-12b-local"]


def test_direct_construction_defaults_match_the_toml() -> None:
    cfg = JudgeConfig()
    assert (cfg.model, cfg.endpoint, cfg.think) == ("gemma4:31b-cloud", "local", False)
    assert (cfg.temperature, cfg.chunk_regions, cfg.max_repair_rounds) == (0.2, 20, 1)
    assert (cfg.agree_threshold, cfg.always_judge, cfg.prefer) == (0.9, False, [])


def test_later_file_overrides_only_its_keys_and_missing_files_are_skipped(tmp_path: Path) -> None:
    first = write_judge(tmp_path / "a.toml", '[judge]\ntemperature = 0.5\nprefer = ["x"]\n')
    second = write_judge(tmp_path / "b.toml", '[judge]\nmodel = "m2"\n')
    cfg = load_judge_config([first, tmp_path / "missing.toml", second])
    assert cfg.temperature == 0.5  # set by the first file, untouched by the second
    assert cfg.prefer == ["x"]
    assert cfg.model == "m2"
    assert cfg.chunk_regions == 20  # untouched default
    assert cfg.think is False


def test_no_existing_files_load_the_defaults(tmp_path: Path) -> None:
    assert load_judge_config([tmp_path / "nope.toml"]) == JudgeConfig()


def test_default_judge_paths_order() -> None:
    assert default_judge_paths() == [
        Path(DEFAULT_TOML.parent / "judge.toml"),
        Path(USER_TOML.parent / "judge.toml"),
    ]


@pytest.mark.parametrize(
    "body",
    [
        "[judge]\ntemperature = 3.0\n",
        "[judge]\nchunk_regions = 0\n",
        "[judge]\nagree_threshold = 1.5\n",
        "[judge]\nmax_repair_rounds = -1\n",
        '[judge]\nendpoint = "remote"\n',
        "[judge]\nunknown_key = 1\n",
        '[judge]\nthink = "maybe"\n',
    ],
    ids=[
        "high-temperature",
        "zero-chunk",
        "threshold-above-one",
        "negative-repair-rounds",
        "unknown-endpoint",
        "unknown-key",
        "think-type",
    ],
)
def test_invalid_values_raise_value_error_naming_the_file(tmp_path: Path, body: str) -> None:
    path = write_judge(tmp_path / "judge.toml", body)
    with pytest.raises(ValueError, match=r"judge\.toml"):
        load_judge_config([path])


def test_invalid_toml_raises_value_error_naming_the_file(tmp_path: Path) -> None:
    path = write_judge(tmp_path / "broken.toml", "[judge\nmodel = ")
    with pytest.raises(ValueError, match=r"broken\.toml"):
        load_judge_config([path])
