"""Settings service: validated reads/writes for the settings page (Qt-free).

Global values go through `core.config.set_user_setting` (validated against the config models);
`clear_global` complements it for the keys a GUI must be able to unset (the runner-facing writer
can only set). Per-series overrides use `set_series_setting` (only `SERIES_SECTIONS`); removing one
rewrites `<series>/series.toml` without the key. All writers raise `SettingError` with a one-line
message the Settings page shows inline.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from omniscan.core.config import (
    SERIES_SECTIONS,
    USER_TOML,
    Config,
    SettingError,
    dumps_toml,
    set_series_setting,
    set_user_setting,
)

type FieldKind = Literal["path", "bool", "choice", "model", "float"]

_DEVICE_RE = re.compile(r"^(auto|cpu|mps|xpu(?::\d+)?|cuda(?::\d+)?)$")


@dataclass(frozen=True, slots=True)
class SettingField:
    """One editable global setting: the config key it writes and the editor to render."""

    section: str
    key: str
    kind: FieldKind
    choices: tuple[str, ...] = ()  # "choice": fixed options; editable combos allow more
    editable: bool = False  # "choice": the user may type a value outside `choices` (gpu device)
    low: float = 0.0  # "float": spin range
    high: float = 1.0
    decimals: int = 2


GLOBAL_FIELDS: tuple[SettingField, ...] = (
    SettingField("paths", "library_root", "path"),
    SettingField("paths", "work_root", "path"),
    SettingField("paths", "output_root", "path"),
    SettingField("paths", "models_dir", "path"),
    SettingField("gpu", "device", "choice", choices=("auto", "cpu", "mps", "cuda"), editable=True),
    SettingField("gpu", "warmup", "bool"),
    SettingField("ocr", "engine", "choice", choices=("ppocr", "manga_ocr", "paddleocr_vl")),
    SettingField("ocr", "det_model", "model"),  # catalog id; empty means the default (None)
    SettingField("ocr", "rec_model", "model"),
    SettingField("slicer", "strategy", "choice", choices=("smart", "page", "fixed", "simple_gutter")),
    SettingField("filter", "enabled", "bool"),
    SettingField("filter", "threshold", "float", low=0.0, high=1.0),
)


def current_value(cfg: Config, section: str, key: str) -> Any:
    """The effective value of one setting as JSON-able data (Paths as strings, None stays None)."""
    return cfg.model_dump(mode="json")[section][key]


def section_keys(section: str) -> tuple[str, ...]:
    """The setting keys a config section accepts (for the per-series override editor)."""
    return tuple(Config().model_dump()[section])


def set_global(section: str, key: str, value: Any, *, path: Path | None = None) -> Path:
    """Validate and write one global setting to the user config; raises SettingError when refused."""
    if section == "gpu" and key == "device" and isinstance(value, str) and not _DEVICE_RE.match(value):
        raise SettingError(f"gpu.device: {value!r} is not auto | cpu | mps | xpu[:N] | cuda[:N]")
    return set_user_setting(section, key, value, path=path)


def clear_global(section: str, key: str, *, path: Path | None = None) -> bool:
    """Drop one key from the user config so the built-in default applies again; True when changed."""
    file = path or USER_TOML
    data = _read_toml(file)
    if key not in data.get(section, {}):
        return False
    del data[section][key]
    if not data[section]:  # the table held only this key: remove the empty table too
        del data[section]
    _write_toml(file, data)
    return True


def series_overrides(series_dir: Path) -> dict[str, dict[str, Any]]:
    """The per-series overrides that exist in `<series_dir>/series.toml`, allowed sections only."""
    data = _read_toml(series_dir / "series.toml")
    return {section: dict(table) for section, table in data.items() if section in SERIES_SECTIONS}


def set_series_override(series_dir: Path, section: str, key: str, value: Any) -> Path:
    """Write one per-series override (validated); raises SettingError when the section/key is unknown."""
    return set_series_setting(series_dir, section, key, value)


def remove_series_override(series_dir: Path, section: str, key: str) -> bool:
    """Delete one override from `<series_dir>/series.toml` so the machine value applies; True when changed."""
    file = series_dir / "series.toml"
    data = _read_toml(file)
    if key not in data.get(section, {}):
        return False
    del data[section][key]
    if not data[section]:
        del data[section]
    if data:
        _write_toml(file, data)
    else:
        file.unlink(missing_ok=True)  # nothing left to override: an empty file is worse than none
    return True


def parse_value(text: str) -> Any:
    """A settings value typed as text: bool/int/float/list via JSON, everything else a string."""
    if not text.strip():
        raise ValueError("enter a value")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def model_ids() -> tuple[str, ...]:
    """Every catalog model id (sorted), for the `ocr.*_model` combos."""
    from omniscan.models.catalog import load_catalog

    return tuple(sorted(entry.id for entry in load_catalog()))


def translation_profiles(paths: Sequence[Path] | None = None) -> list[Any]:
    """The translation profiles in effect (repo defaults + user overrides), in file order.

    Returns `TranslationProfile`s; typed `Any` here to keep the GUI service free of that import.
    """
    from omniscan.translate.profiles import default_profile_paths, load_profiles

    return list(load_profiles(list(paths) if paths is not None else default_profile_paths()).values())


def set_profile_enabled(name: str, enabled: bool, *, user_path: Path | None = None) -> Path:
    """Enable/disable one profile by name in the user's `translation_profiles.toml`; returns the file.

    The profile must exist in the repo or user file; a user entry keeps its other keys. There is no
    `Config` section for profiles, so this is the settings page's own writer (validated by
    re-parsing the profile model). The file layout is `[profiles.<name>]`, as `load_profiles` reads it.
    """
    from omniscan.translate.profiles import TranslationProfile, default_profile_paths, load_profiles

    paths = default_profile_paths()
    if user_path is None:
        user_path = paths[1]
    if name not in load_profiles(paths):
        raise SettingError(f"unknown translation profile {name!r}")
    data = _read_toml(user_path)
    tables = dict(data.get("profiles", {}))
    table = dict(tables.get(name) or _read_toml(paths[0]).get("profiles", {}).get(name, {}))
    table["enabled"] = enabled
    try:
        TranslationProfile(name=name, **table)  # the table key is the name (tables must not set 'name')
    except ValidationError as error:
        raise SettingError(f"profile {name!r}: {error.errors()[0].get('msg', 'invalid value')}") from error
    tables[name] = table
    _write_profiles_toml(user_path, {"profiles": tables})
    return user_path


def _write_profiles_toml(path: Path, data: dict[str, Any]) -> None:
    """Write the user profile file (`dumps_toml` has no nested tables; profile tables are flat)."""
    lines: list[str] = []
    for name, table in data.get("profiles", {}).items():
        if lines:
            lines.append("")
        lines.append(f"[profiles.{name}]")
        for key, value in table.items():
            lines.append(f"{key} = {json.dumps(value)}")  # str/bool/number json literals are valid TOML
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _write_toml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps_toml(dict(data)), encoding="utf-8", newline="\n")
    tmp.replace(path)
