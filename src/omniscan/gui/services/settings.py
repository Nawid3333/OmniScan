"""Settings service: validated reads/writes for the settings page (Qt-free).

Global values go through `core.config.set_user_setting` (validated against the config models);
`clear_global` complements it for the keys a GUI must be able to unset (the runner-facing writer
can only set). Per-series overrides use `set_series_setting` (only `SERIES_SECTIONS`); removing one
rewrites `<series>/series.toml` without the key. Translation profiles are written to the user's
`translation_profiles.toml` (enable, add your own model, remove one you added), and an API key the user types
goes into `secrets.env` (never read back or shown). All writers raise `SettingError` with a one-line
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

import httpx
from pydantic import ValidationError

from omniscan.core.config import (
    SECRETS_ENV,
    SERIES_SECTIONS,
    USER_TOML,
    Config,
    SettingError,
    api_key,
    dumps_toml,
    set_series_setting,
    set_user_setting,
)

type FieldKind = Literal["path", "bool", "choice", "model", "float", "int"]

_DEVICE_RE = re.compile(r"^(auto|cpu|mps|xpu(?::\d+)?|cuda(?::\d+)?)$")


@dataclass(frozen=True, slots=True)
class SettingField:
    """One editable global setting: the config key it writes and the editor to render."""

    section: str
    key: str
    kind: FieldKind
    choices: tuple[str, ...] = ()  # "choice": fixed options; editable combos allow more
    editable: bool = False  # "choice": the user may type a value outside `choices` (gpu device)
    low: float = 0.0  # "float" / "int": spin range
    high: float = 1.0
    decimals: int = 2
    label: str = ""  # the plain name the Settings page shows ("" = section.key)
    help: str = ""  # one short line under the editor; also what the Settings search matches


GLOBAL_FIELDS: tuple[SettingField, ...] = (
    SettingField("paths", "library_root", "path", label="Library folder", help="Where your raw series live"),
    SettingField("paths", "work_root", "path", label="Work folder", help="Stage files, caches and logs"),
    SettingField("paths", "output_root", "path", label="Output folder", help="Finished, translated chapters"),
    SettingField("paths", "models_dir", "path", label="Models folder", help="Downloaded AI models"),
    SettingField(
        "gpu",
        "device",
        "choice",
        choices=("auto", "cpu", "cuda", "xpu", "mps"),
        editable=True,
        label="Graphics card",
        help="auto picks the fastest GPU; cuda = NVIDIA or AMD, xpu = Intel, mps = Apple",
    ),
    SettingField(
        "gpu",
        "usage",
        "choice",
        choices=("full", "balanced", "background"),
        label="Hardware usage",
        help="full = as fast as possible; balanced leaves room for other apps; background stays out of the way",
    ),
    SettingField("gpu", "warmup", "bool", label="Warm up models", help="Load models before the first page"),
    SettingField(
        "ocr",
        "engine",
        "choice",
        choices=("ppocr", "manga_ocr", "paddleocr_vl"),
        label="Text reader (OCR)",
        help="The model that reads the source text",
    ),
    SettingField("ocr", "det_model", "model", label="OCR line finder", help="Empty uses the default"),
    SettingField("ocr", "rec_model", "model", label="OCR reader model", help="Empty uses the default"),
    SettingField(
        "slicer",
        "strategy",
        "choice",
        choices=("smart", "page", "fixed", "simple_gutter"),
        label="Page slicing",
        help="How long strips are cut into pages",
    ),
    SettingField(
        "filter", "enabled", "bool", label="Skip promo pages", help="Leave out ads and credits pages"
    ),
    SettingField(
        "filter",
        "threshold",
        "float",
        low=0.0,
        high=1.0,
        label="Promo match strictness",
        help="Higher skips less",
    ),
    SettingField(
        "translate",
        "cloud_request_budget",
        "int",
        low=0,
        high=1_000_000,
        label="Cloud requests per run",
        help="Most requests one run sends to cloud models; then the local fallback translates (0 = no limit)",
    ),
    SettingField(
        "share",
        "enabled",
        "bool",
        label="Share corrections",
        help="Allow exporting your corrections and checked lines to improve OmniScan (off: no series exports)",
    ),
)

SECTION_TITLES = {
    "paths": "Folders",
    "gpu": "Hardware",
    "ocr": "Text reading",
    "slicer": "Pages",
    "filter": "Promo filter",
    "translate": "Translation",
    "share": "Sharing",
}

# Settings → Profiles: where a model you add runs ((label, endpoint, key variable, base URL))
PROVIDERS: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("Ollama (a model on this PC, or an Ollama Cloud model)", "local", None, None),
    ("OpenAI", "openai", "OPENAI_API_KEY", None),
    ("Anthropic (Claude)", "anthropic", "ANTHROPIC_API_KEY", None),
    ("OpenAI-compatible server (OpenRouter, Groq, LM Studio, …)", "openai", "", ""),
)
_PROFILE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


def profile_paths(user_path: Path | None = None) -> list[Path]:
    """The shipped profile file, then `user_path` (default: the user's own translation_profiles.toml)."""
    from omniscan.translate.profiles import default_profile_paths

    shipped, user = default_profile_paths()
    return [shipped, user_path or user]


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


def suggest_profile_name(endpoint: str, model: str) -> str:
    """A profile name for a model you add: `my-<model>` with every character a name may not hold made a dash."""
    slug = _PROFILE_SLUG_RE.sub("-", model.replace(":", "-").replace("/", "-")).strip("-._")
    suffix = "" if endpoint == "local" else f"-{endpoint}"
    return f"my-{slug or 'model'}{suffix}"


def add_profile(
    name: str,
    *,
    endpoint: str,
    model: str,
    style: str = "chat_json",
    base_url: str | None = None,
    api_key_env: str | None = None,
    only: bool = False,
    user_path: Path | None = None,
) -> Path:
    """Add (or replace) an enabled profile in the user's `translation_profiles.toml`; with `only`, every other
    profile is switched off there, so this model alone translates. Returns the file; SettingError when the
    profile is invalid (unknown endpoint, a bad name, base URL or key variable)."""
    from omniscan.translate.profiles import TranslationProfile, default_profile_paths, load_profiles

    paths = default_profile_paths()
    if user_path is None:
        user_path = paths[1]
    table: dict[str, Any] = {"enabled": True, "endpoint": endpoint, "model": model.strip(), "style": style}
    if endpoint == "local" and style == "chat_json":
        table["think"] = False  # a thinking model would spend the request reasoning
    if base_url:
        table["base_url"] = base_url.strip()
    if api_key_env:
        table["api_key_env"] = api_key_env.strip()
    if not table["model"]:
        raise SettingError("type the model's name")
    try:
        TranslationProfile(name=name, **table)
    except ValidationError as error:
        raise SettingError(f"profile {name!r}: {_first_error(error)}") from error
    tables = dict(_read_toml(user_path).get("profiles", {}))
    if only:
        repo = _read_toml(paths[0]).get("profiles", {})
        for other, profile in load_profiles([*paths[:1], user_path]).items():
            if other != name and profile.enabled:
                tables[other] = {**(tables.get(other) or repo.get(other, {})), "enabled": False}
    tables[name] = table
    _write_profiles_toml(user_path, {"profiles": tables})
    return user_path


def remove_profile(name: str, *, user_path: Path | None = None) -> Path:
    """Remove a profile you added from the user's `translation_profiles.toml`; SettingError for a shipped one
    (switch it off instead) or an unknown name."""
    from omniscan.translate.profiles import default_profile_paths

    paths = default_profile_paths()
    if user_path is None:
        user_path = paths[1]
    if name in _read_toml(paths[0]).get("profiles", {}):
        raise SettingError(f"{name} ships with OmniScan: switch it off instead")
    tables = dict(_read_toml(user_path).get("profiles", {}))
    if tables.pop(name, None) is None:
        raise SettingError(f"unknown translation profile {name!r}")
    _write_profiles_toml(user_path, {"profiles": tables})
    return user_path


def user_profile_names(user_path: Path | None = None) -> set[str]:
    """The profiles only the user's file defines (the ones Remove can take away)."""
    from omniscan.translate.profiles import default_profile_paths

    paths = default_profile_paths()
    repo = set(_read_toml(paths[0]).get("profiles", {}))
    return set(_read_toml(user_path or paths[1]).get("profiles", {})) - repo


def ollama_models(cfg: Config, *, http: httpx.Client | None = None) -> list[str]:
    """The models the local Ollama has (its /api/tags), sorted; SettingError when it does not answer within 3 s."""
    url = f"{cfg.ollama.local_url}/api/tags"
    try:
        response = (http or httpx).get(url, timeout=3.0)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (httpx.HTTPError, ValueError) as error:
        raise SettingError(
            f"Ollama does not answer at {cfg.ollama.local_url}: start it, then list again"
        ) from error
    return sorted(str(m["name"]) for m in models if isinstance(m, dict) and "name" in m)


def has_api_key(name: str, *, env_file: Path | None = None) -> bool:
    """Whether the key variable `name` is set (environment or secrets.env); the value is never returned."""
    return api_key(name, env_file=env_file) is not None


def save_api_key(name: str, value: str, *, env_file: Path | None = None) -> Path:
    """Store an API key as `name=value` in secrets.env (replacing an earlier one, keeping every other line);
    returns the file. SettingError for an invalid variable name or a key with spaces or line breaks."""
    key = value.strip()
    if not _ENV_NAME_RE.fullmatch(name):
        raise SettingError(f"{name!r} is not a variable name (letters, digits, _)")
    if not key or any(ch.isspace() for ch in key) or any(ch in key for ch in "\"'#"):
        raise SettingError("paste the key itself: no spaces, quotes or line breaks")
    path = env_file if env_file is not None else SECRETS_ENV
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    prefixes = (f"{name}=", f"export {name}=")
    kept = [line for line in lines if not line.strip().startswith(prefixes)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join([*kept, f"{name}={key}"]) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


def _first_error(error: ValidationError) -> str:
    """The first validation message, without pydantic's "Value error, " prefix."""
    return str(error.errors()[0].get("msg", "invalid value")).removeprefix("Value error, ")


def _write_profiles_toml(path: Path, data: dict[str, Any]) -> None:
    """Write the user profile file (`dumps_toml` has no nested tables; profile tables are flat). A name with
    a dot (`gpt-5.4`) is quoted, or TOML would read it as nested tables."""
    lines: list[str] = []
    for name, table in data.get("profiles", {}).items():
        if lines:
            lines.append("")
        lines.append(f"[profiles.{name if _BARE_KEY_RE.fullmatch(name) else json.dumps(name)}]")
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
