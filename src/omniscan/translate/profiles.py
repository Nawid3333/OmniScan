"""Translation profiles: a model + prompt style + request knobs, loaded from TOML files.

Later files in the load order replace an earlier profile of the same name entirely (no field-level
merge). The shipped Ollama profiles use `endpoint = "local"` — the local Ollama daemon also proxies the
`*-cloud` models, so no API key is needed for them. `endpoint = "openai"` (any OpenAI-compatible server, with
`base_url`) and `"anthropic"` send the requests to that API instead (llm/api.py), with the key named by
`api_key_env` (default OPENAI_API_KEY / ANTHROPIC_API_KEY) from the environment or secrets.env.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from omniscan.core.config import DEFAULT_TOML, USER_TOML

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
Endpoint = Literal["local", "cloud", "openai", "anthropic"]
API_ENDPOINTS: tuple[Endpoint, ...] = ("openai", "anthropic")


def unkeyed_fields(settings: BaseModel) -> set[str]:
    """The provider fields a profile's or the judge's fingerprint leaves out: where the API key lives never
    changes a translation, and an unset base URL keeps every key made before API endpoints existed."""
    return {"api_key_env"} | ({"base_url"} if getattr(settings, "base_url", None) is None else set())


def check_provider(endpoint: str, base_url: str | None, api_key_env: str | None) -> None:
    """ValueError when `base_url` or `api_key_env` is set for an Ollama endpoint, or the variable name is invalid."""
    if endpoint not in API_ENDPOINTS and (base_url is not None or api_key_env is not None):
        raise ValueError(
            f'base_url and api_key_env need endpoint = "openai" or "anthropic", not {endpoint!r}'
        )
    if base_url is not None and not base_url.startswith(("http://", "https://")):
        raise ValueError(f"base_url must start with http:// or https://, not {base_url!r}")
    if api_key_env is not None and not _ENV_RE.fullmatch(api_key_env):
        raise ValueError(f"api_key_env must be an environment variable name, not {api_key_env!r}")


class TranslationProfile(BaseModel):
    """One named way to translate a chapter's regions (file stem of the resulting candidate run)."""

    model_config = ConfigDict(extra="forbid")

    name: str  # file stem of the run; the [profiles.<name>] table key
    enabled: bool = True  # part of the default set of `omniscan translate` without --profile
    endpoint: Endpoint
    model: str
    style: Literal["chat_json", "translategemma"]
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    think: bool | None = None  # None = do not send the parameter at all
    chunk_regions: int = Field(default=30, ge=1)  # chat_json only: max regions per request
    # Name of the profile that translates instead when this one hits the Ollama rate limit (cloud
    # tokens exhausted). It may be disabled: it then only ever runs as this fallback.
    fallback: str | None = None
    # chat_json only: also send the page images the regions sit on, so a vision model sees who speaks and what
    # a line points at (translate/images.py). Each image's long side is scaled down to `image_side` px, and a
    # request covers at most `images_per_request` images (its regions are chunked to fit).
    images: bool = False
    image_side: int = Field(default=1280, ge=256, le=4096)
    images_per_request: int = Field(default=3, ge=1, le=16)
    base_url: str | None = (
        None  # openai/anthropic: another server speaking that API (default: the vendor's own)
    )
    api_key_env: str | None = (
        None  # openai/anthropic: the variable holding the key (default OPENAI_API_KEY / ...)
    )

    @model_validator(mode="after")
    def _images_need_chat_json(self) -> Self:
        if self.images and self.style != "chat_json":
            raise ValueError(f'profile {self.name!r}: images need style = "chat_json"')
        return self

    @model_validator(mode="after")
    def _provider_fields(self) -> Self:
        try:
            check_provider(self.endpoint, self.base_url, self.api_key_env)
        except ValueError as exc:
            raise ValueError(f"profile {self.name!r}: {exc}") from exc
        return self

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError(f"invalid profile name {value!r} (must match ^[A-Za-z0-9][A-Za-z0-9._-]*$)")
        return value


def resolve_fallbacks(
    profiles: Sequence[TranslationProfile], known: dict[str, TranslationProfile]
) -> dict[str, TranslationProfile]:
    """profile name -> its fallback profile for every profile in `profiles` that names one in `known`.

    A fallback that is missing, names itself, or has a fallback of its own raises ValueError: the chain
    is one level deep so a rate limit can never loop.
    """
    resolved: dict[str, TranslationProfile] = {}
    for profile in profiles:
        if profile.fallback is None:
            continue
        fallback = known.get(profile.fallback)
        if fallback is None:
            raise ValueError(f"profile {profile.name!r}: unknown fallback profile {profile.fallback!r}")
        if fallback.name == profile.name or fallback.fallback is not None:
            raise ValueError(
                f"profile {profile.name!r}: fallback {fallback.name!r} must be another profile without a fallback"
            )
        resolved[profile.name] = fallback
    return resolved


def default_profile_paths() -> list[Path]:
    """Shipped repo profile file, then the per-user one (later wins)."""
    return [DEFAULT_TOML.parent / "translation_profiles.toml", USER_TOML.parent / "translation_profiles.toml"]


def load_profiles(paths: Sequence[Path]) -> dict[str, TranslationProfile]:
    """Read the existing profile files in order; a later file replaces an earlier same-name profile."""
    profiles: dict[str, TranslationProfile] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: invalid TOML: {exc}") from exc
        tables = data.get("profiles", {})
        if not isinstance(tables, dict):
            raise ValueError(f"{path}: expected a [profiles.<name>] table")
        for key, table in tables.items():
            if not isinstance(table, dict):
                raise ValueError(f"{path}: profile {key!r} must be a table")
            if "name" in table:
                raise ValueError(f"{path}: profile {key!r} must not set 'name' (the table key is the name)")
            try:
                profile = TranslationProfile(name=str(key), **table)
            except ValidationError as exc:
                raise ValueError(f"{path}: invalid profile {key!r}: {exc}") from exc
            profiles[profile.name] = profile
    return profiles
