"""Judge configuration: the judge model, request knobs and candidate priority, loaded from TOML files.

Later files in the load order override the keys they set (field-level merge, unlike translation
profiles which replace a whole profile). The judge model is only asked about regions where the
candidate runs disagree or a locked glossary term is violated; `always_judge` widens that to every
region with at least two unique candidates.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omniscan.core.config import DEFAULT_TOML, USER_TOML
from omniscan.translate.profiles import Endpoint, check_provider


class JudgeConfig(BaseModel):
    """How `omniscan judge` picks/merges/rewrites between a chapter's candidate runs."""

    model_config = ConfigDict(extra="forbid")

    model: str = "gemma4:31b-cloud"
    endpoint: Endpoint = "local"
    think: bool | None = False  # None = do not send the parameter at all
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    chunk_regions: int = Field(default=20, ge=1)  # max regions per judge request
    max_repair_rounds: int = Field(default=1, ge=0)  # extra rounds for rejected answers
    agree_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    always_judge: bool = False  # True: ask the judge about every region with >= 2 unique candidates
    prefer: list[str] = Field(default_factory=list)  # run ids in priority order
    base_url: str | None = None  # openai/anthropic endpoint: another server speaking that API
    api_key_env: str | None = None  # openai/anthropic endpoint: the variable holding the key

    @model_validator(mode="after")
    def _provider_fields(self) -> Self:
        check_provider(self.endpoint, self.base_url, self.api_key_env)
        return self


def default_judge_paths() -> list[Path]:
    """Shipped repo judge file, then the per-user one (later wins)."""
    return [DEFAULT_TOML.parent / "judge.toml", USER_TOML.parent / "judge.toml"]


def load_judge_config(paths: Sequence[Path]) -> JudgeConfig:
    """Read the existing judge files in order; a later file overrides the keys it sets."""
    values: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: invalid TOML: {exc}") from exc
        table = data.get("judge", {})
        if not isinstance(table, dict):
            raise ValueError(f"{path}: expected a [judge] table")
        try:
            validated = JudgeConfig(**table)
        except ValidationError as exc:
            raise ValueError(f"{path}: invalid judge config: {exc}") from exc
        values.update(validated.model_dump(exclude_unset=True))
    return JudgeConfig(**values)
