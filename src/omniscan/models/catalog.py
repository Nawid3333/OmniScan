"""Model catalog: every downloadable model OmniScan knows, loaded from TOML.

The shipped file is `config/models.toml`; a per-machine `~/.config/omniscan/models.toml` replaces
same-id entries and may add new ones (same rule as the translation profiles). The catalog only
DESCRIBES the models — the pipeline still loads them its own way until card U2b.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omniscan.core.config import DEFAULT_TOML, USER_CONFIG_DIR

ModelKind = Literal["vision", "ocr", "inpaint", "llm"]
ModelFormat = Literal["zip", "file", "ollama", "cloud"]
Backend = Literal["cuda", "rocm", "mps", "xpu", "cpu"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Fields every format must set beyond the common ones.
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "zip": ("mirror_url", "sha256", "bytes", "upstream_repo", "upstream_revision"),
    "file": ("mirror_url", "sha256", "bytes", "upstream_url", "install_path"),
    "ollama": ("ollama_name",),
    "cloud": ("ollama_name",),
}


class ModelEntry(BaseModel):
    """One catalog entry: what the model is for, its size and licence, and how to get it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: ModelKind
    required: bool = False  # the pipeline cannot run without it
    format: ModelFormat
    size_mb: int  # approximate download size shown in the settings screen
    license: str
    description: str
    used_by: list[str] = Field(default_factory=list)
    # zip / file
    mirror_url: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    upstream_repo: str | None = None  # zip only
    upstream_revision: str | None = None  # zip only
    upstream_url: str | None = None  # file only
    install_path: str | None = None  # file only, relative to models_dir
    # ollama / cloud
    ollama_name: str | None = None
    # hardware requirements (card H1): what the model needs to run well; `hw.assess` turns
    # them into ok / slow / warn / incompatible against a detected machine
    min_vram_gb: float | None = None  # GPU memory needed to run comfortably
    min_ram_gb: float | None = None
    backends: list[Backend] = Field(default_factory=list)  # empty = all
    cpu_ok: bool = True  # usable without a GPU at all
    cpu_speed: Literal["fast", "ok", "slow", "unusable"] = "ok"  # how it feels on CPU
    notes: str = ""

    def validate_for_format(self) -> None:
        """Raise ValueError naming the first format-specific field the entry is missing."""
        missing = [f for f in _REQUIRED_FIELDS[self.format] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.format} model {self.id!r} needs {missing[0]}")
        if self.format in ("zip", "file") and (self.sha256 is None or not _SHA256_RE.fullmatch(self.sha256)):
            raise ValueError(f"{self.format} model {self.id!r} needs a 64-hex sha256")


def default_catalog_path() -> Path:
    """Shipped catalog: `<repo root>/config/models.toml`."""
    return DEFAULT_TOML.parent / "models.toml"


def machine_catalog_path() -> Path:
    """Per-machine override: `~/.config/omniscan/models.toml` (replaces same-id entries, may add)."""
    return USER_CONFIG_DIR / "models.toml"


def load_catalog(path: Path | None = None) -> list[ModelEntry]:
    """Read the catalog file(s): repo file order, then machine-only additions.

    With an explicit `path` only that file is read. Otherwise the shipped catalog is read first and
    the machine override second; a machine entry with the same id replaces the shipped one in place.
    Duplicate ids in one file, unknown keys or a format-missing field raise ValueError.
    """
    paths = (path,) if path is not None else (default_catalog_path(), machine_catalog_path())
    entries: dict[str, ModelEntry] = {}
    for file_path in paths:
        if not file_path.is_file():
            continue
        try:
            with file_path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{file_path}: invalid TOML: {exc}") from exc
        tables = data.get("model", [])
        if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
            raise ValueError(f"{file_path}: expected [[model]] array-of-tables")
        seen: set[str] = set()
        for table in tables:
            try:
                entry = ModelEntry(**table)
            except ValidationError as exc:
                raise ValueError(f"{file_path}: invalid model entry: {exc}") from exc
            if entry.id in seen:
                raise ValueError(f"{file_path}: duplicate model id {entry.id!r}")
            seen.add(entry.id)
            try:
                entry.validate_for_format()
            except ValueError as exc:
                raise ValueError(f"{file_path}: {exc}") from exc
            entries[entry.id] = entry  # replacing a key keeps its position; new ids go to the end
    return list(entries.values())
