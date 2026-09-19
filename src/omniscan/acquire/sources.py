"""Chapter sources: explicit `[[chapter]]` URLs plus a `{n}` URL template for numbered ranges."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MAX_TEMPLATE_CHAPTERS = 1000


@dataclass(frozen=True, slots=True)
class ChapterSource:
    """One chapter to acquire: its display name and the URL of its page."""

    name: str
    url: str


def sources_path(library_root: Path, series: str) -> Path:
    """sources.toml location of a series inside the library root."""
    return library_root / series / "sources.toml"


def load_sources(path: Path) -> list[ChapterSource]:
    """Parse sources.toml into chapter sources in acquisition order; raises ValueError when invalid."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    seen: set[str] = set()
    sources: list[ChapterSource] = []
    for index, entry in enumerate(data.get("chapter", []), start=1):
        name = entry.get("name")
        url = entry.get("url")
        if name is None:
            raise ValueError(f"chapter #{index}: missing 'name'")
        if url is None:
            raise ValueError(f"chapter #{index}: missing 'url'")
        url = _check_url(url, f"chapter #{index}")
        name = _check_name(name)
        _check_duplicate(name, seen)
        sources.append(ChapterSource(name=name, url=url))
    template = data.get("template")
    if template:
        sources.extend(_template_sources(template, seen))
    return sources


def _template_sources(template: dict[str, object], seen: set[str]) -> list[ChapterSource]:
    """Chapters generated from the [template] table, validated like explicit ones."""
    for key in ("url", "first", "last"):
        if template.get(key) is None:
            raise ValueError(f"template: missing {key!r}")
    first = template["first"]
    last = template["last"]
    if not isinstance(first, int) or not isinstance(last, int):
        raise ValueError("template: 'first' and 'last' must be integers")
    if first > last:
        raise ValueError(f"template: first ({first}) is greater than last ({last})")
    if last - first + 1 > MAX_TEMPLATE_CHAPTERS:
        raise ValueError(f"template: more than {MAX_TEMPLATE_CHAPTERS} chapters")
    url = _check_url(template["url"], "template")
    name_template = template.get("name", "Chapter {n}")
    if not isinstance(name_template, str):
        raise ValueError(f"unsafe chapter name {name_template!r}")
    sources = []
    for number in range(first, last + 1):
        name = _check_name(name_template.format(n=number))
        _check_duplicate(name, seen)
        sources.append(ChapterSource(name=name, url=url.format(n=number)))
    return sources


def _check_url(url: object, label: str) -> str:
    """HTTP(S) URL as a string; ValueError when it is not one."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError(f"{label}: url must be http(s)")
    return url


def _check_name(name: object) -> str:
    """Chapter name as a string usable as a folder name; ValueError when unsafe."""
    if not isinstance(name, str) or not name.strip() or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"unsafe chapter name {name!r}")
    return name


def _check_duplicate(name: str, seen: set[str]) -> None:
    """ValueError when the name was already used in this file."""
    if name in seen:
        raise ValueError(f"duplicate chapter name {name!r}")
    seen.add(name)
