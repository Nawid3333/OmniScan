"""model-watch: report upstream Hugging Face drift for the model catalog (card W1).

Runs as a scheduled GitHub Action (`.github/workflows/model-watch.yml`, Mondays 06:00 UTC). The job
only COMPARES metadata — it never downloads weights and never edits `config/models.toml`. Two checks:

- `check_entries`: every `hf` (and mirrored `zip`) catalog entry's pinned `upstream_revision` is
  compared with the upstream repo's head commit — `updated` when they differ, `missing` on 404.
- `find_new`: the listing API of the watched orgs (`config/model_watch.toml`) is searched for repos
  matching the configured patterns that the catalog does not know yet — `new`.

The report (markdown + JSON) goes to stdout and, in CI, onto a `model-watch` issue. A candidate
found there is qualified by hand on a GPU machine (`scripts/qualify_ocr.py`, card O1c) before the
catalog changes. The default fetch talks to the public Hugging Face API with 2 retries on
5xx/transport errors; an optional `HF_TOKEN` raises the rate limit and is never printed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx

if TYPE_CHECKING:
    from omniscan.models.catalog import ModelEntry

API_BASE = "https://huggingface.co/api/models"
USER_AGENT = "OmniScan-model-watch"
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2  # retries after the first attempt: 3 requests worst case
BACKOFF_BASE = 0.5  # sleep before retry `attempt` is 0.5 * 2**attempt seconds
LISTING_LIMIT = "100"

Fetch = Callable[[str, Mapping[str, str] | None], tuple[int, Any]]
Sleep = Callable[[float], None]

Kind = Literal["updated", "new", "missing"]


class FetchError(RuntimeError):
    """A Hugging Face request failed (HTTP 5xx or transport error) after all retries."""


@dataclass(frozen=True, slots=True)
class Change:
    """One upstream difference the watch found (one row of the report)."""

    kind: Kind
    repo: str
    catalog_id: str | None  # for "updated"/"missing"
    old_revision: str | None
    new_revision: str | None
    last_modified: str | None  # ISO string from the API
    downloads: int | None
    note: str  # one human sentence


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """The four catalog fields the watch reads (the CI env runs `--no-project`: no pydantic there)."""

    id: str
    format: str
    upstream_repo: str | None
    upstream_revision: str | None


@dataclass(frozen=True, slots=True)
class OrgWatch:
    """What to watch in one Hugging Face org: listing search terms and repo-name patterns."""

    search: tuple[str, ...]
    patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WatchConfig:
    """Parsed `config/model_watch.toml`: watched orgs plus repo ids silenced forever."""

    orgs: dict[str, OrgWatch]
    ignore: frozenset[str]


def http_fetch(
    url: str,
    params: Mapping[str, str] | None = None,
    *,
    sleep: Sleep = time.sleep,
    transport: httpx.BaseTransport | None = None,
) -> tuple[int, Any]:
    """Default Fetch: GET with UA + optional HF_TOKEN bearer, 2 retries on 5xx/transport errors."""
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(
        timeout=TIMEOUT_SECONDS, headers=headers, transport=transport, follow_redirects=True
    ) as client:
        attempt = 0
        while True:
            try:
                response = client.get(url, params=params)
            except httpx.HTTPError as exc:
                if attempt >= MAX_RETRIES:
                    raise FetchError(f"{type(exc).__name__} for {url} after {MAX_RETRIES} retries") from exc
            else:
                if not 500 <= response.status_code <= 599:
                    try:
                        return response.status_code, response.json()
                    except ValueError:
                        return response.status_code, None
                if attempt >= MAX_RETRIES:
                    raise FetchError(f"HTTP {response.status_code} for {url} after {MAX_RETRIES} retries")
            sleep(BACKOFF_BASE * 2**attempt)
            attempt += 1


def load_watch_config(path: Path) -> WatchConfig:
    """Parse `config/model_watch.toml`; a ValueError names the offending field."""
    data = _toml(path)
    orgs_raw = data.get("orgs")
    if not isinstance(orgs_raw, dict):
        raise ValueError(f"{path}: 'orgs' table is required")
    orgs: dict[str, OrgWatch] = {}
    for name, table in orgs_raw.items():
        if not name or "/" in name:
            raise ValueError(f"{path}: orgs: '{name}' must be a non-empty org id without '/'")
        if not isinstance(table, dict):
            raise ValueError(f"{path}: orgs.{name}: expected a table")
        orgs[name] = OrgWatch(
            search=tuple(_string_list(table, "search", f"{path}: orgs.{name}")),
            patterns=tuple(_string_list(table, "patterns", f"{path}: orgs.{name}")),
        )
    ignore_raw = data.get("ignore", [])
    if not isinstance(ignore_raw, list) or not all(isinstance(item, str) for item in ignore_raw):
        raise ValueError(f"{path}: 'ignore' must be a list of repo ids")
    return WatchConfig(orgs=orgs, ignore=frozenset(ignore_raw))


def _toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML: {exc}") from exc


def _string_list(table: Mapping[str, Any], field: str, where: str) -> list[str]:
    value = table.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{where}: '{field}' must be a list of strings")
    return list(value)


def load_catalog_entries(path: Path) -> list[CatalogEntry]:
    """Minimal read of a models.toml catalog (`[[model]]` id/format/upstream fields only)."""
    data = _toml(path)
    tables = data.get("model", [])
    if not isinstance(tables, list) or not all(isinstance(table, dict) for table in tables):
        raise ValueError(f"{path}: expected [[model]] array-of-tables")
    entries: list[CatalogEntry] = []
    for table in tables:
        entry_id = table.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"{path}: every model entry needs a non-empty 'id' string")
        fmt = table.get("format")
        if not isinstance(fmt, str) or not fmt:
            raise ValueError(f"{path}: model entry {entry_id!r} needs a 'format' string")
        for field in ("upstream_repo", "upstream_revision"):
            value = table.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{path}: model entry {entry_id!r}: '{field}' must be a string")
        entries.append(
            CatalogEntry(
                id=entry_id,
                format=fmt,
                upstream_repo=table.get("upstream_repo"),
                upstream_revision=table.get("upstream_revision"),
            )
        )
    return entries


def check_entries(entries: Sequence[ModelEntry], fetch: Fetch) -> list[Change]:
    """Compare every hf/zip catalog entry's pinned revision with the upstream head sha."""
    by_repo: dict[str, list[ModelEntry]] = {}
    for entry in entries:
        if entry.format not in ("hf", "zip") or not entry.upstream_repo or not entry.upstream_revision:
            continue
        by_repo.setdefault(entry.upstream_repo, []).append(entry)
    changes: list[Change] = []
    for repo, repo_entries in by_repo.items():
        try:
            status, payload = fetch(f"{API_BASE}/{repo}", None)
        except FetchError as exc:
            _warn(f"{repo}: {exc}")
            continue
        if status == 404:
            for entry in repo_entries:
                changes.append(
                    Change(
                        kind="missing",
                        repo=repo,
                        catalog_id=entry.id,
                        old_revision=entry.upstream_revision,
                        new_revision=None,
                        last_modified=None,
                        downloads=None,
                        note="repo not found (renamed or removed?)",
                    )
                )
            continue
        if status != 200 or not isinstance(payload, dict):
            _warn(f"{repo}: HTTP {status} (skipping)")
            continue
        sha = payload.get("sha")
        if not isinstance(sha, str) or not sha:
            _warn(f"{repo}: response has no head sha (skipping)")
            continue
        for entry in repo_entries:
            if sha != entry.upstream_revision:
                changes.append(
                    Change(
                        kind="updated",
                        repo=repo,
                        catalog_id=entry.id,
                        old_revision=entry.upstream_revision,
                        new_revision=sha,
                        last_modified=_opt_str(payload.get("lastModified")),
                        downloads=_opt_int(payload.get("downloads")),
                        note=f"{entry.id}: upstream has a newer commit",
                    )
                )
    return changes


def find_new(entries: Sequence[ModelEntry], cfg: WatchConfig, fetch: Fetch) -> list[Change]:
    """Watched-org repos matching a pattern that are neither catalogued nor ignored."""
    known = {entry.upstream_repo for entry in entries if entry.upstream_repo}
    seen: dict[str, dict[str, Any]] = {}  # repo -> its listing item (first term that found it)
    for org, watch in cfg.orgs.items():
        for term in watch.search:
            params = {
                "author": org,
                "search": term,
                "limit": LISTING_LIMIT,
                "sort": "lastModified",
                "direction": "-1",
            }
            try:
                status, payload = fetch(API_BASE, params)
            except FetchError as exc:
                _warn(f"listing {org} (search {term!r}): {exc}")
                continue
            if status != 200 or not isinstance(payload, list):
                _warn(f"listing {org} (search {term!r}): HTTP {status} (skipping)")
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                repo = item.get("id")
                if isinstance(repo, str) and repo and repo not in seen:
                    seen[repo] = item
    changes: list[Change] = []
    for repo, item in seen.items():
        org, _, name = repo.partition("/")
        watch = cfg.orgs.get(org)
        if watch is None:
            continue
        if not any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in watch.patterns):
            continue  # patterns match the repo name without the org, case-insensitively
        if repo in known or repo in cfg.ignore:
            continue
        changes.append(
            Change(
                kind="new",
                repo=repo,
                catalog_id=None,
                old_revision=None,
                new_revision=None,
                last_modified=_opt_str(item.get("lastModified")),
                downloads=_opt_int(item.get("downloads")),
                note="not in the catalog",
            )
        )
    changes.sort(key=lambda change: change.last_modified or "", reverse=True)
    return changes


_SECTIONS: tuple[tuple[Kind, str], ...] = (
    ("updated", "Updated upstream"),
    ("new", "New upstream models"),
    ("missing", "Missing upstream"),
)
_TABLE_HEAD = (
    "| repo | catalog id | old → new | last modified | downloads | note |",
    "|---|---|---|---|---|---|",
)
_WHAT_TO_DO = (
    "## What to do",
    "",
    "- run scripts/qualify_ocr.py --only <candidate> on a GPU machine",
    "- update config/models.toml only if the qualification improves",
    "- add the repo to the ignore list in config/model_watch.toml to silence it",
)


def render_report(changes: Sequence[Change], *, generated: str) -> str:
    """Deterministic markdown report; without changes a one-line "no changes" report."""
    lines = ["# Model watch report", "", f"Generated: {generated}"]
    if not changes:
        return "\n".join([*lines, "", "No changes."]) + "\n"
    ordered = sorted(changes, key=lambda change: (change.repo, change.catalog_id or ""))
    for kind, title in _SECTIONS:
        group = [change for change in ordered if change.kind == kind]
        if not group:
            continue
        lines += ["", f"## {title}", "", *_TABLE_HEAD]
        lines += [_row(change) for change in group]
    lines += ["", *_WHAT_TO_DO]
    return "\n".join(lines) + "\n"


def _row(change: Change) -> str:
    old_new = f"{_rev(change.old_revision)} → {_rev(change.new_revision)}" if _has_revision(change) else "—"
    cells = [
        f"`{change.repo}`",
        f"`{change.catalog_id}`" if change.catalog_id else "—",
        old_new,
        change.last_modified or "—",
        str(change.downloads) if change.downloads is not None else "—",
        change.note,
    ]
    return "| " + " | ".join(cells) + " |"


def _has_revision(change: Change) -> bool:
    return change.old_revision is not None or change.new_revision is not None


def _rev(revision: str | None) -> str:
    return f"`{revision[:8]}`" if revision else "—"


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _opt_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _use_utf8_stdio() -> None:
    """The report uses → and —; Windows consoles default to a legacy codepage that cannot show them."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None, *, fetch: Fetch | None = None) -> int:
    """CLI entry point: 0 no changes, 3 changes (with --fail-on-change), 2 bad arguments/config."""
    parser = argparse.ArgumentParser(
        description="Report upstream Hugging Face drift for the catalog (card W1)."
    )
    parser.add_argument(
        "--catalog", type=Path, default=Path("config/models.toml"), help="path to models.toml"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/model_watch.toml"), help="path to model_watch.toml"
    )
    parser.add_argument("--markdown", type=Path, help="also write the markdown report to this path")
    parser.add_argument("--json", dest="json_out", type=Path, help="also write the changes as a JSON list")
    parser.add_argument(
        "--fail-on-change",
        dest="fail_on_change",
        action="store_true",
        help="exit 3 when any change was found",
    )
    parser.add_argument(
        "--offline", action="store_true", help="no network: only validate both files and print ok"
    )
    args = parser.parse_args(argv)
    _use_utf8_stdio()
    try:
        entries = load_catalog_entries(args.catalog)
        watch_cfg = load_watch_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.offline:
        print("ok")
        return 0
    active = fetch if fetch is not None else http_fetch
    light = cast("Sequence[ModelEntry]", entries)  # check_entries only reads id/format/upstream fields
    changes = check_entries(light, active)
    changes += find_new(light, watch_cfg, active)
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    report = render_report(changes, generated=generated)
    try:
        if args.markdown is not None:
            args.markdown.write_text(report, encoding="utf-8")
        if args.json_out is not None:
            args.json_out.write_text(
                json.dumps([asdict(c) for c in changes], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report, end="")
    return 3 if args.fail_on_change and changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
