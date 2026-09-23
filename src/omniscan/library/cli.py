"""`omniscan library` sub-app: covers and metadata for the series in the library (card L1)."""

from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer

from omniscan.core.config import get_config
from omniscan.library.metadata import (
    PROVIDERS,
    Candidate,
    MetadataClient,
    Provider,
    SearchResult,
    best_match,
    download_cover,
    find_cover,
    match_score,
    meta_dir,
    read_series_meta,
    set_cover_from_file,
)

library_app = typer.Typer(no_args_is_help=True, help="Series covers and metadata from free APIs.")


def make_http_client() -> httpx.Client:
    """HTTP client for the library commands (tests replace this to mock the network)."""
    return httpx.Client(follow_redirects=True, timeout=30.0)


class CoverProvider(enum.StrEnum):
    """Choice of `library cover --provider` (typer cannot build a click option from list[Literal])."""

    ALL = "all"
    ANILIST = "anilist"
    MANGADEX = "mangadex"
    JIKAN = "jikan"


class CoverSize(enum.StrEnum):
    """Choice of `library cover --size` (typer cannot build a click option from list[Literal])."""

    LARGE = "large"
    SMALL = "small"


def _echo_text(text: str) -> None:
    """Echo text that may hold Korean/Japanese; Windows pipes (cp1252) must not crash on it."""
    import sys

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None and (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        reconfigure(encoding="utf-8", errors="replace")
    typer.echo(text)


def _candidate_json(candidate: Candidate, score: float) -> dict[str, Any]:
    """The machine-readable form of one candidate list entry."""
    return {
        "provider": candidate.provider,
        "id": candidate.id,
        "title": candidate.title,
        "year": candidate.year,
        "country": candidate.country,
        "score": score,
        "cover_url": candidate.cover_url,
    }


def _json_payload(
    series: str,
    query: str,
    scored: list[tuple[Candidate, float]],
    result: SearchResult,
    chosen: tuple[Candidate, float] | None,
    cover: Path | None,
) -> dict[str, Any]:
    """The `--json` object of `library cover`."""
    return {
        "series": series,
        "query": query,
        "candidates": [_candidate_json(candidate, score) for candidate, score in scored],
        "errors": dict(result.errors),
        "chosen": _candidate_json(*chosen) if chosen is not None else None,
        "cover": str(cover) if cover is not None else None,
    }


def _print_list(scored: list[tuple[Candidate, float]], result: SearchResult) -> None:
    """The numbered candidate list (score against the query) plus one line per provider error."""
    for number, (candidate, score) in enumerate(scored, 1):
        _echo_text(
            f"{number}. {candidate.provider:<9}{candidate.title}"
            f"  {candidate.year if candidate.year is not None else '?'}"
            f"  {candidate.country or '?'}  {score:.3f}"
        )
    for name, error in result.errors.items():
        typer.echo(f"{name}: {error}", err=True)


@library_app.command("cover")
def cover(
    series: Annotated[str, typer.Argument()],
    title: Annotated[
        str | None, typer.Option("--title", help="Search text. Default: the series name.")
    ] = None,
    provider: Annotated[
        CoverProvider, typer.Option("--provider", case_sensitive=False, help="Metadata provider.")
    ] = CoverProvider.ALL,
    pick: Annotated[
        int | None, typer.Option("--pick", min=1, help="1-based number from the candidate list.")
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", exists=True, dir_okay=False, help="Use this image as the cover (no network)."),
    ] = None,
    size: Annotated[
        CoverSize, typer.Option("--size", case_sensitive=False, help="Cover resolution.")
    ] = CoverSize.LARGE,
    as_json: Annotated[bool, typer.Option("--json", help="Emit one JSON object.")] = False,
) -> None:
    """Find the series on AniList/MangaDex/Jikan and save the matching cover into <series>/_meta."""
    mdir = meta_dir(get_config().paths.library_root, series)
    if file is not None:
        try:
            set_cover_from_file(mdir, file)
        except ValueError as exc:
            typer.echo(f"cover: {exc}", err=True)
            raise typer.Exit(2) from exc
        typer.echo(f"cover set from {file}")
        return
    query = (title or series).strip()
    if not query:
        typer.echo("cover: empty search text", err=True)
        raise typer.Exit(2)
    providers = PROVIDERS if provider is CoverProvider.ALL else (cast("Provider", provider.value),)
    http = make_http_client()
    try:
        result = MetadataClient(client=http).search(query, providers=providers)
        scored = [(candidate, match_score(query, candidate)) for candidate in result.candidates]
        if not as_json:
            _print_list(scored, result)
        chosen = _pick_candidate(scored, pick) if pick is not None else best_match(query, result.candidates)
        if chosen is None:
            if as_json:
                typer.echo(json.dumps(_json_payload(series, query, scored, result, None, None), indent=2))
            else:
                typer.echo("no confident match — choose one with --pick N", err=True)
            raise typer.Exit(1)
        try:
            cover_path = download_cover(
                chosen[0], mdir, client=http, size=cast("Literal['large', 'small']", size.value)
            )
        except (ValueError, httpx.HTTPError) as exc:
            typer.echo(f"cover: {exc}", err=True)
            if as_json:
                typer.echo(json.dumps(_json_payload(series, query, scored, result, chosen, None), indent=2))
            raise typer.Exit(1) from exc
        if as_json:
            typer.echo(json.dumps(_json_payload(series, query, scored, result, chosen, cover_path), indent=2))
        else:
            typer.echo(f"cover saved: {cover_path} ({chosen[0].provider})")
    finally:
        http.close()


def _pick_candidate(scored: list[tuple[Candidate, float]], pick: int) -> tuple[Candidate, float]:
    """The 1-based picked candidate from the printed list; out of range is a usage error."""
    if not 1 <= pick <= len(scored):
        typer.echo(f"cover: no candidate {pick} (1..{len(scored)})", err=True)
        raise typer.Exit(2)
    return scored[pick - 1]


@library_app.command("info")
def info(
    series: Annotated[str, typer.Argument()],
    as_json: Annotated[bool, typer.Option("--json", help="Emit series.json as JSON.")] = False,
) -> None:
    """Print a series' stored cover metadata (or 'no metadata' when there is none)."""
    mdir = meta_dir(get_config().paths.library_root, series)
    meta = read_series_meta(mdir)
    if meta is None:
        typer.echo(f"no metadata for {series}", err=True)
        raise typer.Exit(1)
    if as_json:
        typer.echo(json.dumps(meta, indent=2, ensure_ascii=False))
        return
    for key in ("title", "provider", "year", "country", "status", "credit"):
        _echo_text(f"{key}: {meta.get(key)}")
    cover_path = find_cover(mdir)
    _echo_text(f"cover: {cover_path if cover_path is not None else '-'}")
