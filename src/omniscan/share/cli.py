"""`omniscan contribute` sub-app: write a series' hand corrections to a contribution archive (share/contribution.py).

Nothing is sent anywhere: the archive is a local file the user can inspect and share. A series that opted out of
sharing (`[share] enabled = false` in config.toml or its series.toml) exports nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from omniscan.core.config import SeriesConfigError, get_config
from omniscan.core.paths import SeriesPaths
from omniscan.share.contribution import ShareOptOutError, Summary, build, summarize, write_archive

contribute_app = typer.Typer(
    no_args_is_help=True,
    help="Share hand corrections so OmniScan improves (a local archive; nothing is uploaded).",
)


@contribute_app.callback()
def contribute_group() -> None:
    """Share hand corrections so OmniScan improves (a local archive; nothing is uploaded)."""


def _fail(message: str) -> typer.Exit:
    """Print `message` as a contribute error and return the exit to raise (code 2)."""
    typer.echo(f"contribute: {message}", err=True)
    return typer.Exit(2)


def describe(summary: Summary) -> str:
    """One line saying what a contribution holds."""
    fixes = ", ".join(
        f"{count} {one if count == 1 else many}"
        for count, one, many in (
            (summary.ocr_fixes, "OCR text", "OCR texts"),
            (summary.kinds, "region type", "region types"),
            (summary.added, "added box", "added boxes"),
            (summary.deleted, "deleted box", "deleted boxes"),
            (summary.english, "English line", "English lines"),
            (summary.lettering, "lettering", "letterings"),
            (summary.other, "other region edit", "other region edits"),
        )
        if count
    )
    return (
        f"{summary.chapters} chapter(s), {summary.pages} page(s), {summary.regions} region(s) and"
        f" {summary.terms} glossary term(s); corrections: {fixes}"
    )


def _size(size: int) -> str:
    """A file size for people: kB below a megabyte."""
    return f"{size / 1000:.0f} kB" if size < 1_000_000 else f"{size / 1_000_000:.1f} MB"


@contribute_app.command("export")
def contribute_export(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Only this chapter; repeatable. Default: every chapter."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="The archive to write. Default: omniscan-contribution-<id>.zip here."
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only say what the archive would hold.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Write the pages a series' hand corrections were made on, with the corrections and its locked glossary
    terms, to one zip archive (no file names, folder paths or image metadata)."""
    cfg = get_config()
    series_paths = SeriesPaths.from_config(cfg, series)
    known = series_paths.chapters()
    unknown = [name for name in chapter or [] if name not in known]
    if unknown:
        raise _fail(f"no chapter {unknown[0]!r} in {series}")
    try:
        contribution, pages = build(series_paths, cfg, chapter)
    except (ShareOptOutError, SeriesConfigError) as exc:
        raise _fail(f"{exc}; nothing exported") from exc
    summary = summarize(contribution)
    path = (
        output if output is not None else Path.cwd() / f"omniscan-contribution-{contribution.series_id}.zip"
    )
    size = None
    if pages and not dry_run:
        try:
            size = write_archive(path, contribution, pages)
        except FileNotFoundError as exc:
            raise _fail(f"{exc}; nothing exported") from exc
    if as_json:
        typer.echo(
            json.dumps({**asdict(summary), "path": str(path) if size is not None else None, "bytes": size})
        )
    elif not pages:
        typer.echo(f"contribute: {series} has no hand corrections to share; nothing exported")
    elif size is None:
        typer.echo(f"contribute: would write {describe(summary)}")
    else:
        typer.echo(f"contribute: wrote {describe(summary)} to {path} ({_size(size)})")
