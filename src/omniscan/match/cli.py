"""`omniscan match` subcommands: chapter alignment between two independently-sourced chapter sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from omniscan.match.chapters import ChapterMapping, Thresholds, match_chapters
from omniscan.match.duplicates import DuplicateReport, DuplicateThresholds, find_duplicate_chapters

match_app = typer.Typer(no_args_is_help=True, help="Align two independently-sourced chapter sets.")

DEFAULT_OUT = "chapter-match.json"
DEFAULT_DUPLICATES_OUT = "chapter-duplicates.json"


@match_app.command("chapters")
def chapters(
    dir_a: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="First chapter set (one folder per chapter).")
    ],
    dir_b: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Second chapter set (one folder per chapter)."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help=f"Mapping artifact path. Default: {DEFAULT_OUT} in the current folder."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing mapping (it is meant to be hand-edited).")
    ] = False,
    page_similarity: Annotated[
        float, typer.Option("--page-similarity", help="dhash similarity at/above which two pages may align.")
    ] = 0.75,
    page_gap: Annotated[
        float, typer.Option("--page-gap", help="Penalty for leaving a page unmatched inside a chapter pair.")
    ] = 0.25,
    chapter_gap: Annotated[
        float, typer.Option("--chapter-gap", help="Penalty for leaving a chapter unmatched.")
    ] = 0.6,
    min_quality: Annotated[
        float, typer.Option("--min-quality", help="Page-coverage quality at/above which chapters may match.")
    ] = 0.3,
    review_quality: Annotated[
        float, typer.Option("--review-quality", help="Matches below this quality are flagged for review.")
    ] = 0.5,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full mapping JSON to stdout as well.")
    ] = False,
) -> None:
    """Map the chapters of dir_b onto the chapters of dir_a by page art (dhash), never by name.

    Writes the hand-editable mapping artifact and prints counts, the unmatched chapters and the
    lowest-confidence matches. Refuses to overwrite an existing mapping without --force."""
    thresholds = Thresholds(
        page_similarity=page_similarity,
        page_gap=page_gap,
        chapter_gap=chapter_gap,
        min_quality=min_quality,
        review_quality=review_quality,
    )
    dest = out or Path(DEFAULT_OUT)
    if dest.is_file() and not force:
        typer.echo(f"match: {dest} already exists (pass --force to overwrite it)", err=True)
        raise typer.Exit(2)
    try:
        mapping = match_chapters(dir_a, dir_b, thresholds)
    except ValueError as exc:
        typer.echo(f"match: {exc}", err=True)
        raise typer.Exit(2) from exc
    mapping.save(dest)
    if as_json:
        typer.echo(json.dumps(mapping.model_dump(), indent=2))
    else:
        _summary(mapping, dest)


def _summary(mapping: ChapterMapping, dest: Path) -> None:
    """Human-readable result: counts, the unmatched chapters (eyeball these), and the weakest matches."""
    typer.echo(
        f"match: {len(mapping.matched)} matched, {len(mapping.unmatched_a)} unmatched in A, "
        f"{len(mapping.unmatched_b)} unmatched in B -> {dest}"
    )
    for name in mapping.unmatched_a:
        typer.echo(f"match:   A only: {mapping.dir_a}/{name}")
    for name in mapping.unmatched_b:
        typer.echo(f"match:   B only: {mapping.dir_b}/{name}")
    for match in sorted(mapping.matched, key=lambda m: m.quality)[:10]:
        if not match.review:
            break
        runner_up = (
            f" (runner-up {match.runner_up.name} {match.runner_up.quality:.2f})" if match.runner_up else ""
        )
        typer.echo(f"match:   review: A {match.a!r} <-> B {match.b!r} quality {match.quality:.2f}{runner_up}")


@match_app.command("duplicates")
def duplicates(
    root: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, help="The chapter set to scan (one folder per chapter)."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", help=f"Report artifact path. Default: {DEFAULT_DUPLICATES_OUT} in the current folder."
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing report (it is meant to be hand-edited).")
    ] = False,
    page_similarity: Annotated[
        float, typer.Option("--page-similarity", help="dhash similarity at/above which two pages may align.")
    ] = 0.75,
    page_gap: Annotated[
        float, typer.Option("--page-gap", help="Penalty for leaving a page unmatched inside a chapter pair.")
    ] = 0.25,
    min_quality: Annotated[
        float,
        typer.Option(
            "--min-quality", help="Page-coverage quality at/above which a chapter pair is a duplicate."
        ),
    ] = 0.9,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full report JSON to stdout as well.")
    ] = False,
) -> None:
    """Flag pairs of chapters under root that are likely duplicates of each other (same pages imported twice).

    Writes the hand-editable report artifact and prints one line per flagged pair, highest quality first.
    Refuses to overwrite an existing report without --force."""
    thresholds = DuplicateThresholds(
        page_similarity=page_similarity,
        page_gap=page_gap,
        min_quality=min_quality,
    )
    dest = out or Path(DEFAULT_DUPLICATES_OUT)
    if dest.is_file() and not force:
        typer.echo(f"match: {dest} already exists (pass --force to overwrite it)", err=True)
        raise typer.Exit(2)
    try:
        report = find_duplicate_chapters(root, thresholds)
    except ValueError as exc:
        typer.echo(f"match: {exc}", err=True)
        raise typer.Exit(2) from exc
    report.save(dest)
    if as_json:
        typer.echo(json.dumps(report.model_dump(), indent=2))
    else:
        _duplicates_summary(report, dest)


def _duplicates_summary(report: DuplicateReport, dest: Path) -> None:
    """Human-readable result: the pair count and one line per flagged pair (already quality-sorted)."""
    typer.echo(f"match: {len(report.duplicates)} likely duplicate pair(s) -> {dest}")
    for pair in report.duplicates:
        typer.echo(f"match:   {pair.a!r} <-> {pair.b!r} quality {pair.quality:.2f}")
