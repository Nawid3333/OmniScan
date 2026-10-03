"""Interchange sub-apps: LabelPlus files, layered PSD pages, BallonsTranslator projects and manga-image-translator
text files — a chapter out to another tool and that tool's work back in as edits (`interchange.actions`, shared
with the desktop Library page)."""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from omniscan.core.config import get_config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import ProjectFile
from omniscan.interchange import actions, project
from omniscan.interchange.actions import InterchangeError
from omniscan.interchange.blocks import Block
from omniscan.packaging.names import safe_filename

labelplus_app = typer.Typer(
    no_args_is_help=True, help="LabelPlus files: export a chapter, import a translation."
)
psd_app = typer.Typer(
    no_args_is_help=True, help="Layered Photoshop files (raw, clean, text) of a chapter's pages."
)


ballons_app = typer.Typer(
    no_args_is_help=True,
    help="BallonsTranslator projects: import one into a chapter, export a chapter as one.",
)
mit_app = typer.Typer(
    no_args_is_help=True, help="manga-image-translator text files (--save-text): import into a chapter."
)


@psd_app.callback()
def psd_group() -> None:
    """Layered Photoshop files (raw, clean, text) of a chapter's pages."""


@mit_app.callback()
def mit_group() -> None:
    """manga-image-translator text files (--save-text): import into a chapter."""


class ExportText(enum.StrEnum):
    """Which text the exported labels carry."""

    ENGLISH = "english"
    SOURCE = "source"


def _fail(message: str, tool: str = "labelplus") -> typer.Exit:
    """Print `message` as an error of `tool` and return the exit to raise (code 2)."""
    typer.echo(f"{tool}: {message}", err=True)
    return typer.Exit(2)


def _attempt[T](tool: str, action: Callable[[], T]) -> T:
    """Run an interchange action; its refusal is printed as an error of `tool` (exit 2)."""
    try:
        return action()
    except InterchangeError as exc:
        raise _fail(str(exc), tool) from exc


@labelplus_app.command("export")
def labelplus_export(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    out: Annotated[
        Path | None, typer.Option("--out", help="Default: <output_root>/<series>/_labelplus/<chapter>.txt")
    ] = None,
    text: Annotated[
        ExportText, typer.Option("--text", help="english (final lines) or source (OCR text).")
    ] = (ExportText.ENGLISH),
) -> None:
    """Write the chapter's regions as a LabelPlus file: one label per region, on the raw page it sits on."""
    series_paths = SeriesPaths.from_config(get_config(), series)
    which: actions.ExportText = "english" if text is ExportText.ENGLISH else "source"
    done = _attempt("labelplus", lambda: actions.export_labelplus(series_paths, chapter, text=which, out=out))
    typer.echo(f"labelplus: {done.labels} label(s) on {done.pages} page(s) -> {done.path}")


@labelplus_app.command("import")
def labelplus_import(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="The LabelPlus .txt file.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only report what would change.")] = False,
) -> None:
    """Take a LabelPlus file's texts as the English lines of the regions its labels point into (recorded as
    hand-written lines in edits.json, kept by every re-run and remembered by learning)."""
    cfg = get_config()
    series_paths = SeriesPaths.from_config(cfg, series)
    found = _attempt(
        "labelplus", lambda: actions.import_labelplus(cfg, series_paths, chapter, file, dry_run=dry_run)
    )
    verb = "would import" if dry_run else "imported"
    typer.echo(
        f"labelplus: {verb} {found.changed} line(s); {found.same} already the same"
        + (
            f"; {found.joined} label(s) joined into a region another label pointed into"
            if found.joined
            else ""
        )
    )
    for page, label in found.unmatched:
        typer.echo(
            f"labelplus: {page} label {label.number} ({label.x:.3f}, {label.y:.3f}) is outside every region"
        )
    for page in found.unknown_pages:
        typer.echo(f"labelplus: page {page!r} is not in this chapter")


@psd_app.command("export")
def psd_export(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    page: Annotated[
        list[int] | None,
        typer.Option("--page", "-p", help="Page index (as in ingest.json); repeatable. Default: all."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Default: <output_root>/<series>/_psd/<chapter>/")
    ] = None,
) -> None:
    """Write one layered PSD per page: raw, clean (cleaning and hand cleanup) and text (the lettering), with the
    finished page as the composite. Uses layout.json (run typeset first for the text layer)."""
    series_paths = SeriesPaths.from_config(get_config(), series)
    done = _attempt("psd", lambda: actions.export_psd(series_paths, chapter, pages=page, out=out))
    if not done.has_text:
        typer.echo("psd: no layout.json yet — the text layers stay empty (run typeset first)", err=True)
    typer.echo(f"psd: {done.pages} page(s) -> {done.folder}")


Source = Annotated[
    bool, typer.Option("--source", help="Also take the project's source text as each region's OCR text.")
]
Add = Annotated[
    bool, typer.Option("--add", help="Add a region for every block over no region (text and translation).")
]
DryRun = Annotated[bool, typer.Option("--dry-run", help="Only report what would change.")]


def _import_blocks(
    tool: str, series: str, chapter: str, blocks: Sequence[Block], *, source: bool, add: bool, dry_run: bool
) -> None:
    """Take a foreign project's blocks into the chapter as hand edits in edits.json (`actions.import_blocks`).
    Prints what changed and what matched nothing."""
    cfg = get_config()
    series_paths = SeriesPaths.from_config(cfg, series)
    found = _attempt(
        tool,
        lambda: actions.import_blocks(
            cfg, series_paths, chapter, blocks, source=source, add=add, dry_run=dry_run
        ),
    )
    verb = "would import" if dry_run else "imported"
    summary = f"{tool}: {verb} {found.lines} English line(s)"
    if source:
        summary += f", {found.sources} source text(s)"
    summary += f"; {found.same} line(s) already the same"
    if found.joined:
        summary += f"; {found.joined} block(s) joined into a region another block is in"
    typer.echo(summary)
    for block, box, note in found.unmatched:
        typer.echo(
            f"{tool}: {block.page} block at {box.x0},{box.y0}-{box.x1},{box.y1} (strip) is over no region{note}"
        )
    for page in found.unknown_pages:
        typer.echo(f"{tool}: page {page!r} is not in this chapter")


@ballons_app.command("import")
def ballons_import(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    project: Annotated[
        Path,
        typer.Argument(exists=True, help="The project's imgtrans_*.json, or the page folder that holds it."),
    ],
    source: Source = False,
    add: Add = False,
    dry_run: DryRun = False,
) -> None:
    """Take a BallonsTranslator project's translations as the English lines of the regions its text blocks lie
    on (hand-written lines in edits.json: every re-run keeps them, learning remembers them)."""
    blocks = _attempt("ballons", lambda: actions.read_ballons(project))
    _import_blocks("ballons", series, chapter, blocks, source=source, add=add, dry_run=dry_run)


@ballons_app.command("export")
def ballons_export(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="The project folder. Default: <output_root>/<series>/_ballons/<chapter>/"),
    ] = None,
) -> None:
    """Write the chapter as a BallonsTranslator project: copies of its pages, the cleaned pages (inpainted/)
    and one text block per region with its source text, English line and lettering size and colours."""
    series_paths = SeriesPaths.from_config(get_config(), series)
    done = _attempt("ballons", lambda: actions.export_ballons(series_paths, chapter, out=out))
    typer.echo(
        f"ballons: {done.blocks} text block(s) on {done.pages} page(s)"
        + ("" if done.cleaned else " (not cleaned yet: no inpainted pages)")
        + f" -> {done.folder}"
    )


@mit_app.command("import")
def mit_import(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    files: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="One or more *_translations.txt files."),
    ],
    source: Source = False,
    add: Add = False,
    dry_run: DryRun = False,
) -> None:
    """Take manga-image-translator's translations (its --save-text files) as the English lines of the regions
    its text regions lie on (hand-written lines in edits.json: every re-run keeps them)."""
    blocks = _attempt("mit", lambda: actions.read_mit(files))
    _import_blocks("mit", series, chapter, blocks, source=source, add=add, dry_run=dry_run)


project_app = typer.Typer(
    no_args_is_help=True,
    help="Chapter projects: one chapter's raw pages and all its work in one file, to pass to another OmniScan user.",
)


@project_app.callback()
def project_group() -> None:
    """Chapter projects: one chapter's raw pages and all its work in one file, to pass to another OmniScan user."""


def _size(size: int) -> str:
    """A file size for people: kB below a megabyte."""
    return f"{size / 1000:.0f} kB" if size < 1_000_000 else f"{size / 1_000_000:.1f} MB"


def _parts(files: Sequence[ProjectFile]) -> str:
    """How many files a project holds per part (raw pages, work files, series files, finished pages)."""
    counts: dict[str, int] = {}
    for file in files:
        part = file.path.partition("/")[0]
        counts[part] = counts.get(part, 0) + 1
    labels = {
        "raw": "raw page file(s)",
        "work": "work file(s)",
        "series": "series file(s)",
        "output": "finished page(s)",
    }
    return ", ".join(f"{counts[part]} {label}" for part, label in labels.items() if part in counts)


@project_app.command("pack")
def project_pack(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="The file to write. Default: '<series> - <chapter>.omniscan' here."
        ),
    ] = None,
    with_output: Annotated[
        bool, typer.Option("--with-output", help="Also pack the finished pages (for the quality check).")
    ] = False,
) -> None:
    """Pack a chapter — raw pages, every stage's work, hand edits with their undo history, hand cleanup, and the
    series' series.toml / voices.toml — into one file another OmniScan user unpacks and continues."""
    series_paths = SeriesPaths.from_config(get_config(), series)
    if chapter not in series_paths.chapters():
        raise _fail(f"no chapter {chapter!r} in {series}", "project")
    dest = (
        output
        if output is not None
        else Path.cwd() / f"{safe_filename(series)} - {safe_filename(chapter)}{project.SUFFIX}"
    )
    try:
        packed = project.pack(series_paths, chapter, dest, with_output=with_output)
    except project.ProjectError as exc:
        raise _fail(str(exc), "project") from exc
    typer.echo(f"project: {_parts(packed.files)} -> {dest} ({_size(dest.stat().st_size)})")


@project_app.command("unpack")
def project_unpack(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A .omniscan chapter project.")],
    series: Annotated[str | None, typer.Option("--series", help="Unpack into this series instead.")] = None,
    chapter: Annotated[str | None, typer.Option("--chapter", help="Unpack as this chapter instead.")] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Replace the chapter if it exists (its raw pages, work and finished pages)."
        ),
    ] = False,
) -> None:
    """Unpack a chapter project into your library and work folders; every stage done and every hand edit comes
    along. The series files it carries are added only where the series has none, and of its series.toml only
    numbers, switches and fixed choices (never a download address, a file, a model or a font)."""
    try:
        done = project.unpack(file, get_config(), series=series, chapter=chapter, force=force)
    except (project.ProjectError, OSError) as exc:
        raise _fail(str(exc), "project") from exc
    notes = [f"added {', '.join(done.series_files)}"] if done.series_files else []
    notes += [f"kept your {', '.join(done.kept_series_files)}"] if done.kept_series_files else []
    notes += [f"left out {', '.join(done.dropped_settings)}"] if done.dropped_settings else []
    verb = "replaced" if done.replaced else "unpacked"
    typer.echo(
        f"project: {verb} {done.paths.series}/{done.paths.chapter}"
        + (f" ({'; '.join(notes)})" if notes else "")
    )


@project_app.command("show")
def project_show(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A .omniscan chapter project.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print its project.json.")] = False,
) -> None:
    """Say what a chapter project holds, without unpacking it (checked like `unpack` checks it)."""
    try:
        shown = project.show(file)
    except project.ProjectError as exc:
        raise _fail(str(exc), "project") from exc
    if as_json:
        typer.echo(shown.model_dump_json(indent=2))
        return
    total = sum(entry.bytes for entry in shown.files)
    typer.echo(
        f"project: {shown.series}/{shown.chapter} (OmniScan {shown.app_version}): {_parts(shown.files)},"
        f" {_size(total)} unpacked"
    )
