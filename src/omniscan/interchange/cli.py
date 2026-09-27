"""`omniscan labelplus` sub-app: a chapter's lines out to a LabelPlus file and back in as English lines."""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Annotated

import typer

from omniscan.core.config import SeriesConfigError, get_config, series_config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import IngestArtifact
from omniscan.edits import store
from omniscan.interchange.labelplus import export_labels, match_labels, parse, write
from omniscan.translate.on_demand import english_lines

labelplus_app = typer.Typer(
    no_args_is_help=True, help="LabelPlus files: export a chapter, import a translation."
)


class ExportText(enum.StrEnum):
    """Which text the exported labels carry."""

    ENGLISH = "english"
    SOURCE = "source"


def _fail(message: str) -> typer.Exit:
    """Print `message` as a labelplus error and return the exit to raise (code 2)."""
    typer.echo(f"labelplus: {message}", err=True)
    return typer.Exit(2)


def _ingest(path: Path) -> IngestArtifact:
    """The chapter's ingest.json (the page geometry); exit 2 when it is missing."""
    if not path.is_file():
        raise _fail("ingest.json not found — run the ingest stage first")
    return IngestArtifact.load(path)


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
    paths = series_paths.chapter(chapter)
    ingest = _ingest(paths.artifact("ingest.json"))
    regions = store.current_regions(paths)
    texts = english_lines(paths) if text is ExportText.ENGLISH else {r.id: r.text for r in regions}
    doc = export_labels(ingest, regions, texts, comment=f"OmniScan: {series} / {chapter} ({text.value})")
    target = out or series_paths.output_dir / "_labelplus" / f"{chapter}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(write(doc), encoding="utf-8-sig")  # LabelPlus itself writes a BOM
    count = sum(len(labels) for labels in doc.pages.values())
    typer.echo(f"labelplus: {count} label(s) on {len(doc.pages)} page(s) -> {target}")


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
    paths = series_paths.chapter(chapter)
    try:
        direction = series_config(cfg, series_paths.library_dir).detect.reading_direction
        doc = parse(file.read_text(encoding="utf-8-sig"))
    except (SeriesConfigError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    found = match_labels(doc, _ingest(paths.artifact("ingest.json")), store.current_regions(paths))
    current = english_lines(paths)
    changed = {rid: text for rid, text in found.texts.items() if current.get(rid) != text}
    if not dry_run and changed:
        try:  # one write and one rebuild: the file lands whole or not at all
            store.set_translations(paths, changed, direction=direction)
        except store.EditNotFoundError as exc:
            raise _fail(str(exc)) from exc
    verb = "would import" if dry_run else "imported"
    typer.echo(
        f"labelplus: {verb} {len(changed)} line(s); {len(found.texts) - len(changed)} already the same"
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
