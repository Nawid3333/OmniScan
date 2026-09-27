"""`omniscan labelplus` sub-app: a chapter's lines out to a LabelPlus file and back in as English lines."""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Annotated

import typer

from omniscan.core.config import SeriesConfigError, get_config, series_config
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import IngestArtifact, LayoutArtifact
from omniscan.edits import store
from omniscan.interchange.labelplus import export_labels, match_labels, parse, write
from omniscan.interchange.psd import page_psd
from omniscan.translate.on_demand import english_lines

labelplus_app = typer.Typer(
    no_args_is_help=True, help="LabelPlus files: export a chapter, import a translation."
)
psd_app = typer.Typer(
    no_args_is_help=True, help="Layered Photoshop files (raw, clean, text) of a chapter's pages."
)


@psd_app.callback()
def psd_group() -> None:
    """Layered Photoshop files (raw, clean, text) of a chapter's pages."""


class ExportText(enum.StrEnum):
    """Which text the exported labels carry."""

    ENGLISH = "english"
    SOURCE = "source"


def _fail(message: str, tool: str = "labelplus") -> typer.Exit:
    """Print `message` as an error of `tool` and return the exit to raise (code 2)."""
    typer.echo(f"{tool}: {message}", err=True)
    return typer.Exit(2)


def _ingest(path: Path, tool: str = "labelplus") -> IngestArtifact:
    """The chapter's ingest.json (the page geometry); exit 2 when it is missing."""
    if not path.is_file():
        raise _fail("ingest.json not found — run the ingest stage first", tool)
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
    paths = series_paths.chapter(chapter)
    ingest = _ingest(paths.artifact("ingest.json"), "psd")
    layout_path = paths.artifact("layout.json")
    items = LayoutArtifact.load(layout_path).items if layout_path.is_file() else []
    if not items:
        typer.echo("psd: no layout.json yet — the text layers stay empty (run typeset first)", err=True)
    pages = [f for f in ingest.files if not f.filtered and f.y1 > f.y0]
    if page:
        unknown = sorted(set(page) - {f.index for f in pages})
        if unknown:
            raise _fail(f"no page {unknown[0]} in this chapter", "psd")
        pages = [f for f in pages if f.index in set(page)]
    folder = out or series_paths.output_dir / "_psd" / chapter
    folder.mkdir(parents=True, exist_ok=True)
    for source in pages:
        target = folder / f"{Path(source.name).stem}.psd"
        target.write_bytes(page_psd(paths, ingest, source.index, items))
    typer.echo(f"psd: {len(pages)} page(s) -> {folder}")
