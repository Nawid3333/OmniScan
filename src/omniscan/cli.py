"""OmniScan command line."""

import enum
import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from omniscan.core.config import Config, get_config, get_secrets
from omniscan.core.paths import ChapterPaths, SeriesPaths, chapter_number, list_chapters, list_images
from omniscan.core.schemas import FilterArtifact, IngestArtifact, SlicesArtifact
from omniscan.doctor import run_all_checks
from omniscan.filter.decide import decide_files, decide_slices, load_examples, restore
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml, import_yaml
from omniscan.importer.execute import execute_import
from omniscan.importer.plan import ImportPlanError, plan_import
from omniscan.log import setup_logging
from omniscan.packaging import pack_cbz, pack_pdf, safe_filename
from omniscan.watermark.store import WatermarkStore

app = typer.Typer(help="OmniScan — manhwa/manga translator", no_args_is_help=True)

STATUS_STYLES = {"OK": "green", "WARN": "yellow", "FAIL": "red"}

_STUB_COMMANDS = (
    "acquire",
    "ingest",
    "slice",
    "detect",
    "ocr",
    "translate",
    "judge",
    "inpaint",
    "typeset",
    "export",
    "run",
    "reference",
)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug log output.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Warnings only.")] = False,
) -> None:
    """OmniScan global options."""
    setup_logging("DEBUG" if verbose else "WARNING" if quiet else "INFO")


@app.command()
def version() -> None:
    """Print the OmniScan version."""
    from omniscan import __version__

    typer.echo(__version__)


@app.command()
def doctor(
    as_json: Annotated[bool, typer.Option("--json", help="Emit a JSON array instead of a table.")] = False,
) -> None:
    """Check this machine is ready for OmniScan."""
    results = run_all_checks(get_config(), get_secrets())
    if as_json:
        typer.echo(
            json.dumps(
                [{"name": r.name, "status": r.status, "detail": r.detail} for r in results],
                indent=2,
            )
        )
    else:
        console = Console()
        table = Table(title="omniscan doctor")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        for r in results:
            table.add_row(r.name, f"[{STATUS_STYLES[r.status]}]{r.status}[/]", r.detail)
        console.print(table)
        counts = {s: sum(1 for r in results if r.status == s) for s in ("OK", "WARN", "FAIL")}
        console.print(f"{counts['OK']} ok, {counts['WARN']} warn, {counts['FAIL']} fail")
    if any(r.status == "FAIL" for r in results):
        raise typer.Exit(1)


def cmd_import(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    series: Annotated[str | None, typer.Option("--series")] = None,
    chapter: Annotated[str | None, typer.Option("--chapter")] = None,
    move: Annotated[
        bool, typer.Option("--move", help="Move instead of copy; delete source after import.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan without writing anything.")
    ] = False,
) -> None:
    """Import raw chapter images from a local folder into the library."""
    try:
        plan = plan_import(source, series=series, chapter=chapter)
        if dry_run:
            typer.echo(f"import: plan for series '{plan.series}' — {len(plan.items)} chapter(s)")
            for item in plan.items:
                typer.echo(f"import:   {item.chapter}: {len(item.files)} file(s)")
            for warning in plan.warnings:
                typer.echo(f"import:   {warning}", err=True)
            return
        result = execute_import(plan, get_config().paths.library_root, move=move)
    except ImportPlanError as exc:
        typer.echo(f"import: {exc}", err=True)
        raise typer.Exit(2) from exc
    for chapter_written in result.chapters_written:
        typer.echo(f"import: {plan.series}/{chapter_written}")
    for warning in plan.warnings:
        typer.echo(f"import: {warning}", err=True)
    typer.echo(
        f"import: {result.files_copied} file(s) copied, "
        f"{result.files_skipped_duplicate} duplicate file(s) skipped"
    )


app.command("import")(cmd_import)


class PackFormat(enum.StrEnum):
    """Output format of `omniscan pack` (typer cannot build a click option from list[Literal])."""

    CBZ = "cbz"
    PDF = "pdf"


def cmd_pack(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    fmt: Annotated[
        list[PackFormat] | None,
        typer.Option("--format", "-f", help="cbz or pdf; repeatable. Default: cbz."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output folder. Default: <output_root>/<series>/_packaged")
    ] = None,
) -> None:
    """Package finished chapters (output_root/<series>/<chapter>/*.jpg) into CBZ and/or PDF files."""
    cfg = get_config()
    sp = SeriesPaths.from_config(cfg, series)
    chapters = chapter or [p.name for p in list_chapters(sp.output_dir)]
    formats = list(dict.fromkeys(fmt or [PackFormat.CBZ]))
    dest_dir = out or sp.output_dir / "_packaged"
    written = 0
    for chap in chapters:
        images = list_images(sp.chapter(chap).output_dir)
        if not images:
            typer.echo(f"pack: no images in {series}/{chap}, skipped", err=True)
            continue
        n = chapter_number(chap)
        stem = safe_filename(f"{series} - {chap}")
        for extension in formats:
            dest = dest_dir / f"{stem}.{extension.value}"
            if extension is PackFormat.CBZ:
                pack_cbz(
                    images, dest, title=chap, series=series, number=(f"{n:g}" if n is not None else None)
                )
            else:
                pack_pdf(images, dest)
            typer.echo(f"pack: {dest}")
            written += 1
    if not written:
        typer.echo(f"pack: nothing to pack for series {series!r}", err=True)
        raise typer.Exit(2)


app.command("pack")(cmd_pack)


def _stub(name: str, series: str | None) -> None:
    typer.echo(f"{name}: not implemented yet", err=True)
    raise typer.Exit(2)


def cmd_acquire(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("acquire", series)


def cmd_ingest(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("ingest", series)


def cmd_slice(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("slice", series)


def cmd_detect(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("detect", series)


def cmd_ocr(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("ocr", series)


def cmd_translate(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("translate", series)


def cmd_judge(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("judge", series)


def cmd_inpaint(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("inpaint", series)


def cmd_typeset(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("typeset", series)


def cmd_export(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("export", series)


def cmd_run(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("run", series)


def cmd_serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the web debug tool's API (pair with `npm run dev` in webui/ for the UI)."""
    import uvicorn

    from omniscan.web.app import create_app

    uvicorn.run(create_app(get_config()), host=host, port=port, reload=reload)


app.command("serve")(cmd_serve)


def cmd_reference(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("reference", series)


def _register_stubs() -> None:
    for name in _STUB_COMMANDS:
        app.command(name)(globals()[f"cmd_{name}"])


_register_stubs()


def _chapter_paths(cfg: Config, series: str, chapter: str) -> ChapterPaths:
    return SeriesPaths.from_config(cfg, series).chapter(chapter)


def _assemble_strip(paths: ChapterPaths, ingest: IngestArtifact) -> Image.Image:
    """Paste each raw file (resized to its strip-space y-range) into one canvas for hashing."""
    strip = Image.new("RGB", (ingest.strip_width, ingest.strip_height), (255, 255, 255))
    for source_file in ingest.files:
        with Image.open(paths.raw_dir / source_file.name) as img:
            img = img.convert("RGB").resize((ingest.strip_width, source_file.y1 - source_file.y0))
            strip.paste(img, (0, source_file.y0))
    return strip


filter_app = typer.Typer(no_args_is_help=True, help="Promo filter: pHash match against example images.")


@filter_app.command("run")
def filter_run(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    threshold: Annotated[float, typer.Option(help="Similarity threshold for a filtered verdict.")] = 0.90,
) -> None:
    """Filter promo files/slices of a chapter against the user's promo examples."""
    cfg = get_config()
    paths = _chapter_paths(cfg, series, chapter)
    ingest_path = paths.artifact("ingest.json")
    if not ingest_path.is_file():
        typer.echo(f"filter: no ingest.json for {series}/{chapter} — run ingest first", err=True)
        raise typer.Exit(2)
    slices_path = paths.artifact("slices.json")
    if not slices_path.is_file():
        typer.echo(f"filter: no slices.json for {series}/{chapter} — run slice first", err=True)
        raise typer.Exit(2)

    ingest = IngestArtifact.load(ingest_path)
    slices = SlicesArtifact.load(slices_path)
    strip = _assemble_strip(paths, ingest)
    examples = load_examples(cfg.paths.promo_examples, series)
    decisions = decide_files(paths, ingest, examples, threshold) + decide_slices(
        paths, chapter, strip, slices, examples, threshold
    )
    FilterArtifact(decisions=decisions).save(paths.artifact("filter.json"))
    n_keep = sum(1 for d in decisions if d.decision == "keep")
    n_file = sum(1 for d in decisions if d.decision == "filtered" and d.target == "file")
    n_slice = sum(1 for d in decisions if d.decision == "filtered" and d.target == "slice")
    typer.echo(f"filter: kept {n_keep}, filtered {n_file} file(s), {n_slice} slice(s)")


@filter_app.command("restore")
def filter_restore(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    target: Annotated[str, typer.Argument()],
    index: Annotated[int, typer.Argument()],
) -> None:
    """Restore a previously filtered file or slice (metadata override; nothing is deleted)."""
    if target not in ("file", "slice"):
        raise typer.BadParameter("target must be 'file' or 'slice'")
    paths = _chapter_paths(get_config(), series, chapter)
    decision = restore(paths, target, index)  # type: ignore[arg-type]
    typer.echo(
        f"filter: {decision.decision} {decision.target} {decision.index} "
        f"(score {decision.score}, method {decision.method})"
    )


app.add_typer(filter_app, name="filter")

GlossaryStatus = Literal["proposed", "locked", "rejected"]

glossary_app = typer.Typer(
    no_args_is_help=True, help="Per-series glossary: SQLite working copy + review YAML."
)


def _series_paths(series: str) -> SeriesPaths:
    return SeriesPaths.from_config(get_config(), series)


def _open_glossary(paths: SeriesPaths) -> GlossaryStore:
    if not paths.db.is_file():
        typer.echo(f"glossary: no db for series {paths.series!r} at {paths.db}", err=True)
        raise typer.Exit(2)
    return GlossaryStore(paths.db)


@glossary_app.command("list")
def glossary_list(
    series: Annotated[str, typer.Argument()],
    status: Annotated[
        GlossaryStatus | None, typer.Option("--status", help="Only entries with this status.")
    ] = None,
) -> None:
    """Print the glossary of a series as a table (source, target, type, status, count)."""
    with _open_glossary(_series_paths(series)) as store:
        entries = store.list(status=status)
    table = Table(title=f"glossary: {series} ({len(entries)} entries)")
    table.add_column("Source")
    table.add_column("Target")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for entry in entries:
        table.add_row(entry.source, entry.target, entry.type, entry.status, str(entry.count))
    Console().print(table)


@glossary_app.command("export")
def glossary_export(series: Annotated[str, typer.Argument()]) -> None:
    """Write the full glossary of a series to its glossary.yaml (hand-editable)."""
    paths = _series_paths(series)
    with _open_glossary(paths) as store:
        export_yaml(store, paths.glossary_yaml)
        count = len(store.list())
    typer.echo(f"glossary: exported {count} entries -> {paths.glossary_yaml}")


@glossary_app.command("import")
def glossary_import(
    series: Annotated[str, typer.Argument()],
    mode: Annotated[
        Literal["merge", "replace"], typer.Option("--mode", help="merge updates by source, replace rebuilds.")
    ] = "merge",
) -> None:
    """Import glossary entries from the series' glossary.yaml into its db."""
    paths = _series_paths(series)
    if not paths.glossary_yaml.is_file():
        typer.echo(f"glossary: no glossary.yaml for series {series!r} at {paths.glossary_yaml}", err=True)
        raise typer.Exit(2)
    with _open_glossary(paths) as store:
        written = import_yaml(store, paths.glossary_yaml, mode=mode)
    typer.echo(f"glossary: wrote {written} entries ({mode}) from {paths.glossary_yaml}")


app.add_typer(glossary_app, name="glossary")

watermark_app = typer.Typer(no_args_is_help=True, help="Per-series fixed-position watermark regions.")


@watermark_app.command("add")
def watermark_add(
    series: Annotated[str, typer.Argument()],
    x0: Annotated[float, typer.Option("--x0", min=0.0, max=1.0, help="Left edge, fraction of page width.")],
    y0: Annotated[float, typer.Option("--y0", min=0.0, max=1.0, help="Top edge, fraction of page height.")],
    x1: Annotated[float, typer.Option("--x1", min=0.0, max=1.0, help="Right edge, fraction of page width.")],
    y1: Annotated[
        float, typer.Option("--y1", min=0.0, max=1.0, help="Bottom edge, fraction of page height.")
    ],
    note: Annotated[
        str | None, typer.Option("--note", help="Free-text reminder of what the region holds.")
    ] = None,
) -> None:
    """Record a fixed-position watermark region for a series (fractions of every raw page)."""
    try:
        region = WatermarkStore(_series_paths(series).work_dir).add(x0, y0, x1, y1, note=note)
    except ValueError as exc:
        typer.echo(f"watermark: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"watermark: added region {region.index} to {series}")


@watermark_app.command("list")
def watermark_list(series: Annotated[str, typer.Argument()]) -> None:
    """Print the watermark regions of a series as a table."""
    regions = WatermarkStore(_series_paths(series).work_dir).list()
    table = Table(title=f"watermark: {series} ({len(regions)} region(s))")
    table.add_column("Index", justify="right")
    table.add_column("x0-x1")
    table.add_column("y0-y1")
    table.add_column("Note")
    for region in regions:
        table.add_row(
            str(region.index),
            f"{region.x0_frac:.3f}-{region.x1_frac:.3f}",
            f"{region.y0_frac:.3f}-{region.y1_frac:.3f}",
            region.note or "",
        )
    Console().print(table)


@watermark_app.command("remove")
def watermark_remove(
    series: Annotated[str, typer.Argument()],
    index: Annotated[int, typer.Argument()],
) -> None:
    """Remove a watermark region of a series by index (remaining indices keep their values)."""
    try:
        WatermarkStore(_series_paths(series).work_dir).remove(index)
    except KeyError:
        typer.echo(f"watermark: no region {index} for {series}", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"watermark: removed region {index} from {series}")


app.add_typer(watermark_app, name="watermark")
