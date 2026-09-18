"""OmniScan command line."""

import json
from typing import Annotated

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from omniscan.core.config import Config, get_config, get_secrets
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import FilterArtifact, IngestArtifact, SlicesArtifact
from omniscan.doctor import run_all_checks
from omniscan.filter.decide import decide_files, decide_slices, load_examples, restore
from omniscan.log import setup_logging

app = typer.Typer(help="OmniScan — manhwa/manga translator", no_args_is_help=True)

STATUS_STYLES = {"OK": "green", "WARN": "yellow", "FAIL": "red"}

_STUB_COMMANDS = (
    "acquire",
    "ingest",
    "slice",
    "detect",
    "ocr",
    "glossary",
    "translate",
    "judge",
    "inpaint",
    "typeset",
    "export",
    "run",
    "serve",
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


def cmd_glossary(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("glossary", series)


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


def cmd_serve(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("serve", series)


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
