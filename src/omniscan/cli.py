"""OmniScan command line."""

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from omniscan.core.config import get_config, get_secrets
from omniscan.doctor import run_all_checks
from omniscan.log import setup_logging

app = typer.Typer(help="OmniScan — manhwa/manga translator", no_args_is_help=True)

STATUS_STYLES = {"OK": "green", "WARN": "yellow", "FAIL": "red"}

_STUB_COMMANDS = (
    "acquire",
    "ingest",
    "slice",
    "filter",
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


def cmd_filter(series: Annotated[str | None, typer.Argument()] = None) -> None:
    """Not implemented yet."""
    _stub("filter", series)


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
