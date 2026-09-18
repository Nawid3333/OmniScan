"""OmniScan command line. Placeholder until card B1 builds the full CLI."""

import typer

app = typer.Typer(help="OmniScan — manhwa/manga translator", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the OmniScan version."""
    from omniscan import __version__

    typer.echo(__version__)
