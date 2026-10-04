"""`omniscan runtime`: the packaged app's GPU runtime — which torch runs, which one this PC should run, downloading
it (#44)."""

from __future__ import annotations

import sys
from typing import Annotated, cast, get_args

import typer

from omniscan import runtime
from omniscan.hw.tune import TorchExtra

runtime_app = typer.Typer(
    no_args_is_help=True,
    help="The packaged app's GPU runtime: the PyTorch build for this PC's graphics card.",
)

BACKENDS = get_args(TorchExtra)


def _fail(message: str) -> typer.Exit:
    """Print `message` as a runtime error and return the exit to raise (code 2)."""
    typer.echo(f"runtime: {message}", err=True)
    return typer.Exit(2)


def _recommendation() -> runtime.Recommendation:
    """What this PC should run (hardware detection plus the cards the OS reports)."""
    from omniscan.hw.detect import detect_hardware

    return runtime.recommend(detect_hardware())


@runtime_app.command("status")
def runtime_status() -> None:
    """Show which torch runs now and from where, the runtimes installed, and the one this PC should run."""
    folder = runtime.active()
    typer.echo(f"active runtime: {folder.name if folder else 'none (the torch shipped with the app)'}")
    try:
        import torch

        build = "rocm" if getattr(torch.version, "hip", None) else "cuda" if torch.version.cuda else "cpu"
        typer.echo(f"torch {torch.__version__} ({build}) from {torch.__file__}")
    except Exception as exc:  # a broken runtime must still let the command report
        typer.echo(f"torch cannot be imported: {exc}")
    for name in runtime.installed():
        typer.echo(f"installed: {name}")
    best = _recommendation()
    typer.echo(
        f"this PC: {best.backend}"
        + (f" for {best.gpu}" if best.gpu else "")
        + (f" ({best.note})" if best.note else "")
    )


@runtime_app.command("install")
def runtime_install(
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help=f"auto (what this PC needs) or one of: {', '.join(BACKENDS)}."),
    ] = "auto",
    force: Annotated[bool, typer.Option("--force", help="Also in a source checkout (for testing).")] = False,
) -> None:
    """Download the PyTorch build for this PC's graphics card (a few GB for a GPU build) and use it from the next
    start of OmniScan on."""
    if not getattr(sys, "frozen", False) and not force:
        raise _fail(
            "a source checkout picks its torch with `uv sync --extra <backend>`; this is for the packaged app"
        )
    chosen = _recommendation().backend if backend == "auto" else backend
    if chosen not in BACKENDS:
        raise _fail(f"unknown backend {backend!r} (auto or one of {', '.join(BACKENDS)})")
    try:
        folder = runtime.install(cast(TorchExtra, chosen))
    except runtime.RuntimeSetupError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"runtime: installed {folder.name} in {folder.parent}; restart OmniScan to use it")


@runtime_app.command("use")
def runtime_use(
    name: Annotated[str, typer.Argument(help="An installed runtime (see `runtime status`), or `bundled`.")],
) -> None:
    """Choose the runtime the next start uses (`bundled`: the torch shipped with the app)."""
    try:
        runtime.use(None if name == "bundled" else name)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"runtime: {name} from the next start on")


@runtime_app.command("remove")
def runtime_remove(
    name: Annotated[str, typer.Argument(help="An installed runtime (see `runtime status`).")],
) -> None:
    """Delete an installed runtime (the active one first falls back to the bundled torch)."""
    try:
        runtime.remove(name)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"runtime: removed {name}")
