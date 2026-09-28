import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

OD = "src/omniscan/ocr/on_demand.py"
W = "src/omniscan/web/app.py"
E = "src/omniscan/edits/cli.py"
MUTANTS = [
    (
        OD,
        "        x1=min(width, box.x1 + margin),",
        "        x1=box.x1 + margin,",
        "the crop can leave the strip",
    ),
    (OD, "        y0=max(0, box.y0 - margin),", "        y0=box.y0,", "no margin above the box"),
    (
        OD,
        "y0=box.y0 - crop.y0, x1=box.x1 - crop.x0",
        "y0=box.y0, x1=box.x1 - crop.x0",
        "the box not moved into the crop",
    ),
    (
        OD,
        "        raise store.EditNotFoundError(",
        "        return None  # type: ignore[return-value]\n        raise store.EditNotFoundError(",
        "an unknown region read anyway",
    ),
    (
        W,
        "        applied = body.apply and bool(reading.text.strip())",
        "        applied = body.apply",
        "an empty reading blanks the text (web)",
    ),
    (
        W,
        "        applied = body.apply and bool(reading.text.strip())",
        "        applied = False",
        "apply ignored (web)",
    ),
    (
        W,
        "        except (ImportError, OSError, RuntimeError, ValueError) as exc:  # no torch backend, no models, no GPU\n"
        '            raise HTTPException(status_code=503, detail=f"the OCR could not run: {exc}") from exc\n',
        "",
        "an OCR failure crashes the request instead of a 503",
    ),
    (E, "    if apply and reading.text.strip():", "    if apply:", "an empty reading blanks the text (CLI)"),
    (
        E,
        '        raise typer.Exit(1) from exc\n    typer.echo(\n        f"edit: {region} reads',
        '        raise typer.Exit(0) from exc\n    typer.echo(\n        f"edit: {region} reads',
        "an OCR failure exits 0",
    ),
]
