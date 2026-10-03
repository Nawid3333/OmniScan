"""`omniscan workflow`: a group's chapter status, hand-overs, notes on regions and each role's to-do list (#38)."""

from __future__ import annotations

import json
from typing import Annotated, cast

import typer

from omniscan.core.config import get_config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import WORKFLOW_STEPS, WorkflowStep
from omniscan.workflow.notes import add_note, load_notes, resolve_note
from omniscan.workflow.status import ROLES, Role, describe, hand_over, load_status, mark_step, next_step
from omniscan.workflow.todo import todo

workflow_app = typer.Typer(
    no_args_is_help=True,
    help="A group's work on a chapter: steps done, who has it, notes on regions, what each role has left.",
)

Series = Annotated[str, typer.Argument()]
Chapter = Annotated[str, typer.Argument()]


def _fail(message: str) -> typer.Exit:
    """Print `message` as a workflow error and return the exit to raise (code 2)."""
    typer.echo(f"workflow: {message}", err=True)
    return typer.Exit(2)


def _paths(series: str, chapter: str) -> ChapterPaths:
    """The chapter's paths; exit 2 when the series has no such chapter."""
    sp = SeriesPaths.from_config(get_config(), series)
    if chapter not in sp.chapters():
        raise _fail(f"no chapter {chapter!r} in series {series!r}")
    return sp.chapter(chapter)


def _me() -> str | None:
    """The `[user] name` this machine records, or None."""
    return get_config().user.name or None


@workflow_app.command("status")
def workflow_status(
    series: Series,
    chapter: Annotated[
        str | None, typer.Argument(help="One chapter; default: every chapter of the series.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the status as JSON.")] = False,
) -> None:
    """Show which steps are done and who has each chapter (one line per chapter), or one chapter's history."""
    sp = SeriesPaths.from_config(get_config(), series)
    names = [chapter] if chapter is not None else sp.chapters()
    if chapter is not None:
        _paths(series, chapter)
    try:
        statuses = {name: load_status(sp.chapter(name)) for name in names}
    except ValueError as exc:
        raise _fail(f"damaged chapter_status.json: {exc}") from exc
    if as_json:
        typer.echo(json.dumps({name: s.model_dump(mode="json") for name, s in statuses.items()}, indent=2))
        return
    for name, status in statuses.items():
        upcoming = next_step(status)
        typer.echo(f"{name}: {describe(status)}" + (f"; next: {upcoming}" if upcoming else ""))
        if chapter is not None:
            for event in status.events:
                what = f"{event.kind} {event.step}" if event.step else f"handed to {event.to or 'nobody'}"
                who = f" by {event.by}" if event.by else ""
                note = f" — {event.note}" if event.note else ""
                typer.echo(f"  {event.at:%Y-%m-%d %H:%M} {what}{who}{note}")


@workflow_app.command("done")
def workflow_done(
    series: Series,
    chapter: Chapter,
    step: Annotated[str, typer.Argument(help=f"One of: {', '.join(WORKFLOW_STEPS)}.")],
    undo: Annotated[bool, typer.Option("--undo", help="Mark the step not done any more.")] = False,
    note: Annotated[str | None, typer.Option("--note", help="A remark kept with the change.")] = None,
) -> None:
    """Mark a step of the chapter done (translated, proofread, cleaned, lettered, qc_passed)."""
    if step not in WORKFLOW_STEPS:
        raise _fail(f"unknown step {step!r} (one of {', '.join(WORKFLOW_STEPS)})")
    status = mark_step(_paths(series, chapter), cast(WorkflowStep, step), done=not undo, by=_me(), note=note)
    typer.echo(f"{chapter}: {describe(status)}")


@workflow_app.command("hand")
def workflow_hand(
    series: Series,
    chapter: Chapter,
    to: Annotated[str, typer.Argument(help='Who has the chapter now ("" for nobody in particular).')],
    note: Annotated[str | None, typer.Option("--note", help="A remark for the next person.")] = None,
) -> None:
    """Hand the chapter over to the next person (send them the chapter with `omniscan project pack`)."""
    status = hand_over(_paths(series, chapter), to, by=_me(), note=note)
    typer.echo(f"{chapter}: {describe(status)}")


@workflow_app.command("todo")
def workflow_todo(
    series: Series,
    chapter: Chapter,
    role: Annotated[str, typer.Option("--role", "-r", help=f"One of: {', '.join(ROLES)}.")],
) -> None:
    """List what a role still has to look at in the chapter (one line per region and reason)."""
    if role not in ROLES:
        raise _fail(f"unknown role {role!r} (one of {', '.join(ROLES)})")
    items = todo(_paths(series, chapter), cast(Role, role))
    for item in items:
        typer.echo(f"{item.region_id}\tpage {item.page + 1}\t{item.what}")
    typer.echo(f"{len(items)} item(s) for the {role}", err=True)


@workflow_app.command("note")
def workflow_note(
    series: Series,
    chapter: Chapter,
    region_id: Annotated[str, typer.Argument(help="Region id as `omniscan edit show` lists it.")],
    text: Annotated[str, typer.Argument(help="The note.")],
) -> None:
    """Leave a note on a region for the next person (the line itself is not changed)."""
    try:
        note = add_note(_paths(series, chapter), region_id, text, by=_me())
    except KeyError as exc:
        raise _fail(f"no region {region_id!r} in {chapter!r}") from exc
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"{note.id} on {note.region_id}")


@workflow_app.command("notes")
def workflow_notes(
    series: Series,
    chapter: Chapter,
    show_all: Annotated[bool, typer.Option("--all", help="Resolved notes too.")] = False,
) -> None:
    """List the chapter's open notes (with --all, the resolved ones too)."""
    for note in load_notes(_paths(series, chapter)):
        if note.resolved and not show_all:
            continue
        who = f" ({note.by})" if note.by else ""
        state = " [resolved]" if note.resolved else ""
        typer.echo(f"{note.id}\t{note.region_id}\t{note.text}{who}{state}")


@workflow_app.command("resolve")
def workflow_resolve(
    series: Series,
    chapter: Chapter,
    note_id: Annotated[str, typer.Argument(help="Note id as `workflow notes` lists it (n0001, …).")],
    reopen: Annotated[bool, typer.Option("--reopen", help="Open the note again.")] = False,
) -> None:
    """Mark a note resolved (or open it again)."""
    try:
        note = resolve_note(_paths(series, chapter), note_id, resolved=not reopen)
    except KeyError as exc:
        raise _fail(f"no note {note_id!r} in {chapter!r}") from exc
    typer.echo(f"{note.id}: {'resolved' if note.resolved else 'open'}")
