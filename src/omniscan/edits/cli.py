"""`omniscan edit` sub-app: the Studio's hand edits from the command line (scripts, batch fixes, no browser).

Every command goes through the same edit operations as the Studio (edits/store.py), so an edit made here is
recorded in edits.json, applied to ocr.json / final.json at once and survives every pipeline re-run.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Callable
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from omniscan.core.config import Config, SeriesConfigError, get_config, get_secrets, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, RegionKind
from omniscan.edits import store
from omniscan.edits.replace import FindReplace, apply_changes, plan
from omniscan.llm.ollama import OllamaClient, OllamaError
from omniscan.translate.on_demand import english_lines, translate_now
from omniscan.translate.profiles import default_profile_paths, load_profiles
from omniscan.translate.run import ChatClient
from omniscan.translate.voices import VOICES_FILE, character_named, load_voices

edit_app = typer.Typer(
    no_args_is_help=True, help="Edit a chapter by hand: regions, source text, English, cuts."
)


class Kind(enum.StrEnum):
    """A region kind (typer cannot build a click choice from a Literal)."""

    BUBBLE_TEXT = "bubble_text"
    FREE_TEXT = "free_text"
    SFX = "sfx"
    WATERMARK = "watermark"


def make_chat_client(cfg: Config) -> ChatClient:
    """The chat client of `edit translate` (tests replace this to fake the model)."""
    return OllamaClient(cfg.ollama, get_secrets())


def _fail(message: str) -> typer.Exit:
    """Print `message` as an edit error and return the exit to raise (code 2)."""
    typer.echo(f"edit: {message}", err=True)
    return typer.Exit(2)


def _chapter(series: str, chapter: str) -> tuple[Config, SeriesPaths, ChapterPaths]:
    """The series' settings (series.toml applied) and the paths of the series and the chapter."""
    cfg = get_config()
    paths = SeriesPaths.from_config(cfg, series)
    try:
        scfg = series_config(cfg, paths.library_dir)
    except SeriesConfigError as exc:
        raise _fail(str(exc)) from exc
    return scfg, paths, paths.chapter(chapter)


def _run[T](operation: Callable[[], T]) -> T:
    """Run one edit operation; a missing artifact or region and a bad value exit 2 with the reason."""
    try:
        return operation()
    except (store.EditNotFoundError, FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


Series = Annotated[str, typer.Argument()]
Chapter = Annotated[str, typer.Argument()]
RegionId = Annotated[str, typer.Argument(help="Region id as `edit show` lists it (r0001, m0001, …).")]


@edit_app.command("show")
def edit_show(
    series: Series,
    chapter: Chapter,
    as_json: Annotated[bool, typer.Option("--json", help="Print the regions as JSON.")] = False,
) -> None:
    """List the chapter's regions: id, kind, source text, English line, what was edited by hand and each line's
    status (todo, edited, checked)."""
    _cfg, _series, paths = _chapter(series, chapter)
    regions = store.current_regions(paths)
    english = english_lines(paths)
    edited, translated = (set(ids) for ids in store.edited_ids(paths))
    deleted = store.deleted_regions(paths)
    status = store.line_statuses(paths, edited | translated)
    if as_json:
        rows = [
            {
                "id": r.id,
                "kind": r.kind,
                "text": r.text,
                "english": english.get(r.id),
                "speaker": r.speaker,
                "edited": r.id in edited,
                "hand_translated": r.id in translated,
                "status": status[r.id],
            }
            for r in regions
        ]
        gone = [{"id": r.id, "text": r.text} for r in deleted]
        typer.echo(json.dumps({"regions": rows, "deleted": gone}, ensure_ascii=False, indent=2))
        return
    table = Table(title=f"edit: {series} / {chapter} ({len(regions)} region(s))")
    for column in ("Id", "Kind", "Speaker", "Source", "English", "Edited", "Status"):
        table.add_column(column)
    for r in regions:
        marks = [label for label, ids in (("region", edited), ("English", translated)) if r.id in ids]
        table.add_row(
            r.id, r.kind, r.speaker or "", r.text, english.get(r.id, ""), ", ".join(marks), status[r.id]
        )
    Console().print(table)
    if deleted:
        typer.echo("deleted by hand: " + ", ".join(f"{r.id} ({r.text})" for r in deleted))


@edit_app.command("check")
def edit_check(
    series: Series,
    chapter: Chapter,
    regions: Annotated[list[str] | None, typer.Argument(help="Region ids as `edit show` lists them.")] = None,
    every: Annotated[bool, typer.Option("--all", help="Every region of the chapter.")] = False,
    uncheck: Annotated[bool, typer.Option("--uncheck", help="Unmark the lines instead.")] = False,
) -> None:
    """Mark lines checked: their source text and English as they are now are approved (a later change to either
    unchecks them); with --uncheck unmark them. One undo step."""
    _cfg, _series, paths = _chapter(series, chapter)
    if every and regions:
        raise _fail("name regions or pass --all, not both")
    ids = [region.id for region in store.current_regions(paths)] if every else list(regions or [])
    if not ids:
        raise _fail("name the regions to check, or pass --all")
    changed = _run(lambda: store.set_checked(paths, ids, checked=not uncheck))
    status = store.line_statuses(paths)
    done = sum(1 for value in status.values() if value == "checked")
    typer.echo(
        f"edit: {len(changed)} line(s) {'unchecked' if uncheck else 'checked'}; {done} of {len(status)} checked"
    )


@edit_app.command("text")
def edit_text(
    series: Series, chapter: Chapter, region: RegionId, text: Annotated[str, typer.Argument()]
) -> None:
    """Correct a region's source text (the OCR's reading stays in ocr_auto.json)."""
    cfg, _series, paths = _chapter(series, chapter)
    _run(lambda: store.update_region(paths, region, direction=cfg.detect.reading_direction, text=text))
    typer.echo(f"edit: {region} source text saved")


@edit_app.command("kind")
def edit_kind(
    series: Series, chapter: Chapter, region: RegionId, kind: Annotated[Kind, typer.Argument()]
) -> None:
    """Change a region's kind (a watermark is erased and never translated)."""
    cfg, _series, paths = _chapter(series, chapter)
    new_kind = cast(RegionKind, kind.value)  # the enum's values are the RegionKind literals
    _run(lambda: store.update_region(paths, region, direction=cfg.detect.reading_direction, kind=new_kind))
    typer.echo(f"edit: {region} is now {kind.value}")


@edit_app.command("speaker")
def edit_speaker(
    series: Series,
    chapter: Chapter,
    region: RegionId,
    name: Annotated[
        str, typer.Argument(help='A character of the series\' voices.toml, any name, or "" for nobody.')
    ],
) -> None:
    """Say who speaks a region's line; the translation keeps that character's voice (voices.toml)."""
    cfg, series_paths, paths = _chapter(series, chapter)
    _run(lambda: store.update_region(paths, region, direction=cfg.detect.reading_direction, speaker=name))
    if not name.strip():
        typer.echo(f"edit: {region} has no speaker")
        return
    try:
        known = character_named(name, load_voices(series_paths)) is not None
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    note = "" if known else f" (not in {VOICES_FILE}: no voice to keep)"
    typer.echo(f"edit: {region} is said by {name.strip()}{note}")


def _box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    """A strip box from its corners; exit 2 when they are not ordered."""
    try:
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValueError as exc:
        raise _fail(
            f"box {x0},{y0},{x1},{y1}: the second corner must be right of and below the first"
        ) from exc


Coord = Annotated[int, typer.Argument()]


@edit_app.command("box")
def edit_box(
    series: Series, chapter: Chapter, region: RegionId, x0: Coord, y0: Coord, x1: Coord, y1: Coord
) -> None:
    """Move or resize a region's text box (strip pixels); its English line and lettering follow it."""
    cfg, _series, paths = _chapter(series, chapter)
    box = _box(x0, y0, x1, y1)
    moved = _run(lambda: store.update_region(paths, region, direction=cfg.detect.reading_direction, bbox=box))
    b = moved.bbox
    typer.echo(f"edit: {region} box {b.x0},{b.y0},{b.x1},{b.y1}")


@edit_app.command("add")
def edit_add(
    series: Series,
    chapter: Chapter,
    x0: Coord,
    y0: Coord,
    x1: Coord,
    y1: Coord,
    kind: Annotated[Kind, typer.Option("--kind")] = Kind.BUBBLE_TEXT,
    text: Annotated[str, typer.Option("--text", help="Its source text.")] = "",
) -> None:
    """Add a region the detector missed (strip pixels); prints its new id."""
    cfg, _series, paths = _chapter(series, chapter)
    box = _box(x0, y0, x1, y1)
    new_kind = cast(RegionKind, kind.value)  # the enum's values are the RegionKind literals
    added = _run(
        lambda: store.add_region(
            paths, box, direction=cfg.detect.reading_direction, kind=new_kind, text=text, lang=cfg.ocr.lang
        )
    )
    typer.echo(f"edit: added {added.id}")


@edit_app.command("delete")
def edit_delete(series: Series, chapter: Chapter, region: RegionId) -> None:
    """Remove a region (a false detection: its text stays on the page untranslated); `revert` restores it."""
    cfg, _series, paths = _chapter(series, chapter)
    _run(lambda: store.delete_region(paths, region, direction=cfg.detect.reading_direction))
    typer.echo(f"edit: {region} deleted")


@edit_app.command("revert")
def edit_revert(
    series: Series,
    chapter: Chapter,
    region: RegionId,
    english: Annotated[bool, typer.Option("--english", help="Revert the English line instead.")] = False,
) -> None:
    """Drop a region's hand edits (box, text, kind; a deleted one comes back), or with --english its line."""
    cfg, _series, paths = _chapter(series, chapter)
    direction = cfg.detect.reading_direction
    if english:
        line = _run(lambda: store.revert_translation(paths, region, direction=direction))
        typer.echo(f"edit: {region} English is the judge's again: {line.text if line else '(none)'}")
        return
    restored = _run(lambda: store.revert_region(paths, region, direction=direction))
    typer.echo(f"edit: {region} reverted" if restored else f"edit: {region} (drawn by hand) removed")


@edit_app.command("english")
def edit_english(
    series: Series, chapter: Chapter, region: RegionId, text: Annotated[str, typer.Argument()]
) -> None:
    """Write a region's English line; every later judge run keeps it."""
    cfg, _series, paths = _chapter(series, chapter)
    _run(lambda: store.set_translation(paths, region, text, direction=cfg.detect.reading_direction))
    typer.echo(f"edit: {region} English saved")


@edit_app.command("translate")
def edit_translate(
    series: Series,
    chapter: Chapter,
    regions: Annotated[list[str], typer.Argument(help="One or more region ids.")],
    profile: Annotated[
        str | None, typer.Option("--profile", help="Translation profile (default: every enabled one).")
    ] = None,
    apply: Annotated[bool, typer.Option("--apply", help="Keep each region's first suggestion.")] = False,
) -> None:
    """Translate a few regions now (neighbouring lines, glossary, story and learned memory applied) and print
    every profile's suggestion; nothing is written unless --apply."""
    cfg, series_paths, paths = _chapter(series, chapter)
    try:
        known = load_profiles(default_profile_paths())
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    client = make_chat_client(cfg)
    try:
        result = _run(
            lambda: translate_now(
                client, cfg, series_paths, paths, regions, known, profile=profile, apply=apply
            )
        )
    except OllamaError as exc:
        typer.echo(f"edit: Ollama failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    for item in result.suggestions:
        typer.echo(f"{item.region_id} [{item.profile}] {item.text}")
    for line in result.applied:
        typer.echo(f"edit: {line.region_id} English kept")


@edit_app.command("replace")
def edit_replace(
    series: Series,
    find: Annotated[str, typer.Argument(help="The text to find (a regular expression with --regex).")],
    replace: Annotated[str, typer.Argument(help="What to put instead (with --regex, \\1 … are its groups).")],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Only this chapter; repeatable. Default: every chapter."),
    ] = None,
    source: Annotated[
        bool, typer.Option("--source", help="Replace in the source (OCR) texts instead of the English lines.")
    ] = False,
    regex: Annotated[bool, typer.Option("--regex", help="FIND is a regular expression.")] = False,
    word: Annotated[bool, typer.Option("--word", help="Whole words only.")] = False,
    ignore_case: Annotated[
        bool, typer.Option("--ignore-case", "-i", help="Any case; the replacement takes each match's case.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only list what would change.")] = False,
) -> None:
    """Find and replace across the English lines (or with --source the source texts) of a series or some of
    its chapters; every change is a hand edit, and each chapter's changes are one undo step."""
    cfg = get_config()
    series_paths = SeriesPaths.from_config(cfg, series)
    try:
        direction = series_config(cfg, series_paths.library_dir).detect.reading_direction
    except SeriesConfigError as exc:
        raise _fail(str(exc)) from exc
    known = series_paths.chapters()
    unknown = [name for name in chapter or [] if name not in known]
    if unknown:
        raise _fail(f"no chapter {unknown[0]!r} in {series}")
    rule = FindReplace(
        replace=replace, find=find, regex=regex, whole_word=word, case_sensitive=not ignore_case
    )
    target = "source" if source else "english"
    total, touched = 0, 0
    for name in chapter or known:
        paths = series_paths.chapter(name)
        changes = _run(lambda paths=paths: plan(paths, rule, target))
        for change in changes:
            typer.echo(f"{change.chapter} {change.region_id}: {change.before} → {change.after}")
        if changes and not dry_run:
            _run(
                lambda paths=paths, changes=changes: apply_changes(
                    paths, changes, target, direction=direction
                )
            )
        total, touched = total + len(changes), touched + bool(changes)
    verb = "would replace" if dry_run else "replaced"
    typer.echo(
        f"replace: {verb} {total} {'source text' if source else 'English line'}(s) in {touched} chapter(s)"
    )


def _history(paths: ChapterPaths) -> str:
    """How many steps are left to undo and to redo, for a message."""
    back, forward = store.history_steps(paths)
    return f"{back} more to undo, {forward} to redo"


@edit_app.command("undo")
def edit_undo(series: Series, chapter: Chapter) -> None:
    """Take back the last hand edit of the chapter (an import or a desktop save counts as one)."""
    cfg, _series, paths = _chapter(series, chapter)
    _run(lambda: store.undo(paths, direction=cfg.detect.reading_direction))
    typer.echo(f"edit: undone ({_history(paths)})")


@edit_app.command("redo")
def edit_redo(series: Series, chapter: Chapter) -> None:
    """Make the last undone edit again (a new edit since the undo forgets what could be redone)."""
    cfg, _series, paths = _chapter(series, chapter)
    _run(lambda: store.redo(paths, direction=cfg.detect.reading_direction))
    typer.echo(f"edit: redone ({_history(paths)})")


@edit_app.command("cuts")
def edit_cuts(
    series: Series,
    chapter: Chapter,
    rows: Annotated[
        list[int] | None, typer.Argument(help="Strip rows where the output images split.")
    ] = None,
    reset: Annotated[bool, typer.Option("--reset", help="Back to one image per slice.")] = False,
) -> None:
    """Show or set where the exported images split (strip rows); --reset goes back to one image per slice."""
    _cfg, _series, paths = _chapter(series, chapter)
    if reset or rows:
        stored = _run(lambda: store.set_cuts(paths, None if reset else list(rows or [])))
    else:
        stored = store.load_edits(paths).cuts
    typer.echo("cuts: " + (", ".join(str(c) for c in stored) if stored else "one image per slice"))
