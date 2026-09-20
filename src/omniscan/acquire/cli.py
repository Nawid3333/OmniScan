"""`omniscan acquire` — plan chapter costs, download chapters completely and check their completeness."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Annotated, Literal, NoReturn

import httpx
import typer

from omniscan.acquire.completeness import (
    Finding,
    PageInfo,
    check_numbering,
    check_pages,
    inspect_chapter,
    verdict,
)
from omniscan.acquire.drm import NOTICE, DrmPlatformError, check_allowed
from omniscan.acquire.extractor import Extractor
from omniscan.acquire.extractpics import ExtractPicsClient
from omniscan.acquire.plan import AcquirePlan, build_plan, to_json
from omniscan.acquire.run import ChapterOutcome, acquire_chapters
from omniscan.acquire.selection import read_url_list, select_chapters
from omniscan.acquire.sources import (
    ChapterSource,
    load_sources,
    named_chapters,
    sources_path,
    template_chapters,
)
from omniscan.core.config import get_config, get_secrets
from omniscan.core.paths import list_chapters

acquire_app = typer.Typer(no_args_is_help=True, help="Acquire chapters: plan, run and check downloads.")

_SELECT_HELP = "Subset spec: 'N', 'A-B', 'A-', '-B', comma-separated. Default: all."
_MODE_HELP = "Extraction mode: basic costs 1 credit per chapter, advanced 2."
_URLS_HELP = "Text file with one 'URL' or 'NAME | URL' line per chapter ('-' = stdin)."


def make_extractor(api_key: str) -> Extractor:
    """The extraction back-end of a run (tests monkeypatch this to inject a fake)."""
    return ExtractPicsClient(api_key)


def make_image_client() -> httpx.Client:
    """The HTTP client for the image downloads (tests monkeypatch this to use a mock transport)."""
    return httpx.Client(follow_redirects=True, timeout=30.0)


@acquire_app.command("plan")
def acquire_plan(
    series: Annotated[str, typer.Argument()],
    select: Annotated[str | None, typer.Option("--select", help=_SELECT_HELP)] = None,
    mode: Annotated[Literal["basic", "advanced"], typer.Option("--mode", help=_MODE_HELP)] = "basic",
    force: Annotated[bool, typer.Option("--force", help="Treat already-acquired chapters as to do.")] = False,
    urls: Annotated[Path | None, typer.Option("--urls", help=_URLS_HELP)] = None,
    link: Annotated[
        list[str] | None, typer.Option("--link", help="Chapter page URL; repeatable, numbered by position.")
    ] = None,
    first_number: Annotated[
        int, typer.Option("--first-number", help="Number of the first --link chapter.")
    ] = 1,
    template: Annotated[
        str | None, typer.Option("--template", help="Chapter page URL template with '{n}'.")
    ] = None,
    first: Annotated[
        int | None, typer.Option("--first", help="First template chapter number (with --template).")
    ] = None,
    last: Annotated[
        int | None, typer.Option("--last", help="Last template chapter number (with --template).")
    ] = None,
    name: Annotated[
        str, typer.Option("--name", help="Chapter name template with '{n}' (with --template).")
    ] = "Chapter {n}",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the machine-readable plan instead of the table.")
    ] = False,
) -> None:
    """Print the acquire plan of a series: which chapters are done and what the extraction costs."""
    sources = _resolve_sources(
        series,
        urls=urls,
        link=link,
        first_number=first_number,
        template=template,
        first=first,
        last=last,
        name=name,
    )
    sources = _selected(sources, select)
    _refuse_drm(sources)
    plan = build_plan(sources, _series_dir(series), mode=mode, force=force)
    if as_json:
        typer.echo(json.dumps({"series": series, **to_json(plan)}, indent=2, ensure_ascii=False))
        return
    for line in _plan_lines(plan):
        typer.echo(line)


@acquire_app.command("run")
def acquire_run(
    series: Annotated[str, typer.Argument()],
    select: Annotated[str | None, typer.Option("--select", help=_SELECT_HELP)] = None,
    mode: Annotated[Literal["basic", "advanced"], typer.Option("--mode", help=_MODE_HELP)] = "basic",
    force: Annotated[bool, typer.Option("--force", help="Re-acquire chapters already marked done.")] = False,
    urls: Annotated[Path | None, typer.Option("--urls", help=_URLS_HELP)] = None,
    link: Annotated[
        list[str] | None, typer.Option("--link", help="Chapter page URL; repeatable, numbered by position.")
    ] = None,
    first_number: Annotated[
        int, typer.Option("--first-number", help="Number of the first --link chapter.")
    ] = 1,
    template: Annotated[
        str | None, typer.Option("--template", help="Chapter page URL template with '{n}'.")
    ] = None,
    first: Annotated[
        int | None, typer.Option("--first", help="First template chapter number (with --template).")
    ] = None,
    last: Annotated[
        int | None, typer.Option("--last", help="Last template chapter number (with --template).")
    ] = None,
    name: Annotated[
        str, typer.Option("--name", help="Chapter name template with '{n}' (with --template).")
    ] = "Chapter {n}",
    max_credits: Annotated[
        int | None, typer.Option("--max-credits", help="Refuse to spend more than this many credits.")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the spend confirmation.")] = False,
    no_page_run: Annotated[
        bool, typer.Option("--no-page-run", help="Keep every extracted image, not just the page run.")
    ] = False,
    no_filter: Annotated[
        bool, typer.Option("--no-filter", help="Keep images the size/content filter would drop.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print only the final JSON on stdout; progress goes to stderr.")
    ] = False,
) -> None:
    """Download every selected chapter completely and record its acceptance."""
    sources = _resolve_sources(
        series,
        urls=urls,
        link=link,
        first_number=first_number,
        template=template,
        first=first,
        last=last,
        name=name,
    )
    sources = _selected(sources, select)
    _refuse_drm(sources)
    series_dir = _series_dir(series)
    plan = build_plan(sources, series_dir, mode=mode, force=force)
    progress_err = as_json  # --json keeps stdout to the final JSON alone; progress goes to stderr
    typer.echo(NOTICE, err=progress_err)
    for line in _plan_lines(plan):
        typer.echo(line, err=progress_err)
    if plan.todo == 0:
        typer.echo(
            f"acquire: nothing to do — all {plan.done} selected chapter(s) are already acquired",
            err=progress_err,
        )
        if as_json:
            _echo_run_json(series, plan.mode, [], 0)
        return
    if not yes:
        try:
            confirmed = typer.confirm(f"Spend up to {plan.credits_total} credit(s)?", err=progress_err)
        except typer.Abort:
            confirmed = False
        if not confirmed:
            typer.echo("acquire: aborted", err=True)
            raise typer.Exit(2) from None
    api_key = get_secrets().extractpics_api_key
    if api_key is None or not api_key.get_secret_value():
        typer.echo("EXTRACTPICS_API_KEY is not set; put it in ~/.config/omniscan/secrets.env", err=True)
        raise typer.Exit(2) from None
    extractor = make_extractor(api_key.get_secret_value())
    client = make_image_client()

    def on_event(outcome: ChapterOutcome) -> None:
        """Show the chapter outcome as soon as it is known."""
        typer.echo(_event_line(outcome, err=progress_err), err=progress_err)

    try:
        outcomes = acquire_chapters(
            plan,
            series_dir,
            extractor,
            client=client,
            max_credits=max_credits,
            page_run=not no_page_run,
            apply_filters=not no_filter,
            on_event=on_event,
        )
    finally:
        client.close()
        closer = getattr(extractor, "close", None)
        if callable(closer):
            closer()
    counts = {
        status: sum(1 for outcome in outcomes if outcome.status == status)
        for status in ("ok", "review", "failed", "skipped")
    }
    credits_used = sum(outcome.credits for outcome in outcomes)
    typer.echo(
        f"ok {counts['ok']}, review {counts['review']}, failed {counts['failed']}, "
        f"skipped {counts['skipped']} — credits used {credits_used}",
        err=progress_err,
    )
    if as_json:
        _echo_run_json(series, plan.mode, outcomes, credits_used)
    if counts["failed"]:
        raise typer.Exit(1) from None


@acquire_app.command("check")
def acquire_check(
    series: Annotated[str, typer.Argument()],
    as_json: Annotated[bool, typer.Option("--json", help="Emit the machine-readable report.")] = False,
) -> None:
    """Check the completeness of every chapter folder a series already has in the library."""
    series_dir = _series_dir(series)
    chapters = list_chapters(series_dir)
    if not chapters:
        typer.echo(f"acquire check: no chapters for series {series!r} at {series_dir}", err=True)
        raise typer.Exit(1) from None
    inspected = {chapter.name: inspect_chapter(chapter) for chapter in chapters}
    counts = [len(inspected[chapter.name][0]) for chapter in chapters]
    reports = [
        _check_report(index, chapter.name, inspected[chapter.name], counts)
        for index, chapter in enumerate(chapters)
    ]
    series_findings = check_numbering([chapter.name for chapter in chapters])
    if as_json:
        payload = {
            "series": series,
            "chapters": [
                {
                    "name": name,
                    "pages": pages,
                    "verdict": result,
                    "findings": [_finding_json(f) for f in findings],
                }
                for name, pages, result, findings in reports
            ],
            "series_findings": [_finding_json(finding) for finding in series_findings],
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for name, pages, result, findings in reports:
            typer.echo(f"{name}: {pages} pages — {result}")
            for finding in findings:
                typer.echo(f"    {finding.level} {finding.code}: {finding.message}")
        for finding in series_findings:
            typer.echo(f"series: {finding.level} {finding.code}: {finding.message}")
    if any(result == "failed" for _, _, result, _ in reports):
        raise typer.Exit(1) from None


type _Report = tuple[str, int, Literal["ok", "review", "failed"], list[Finding]]


def _check_report(
    index: int, name: str, inspected: tuple[list[PageInfo], list[Finding]], counts: list[int]
) -> _Report:
    """(name, decodable pages, verdict, findings) of one chapter folder against the other chapters."""
    pages, findings = inspected
    others = [count for position, count in enumerate(counts) if position != index]
    findings = [
        *findings,
        *check_pages(
            pages,
            median_pages=median(others) if len(others) >= 3 else None,
            previous_pages=counts[index - 1] if index > 0 else None,
        ),
    ]
    return name, len(pages), verdict(findings), findings


def _finding_json(finding: Finding) -> dict[str, str]:
    """The JSON object of one completeness finding."""
    return {"level": finding.level, "code": finding.code, "message": finding.message}


def _plan_lines(plan: AcquirePlan) -> list[str]:
    """The aligned table rows (`#`, name, state, credits, url) plus the credit-estimate summary."""
    name_width = max((len(item.source.name) for item in plan.items), default=4)
    lines = [f"{'#':>4}  {'name':<{name_width}}  {'state':<6}  {'credits':>7}  url"]
    for number, item in enumerate(plan.items, start=1):
        lines.append(
            f"{number:>4}  {item.source.name:<{name_width}}  {item.state:<6}  {item.credits:>7}  {item.source.url}"
        )
    lines.append(
        f"{len(plan.items)} chapter(s): {plan.todo} to do, {plan.done} done"
        f" — estimated credits: {plan.credits_total} (mode {plan.mode})"
    )
    return lines


def _echo_run_json(series: str, mode: str, outcomes: Sequence[ChapterOutcome], credits_used: int) -> None:
    """Print the machine-readable run result on stdout."""
    payload = {
        "series": series,
        "mode": mode,
        "outcomes": [
            {
                "name": outcome.name,
                "status": outcome.status,
                "pages": outcome.pages,
                "credits": outcome.credits,
                "error": outcome.error,
                "findings": [_finding_json(finding) for finding in outcome.findings],
            }
            for outcome in outcomes
        ],
        "credits_used": credits_used,
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


_EVENT_SYMBOLS = {"ok": "✓", "review": "?", "failed": "✗", "skipped": "-"}
_EVENT_ASCII = {"ok": "ok", "review": "review", "failed": "FAILED", "skipped": "skipped"}


def _event_line(outcome: ChapterOutcome, *, err: bool) -> str:
    """One progress line per chapter; ASCII markers when the target stream cannot encode the symbols."""
    line = f"{_EVENT_SYMBOLS[outcome.status]} {outcome.name}: {_event_detail(outcome)}"
    encoding = getattr(sys.stderr if err else sys.stdout, "encoding", None) or "utf-8"
    try:
        line.encode(encoding)
    except UnicodeEncodeError:
        line = f"{_EVENT_ASCII[outcome.status]} {outcome.name}: {_event_detail(outcome)}"
    return line


def _event_detail(outcome: ChapterOutcome) -> str:
    """The status text after 'name: ' of a progress line."""
    if outcome.status == "ok":
        return f"{outcome.pages} pages"
    if outcome.status == "review":
        codes = ", ".join(finding.code for finding in outcome.findings)
        return f"{outcome.pages} pages (review: {codes})"
    if outcome.status == "failed":
        return outcome.error or "failed"
    return f"skipped ({outcome.error})"


def _resolve_sources(
    series: str,
    *,
    urls: Path | None,
    link: Sequence[str] | None,
    first_number: int,
    template: str | None,
    first: int | None,
    last: int | None,
    name: str,
) -> list[ChapterSource]:
    """Chapter sources from the source options; every problem prints on stderr and exits 2."""
    if sum((urls is not None, bool(link), template is not None)) > 1:
        _fail("give at most one of --urls, --link, --template")
    try:
        if urls is not None:
            return read_url_list(_url_list_text(urls))
        if link:
            return named_chapters(
                [(f"Chapter {first_number + index}", url) for index, url in enumerate(link)]
            )
        if template is not None:
            if first is None or last is None:
                _fail("--template needs --first and --last")
            return template_chapters(template, first, last, name)
        path = sources_path(get_config().paths.library_root, series)
        if not path.is_file():
            _fail(f"no chapter sources: give --urls, --link, --template or create {path}")
        return load_sources(path)
    except (ValueError, OSError) as exc:
        _fail(str(exc))


def _url_list_text(urls: Path) -> str:
    """The text of a --urls file, or stdin when the file is '-'."""
    return sys.stdin.read() if urls == Path("-") else urls.read_text(encoding="utf-8")


def _selected(sources: Sequence[ChapterSource], spec: str | None) -> list[ChapterSource]:
    """The sources matching the --select spec; an unusable spec exits 2."""
    try:
        return select_chapters(sources, spec)
    except ValueError as exc:
        _fail(str(exc))


def _refuse_drm(sources: Sequence[ChapterSource]) -> None:
    """Exit 2 naming the platform when any selected URL belongs to a DRM platform."""
    for source in sources:
        try:
            check_allowed(source.url)
        except DrmPlatformError as exc:
            _fail(f"{source.name}: {exc}")


def _series_dir(series: str) -> Path:
    """The series folder of the library root the acquire commands work on."""
    return get_config().paths.library_root / series


def _fail(message: str) -> NoReturn:
    """Print a refusal on stderr and exit 2."""
    typer.echo(f"acquire: {message}", err=True)
    raise typer.Exit(2) from None
