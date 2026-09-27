"""Interchange sub-apps: LabelPlus files, layered PSD pages, BallonsTranslator projects and manga-image-translator
text files — a chapter out to another tool and that tool's work back in as edits."""

from __future__ import annotations

import enum
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from omniscan.cleanup.store import CLEANUP_FILE, current_crop
from omniscan.core.config import SeriesConfigError, get_config, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths, jpeg_paths
from omniscan.core.schemas import IngestArtifact, LayoutArtifact, ProjectFile
from omniscan.edits import store
from omniscan.interchange import ballons, mit, project
from omniscan.interchange.blocks import Block, join_lines, match_blocks
from omniscan.interchange.labelplus import export_labels, match_labels, parse, strip_pages, write
from omniscan.interchange.psd import page_psd
from omniscan.packaging.names import safe_filename
from omniscan.translate.on_demand import english_lines
from omniscan.typeset.page_preview import page_box

labelplus_app = typer.Typer(
    no_args_is_help=True, help="LabelPlus files: export a chapter, import a translation."
)
psd_app = typer.Typer(
    no_args_is_help=True, help="Layered Photoshop files (raw, clean, text) of a chapter's pages."
)


ballons_app = typer.Typer(
    no_args_is_help=True,
    help="BallonsTranslator projects: import one into a chapter, export a chapter as one.",
)
mit_app = typer.Typer(
    no_args_is_help=True, help="manga-image-translator text files (--save-text): import into a chapter."
)


@psd_app.callback()
def psd_group() -> None:
    """Layered Photoshop files (raw, clean, text) of a chapter's pages."""


@mit_app.callback()
def mit_group() -> None:
    """manga-image-translator text files (--save-text): import into a chapter."""


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
        try:  # one write, one rebuild, one undo step: the file lands whole or not at all
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


Source = Annotated[
    bool, typer.Option("--source", help="Also take the project's source text as each region's OCR text.")
]
Add = Annotated[
    bool, typer.Option("--add", help="Add a region for every block over no region (text and translation).")
]
DryRun = Annotated[bool, typer.Option("--dry-run", help="Only report what would change.")]


def _import_blocks(
    tool: str, series: str, chapter: str, blocks: Sequence[Block], *, source: bool, add: bool, dry_run: bool
) -> None:
    """Take a foreign project's blocks into the chapter as hand edits in edits.json: the translation of the
    blocks in each region as its English line (and with `source` their text as its source text); with `add`, a
    new region for each block over none. Prints what changed and what matched nothing."""
    cfg = get_config()
    series_paths = SeriesPaths.from_config(cfg, series)
    paths = series_paths.chapter(chapter)
    try:
        scfg = series_config(cfg, series_paths.library_dir)
    except SeriesConfigError as exc:
        raise _fail(str(exc), tool) from exc
    direction = scfg.detect.reading_direction
    regions = store.current_regions(paths)
    found = match_blocks(blocks, _ingest(paths.artifact("ingest.json"), tool), regions)
    english, texts = english_lines(paths), {r.id: r.text for r in regions}
    lines = {
        rid: " ".join(join_lines([b.translation for b in bs]).split()) for rid, bs in found.matched.items()
    }
    sources = {rid: join_lines([b.text for b in bs]) for rid, bs in found.matched.items()} if source else {}
    new_lines = {rid: line for rid, line in lines.items() if line and english.get(rid) != line}
    new_sources = {rid: text for rid, text in sources.items() if text and texts.get(rid) != text}
    notes: list[str] = []  # per unmatched block: what became of it
    try:
        with store.edit_group(paths):  # the whole import is one undo step
            if not dry_run:
                for region_id, text in new_sources.items():
                    store.update_region(paths, region_id, direction=direction, text=text)
                for region_id, line in new_lines.items():
                    store.set_translation(paths, region_id, line, direction=direction)
            for block, box in found.unmatched:
                if not add:
                    notes.append("")
                elif dry_run:
                    notes.append(" — would add a region")
                else:
                    region = store.add_region(
                        paths, box, direction=direction, text=block.text, lang=scfg.ocr.lang
                    )
                    if block.translation.strip():
                        line = " ".join(block.translation.split())
                        store.set_translation(paths, region.id, line, direction=direction)
                    notes.append(f" — added as {region.id}")
    except (store.EditNotFoundError, ValueError) as exc:
        raise _fail(str(exc), tool) from exc
    joined = sum(len(bs) - 1 for bs in found.matched.values())
    same = sum(1 for rid, line in lines.items() if line and english.get(rid) == line)
    verb = "would import" if dry_run else "imported"
    summary = f"{tool}: {verb} {len(new_lines)} English line(s)"
    if source:
        summary += f", {len(new_sources)} source text(s)"
    summary += f"; {same} line(s) already the same"
    if joined:
        summary += f"; {joined} block(s) joined into a region another block is in"
    typer.echo(summary)
    for (block, box), note in zip(found.unmatched, notes, strict=True):
        typer.echo(
            f"{tool}: {block.page} block at {box.x0},{box.y0}-{box.x1},{box.y1} (strip) is over no region{note}"
        )
    for page in found.unknown_pages:
        typer.echo(f"{tool}: page {page!r} is not in this chapter")


@ballons_app.command("import")
def ballons_import(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    project: Annotated[
        Path,
        typer.Argument(exists=True, help="The project's imgtrans_*.json, or the page folder that holds it."),
    ],
    source: Source = False,
    add: Add = False,
    dry_run: DryRun = False,
) -> None:
    """Take a BallonsTranslator project's translations as the English lines of the regions its text blocks lie
    on (hand-written lines in edits.json: every re-run keeps them, learning remembers them)."""
    file = ballons.project_file(project) if project.is_dir() else project
    try:
        blocks = ballons.read_project(json.loads(file.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise _fail(f"{file}: {exc}", "ballons") from exc
    _import_blocks("ballons", series, chapter, blocks, source=source, add=add, dry_run=dry_run)


def _cleaned(paths: ChapterPaths) -> bool:
    """Whether the chapter was cleaned, automatically or by hand."""
    return any(paths.artifact(name).is_file() for name in ("inpaint.json", "inpaint_lama.json", CLEANUP_FILE))


@ballons_app.command("export")
def ballons_export(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="The project folder. Default: <output_root>/<series>/_ballons/<chapter>/"),
    ] = None,
) -> None:
    """Write the chapter as a BallonsTranslator project: copies of its pages, the cleaned pages (inpainted/)
    and one text block per region with its source text, English line and lettering size and colours."""
    series_paths = SeriesPaths.from_config(get_config(), series)
    paths = series_paths.chapter(chapter)
    ingest = _ingest(paths.artifact("ingest.json"), "ballons")
    layout_path = paths.artifact("layout.json")
    items = LayoutArtifact.load(layout_path).items if layout_path.is_file() else []
    folder = out or series_paths.output_dir / "_ballons" / chapter
    folder.mkdir(parents=True, exist_ok=True)
    jpegs = dict(
        zip(
            (f.index for f in ingest.files),
            jpeg_paths(ingest, paths.raw_dir, paths.work_dir / "converted"),
            strict=True,
        )
    )
    pages = strip_pages(ingest)
    cleaned = _cleaned(paths)
    for page in pages:
        name = ballons.page_name(page)
        shutil.copyfile(paths.raw_dir / page.name if name == page.name else jpegs[page.index], folder / name)
        if cleaned:
            pixels = Image.fromarray(current_crop(paths, ingest, page_box(ingest, page.index)))
            if pixels.size != (page.width, page.height):
                pixels = pixels.resize((page.width, page.height), Image.Resampling.LANCZOS)
            (folder / ballons.INPAINTED_DIR).mkdir(exist_ok=True)
            pixels.save(folder / ballons.INPAINTED_DIR / f"{Path(name).stem}.png")
    blocks = ballons.page_blocks(ingest, store.current_regions(paths), english_lines(paths), items)
    project = ballons.write_project(folder.resolve(), ingest, blocks)
    ballons.project_file(folder).write_text(
        json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(
        f"ballons: {sum(len(b) for b in blocks.values())} text block(s) on {len(pages)} page(s)"
        + ("" if cleaned else " (not cleaned yet: no inpainted pages)")
        + f" -> {folder}"
    )


@mit_app.command("import")
def mit_import(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    files: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="One or more *_translations.txt files."),
    ],
    source: Source = False,
    add: Add = False,
    dry_run: DryRun = False,
) -> None:
    """Take manga-image-translator's translations (its --save-text files) as the English lines of the regions
    its text regions lie on (hand-written lines in edits.json: every re-run keeps them)."""
    blocks: list[Block] = []
    for file in files:
        try:
            blocks += mit.parse(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _fail(f"{file}: {exc}", "mit") from exc
    _import_blocks("mit", series, chapter, blocks, source=source, add=add, dry_run=dry_run)


project_app = typer.Typer(
    no_args_is_help=True,
    help="Chapter projects: one chapter's raw pages and all its work in one file, to pass to another OmniScan user.",
)


@project_app.callback()
def project_group() -> None:
    """Chapter projects: one chapter's raw pages and all its work in one file, to pass to another OmniScan user."""


def _size(size: int) -> str:
    """A file size for people: kB below a megabyte."""
    return f"{size / 1000:.0f} kB" if size < 1_000_000 else f"{size / 1_000_000:.1f} MB"


def _parts(files: Sequence[ProjectFile]) -> str:
    """How many files a project holds per part (raw pages, work files, series files, finished pages)."""
    counts: dict[str, int] = {}
    for file in files:
        part = file.path.partition("/")[0]
        counts[part] = counts.get(part, 0) + 1
    labels = {
        "raw": "raw page file(s)",
        "work": "work file(s)",
        "series": "series file(s)",
        "output": "finished page(s)",
    }
    return ", ".join(f"{counts[part]} {label}" for part, label in labels.items() if part in counts)


@project_app.command("pack")
def project_pack(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="The file to write. Default: '<series> - <chapter>.omniscan' here."
        ),
    ] = None,
    with_output: Annotated[
        bool, typer.Option("--with-output", help="Also pack the finished pages (for the quality check).")
    ] = False,
) -> None:
    """Pack a chapter — raw pages, every stage's work, hand edits with their undo history, hand cleanup, and the
    series' series.toml / voices.toml — into one file another OmniScan user unpacks and continues."""
    series_paths = SeriesPaths.from_config(get_config(), series)
    if chapter not in series_paths.chapters():
        raise _fail(f"no chapter {chapter!r} in {series}", "project")
    dest = (
        output
        if output is not None
        else Path.cwd() / f"{safe_filename(series)} - {safe_filename(chapter)}{project.SUFFIX}"
    )
    packed = project.pack(series_paths, chapter, dest, with_output=with_output)
    typer.echo(f"project: {_parts(packed.files)} -> {dest} ({_size(dest.stat().st_size)})")


@project_app.command("unpack")
def project_unpack(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A .omniscan chapter project.")],
    series: Annotated[str | None, typer.Option("--series", help="Unpack into this series instead.")] = None,
    chapter: Annotated[str | None, typer.Option("--chapter", help="Unpack as this chapter instead.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace the chapter if it exists (its raw pages and work).")
    ] = False,
) -> None:
    """Unpack a chapter project into your library and work folders; every stage done and every hand edit comes
    along. The series files it carries are added only where the series has none."""
    try:
        done = project.unpack(file, get_config(), series=series, chapter=chapter, force=force)
    except (project.ProjectError, FileExistsError) as exc:
        raise _fail(str(exc), "project") from exc
    notes = [f"added {', '.join(done.series_files)}"] if done.series_files else []
    notes += [f"kept your {', '.join(done.kept_series_files)}"] if done.kept_series_files else []
    verb = "replaced" if done.replaced else "unpacked"
    typer.echo(
        f"project: {verb} {done.paths.series}/{done.paths.chapter}"
        + (f" ({'; '.join(notes)})" if notes else "")
    )


@project_app.command("show")
def project_show(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A .omniscan chapter project.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print its project.json.")] = False,
) -> None:
    """Say what a chapter project holds, without unpacking it (checked like `unpack` checks it)."""
    try:
        shown = project.show(file)
    except project.ProjectError as exc:
        raise _fail(str(exc), "project") from exc
    if as_json:
        typer.echo(shown.model_dump_json(indent=2))
        return
    total = sum(entry.bytes for entry in shown.files)
    typer.echo(
        f"project: {shown.series}/{shown.chapter} (OmniScan {shown.app_version}): {_parts(shown.files)},"
        f" {_size(total)} unpacked"
    )
