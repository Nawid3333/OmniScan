"""OmniScan command line."""

import enum
import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from omniscan.acquire.cli import acquire_app
from omniscan.core.config import Config, get_config, get_secrets
from omniscan.core.paths import ChapterPaths, SeriesPaths, chapter_number, list_chapters, list_images
from omniscan.core.schemas import GlossaryEntry, IngestArtifact, SlicesArtifact
from omniscan.doctor import run_all_checks
from omniscan.filter.apply import record_override
from omniscan.filter.decide import EXAMPLE_SUFFIXES
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml, import_yaml
from omniscan.importer.execute import execute_import
from omniscan.importer.plan import ImportPlanError, plan_import
from omniscan.llm.ollama import OllamaClient, OllamaError, OllamaRateLimitError
from omniscan.log import setup_logging
from omniscan.models.rows import build_rows, row_to_json
from omniscan.models.rows import ollama_model_names as _ollama_model_names
from omniscan.packaging import pack_cbz, pack_pdf, safe_filename
from omniscan.queue.executor import stage_executor
from omniscan.queue.notify import combine, log_notifier, webhook_notifier
from omniscan.queue.store import KNOWN_STAGES, STATUSES, JobStatus, QueueStore, queue_db_path
from omniscan.queue.worker import run_queue
from omniscan.update.download import check_for_update, download_update
from omniscan.update.github import DEFAULT_REPO, ReleaseInfo, UpdateError, platform_key, select_asset
from omniscan.update.version import current_version
from omniscan.watermark.store import WatermarkStore

app = typer.Typer(help="OmniScan — manhwa/manga translator", no_args_is_help=True)

STATUS_STYLES = {"OK": "green", "WARN": "yellow", "FAIL": "red"}

_STUB_COMMANDS = ("reference",)


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


@app.command()
def hardware(
    as_json: Annotated[bool, typer.Option("--json", help="Emit the machine-readable form.")] = False,
) -> None:
    """Report this machine's OS, CPU, RAM, GPUs, torch build, best device and free disk."""
    from omniscan.hw.detect import detect_hardware

    hw = detect_hardware(get_config().paths.models_dir)
    if as_json:
        _echo_text(json.dumps(asdict(hw), indent=2, ensure_ascii=False))
        return
    _echo_text(f"OS: {hw.os} / {hw.arch}")
    _echo_text(f"CPU: {hw.cpu_name} ({hw.cpu_cores_physical} physical, {hw.cpu_cores_logical} logical)")
    _echo_text(f"RAM: {hw.ram_gb:.1f} GB")
    _echo_text(f"torch build: {hw.torch_build}")
    _echo_text(f"best device: {hw.best_device}")
    for gpu in hw.gpus:
        mark = "  (integrated)" if gpu.integrated else ""
        _echo_text(f"#{gpu.index} {gpu.name}  {gpu.vram_gb:.1f} GB  {gpu.backend}  device {gpu.device}{mark}")
    providers = ", ".join(hw.onnxruntime_providers) if hw.onnxruntime_providers else "none"
    _echo_text(f"onnxruntime providers: {providers}")
    _echo_text(f"free disk at the models folder: {hw.disk_free_gb:.1f} GB")


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


def _run_stages(
    name: str,
    stages: Sequence[Any],
    series: str,
    chapters: list[str] | None,
    force: bool,
    *,
    progress: bool = True,
) -> None:
    """Run a pipeline over a series' chapters, printing one line per stage outcome (exit 1 if any failed).

    With `progress=False` (machine-readable output) nothing is printed unless a stage fails, and
    failures go to stderr instead of stdout."""
    from omniscan.core.stage import run_series

    cfg = get_config()
    if chapters is None and not SeriesPaths.from_config(cfg, series).chapters():
        typer.echo(f"{name}: no chapters found for series {series!r}", err=True)
        raise typer.Exit(2)
    gpu = None
    if any(stage.gpu_group is not None for stage in stages):
        from omniscan.gpu.groups import build_vram_manager

        gpu = build_vram_manager(cfg)
    try:
        results = run_series(stages, cfg, series, chapters, force=force, gpu=gpu)
    finally:
        if gpu is not None:
            gpu.release()  # the models leave VRAM when the command ends
    failed = False
    for chapter, outcomes in results.items():
        for outcome in outcomes:
            if outcome.status != "failed" and not progress:
                continue
            typer.echo(
                f"{series}/{chapter} {outcome.stage}: {outcome.status} ({outcome.seconds:.2f}s)",
                err=not progress,
            )
            if outcome.status == "failed":
                failed = True
                typer.echo(f"    {outcome.error}", err=not progress)
    if failed:
        raise typer.Exit(1)


def cmd_ingest(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if up to date.")] = False,
) -> None:
    """Normalise raw chapter images to JPEG and record the strip layout (ingest.json)."""
    from omniscan.ingest.stage import IngestStage

    _run_stages("ingest", [IngestStage()], series, chapter, force)


def cmd_slice(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if up to date.")] = False,
) -> None:
    """Cut chapter strips into slices (slices.json). Runs ingest first if needed."""
    from omniscan.ingest.stage import IngestStage
    from omniscan.slicer.stage import SliceStage

    _run_stages("slice", [IngestStage(), SliceStage()], series, chapter, force)


app.command("ingest")(cmd_ingest)
app.command("slice")(cmd_slice)


def cmd_slice_compare(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        str | None, typer.Option("--chapter", "-c", help="Chapter folder name. Default: the first chapter.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the per-strategy summaries as one JSON object.")
    ] = False,
) -> None:
    """Compare every slicer strategy's cuts on one chapter's strip (nothing is written)."""
    from omniscan.core.schemas import IngestArtifact
    from omniscan.core.stage import make_context, run_stage
    from omniscan.ingest.stage import IngestStage
    from omniscan.ingest.strip import load_strip
    from omniscan.slicer.compare import compare_strategies, summarize

    cfg = get_config()
    sp = SeriesPaths.from_config(cfg, series)
    chapters = sp.chapters()
    if chapter is None:
        if not chapters:
            typer.echo(f"slice-compare: no chapters found for series {series!r}", err=True)
            raise typer.Exit(1)
        chapter = chapters[0]
    elif chapter not in chapters:
        typer.echo(f"slice-compare: unknown chapter {chapter!r} for series {series!r}", err=True)
        raise typer.Exit(1)

    ctx = make_context(cfg, series, chapter)  # ctx.cfg carries the series.toml [slicer] overrides
    ingest_path = ctx.paths.artifact("ingest.json")
    if not ingest_path.is_file() and run_stage(IngestStage(), ctx).status != "done":
        typer.echo(f"slice-compare: ingest failed for {series}/{chapter}", err=True)
        raise typer.Exit(1)
    ingest = IngestArtifact.load(ingest_path)
    strip = load_strip(ctx, ingest)  # exactly the strip the slice stage builds
    artifacts = compare_strategies(strip, ctx.cfg.slicer, ingest.files)
    summaries = [summarize(name, artifact) for name, artifact in artifacts.items()]
    if as_json:
        payload = {
            "series": series,
            "chapter": chapter,
            "strip_height": artifacts[next(iter(artifacts))].strip_height,
            "strategies": [asdict(s) for s in summaries],
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"{'strategy':<14}{'slices':>7}{'min':>7}{'median':>7}{'max':>7}{'forced':>7}{'blank':>7}")
    for s in summaries:
        typer.echo(
            f"{s.strategy:<14}{s.slices:>7}{s.min_height:>7}{s.median_height:>7}"
            f"{s.max_height:>7}{s.forced:>7}{s.blank:>7}"
        )


app.command("slice-compare")(cmd_slice_compare)


def cmd_detect(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if up to date.")] = False,
) -> None:
    """Detect bubbles and text regions in chapter strips (regions.json). Runs ingest and slice first if needed."""
    from omniscan.detect.stage import DetectStage
    from omniscan.ingest.stage import IngestStage
    from omniscan.slicer.stage import SliceStage

    _run_stages("detect", [IngestStage(), SliceStage(), DetectStage()], series, chapter, force)


app.command("detect")(cmd_detect)


def cmd_typeset(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if up to date.")] = False,
) -> None:
    """Fit final English lines into their regions' target boxes (layout.json). Needs ocr/final/inpaint."""
    from omniscan.typeset.stage import TypesetStage

    _run_stages("typeset", [TypesetStage()], series, chapter, force)


app.command("typeset")(cmd_typeset)


def cmd_ocr(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if up to date.")] = False,
) -> None:
    """Read the text of detected regions with PP-OCRv5 (ocr.json). Runs ingest, slice and detect first if needed."""
    from omniscan.detect.stage import DetectStage
    from omniscan.ingest.stage import IngestStage
    from omniscan.ocr.stage import OcrStage
    from omniscan.slicer.stage import SliceStage

    _run_stages("ocr", [IngestStage(), SliceStage(), DetectStage(), OcrStage()], series, chapter, force)


app.command("ocr")(cmd_ocr)


def _echo_text(text: str) -> None:
    """Echo text that may hold Korean/Japanese; Windows pipes (cp1252) must not crash on it."""
    import sys

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None and (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        reconfigure(encoding="utf-8", errors="replace")
    typer.echo(text)


def _ratio(value: float | None, numerator: int, denominator: int, suffix: str = "") -> str:
    """'0.874 (76/87)' for a score, 'n/a (0/0)' when the denominator was zero."""
    text = f"{value:.3f}" if value is not None else "n/a"
    return f"{text} ({numerator}/{denominator}{suffix})"


def _num(value: float | None) -> str:
    """A metric with three decimals, or 'n/a' when it could not be computed."""
    return f"{value:.3f}" if value is not None else "n/a"


def _short(text: str) -> str:
    return text[:40]


def _box_entry(result: Any) -> str:
    return (
        f"p{result.page:02d} [{result.bbox.x0},{result.bbox.y0},{result.bbox.x1},{result.bbox.y1}]"
        f' "{_short(result.text)}"'
    )


def _eval_block(report: Any, with_translation: bool) -> str:
    """The human-readable score block of one chapter."""
    approx = f", {report.approx_boxes} approximate" if report.approx_boxes else ""
    lines = [
        f"{report.series}/{report.chapter}: {report.pages} pages, {report.truth_boxes} truth boxes "
        f"({report.ignored_boxes} ignored, {report.dropped_boxes} dropped{approx})"
    ]
    if report.truth_boxes == 0:  # no usable truth boxes: nothing to score detection/OCR against
        lines.append(f"  {'detection':<13}n/a (no usable truth boxes)")
    else:
        lines.append(
            f"  {'detection':<13}recall {_ratio(report.recall, report.detected_boxes, report.truth_boxes)}"
            f"  chars {_num(report.recall_chars)}"
            f"  precision {_ratio(report.precision, report.assigned_regions, report.regions, ' regions')}"
        )
        lines.append(
            f"  {'OCR':<13}CER macro {_num(report.cer_macro)}  micro {_num(report.cer_micro)}"
            f"  ({report.cer_boxes} boxes)  page chrF {_num(report.ocr_chrf_mean)}"
            f" ({report.ocr_chrf_pages} pages)"
        )
    if with_translation:
        lines.append(f"  {'translation':<13}chrF {_num(report.chrf_mean)} ({report.chrf_pages} pages)")
    if report.missed:
        lines.append("  missed: " + " … ".join(_box_entry(r) for r in report.missed[:5]))
    if report.worst_cer:
        lines.append(
            "  worst CER: "
            + " … ".join(f'p{r.page:02d} {r.cer:.2f} "{_short(r.text)}"' for r in report.worst_cer[:5])
        )
    return "\n".join(lines)


def cmd_eval(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    lang: Annotated[str, typer.Option("--lang", help="Ground-truth language code (kr, cn, ja...).")] = "kr",
    as_json: Annotated[
        bool, typer.Option("--json", help="One JSON object per chapter instead of the text block.")
    ] = False,
) -> None:
    """Score chapters' ocr.json/final.json against ground-truth SVG text layers (eval.json)."""
    from omniscan.core.schemas import FinalArtifact, RegionsArtifact
    from omniscan.eval.score import score_chapter, to_json_lines
    from omniscan.eval.truth import load_english_pages, load_truth

    cfg = get_config()
    sp = SeriesPaths.from_config(cfg, series)
    chapters = chapter or sp.chapters()
    if not chapters:
        typer.echo(f"eval: no chapters found for series {series!r}", err=True)
        raise typer.Exit(2)
    missing = 0
    for chap in chapters:
        work = sp.chapter(chap).work_dir
        ocr_path = work / "ocr.json"
        if not ocr_path.is_file():
            typer.echo(f"eval: {chap}: ocr.json missing — run the ocr stage first", err=True)
            missing += 1
            continue
        if not (work / "ingest.json").is_file():
            typer.echo(f"eval: {chap}: ingest.json missing — run the ingest stage first", err=True)
            missing += 1
            continue
        check_dir = cfg.paths.library_root.parent / "translated-check" / series / chap / "truth"
        truth_dir = check_dir / lang
        if not truth_dir.is_dir():
            typer.echo(f"eval: {chap}: no ground truth at {truth_dir}", err=True)
            raise typer.Exit(2)
        ingest = IngestArtifact.load(work / "ingest.json")
        regions = RegionsArtifact.load(ocr_path)
        final_path = work / "final.json"
        final = FinalArtifact.load(final_path) if final_path.is_file() else None
        truth, stats = load_truth(check_dir, lang, ingest)
        english = load_english_pages(check_dir, ingest)
        report = score_chapter(series, chap, ingest, regions, final, truth, english, stats)
        (work / "eval.json").write_text(report.to_json(), encoding="utf-8")
        if as_json:
            _echo_text(to_json_lines([report]))
        else:
            _echo_text(_eval_block(report, with_translation=final is not None))
    if missing:
        raise typer.Exit(1)


app.command("eval")(cmd_eval)


def cmd_translate(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    profile: Annotated[
        list[str] | None,
        typer.Option("--profile", "-p", help="Profile name; repeatable. Default: every enabled one."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-run even if the run file already exists.")
    ] = False,
) -> None:
    """Translate chapters' OCR text into candidate runs (one file per translation profile)."""
    from omniscan.translate.chapter import translate_chapter
    from omniscan.translate.profiles import default_profile_paths, load_profiles

    cfg = get_config()
    sp = SeriesPaths.from_config(cfg, series)
    chapters = chapter or sp.chapters()
    if not chapters:
        typer.echo(f"translate: no chapters found for series {series!r}", err=True)
        raise typer.Exit(2)
    profiles = load_profiles(default_profile_paths())
    if profile:
        unknown = next((name for name in profile if name not in profiles), None)
        if unknown is not None:
            typer.echo(f"translate: unknown profile {unknown!r} (known: {', '.join(profiles)})", err=True)
            raise typer.Exit(2)
        selected = [profiles[name] for name in profile]
    else:
        selected = [p for p in profiles.values() if p.enabled]
        if not selected:
            typer.echo("translate: no enabled profiles in translation_profiles.toml", err=True)
            raise typer.Exit(2)
    entries = _glossary_entries(sp)
    failed = 0
    with OllamaClient(cfg.ollama, get_secrets()) as client:
        for chap in chapters:
            paths = sp.chapter(chap)
            for prof in selected:
                try:
                    status, run = translate_chapter(client, paths, prof, entries, force=force)
                except FileNotFoundError as exc:
                    typer.echo(f"{series}/{chap}: {exc}", err=True)
                    failed += 1
                    break
                except OllamaRateLimitError:
                    typer.echo(
                        "translate: Ollama rate limit reached — partial results are kept; re-run later",
                        err=True,
                    )
                    raise typer.Exit(3) from None
                except OllamaError as exc:
                    typer.echo(f"translate: {exc}", err=True)
                    failed += 1
                    continue
                if status == "done" and run is not None:
                    usage = run.usage
                    typer.echo(
                        f"{series}/{chap} {prof.name}: done ({int(usage['regions'])} regions, "
                        f"{int(usage['missing'])} missing, {usage['seconds']:.1f}s)"
                    )
                else:
                    typer.echo(f"{series}/{chap} {prof.name}: skipped")
    if failed:
        raise typer.Exit(1)


def _glossary_entries(sp: SeriesPaths) -> list[GlossaryEntry]:
    """All glossary entries of the series, or none when its db does not exist (never created here)."""
    if not sp.db.is_file():
        return []
    with GlossaryStore(sp.db) as store:
        return store.list()


app.command("translate")(cmd_translate)


def cmd_judge(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    run: Annotated[
        list[str] | None,
        typer.Option("--run", "-r", help="Candidate run id to judge; repeatable. Default: every run."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if final.json already exists.")] = False,
) -> None:
    """Judge candidate translation runs into final.json (one final line per region)."""
    from omniscan.translate.judge_chapter import judge_chapter
    from omniscan.translate.judge_config import default_judge_paths, load_judge_config

    cfg = get_config()
    sp = SeriesPaths.from_config(cfg, series)
    chapters = chapter or sp.chapters()
    if not chapters:
        typer.echo(f"judge: no chapters found for series {series!r}", err=True)
        raise typer.Exit(2)
    try:
        judge_cfg = load_judge_config(default_judge_paths())
    except ValueError as exc:
        typer.echo(f"judge: {exc}", err=True)
        raise typer.Exit(2) from exc
    entries = _glossary_entries(sp)
    failed = 0
    with OllamaClient(cfg.ollama, get_secrets()) as client:
        for chap in chapters:
            paths = sp.chapter(chap)
            try:
                status, _artifact, stats = judge_chapter(
                    client, paths, judge_cfg, entries, run_ids=run, force=force
                )
            except FileNotFoundError as exc:
                typer.echo(f"{series}/{chap}: {exc}", err=True)
                failed += 1
                continue
            except ValueError as exc:
                typer.echo(f"judge: {exc}", err=True)
                raise typer.Exit(2) from exc
            except OllamaRateLimitError:
                typer.echo("judge: Ollama rate limit reached — re-run later", err=True)
                raise typer.Exit(3) from None
            except OllamaError as exc:
                typer.echo(f"judge: {exc}", err=True)
                failed += 1
                continue
            if status == "done" and stats is not None:
                typer.echo(
                    f"{series}/{chap}: done ({stats.regions} regions, {stats.judged} judged, "
                    f"{stats.auto_picked} auto, {stats.untranslated} untranslated, "
                    f"{stats.violations_left} violations left, {stats.seconds:.1f}s)"
                )
            else:
                typer.echo(f"{series}/{chap}: skipped")
    if failed:
        raise typer.Exit(1)


app.command("judge")(cmd_judge)


def cmd_inpaint(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if up to date.")] = False,
    lama: Annotated[
        bool,
        typer.Option(
            "--lama", help="Also clean regions on textured art with LaMa (downloads 205 MB on first use)."
        ),
    ] = False,
) -> None:
    """Clean OCR regions' text out of the strip with flat fills (inpaint.json + patches.npz)."""
    from omniscan.inpaint.stage import InpaintStage

    stages: list[Any] = [InpaintStage()]
    if lama:
        from omniscan.inpaint.lama_stage import LamaStage

        stages.append(LamaStage())
    _run_stages("inpaint", stages, series, chapter, force)


app.command("inpaint")(cmd_inpaint)


def cmd_export(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if up to date.")] = False,
) -> None:
    """Write the finished English slices to output_root/<series>/<chapter> (export.json)."""
    from omniscan.export.stage import ExportStage

    _run_stages("export", [ExportStage()], series, chapter, force)


app.command("export")(cmd_export)


def cmd_run(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    stage: Annotated[
        list[str] | None,
        typer.Option("--stage", "-s", help="Stage name; repeatable. Default: all ten stages."),
    ] = None,
    no_lama: Annotated[
        bool, typer.Option("--no-lama", help="Skip the LaMa inpaint stage (inpaint_lama).")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-run stages even if up to date.")] = False,
    step: Annotated[
        bool,
        typer.Option("--step", help="Preview one chapter after every stage and ask before continuing."),
    ] = False,
    preview_chapter: Annotated[
        str | None,
        typer.Option("--preview-chapter", help="Which chapter step mode previews. Default: the first."),
    ] = None,
) -> None:
    """Take a series from raw chapters to exported English slices in three passes (vision, text, render)."""
    from omniscan.core.stage import StageOutcome
    from omniscan.pipeline.preview import describe
    from omniscan.pipeline.runner import GateEvent, needs_gpu, run_pipeline
    from omniscan.pipeline.stages import PASS_OF, STAGE_ORDER

    if preview_chapter is not None and not step:
        typer.echo("run: --preview-chapter needs --step", err=True)
        raise typer.Exit(2)
    cfg = get_config()
    if chapter is None and not SeriesPaths.from_config(cfg, series).chapters():
        typer.echo(f"run: no chapters found for series {series!r}", err=True)
        raise typer.Exit(2)
    names = list(stage) if stage else list(STAGE_ORDER)
    unknown = next((name for name in names if name not in STAGE_ORDER), None)
    if unknown is not None:
        typer.echo(f"run: unknown stage {unknown!r} (known: {', '.join(STAGE_ORDER)})", err=True)
        raise typer.Exit(2)
    client = (
        OllamaClient(cfg.ollama, get_secrets()) if any(PASS_OF[name] == "text" for name in names) else None
    )
    gpu = None
    auto_continue = False  # set by an "all" answer: every later gate passes without asking

    def gate(event: GateEvent) -> bool:
        """Show the preview block for this stage outcome and ask to continue."""
        nonlocal auto_continue
        if auto_continue:
            return True
        paths = SeriesPaths.from_config(cfg, series).chapter(event.chapter)
        preview = describe(event.stage, paths, cfg)
        typer.echo(
            f"Preview [{event.position}/{event.total}] {event.chapter} {event.stage}: {event.outcome.status}"
        )
        for line in (preview.summary, *preview.details):
            typer.echo(f"    {line}")
        if event.outcome.status == "failed" and event.outcome.error is not None:
            typer.echo(f"    {event.outcome.error}")
        for _attempt in range(3):
            try:
                answer = typer.prompt(
                    "Continue? [y]es / [n]o stop / [a]ll (finish without asking)", default="y"
                )
            except typer.Abort:  # EOF or Ctrl+C at the prompt: stop the run
                return False
            answer = answer.strip().lower()
            if answer in ("", "y"):
                return True
            if answer == "n":
                return False
            if answer == "a":
                auto_continue = True
                return True
        return False  # three invalid answers: treat them as "no"

    def report(chapter: str, outcome: StageOutcome) -> None:
        typer.echo(f"{series}/{chapter} {outcome.stage}: {outcome.status} ({outcome.seconds:.2f}s)")
        if outcome.status == "failed":
            typer.echo(f"    {outcome.error}")

    try:
        if needs_gpu(names, cfg, client):
            from omniscan.gpu.groups import build_vram_manager

            gpu = build_vram_manager(cfg)
        result = run_pipeline(
            cfg,
            series,
            chapter,
            stages=names,
            lama=not no_lama,
            force=force,
            client=client,
            gpu=gpu,
            report=report,
            mode="step" if step else "auto",
            preview_chapter=preview_chapter,
            gate=gate,
        )
    except ValueError as exc:
        typer.echo(f"run: {exc}", err=True)
        raise typer.Exit(2) from exc
    finally:
        if gpu is not None:
            gpu.release()  # the models leave VRAM when the command ends
        if client is not None:
            client.close()
    if result.aborted == "stopped":
        stopped_chapter, stopped_outcomes = next(iter(result.outcomes.items()))
        typer.echo(f"run: stopped after {stopped_outcomes[-1].stage} of {stopped_chapter}", err=True)
        raise typer.Exit(4) from None
    if result.aborted == "preview failed":
        typer.echo("run: the preview chapter failed — nothing else was run", err=True)
        raise typer.Exit(1) from None
    if result.aborted is not None:
        typer.echo("run: Ollama rate limit reached — re-run later", err=True)
        raise typer.Exit(3) from None
    failed = len(result.failed)
    typer.echo(f"{len(result.outcomes) - failed} chapter(s) ok, {failed} failed")
    if failed:
        raise typer.Exit(1)


app.command("run")(cmd_run)


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


filter_app = typer.Typer(no_args_is_help=True, help="Promo filter: pHash match against example images.")


@filter_app.command("run")
def filter_run(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit one JSON object instead of text lines.")
    ] = False,
) -> None:
    """Run ingest and slice with the promo filter over a series and report what was filtered."""
    from omniscan.ingest.stage import IngestStage
    from omniscan.slicer.stage import SliceStage

    _run_stages("filter", [IngestStage(), SliceStage()], series, chapter, force=False, progress=not as_json)
    sp = SeriesPaths.from_config(get_config(), series)
    chapters = list(chapter) if chapter is not None else sp.chapters()
    per_chapter: list[tuple[str, list[str], list[int]]] = []
    total_files = total_slices = 0
    for name in chapters:
        paths = sp.chapter(name)
        ingest_path = paths.artifact("ingest.json")
        slices_path = paths.artifact("slices.json")
        files = IngestArtifact.load(ingest_path).filtered_files if ingest_path.is_file() else []
        slice_indices = (
            [s.index for s in SlicesArtifact.load(slices_path).slices if s.filtered]
            if slices_path.is_file()
            else []
        )
        total_files += len(files)
        total_slices += len(slice_indices)
        per_chapter.append((name, list(files), slice_indices))
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "series": series,
                    "chapters": [
                        {"chapter": name, "files": files, "slices": slices}
                        for name, files, slices in per_chapter
                    ],
                    "totals": {"files": total_files, "slices": total_slices, "chapters": len(chapters)},
                },
                indent=2,
            )
        )
        return
    for name, files, slice_indices in per_chapter:
        typer.echo(f"{series}/{name}: filtered files: {len(files)}, filtered slices: {len(slice_indices)}")
    typer.echo(
        f"filter: {total_files} file(s), {total_slices} slice(s) filtered in {len(chapters)} chapter(s)"
    )


def _record_override(series: str, chapter: str, target: str, index: int, decision: str) -> None:
    """Validate (target, index) and append the manual decision to the chapter's filter.json."""
    if target not in ("file", "slice"):
        raise typer.BadParameter("target must be 'file' or 'slice'")
    paths = _chapter_paths(get_config(), series, chapter)
    record_override(
        paths,
        cast("Literal['file', 'slice']", target),
        index,
        cast("Literal['filtered', 'restored']", decision),
    )
    typer.echo(
        f"filter: {decision} {target} {index} of {series}/{chapter} (method manual); will apply on the next run"
    )


@filter_app.command("restore")
def filter_restore(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    target: Annotated[str, typer.Argument()],
    index: Annotated[int, typer.Argument()],
) -> None:
    """Restore a filtered file or slice (manual override; applied on the next run)."""
    _record_override(series, chapter, target, index, "restored")


@filter_app.command("force")
def filter_force(
    series: Annotated[str, typer.Argument()],
    chapter: Annotated[str, typer.Argument()],
    target: Annotated[str, typer.Argument()],
    index: Annotated[int, typer.Argument()],
) -> None:
    """Force-filter a file or slice (manual override; applied even without examples)."""
    _record_override(series, chapter, target, index, "filtered")


@filter_app.command("add")
def filter_add(
    series: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    global_: Annotated[
        bool,
        typer.Option("--global", help="Add to the shared global examples instead of this series'."),
    ] = False,
    name: Annotated[
        str | None, typer.Option("--name", help="Destination file name. Default: the source's name.")
    ] = None,
) -> None:
    """Copy a promo example image (JPEG/PNG) into the examples folder for `filter run` to use."""
    if name is not None and Path(name).name != name:
        typer.echo(f"filter: --name must be a plain file name, got {name!r}", err=True)
        raise typer.Exit(2)
    dest = get_config().paths.promo_examples / ("global" if global_ else series) / (name or path.name)
    if dest.suffix.lower() not in EXAMPLE_SUFFIXES:
        typer.echo(f"filter: {dest.name} is not a JPEG/PNG example (allowed: .jpg, .jpeg, .png)", err=True)
        raise typer.Exit(2)
    if dest.exists():
        typer.echo(f"filter: refusing to overwrite {dest}", err=True)
        raise typer.Exit(2)
    try:
        with Image.open(path):
            pass
    except Exception as exc:
        typer.echo(f"filter: {path} is not a readable image: {exc}", err=True)
        raise typer.Exit(2) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    typer.echo(f"filter: added example {dest}")
    typer.echo(f"filter: run 'omniscan filter run {series}' to apply it")


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

queue_app = typer.Typer(no_args_is_help=True, help="Persistent job queue: run pipeline stages over series.")


def _open_queue() -> QueueStore:
    return QueueStore(queue_db_path(get_config()))


_QUEUE_ACTIONS: dict[str, Callable[[QueueStore, int], Any]] = {
    "pause": QueueStore.pause,
    "resume": QueueStore.resume,
    "cancel": QueueStore.cancel,
    "retry": QueueStore.retry,
}


def _queue_job_action(action: str, job_id: int) -> None:
    """Run one store state transition for a job, mapping errors to exit 2 and printing the new status."""
    try:
        with _open_queue() as store:
            job = _QUEUE_ACTIONS[action](store, job_id)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except KeyError:
        typer.echo(f"no such job {job_id}", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"job {job.id}: {job.status}")


@queue_app.command("add")
def queue_add(
    series: Annotated[str, typer.Argument()],
    stage: Annotated[
        list[str] | None,
        typer.Option("--stage", "-s", help="Stage name; repeatable. Default: ingest, slice."),
    ] = None,
    chapter: Annotated[
        list[str] | None,
        typer.Option("--chapter", "-c", help="Chapter folder name; repeatable. Default: all."),
    ] = None,
    priority: Annotated[int, typer.Option("--priority", "-p", help="Higher runs first.")] = 0,
    max_attempts: Annotated[
        int, typer.Option("--max-attempts", help="How often the worker may retry a failing job.")
    ] = 2,
    force: Annotated[bool, typer.Option("--force", help="Re-run stages even if up to date.")] = False,
) -> None:
    """Queue pipeline stages over a series' chapters (executed later by `queue run`)."""
    stages = list(stage) if stage else ["ingest", "slice"]
    unknown = next((name for name in stages if name not in KNOWN_STAGES), None)
    if unknown is not None:
        typer.echo(f"queue: unknown stage {unknown!r} (known: {', '.join(KNOWN_STAGES)})", err=True)
        raise typer.Exit(2)
    try:
        with _open_queue() as store:
            job = store.add(
                series,
                stages,
                chapters=chapter,
                priority=priority,
                max_attempts=max_attempts,
                force=force,
            )
    except ValueError as exc:
        typer.echo(f"queue: {exc}", err=True)
        raise typer.Exit(2) from exc
    chapters_text = ",".join(job.chapters) if job.chapters is not None else "all"
    typer.echo(
        f"queued job {job.id}: {job.series} stages={','.join(job.stages)}"
        f" chapters={chapters_text} priority={job.priority}"
    )


@queue_app.command("list")
def queue_list(
    status: Annotated[str | None, typer.Option("--status", help="Only jobs with this status.")] = None,
) -> None:
    """Print the queue's jobs, one line per job (most recent error first 80 chars)."""
    if status is not None and status not in STATUSES:
        typer.echo(f"queue: unknown status {status!r} (known: {', '.join(STATUSES)})", err=True)
        raise typer.Exit(2)
    with _open_queue() as store:
        jobs = store.list(status=cast("JobStatus | None", status))
    if not jobs:
        typer.echo("queue is empty")
        return
    for job in jobs:
        chapters_text = ",".join(job.chapters) if job.chapters is not None else "all"
        line = (
            f"{job.id:>4}  {job.status:<9} pri={job.priority} try={job.attempts}/{job.max_attempts}"
            f"  {job.series}  {','.join(job.stages)}  {chapters_text}"
        )
        if job.error is not None:
            line += f"  !! {job.error[:80]}"
        typer.echo(line)


@queue_app.command("run")
def queue_run(
    webhook: Annotated[
        str | None,
        typer.Option("--webhook", help="POST job events to this URL.", envvar="OMNISCAN_NOTIFY_WEBHOOK"),
    ] = None,
    max_jobs: Annotated[
        int | None, typer.Option("--max-jobs", help="Stop after this many executed jobs.")
    ] = None,
) -> None:
    """Drain the queue: run queued jobs one at a time until it is empty."""
    cfg = get_config()
    notifier = combine(log_notifier, webhook_notifier(webhook)) if webhook is not None else log_notifier
    with QueueStore(queue_db_path(cfg)) as store:
        summary = run_queue(store, stage_executor(cfg), notifier, max_jobs=max_jobs)
    if not summary.finished:
        typer.echo("queue is empty")
        return
    for job in summary.finished:
        typer.echo(f"job {job.id} {job.status}" + (f": {job.error}" if job.error is not None else ""))
    typer.echo(f"done={summary.done} failed={summary.failed} retried={summary.retried}")
    if summary.failed > 0:
        raise typer.Exit(1)


@queue_app.command("pause")
def queue_pause(job_id: Annotated[int, typer.Argument()]) -> None:
    """Pause a queued job (it will not be claimed until resumed)."""
    _queue_job_action("pause", job_id)


@queue_app.command("resume")
def queue_resume(job_id: Annotated[int, typer.Argument()]) -> None:
    """Resume a paused job (back to queued)."""
    _queue_job_action("resume", job_id)


@queue_app.command("cancel")
def queue_cancel(job_id: Annotated[int, typer.Argument()]) -> None:
    """Cancel a queued or paused job."""
    _queue_job_action("cancel", job_id)


@queue_app.command("retry")
def queue_retry(job_id: Annotated[int, typer.Argument()]) -> None:
    """Reset a failed or cancelled job to queued with a fresh attempt counter."""
    _queue_job_action("retry", job_id)


@queue_app.command("clear")
def queue_clear() -> None:
    """Delete done and cancelled jobs from the queue (failed jobs stay for retry)."""
    with _open_queue() as store:
        removed = store.clear_finished()
    typer.echo(f"removed {removed} finished job(s)")


app.add_typer(queue_app, name="queue")

models_app = typer.Typer(no_args_is_help=True, help="Model catalog: list, download, remove, verify models.")


def _models_progress(steps: dict[str, int]) -> Callable[[str, int, int | None], None]:
    """Progress printer: `id: 42% (66/158 MB)`, at most once per 5 % step per model."""

    def on_progress(model_id: str, done: int, total: int | None) -> None:
        if total is None or total <= 0:
            return
        percent = int(done * 100 / total)
        step = percent // 5
        if step > steps.get(model_id, -1):
            steps[model_id] = step
            typer.echo(f"{model_id}: {percent}% ({done // 1_000_000}/{total // 1_000_000} MB)")

    return on_progress


@models_app.command("list")
def models_list(
    as_json: Annotated[bool, typer.Option("--json", help="Emit a JSON object instead of text rows.")] = False,
    role: Annotated[str | None, typer.Option("--role", help="Only models with this role.")] = None,
    lang: Annotated[str | None, typer.Option("--lang", help="Only models that list this language.")] = None,
) -> None:
    """List every catalog model with its size, purpose, install status and hardware fit."""
    cfg = get_config()
    rows, hw = build_rows(cfg, role=role, lang=lang, ollama_names=_ollama_model_names(cfg.ollama.local_url))
    if as_json:
        payload = {
            "models_dir": str(cfg.paths.models_dir),
            "models": [row_to_json(row) for row in rows],
            "hardware": asdict(hw),
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for row in rows:
        typer.echo(
            f"{row.id}  {row.kind}  {row.role or '-'}  {row.size_mb} MB  "
            f"{'required' if row.required else 'optional'}  {row.status}  "
            f"{','.join(row.langs) or '-'}  {row.description}"
            f"  {row.fit_level}"
        )
    for row in rows:
        if row.fit_level != "ok" and row.status != "installed" and row.fit_messages:
            typer.echo(f"  {row.id}: {'; '.join(row.fit_messages)}")
    missing_required = [row for row in rows if row.required and row.status in ("missing", "corrupt")]
    if missing_required:
        typer.echo(
            f"{len(missing_required)} required model(s) missing, {sum(r.size_mb for r in missing_required)} MB to download"
        )
    else:
        typer.echo("all required models installed")


@models_app.command("download")
def models_download(
    ids: Annotated[
        list[str] | None, typer.Argument(help="Model id(s) from the catalog. Default: none.")
    ] = None,
    required: Annotated[
        bool,
        typer.Option("--required", help="Also download every required model that is not installed."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Download even when the model is incompatible with this machine."),
    ] = False,
) -> None:
    """Download the named models (plus every missing required one with --required)."""
    from omniscan.hw.assess import assess
    from omniscan.hw.detect import detect_hardware
    from omniscan.models.catalog import load_catalog
    from omniscan.models.download import ModelDownloadError, download_model
    from omniscan.models.store import model_status

    cfg = get_config()
    entries = {entry.id: entry for entry in load_catalog()}
    named = list(dict.fromkeys(ids or []))
    unknown = [model_id for model_id in named if model_id not in entries]
    if unknown:
        typer.echo(
            f"models: unknown model id(s) {', '.join(unknown)} (known: {', '.join(entries)})", err=True
        )
        raise typer.Exit(2)
    selected_ids = list(named)
    if required:
        for model_id, entry in entries.items():
            if model_id in selected_ids or not entry.required:
                continue
            status = model_status(entry, cfg.paths.models_dir, ollama_names=None)
            if status == "installed":
                typer.echo(f"{model_id}: already installed")
            else:  # missing or corrupt: (re-)download it
                selected_ids.append(model_id)
    if not selected_ids and not required:
        typer.echo("models: nothing to download (no ids given and no required model is missing)", err=True)
        raise typer.Exit(2)
    ollama_names = _ollama_model_names(cfg.ollama.local_url)
    hw = detect_hardware(cfg.paths.models_dir)
    steps: dict[str, int] = {}
    failed = False
    for model_id in selected_ids:
        entry = entries[model_id]
        if entry.format == "ollama" and ollama_names is not None and entry.ollama_name in ollama_names:
            typer.echo(f"{entry.id}: already installed")
            continue
        compat = assess(entry, hw)
        if compat.level != "ok":
            typer.echo(f"{entry.id}: {compat.level}: {'; '.join(compat.messages)}", err=True)
            if compat.level == "incompatible" and not force:
                failed = True  # refused: nothing is downloaded for this model
                continue
        try:
            source = download_model(
                entry,
                cfg.paths.models_dir,
                ollama_url=cfg.ollama.local_url,
                on_progress=_models_progress(steps),
            )
        except ModelDownloadError as exc:
            typer.echo(f"models: {exc}", err=True)
            failed = True
            continue
        if source == "already installed":
            typer.echo(f"{entry.id}: already installed")
        else:
            typer.echo(f"{entry.id}: installed from {source}")
    if failed:
        raise typer.Exit(1)


@models_app.command("remove")
def models_remove(ids: Annotated[list[str], typer.Argument(help="Model id(s) from the catalog.")]) -> None:
    """Remove installed models (their folder, file, or the Ollama daemon's copy)."""
    from omniscan.models.catalog import load_catalog
    from omniscan.models.download import ModelDownloadError, remove_model

    cfg = get_config()
    entries = {entry.id: entry for entry in load_catalog()}
    unknown = [model_id for model_id in ids if model_id not in entries]
    if unknown:
        typer.echo(
            f"models: unknown model id(s) {', '.join(unknown)} (known: {', '.join(entries)})", err=True
        )
        raise typer.Exit(2)
    failed = False
    for model_id in dict.fromkeys(ids):
        try:
            removed = remove_model(entries[model_id], cfg.paths.models_dir, ollama_url=cfg.ollama.local_url)
        except ModelDownloadError as exc:
            typer.echo(f"models: {exc}", err=True)
            failed = True
            continue
        typer.echo(f"{model_id}: removed" if removed else f"{model_id}: nothing to remove")
    if failed:
        raise typer.Exit(1)


@models_app.command("verify")
def models_verify(
    ids: Annotated[
        list[str] | None, typer.Argument(help="Model id(s). Default: every non-llm model.")
    ] = None,
    deep: Annotated[bool, typer.Option("--deep", help="Rehash every file of the hf models (slow).")] = False,
) -> None:
    """Recompute model statuses (hash checks); exit 1 when one is corrupt."""
    from omniscan.models.catalog import load_catalog
    from omniscan.models.download import verify_models

    cfg = get_config()
    entries = {entry.id: entry for entry in load_catalog()}
    if ids:
        unknown = [model_id for model_id in ids if model_id not in entries]
        if unknown:
            typer.echo(
                f"models: unknown model id(s) {', '.join(unknown)} (known: {', '.join(entries)})", err=True
            )
            raise typer.Exit(2)
        selected = [entry for entry in entries.values() if entry.id in set(ids)]
    else:
        selected = [entry for entry in entries.values() if entry.kind != "llm"]
    statuses = verify_models(selected, cfg.paths.models_dir, deep=deep)
    for entry in selected:
        typer.echo(f"{entry.id}: {statuses[entry.id]}")
    if any(status == "corrupt" for status in statuses.values()):
        raise typer.Exit(1)


app.add_typer(models_app, name="models")


update_app = typer.Typer(no_args_is_help=True, help="Check GitHub Releases for and stage a newer OmniScan.")


class UpdateChannel(enum.StrEnum):
    """Channel of `omniscan update` (typer cannot build a click option from list[Literal])."""

    STABLE = "stable"
    BETA = "beta"


def _http_client() -> httpx.Client:
    """HTTP client for the updater's GitHub requests (redirects on, 30 s timeout)."""
    return httpx.Client(follow_redirects=True, timeout=30.0)


def _update_token() -> str | None:
    """The optional GitHub token from the environment (never printed)."""
    return os.environ.get("GITHUB_TOKEN")


def _newer_release(client: httpx.Client, channel: UpdateChannel, repo: str) -> ReleaseInfo | None:
    """The newest app release for the channel, or None when the install is up to date."""
    return check_for_update(
        client, repo=repo, channel=channel.value, current=current_version(), token=_update_token()
    )


@update_app.command("check")
def update_check(
    channel: Annotated[
        UpdateChannel, typer.Option("--channel", case_sensitive=False, help="stable or beta.")
    ] = UpdateChannel.STABLE,
    repo: Annotated[str, typer.Option("--repo", help="GitHub repository OWNER/NAME.")] = DEFAULT_REPO,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the machine-readable form.")] = False,
) -> None:
    """Report whether a newer OmniScan release is available on GitHub Releases."""
    try:
        with _http_client() as client:
            current = current_version()
            release = _newer_release(client, channel, repo)
    except UpdateError as exc:
        typer.echo(f"update: {exc}", err=True)
        raise typer.Exit(1) from exc
    if as_json:
        asset: str | None = None
        if release is not None:
            try:
                asset = select_asset(release, platform_key()).name
            except UpdateError:
                asset = None
        typer.echo(
            json.dumps(
                {
                    "current": str(current),
                    "latest": str(release.version) if release is not None else None,
                    "update_available": release is not None,
                    "tag": release.tag if release is not None else None,
                    "notes": release.notes if release is not None else None,
                    "asset": asset,
                },
                indent=2,
            )
        )
        return
    if release is None:
        typer.echo(f"update: up to date (v{current})")
        return
    typer.echo(f"update: update available: v{release.version} (current v{current})")
    for line in release.notes.splitlines()[:20]:
        typer.echo(f"update:   {line}")


@update_app.command("download")
def update_download(
    channel: Annotated[
        UpdateChannel, typer.Option("--channel", case_sensitive=False, help="stable or beta.")
    ] = UpdateChannel.STABLE,
    repo: Annotated[str, typer.Option("--repo", help="GitHub repository OWNER/NAME.")] = DEFAULT_REPO,
) -> None:
    """Stage the newest release's verified build for this platform under <user data dir>/updates."""
    updates_dir = get_config().paths.work_root.parent / "updates"
    next_mark = 10

    def on_progress(done: int, total: int | None) -> None:
        """Print a line at every 10 % mark of the download."""
        nonlocal next_mark
        if total and done * 100 >= next_mark * total:
            typer.echo(f"update:   {min(100, done * 100 // total)}%")
            next_mark += 10

    try:
        with _http_client() as client:
            release = _newer_release(client, channel, repo)
            if release is None:
                typer.echo("update: already up to date")
                return
            staged = download_update(client, release, updates_dir, on_progress=on_progress)
    except UpdateError as exc:
        typer.echo(f"update: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"update: staged {staged.path} ({staged.sha256})")


app.add_typer(update_app, name="update")

app.add_typer(acquire_app, name="acquire")
