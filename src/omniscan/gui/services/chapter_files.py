"""Chapter files service: chapter projects, contribution archives and other tools' files for the Library page
(Qt-free).

`send_chapter` packs one chapter into a `.omniscan` project (`interchange.project.pack`), `open_project`
unpacks one into the library (`interchange.project.unpack`) and `export_contribution` writes a series' hand
corrections and checked lines to a contribution archive (`share.contribution`). The `export_*` / `import_*`
functions pass a chapter to and from LabelPlus, Photoshop, BallonsTranslator and manga-image-translator
(`interchange.actions`, the same as the CLI's `omniscan labelplus|psd|ballons|mit`). Each returns the one line the
page shows; the backends' own errors (ProjectError, FileExistsError, ShareOptOutError, InterchangeError, OSError)
pass through for the page to report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.interchange import actions, project
from omniscan.packaging import safe_filename
from omniscan.share.cli import describe
from omniscan.share.contribution import build, digest, install_salt, summarize, write_archive

# a renamed or re-zipped file is still offered (Postel's law): unpack checks what it really is
PROJECT_FILTER = f"OmniScan chapter project (*{project.SUFFIX});;All files (*)"
CONTRIBUTION_FILTER = "Contribution archive (*.zip)"
LABELPLUS_FILTER = "LabelPlus file (*.txt);;All files (*)"
BALLONS_FILTER = "BallonsTranslator project (imgtrans_*.json);;All files (*)"
MIT_FILTER = "manga-image-translator text (*_translations.txt);;Text files (*.txt);;All files (*)"


def project_name(series: str, chapter: str) -> str:
    """The file name a sent chapter gets by default, as `omniscan project pack` names it."""
    return f"{safe_filename(series)} - {safe_filename(chapter)}{project.SUFFIX}"


def contribution_name(cfg: Config, series: str) -> str:
    """The file name a series' contribution gets by default, as `omniscan contribute export` names it."""
    return f"omniscan-contribution-{digest(series, install_salt(cfg.paths.work_root))}.zip"


def send_chapter(cfg: Config, series: str, chapter: str, dest: Path) -> str:
    """Pack `chapter` of `series` into the chapter project `dest`; returns what was written."""
    packed = project.pack(SeriesPaths.from_config(cfg, series), chapter, dest)
    return f"Sent {series} — {chapter}: {len(packed.files)} file(s) in {dest.name}"


def open_project(cfg: Config, path: Path, *, force: bool = False) -> str:
    """Unpack the chapter project at `path` into the library (`force` replaces an existing chapter); returns
    what happened. FileExistsError when the chapter exists and `force` is not set."""
    done = project.unpack(path, cfg, force=force)
    verb = "Replaced" if done.replaced else "Opened"
    notes = [f"added {', '.join(done.series_files)}"] if done.series_files else []
    notes += [f"kept your {', '.join(done.kept_series_files)}"] if done.kept_series_files else []
    notes += [f"left out {', '.join(done.dropped_settings)}"] if done.dropped_settings else []
    return f"{verb} {done.paths.series} — {done.paths.chapter}" + (f" ({'; '.join(notes)})" if notes else "")


def export_contribution(cfg: Config, series: str, dest: Path) -> str:
    """Write the contribution archive of `series` to `dest`; returns what it holds (or that it has nothing to
    share, when no page carries a correction or a checked line: then no file is written)."""
    contribution, pages = build(SeriesPaths.from_config(cfg, series), cfg)
    if not contribution.chapters:
        return f"{series} has no hand corrections or checked lines to share; nothing exported"
    write_archive(dest, contribution, pages)
    return f"Exported {describe(summarize(contribution))} to {dest.name}"


def labelplus_path(cfg: Config, series: str, chapter: str) -> Path:
    """Where a chapter's LabelPlus file goes by default (as `omniscan labelplus export` writes it)."""
    return actions.labelplus_path(SeriesPaths.from_config(cfg, series), chapter)


def psd_folder(cfg: Config, series: str, chapter: str) -> Path:
    """Where a chapter's PSD pages go by default (as `omniscan psd export` writes them)."""
    return actions.psd_folder(SeriesPaths.from_config(cfg, series), chapter)


def ballons_folder(cfg: Config, series: str, chapter: str) -> Path:
    """Where a chapter's BallonsTranslator project goes by default (as `omniscan ballons export` writes it)."""
    return actions.ballons_folder(SeriesPaths.from_config(cfg, series), chapter)


def export_labelplus(cfg: Config, series: str, chapter: str, dest: Path) -> str:
    """Write the chapter's English lines as the LabelPlus file `dest`; returns what was written."""
    done = actions.export_labelplus(SeriesPaths.from_config(cfg, series), chapter, out=dest)
    return f"Wrote {done.labels} label(s) on {done.pages} page(s) to {done.path.name}"


def export_psd(cfg: Config, series: str, chapter: str, folder: Path) -> str:
    """Write one layered PSD per page of the chapter into `folder`; returns what was written."""
    done = actions.export_psd(SeriesPaths.from_config(cfg, series), chapter, out=folder)
    note = "" if done.has_text else " (no lettering yet: run typeset for the text layers)"
    return f"Wrote {done.pages} layered page(s) to {done.folder}{note}"


def export_ballons(cfg: Config, series: str, chapter: str, folder: Path) -> str:
    """Write the chapter as a BallonsTranslator project into `folder`; returns what was written."""
    done = actions.export_ballons(SeriesPaths.from_config(cfg, series), chapter, out=folder)
    note = "" if done.cleaned else " (not cleaned yet: no inpainted pages)"
    return f"Wrote {done.blocks} text block(s) on {done.pages} page(s) to {done.folder}{note}"


def _imported(
    chapter: str, lines: int, same: int, outside: int, unknown_pages: Sequence[str], what: str
) -> str:
    """The status line of an import: lines taken, lines already the same, what matched nothing."""
    notes = [f"{same} already the same"]
    if outside:
        notes.append(f"{outside} {what}(s) over no box")
    if unknown_pages:
        notes.append(f"page(s) not in this chapter: {', '.join(unknown_pages)}")
    return f"Imported {lines} English line(s) into {chapter} ({'; '.join(notes)}); Undo in the Studio takes it back"


def import_labelplus(cfg: Config, series: str, chapter: str, file: Path) -> str:
    """Take a LabelPlus file's texts as the chapter's English lines (one undo step); returns what changed."""
    found = actions.import_labelplus(cfg, SeriesPaths.from_config(cfg, series), chapter, file)
    return _imported(chapter, found.changed, found.same, len(found.unmatched), found.unknown_pages, "label")


def import_ballons(cfg: Config, series: str, chapter: str, file: Path) -> str:
    """Take a BallonsTranslator project's translations as the chapter's English lines (one undo step)."""
    blocks = actions.read_ballons(file)
    found = actions.import_blocks(cfg, SeriesPaths.from_config(cfg, series), chapter, blocks)
    return _imported(chapter, found.lines, found.same, len(found.unmatched), found.unknown_pages, "block")


def import_mit(cfg: Config, series: str, chapter: str, files: Sequence[Path]) -> str:
    """Take manga-image-translator's --save-text translations as the chapter's English lines (one undo step)."""
    blocks = actions.read_mit(files)
    found = actions.import_blocks(cfg, SeriesPaths.from_config(cfg, series), chapter, blocks)
    return _imported(
        chapter, found.lines, found.same, len(found.unmatched), found.unknown_pages, "text region"
    )
