"""Chapter files service: chapter projects and contribution archives for the Library page (Qt-free).

`send_chapter` packs one chapter into a `.omniscan` project (`interchange.project.pack`), `open_project`
unpacks one into the library (`interchange.project.unpack`) and `export_contribution` writes a series' hand
corrections and checked lines to a contribution archive (`share.contribution`). Each returns the one line the
page shows; the backends' own errors (ProjectError, FileExistsError, ShareOptOutError, OSError) pass through for
the page to report.
"""

from __future__ import annotations

from pathlib import Path

from omniscan.core.config import Config
from omniscan.core.paths import SeriesPaths
from omniscan.interchange import project
from omniscan.packaging import safe_filename
from omniscan.share.cli import describe
from omniscan.share.contribution import build, digest, install_salt, summarize, write_archive

# a renamed or re-zipped file is still offered (Postel's law): unpack checks what it really is
PROJECT_FILTER = f"OmniScan chapter project (*{project.SUFFIX});;All files (*)"
CONTRIBUTION_FILTER = "Contribution archive (*.zip)"


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
