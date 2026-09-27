"""The reader format: omniscan-chapter.json beside each exported chapter and omniscan-series.json per series.

A reader app (OmniScan's own reading mode, or a later phone/tablet app) opens an output folder without knowing
anything about the pipeline: the series index lists the chapters in reading order, each chapter file lists its
pages with their pixel sizes (so a reader can lay out the scroll before an image loads). docs/READER_FORMAT.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omniscan.core.paths import chapter_number, list_chapters
from omniscan.core.schemas import ExportFile, ReaderChapter, ReaderPage, ReaderSeries, ReaderSeriesChapter

CHAPTER_INDEX = "omniscan-chapter.json"
SERIES_INDEX = "omniscan-series.json"


def write_chapter_index(
    chapter_dir: Path, series: str, chapter: str, files: Sequence[ExportFile]
) -> ReaderChapter:
    """Write omniscan-chapter.json for one exported chapter and return it."""
    index = ReaderChapter(
        series=series,
        chapter=chapter,
        number=chapter_number(chapter),
        pages=[ReaderPage(file=item.name, width=item.width, height=item.height) for item in files],
    )
    index.save(chapter_dir / CHAPTER_INDEX)
    return index


def write_series_index(series_dir: Path, series: str) -> ReaderSeries:
    """Rebuild omniscan-series.json from every chapter folder that has a readable chapter index."""
    chapters: list[ReaderSeriesChapter] = []
    for folder in list_chapters(series_dir):
        chapter = load_chapter_index(folder)
        if chapter is not None:
            chapters.append(
                ReaderSeriesChapter(
                    folder=folder.name,
                    number=chapter.number,
                    pages=len(chapter.pages),
                    updated_at=chapter.updated_at,
                )
            )
    index = ReaderSeries(series=series, chapters=chapters)
    index.save(series_dir / SERIES_INDEX)
    return index


def load_chapter_index(chapter_dir: Path) -> ReaderChapter | None:
    """A chapter folder's index, or None when it is missing or unreadable."""
    try:
        return ReaderChapter.load(chapter_dir / CHAPTER_INDEX)
    except OSError, ValueError:
        return None
