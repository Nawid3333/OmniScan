"""Synthetic library the GUI tests and screenshot script read: raw + output image stacks and manifests.

`build_library(base)` writes one series with four chapters covering the library states (all done /
partial with failures / only raw / stale records) and returns a `Config` pointed at it. Everything
is generated in code — no real manga is ever downloaded.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    ExportArtifact,
    ExportFile,
    IngestArtifact,
    Manifest,
    Slice,
    SlicesArtifact,
    SourceFile,
    StageRecord,
)

SERIES = "FixtureSeries"
CHAPTERS = ("Episode 01", "Episode 02", "Episode 03", "Episode 04")
PAGES = 3
PAGE_SIZE = (40, 100)  # every raw page and output slice: 40x100
STRIP = (40, 300)  # strip width and height of the synthetic chapters
STAGES = (
    "ingest",
    "slice",
    "detect",
    "ocr",
    "translate",
    "judge",
    "inpaint",
    "inpaint_lama",
    "typeset",
    "export",
)

_PAGE_COLORS = ((240, 60, 60), (60, 240, 60), (60, 60, 240))
_SLICE_COLORS = ((250, 210, 60), (60, 220, 250), (250, 60, 250))
_BASE_TIME = datetime(2026, 9, 1, tzinfo=UTC)
_ARTIFACT_OF = {"ingest": "ingest.json", "slice": "slices.json", "export": "export.json"}


def build_library(base: Path) -> Config:
    """Write the synthetic series under `base` (library/work/output/models) and return its Config."""
    cfg = Config(
        paths=PathsConfig(
            library_root=base / "library",
            work_root=base / "work",
            output_root=base / "output",
            models_dir=base / "models",
        )
    )
    _chapter(cfg, "Episode 01", all_done=True, outputs=PAGES)
    _chapter(
        cfg,
        "Episode 02",
        done_through="ocr",
        failures={"translate": "rate limited"},
        stale_done={"typeset": ["typeset.json"]},
        outputs=2,
    )
    _chapter(cfg, "Episode 03")  # only raws: no manifest, no artifacts, no output
    _chapter(
        cfg,
        "Episode 04",
        done_through="judge",
        reruns={"detect": "slice"},  # slice re-ran after detect: detect shows stale
        failures={"inpaint": "out of memory"},
    )
    return cfg


# --------------------------------------------------------------------- one chapter


def _chapter(
    cfg: Config,
    name: str,
    *,
    all_done: bool = False,
    done_through: str | None = None,
    failures: dict[str, str] | None = None,
    stale_done: dict[str, list[str]] | None = None,
    reruns: dict[str, str] | None = None,
    outputs: int = 0,
) -> None:
    """Write one chapter: raws, artifacts, output slices and the manifest (defaults: raws only)."""
    failures = failures or {}
    stale_done = stale_done or {}
    reruns = reruns or {}
    chapter = _chapter_paths(cfg, name)
    _raw_images(chapter)
    if outputs:
        _output_images(chapter, outputs)

    stages: dict[str, StageRecord] = {}
    for index, stage in enumerate(STAGES):
        if stage in stale_done:
            stages[stage] = _record(chapter, stage, index, stale_done[stage], "done", None)
        elif stage in failures:
            stages[stage] = _record(chapter, stage, index, [], "failed", failures[stage])
        elif all_done or (done_through is not None and index <= STAGES.index(done_through)):
            stages[stage] = _record(chapter, stage, index, _stage_outputs(stage), "done", None)
    for later, earlier in reruns.items():  # the earlier stage finished again after the later one
        stages[earlier] = _record(
            chapter,
            earlier,
            STAGES.index(earlier),
            _stage_outputs(earlier),
            "done",
            None,
            finished_offset=STAGES.index(later) + 0.5,
        )
    if artifact_stages := {stage for stage, record in stages.items() if record.status == "done"} & set(
        _ARTIFACT_OF
    ):
        _artifacts(chapter, artifact_stages)
    for stage, record in stages.items():
        if record.status != "done" or stage in stale_done:  # stale_done keeps its missing output
            continue
        for output in record.outputs:  # a done record's files must exist (or the state reads stale)
            placeholder = chapter.work_dir / output
            if not placeholder.is_file():
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                placeholder.write_text("{}", encoding="utf-8")
    Manifest(series=SERIES, chapter=name, stages=stages).save(chapter.manifest)


def _chapter_paths(cfg: Config, chapter: str) -> ChapterPaths:
    """The ChapterPaths of one synthetic chapter."""
    return ChapterPaths(
        series=SERIES,
        chapter=chapter,
        raw_dir=cfg.paths.library_root / SERIES / chapter / "raw",
        work_dir=cfg.paths.work_root / SERIES / chapter,
        output_dir=cfg.paths.output_root / SERIES / chapter,
        filtered_dir=cfg.paths.library_root / SERIES / chapter / "raw-filtered",
    )


def _stage_outputs(stage: str) -> list[str]:
    """The work-dir file names a done record of `stage` lists (its artifact, else a placeholder)."""
    return [_ARTIFACT_OF.get(stage, f"{stage}.json")]


# --------------------------------------------------------------------- files


def _raw_images(chapter: ChapterPaths) -> list[Path]:
    """Three solid-colour raw pages (red/green/blue) as JPEG."""
    chapter.raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, color in enumerate(_PAGE_COLORS):
        path = chapter.raw_dir / f"{index + 1:04d}.jpg"
        Image.new("RGB", PAGE_SIZE, color).save(path, "JPEG")
        paths.append(path)
    return paths


def _output_images(chapter: ChapterPaths, count: int) -> list[Path]:
    """`count` solid-colour output slices as JPEG."""
    chapter.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = chapter.output_dir / f"{index + 1:04d}.jpg"
        Image.new("RGB", PAGE_SIZE, _SLICE_COLORS[index % len(_SLICE_COLORS)]).save(path, "JPEG")
        paths.append(path)
    return paths


def _artifacts(chapter: ChapterPaths, which: set[str]) -> None:
    """ingest.json / slices.json / export.json for the full 3-page, 3-slice strip."""
    files = [
        SourceFile(
            index=i,
            name=f"{i + 1:04d}.jpg",
            sha256=f"{i:064x}",
            width=PAGE_SIZE[0],
            height=PAGE_SIZE[1],
            y0=i * PAGE_SIZE[1],
            y1=(i + 1) * PAGE_SIZE[1],
        )
        for i in range(PAGES)
    ]
    if "ingest" in which:
        IngestArtifact(
            series=SERIES, chapter=chapter.chapter, strip_width=STRIP[0], strip_height=STRIP[1], files=files
        ).save(chapter.artifact("ingest.json"))
    if "slice" in which:
        SlicesArtifact(
            strip_width=STRIP[0],
            strip_height=STRIP[1],
            bands=[],
            slices=[
                Slice(index=i, y0=i * PAGE_SIZE[1], y1=(i + 1) * PAGE_SIZE[1], source_files=[i])
                for i in range(PAGES)
            ],
            params={},
        ).save(chapter.artifact("slices.json"))
    if "export" in which:
        ExportArtifact(
            quality=90,
            subsampling="420",
            files=[
                ExportFile(
                    name=f"{i + 1:04d}.jpg",
                    slice_index=i,
                    width=PAGE_SIZE[0],
                    height=PAGE_SIZE[1],
                    bytes=1024,
                )
                for i in range(PAGES)
            ],
        ).save(chapter.artifact("export.json"))


# --------------------------------------------------------------------- records


def _record(
    chapter: ChapterPaths,
    stage: str,
    index: int,
    outputs: list[str],
    status: str,
    error: str | None,
    *,
    finished_offset: float | None = None,
) -> StageRecord:
    """One manifest record; `finished_offset` (minutes) overrides the natural stage-order time."""
    finished = _BASE_TIME + timedelta(minutes=finished_offset if finished_offset is not None else index)
    return StageRecord(
        stage=stage,
        version=1,
        input_hash=f"hash-{index}",
        config_hash="hash-config",
        outputs=outputs,
        status="done" if status == "done" else "failed",
        started_at=_BASE_TIME + timedelta(minutes=index),
        finished_at=finished,
        metrics={},
        error=error,
    )
