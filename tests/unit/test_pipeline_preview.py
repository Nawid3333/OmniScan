"""Tests for omniscan.pipeline.preview — exact summaries from hand-made artifacts (card P1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    BBox,
    Candidate,
    CandidateRun,
    ExportArtifact,
    ExportFile,
    FinalArtifact,
    FinalLine,
    IngestArtifact,
    InpaintArtifact,
    InpaintItem,
    LayoutArtifact,
    LayoutItem,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.pipeline.preview import Preview, describe

SERIES = "S"
CHAPTER = "Chapter 1"
BOX = BBox(x0=0, y0=0, x1=100, y1=50)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """CPU-only config with all paths under tmp_path."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    """ChapterPaths laid out like the real roots, inside tmp_path."""
    return ChapterPaths(
        series=SERIES,
        chapter=CHAPTER,
        raw_dir=tmp_path / "library" / SERIES / CHAPTER,
        work_dir=tmp_path / "work" / SERIES / CHAPTER,
        output_dir=tmp_path / "output" / SERIES / CHAPTER,
        filtered_dir=tmp_path / "output" / "_filtered" / SERIES / CHAPTER,
    )


def region(rid: str, text: str = "", confidence: float = 0.9, kind: str = "bubble_text") -> Region:
    """A minimal OCR region for the previews."""
    return Region(
        id=rid,
        slice_index=0,
        kind=kind,  # pyright: ignore[reportArgumentType]
        bbox=BOX,
        text=text,
        confidence=confidence,
    )


def test_ingest_counts_kept_files_and_the_strip_size(paths: ChapterPaths, cfg: Config) -> None:
    IngestArtifact(
        series=SERIES,
        chapter=CHAPTER,
        strip_width=800,
        strip_height=2000,
        files=[
            SourceFile(index=0, name="1.jpg", sha256="a", width=800, height=1000, y0=0, y1=1000),
            SourceFile(
                index=1, name="2.jpg", sha256="b", width=800, height=1000, y0=1000, y1=2000, filtered=True
            ),
        ],
    ).save(paths.artifact("ingest.json"))
    assert describe("ingest", paths, cfg) == Preview("ingest", "1 file(s), strip 800x2000 px", ())


def test_slice_summary_counts_blank_forced_and_filtered(paths: ChapterPaths, cfg: Config) -> None:
    SlicesArtifact(
        strip_width=800,
        strip_height=2000,
        bands=[],
        slices=[
            Slice(index=0, y0=0, y1=1000),
            Slice(index=1, y0=1000, y1=1500, blank=True),
            Slice(index=2, y0=1500, y1=2000, forced_cut=True, filtered=True),
        ],
    ).save(paths.artifact("slices.json"))
    assert describe("slice", paths, cfg) == Preview(
        "slice", "3 slice(s): 1 blank, 1 forced cut(s), 1 filtered", ()
    )


def test_detect_summary_lists_kinds_in_order_and_only_nonzero_ones(paths: ChapterPaths, cfg: Config) -> None:
    RegionsArtifact(
        regions=[
            region("r1", kind="bubble_text"),
            region("r2", kind="sfx"),
            region("r3", kind="free_text"),
            region("r4", kind="bubble_text"),
        ]
    ).save(paths.artifact("regions.json"))
    assert describe("detect", paths, cfg) == Preview(
        "detect", "4 region(s): bubble_text 2, free_text 1, sfx 1", ()
    )


def test_detect_with_zero_regions_says_so(paths: ChapterPaths, cfg: Config) -> None:
    RegionsArtifact(regions=[]).save(paths.artifact("regions.json"))
    assert describe("detect", paths, cfg) == Preview("detect", "0 regions", ())


def test_ocr_summary_counts_text_and_low_confidence_and_shows_first_texts(
    paths: ChapterPaths, cfg: Config
) -> None:
    RegionsArtifact(
        regions=[
            region("r1", "First line", confidence=0.9),
            region("r2", "Second line", confidence=0.5),
        ]
    ).save(paths.artifact("ocr.json"))
    assert describe("ocr", paths, cfg) == Preview(
        "ocr",
        "2 region(s) with text, 1 low-confidence (< 0.85)",
        ("First line", "Second line"),
    )


def test_ocr_details_are_clipped_to_59_characters_plus_an_ellipsis(paths: ChapterPaths, cfg: Config) -> None:
    RegionsArtifact(regions=[region("r1", "x" * 70)]).save(paths.artifact("ocr.json"))
    preview = describe("ocr", paths, cfg)
    assert preview.details == ("x" * 59 + "…",)


def test_translate_counts_runs_and_pairs_source_with_candidate(paths: ChapterPaths, cfg: Config) -> None:
    RegionsArtifact(regions=[region("r1", "Hello there"), region("r2", "Good bye")]).save(
        paths.artifact("ocr.json")
    )
    CandidateRun(
        run_id="p@2026",
        profile="p",
        model="m",
        candidates=[
            Candidate(region_id="r1", text="안녕하세요"),
            Candidate(region_id="r2", text="또 만나요"),
        ],
    ).save(paths.artifact("translations") / "p.json")
    assert describe("translate", paths, cfg) == Preview(
        "translate", "1 candidate run(s)", ("Hello there → 안녕하세요", "Good bye → 또 만나요")
    )


def test_translate_details_stop_at_three_lines(paths: ChapterPaths, cfg: Config) -> None:
    RegionsArtifact(regions=[region(f"r{i}", f"s{i}") for i in range(1, 5)]).save(paths.artifact("ocr.json"))
    for name, indexes in (("first", (1, 2)), ("second", (3, 4))):
        CandidateRun(
            run_id=f"{name}@2026",
            profile=name,
            model="m",
            candidates=[Candidate(region_id=f"r{i}", text=f"t{i}") for i in indexes],
        ).save(paths.artifact("translations") / f"{name}.json")
    preview = describe("translate", paths, cfg)
    assert preview.summary == "2 candidate run(s)"
    assert preview.details == ("s1 → t1", "s2 → t2", "s3 → t3")


def test_translate_ignores_dot_prefixed_partial_files(paths: ChapterPaths, cfg: Config) -> None:
    CandidateRun(run_id="p@2026", profile="p", model="m", candidates=[]).save(
        paths.artifact("translations") / "p.json"
    )
    (paths.artifact("translations") / ".p.partial.json").write_text("{", encoding="utf-8")
    assert describe("translate", paths, cfg) == Preview("translate", "1 candidate run(s)", ())


def test_translate_without_runs_says_no_translations(paths: ChapterPaths, cfg: Config) -> None:
    paths.artifact("translations").mkdir(parents=True)
    assert describe("translate", paths, cfg) == Preview("translate", "no translations found", ())


def test_translate_with_a_broken_run_says_no_translations(paths: ChapterPaths, cfg: Config) -> None:
    paths.artifact("translations").mkdir(parents=True)
    (paths.artifact("translations") / "p.json").write_text("<not json>", encoding="utf-8")
    assert describe("translate", paths, cfg) == Preview("translate", "no translations found", ())


def test_judge_counts_final_lines_and_shows_the_first_five(paths: ChapterPaths, cfg: Config) -> None:
    FinalArtifact(
        judge_model="m",
        lines=[FinalLine(region_id=f"r{i}", text=f"line {i}", decision="pick") for i in range(1, 8)],
    ).save(paths.artifact("final.json"))
    preview = describe("judge", paths, cfg)
    assert preview.summary == "7 final line(s)"
    assert preview.details == tuple(f"line {i}" for i in range(1, 6))


def test_inpaint_and_inpaint_lama_each_read_their_own_file(paths: ChapterPaths, cfg: Config) -> None:
    InpaintArtifact(
        items=[
            InpaintItem(region_id="r1", box=BOX, method="flat"),
            InpaintItem(region_id="r2", box=BOX, method="none"),
        ]
    ).save(paths.artifact("inpaint.json"))
    InpaintArtifact(
        items=[
            InpaintItem(region_id="r1", box=BOX, method="lama"),
            InpaintItem(region_id="r2", box=BOX, method="lama"),
        ]
    ).save(paths.artifact("inpaint_lama.json"))
    assert describe("inpaint", paths, cfg) == Preview("inpaint", "2 item(s) inpainted", ())
    assert describe("inpaint_lama", paths, cfg) == Preview("inpaint_lama", "2 item(s) inpainted", ())


def test_typeset_counts_items_and_unfitted_overflow(paths: ChapterPaths, cfg: Config) -> None:
    LayoutArtifact(
        items=[
            LayoutItem(region_id="r1", font_role="dialogue", font="F", size_px=10, lines=["Hi"], box=BOX),
            LayoutItem(
                region_id="r2", font_role="free", font="F", size_px=10, lines=["Oh"], box=BOX, overflow=True
            ),
        ]
    ).save(paths.artifact("layout.json"))
    assert describe("typeset", paths, cfg) == Preview("typeset", "2 layout item(s), 1 unfitted", ())


def test_export_counts_the_written_files(paths: ChapterPaths, cfg: Config) -> None:
    ExportArtifact(
        quality=95,
        subsampling="444",
        files=[ExportFile(name="0001.jpg", slice_index=0, width=800, height=1000, bytes=1)],
    ).save(paths.artifact("export.json"))
    assert describe("export", paths, cfg) == Preview("export", f"1 file(s) written to {paths.output_dir}", ())


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ("ingest", "no ingest.json found"),
        ("slice", "no slices.json found"),
        ("detect", "no regions.json found"),
        ("ocr", "no ocr.json found"),
        ("translate", "no translations found"),
        ("judge", "no final.json found"),
        ("inpaint", "no inpaint.json found"),
        ("inpaint_lama", "no inpaint_lama.json found"),
        ("typeset", "no layout.json found"),
        ("export", "no export.json found"),
    ],
)
def test_a_missing_artifact_gives_a_summary_and_never_raises(
    paths: ChapterPaths, cfg: Config, stage: str, message: str
) -> None:
    assert describe(stage, paths, cfg) == Preview(stage, message, ())


_STAGE_OF_ARTIFACT = {
    "ingest.json": "ingest",
    "slices.json": "slice",
    "regions.json": "detect",
    "ocr.json": "ocr",
    "final.json": "judge",
    "inpaint.json": "inpaint",
    "inpaint_lama.json": "inpaint_lama",
    "layout.json": "typeset",
    "export.json": "export",
}


@pytest.mark.parametrize(
    "artifact_name",
    [
        "ingest.json",
        "slices.json",
        "regions.json",
        "ocr.json",
        "final.json",
        "inpaint.json",
        "inpaint_lama.json",
        "layout.json",
        "export.json",
    ],
)
def test_an_invalid_artifact_gives_the_same_summary(
    paths: ChapterPaths, cfg: Config, artifact_name: str
) -> None:
    paths.artifact(artifact_name).parent.mkdir(parents=True, exist_ok=True)
    paths.artifact(artifact_name).write_text("<not json>", encoding="utf-8")
    preview = describe(_STAGE_OF_ARTIFACT[artifact_name], paths, cfg)
    assert preview.summary == f"no {artifact_name} found"
    assert preview.details == ()
