"""Tests for omniscan.eval.score."""

from __future__ import annotations

import json

from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    IngestArtifact,
    Region,
    RegionsArtifact,
    SourceFile,
)
from omniscan.eval.score import BoxResult, EvalReport, _ocr_hypothesis, score_chapter
from omniscan.eval.truth import TruthBox, TruthStats

STATS = TruthStats(pages=1, pages_without_truth=0, dropped=0)


def truth_box(page: int, x0: int, y0: int, x1: int, y1: int, *lines: str) -> TruthBox:
    return TruthBox(page=page, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), lines=tuple(lines))


def region(rid: str, x0: int, y0: int, x1: int, y1: int, text: str, kind: str = "bubble_text") -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), text=text)  # type: ignore[arg-type]


def s1_ingest() -> IngestArtifact:
    return IngestArtifact(
        series="S",
        chapter="C",
        strip_width=1000,
        strip_height=3000,
        files=[SourceFile(index=0, name="01.jpg", sha256="x", width=1000, height=3000, y0=0, y1=3000)],
    )


def s1_truth() -> list[TruthBox]:
    return [
        truth_box(1, 100, 100, 500, 300, "안녕하세요"),
        truth_box(1, 600, 100, 900, 300, "잘 가"),
        truth_box(1, 100, 1000, 500, 1200, "미안해요"),
    ]


def s1_regions() -> RegionsArtifact:
    return RegionsArtifact(
        regions=[
            region("r0001", 150, 150, 450, 250, "안녕하세여"),
            region("r0002", 620, 120, 880, 280, "잘가"),
            region("r0003", 700, 2000, 800, 2100, "junk"),
        ]
    )


def score(
    truth: list[TruthBox],
    regions: RegionsArtifact,
    *,
    final: FinalArtifact | None = None,
) -> EvalReport:
    """score_chapter for the S1 fixture."""
    return score_chapter("S", "C", s1_ingest(), regions, final, truth, {}, STATS)


def test_scenario_s1_detection_and_ocr() -> None:
    report = score(s1_truth(), s1_regions())
    assert report.series == "S"
    assert report.chapter == "C"
    assert report.truth_boxes == 3
    assert report.ignored_boxes == 0
    assert report.dropped_boxes == 0
    assert report.pages == 1
    assert report.pages_without_truth == 0
    assert report.regions == 3
    assert report.assigned_regions == 2
    assert report.detected_boxes == 2
    assert report.recall == 2 / 3
    assert report.precision == 2 / 3
    assert report.cer_boxes == 2
    assert report.cer_macro == 0.1
    assert report.cer_micro == 1 / 7
    assert report.chrf_mean is None
    assert report.chrf_pages == 0
    assert [(box.page, box.bbox, box.detected) for box in report.missed] == [
        (1, BBox(x0=100, y0=1000, x1=500, y1=1200), False)
    ]
    assert report.worst_cer[0].bbox == BBox(x0=100, y0=100, x1=500, y1=300)
    assert report.worst_cer[0].cer == 0.2
    assert report.worst_cer[0].read == "안녕하세여"
    assert report.worst_cer[1].cer == 0.0


def test_watermark_regions_are_ignored() -> None:
    regions = s1_regions()
    regions.regions.append(region("r0004", 200, 180, 400, 220, "wm", kind="watermark"))
    report = score(s1_truth(), regions)
    assert report.regions == 3
    assert report.assigned_regions == 2
    assert report.precision == 2 / 3


def test_punctuation_only_truth_boxes_are_ignored() -> None:
    report = score([*s1_truth(), truth_box(1, 800, 100, 900, 200, "...")], s1_regions())
    assert report.ignored_boxes == 1
    assert report.truth_boxes == 3
    assert report.recall == 2 / 3
    assert report.detected_boxes == 2
    assert [box.text for box in report.missed] == ["미안해요"]


def test_nested_boxes_assign_to_the_smallest_container() -> None:
    truth = [truth_box(1, 0, 0, 1000, 1000, "outer"), truth_box(1, 100, 100, 300, 300, "inner")]
    regions = RegionsArtifact(
        regions=[
            region("r0001", 150, 150, 250, 250, "in"),  # center (200, 200)
            region("r0002", 700, 700, 900, 900, "out"),  # center (800, 800)
            region("r0003", 950, 450, 1000, 550, "border"),  # center (975, 500), outer only
        ]
    )
    report = score(truth, regions)
    assert report.assigned_regions == 3
    assert report.detected_boxes == 2
    inner_read = next(r.read for r in report.worst_cer if r.text == "inner")
    outer_read = next(r.read for r in report.worst_cer if r.text == "outer")
    assert inner_read == "in"
    assert outer_read == "border out"


def test_border_center_counts_as_inside() -> None:
    truth = [truth_box(1, 100, 100, 300, 300, "box")]
    regions = RegionsArtifact(regions=[region("r0001", 250, 250, 350, 350, "edge")])
    report = score(truth, regions)  # center (300, 300) is exactly the box corner
    assert report.assigned_regions == 1
    assert report.recall == 1.0


def test_regions_in_one_box_join_in_reading_order() -> None:
    truth = [truth_box(1, 0, 0, 1000, 1000, "하나 둘")]
    regions = RegionsArtifact(
        regions=[region("r0001", 400, 200, 500, 260, "둘"), region("r0002", 100, 100, 200, 160, "하나")]
    )
    report = score(truth, regions)
    assert report.cer_boxes == 1
    assert report.worst_cer[0].read == "하나 둘"
    assert report.cer_macro == 0.0


def test_translation_chrf_of_final_lines() -> None:
    final = FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id="r0001", text="Hello, world!", decision="pick"),
            FinalLine(region_id="r0002", text="Goodbye", decision="pick"),
        ],
    )
    report = score_chapter(
        "S", "C", s1_ingest(), s1_regions(), final, s1_truth(), {1: "hello world goodbye"}, STATS
    )
    assert report.chrf_mean == 1.0
    assert report.chrf_pages == 1


def test_translation_skips_pages_with_empty_reference() -> None:
    report = score_chapter(
        "S", "C", s1_ingest(), s1_regions(), None, s1_truth(), {1: "hello world goodbye"}, STATS
    )
    assert report.chrf_mean is None
    assert report.chrf_pages == 0
    report = score_chapter("S", "C", s1_ingest(), s1_regions(), None, s1_truth(), {1: ""}, STATS)
    assert report.chrf_mean is None
    assert report.chrf_pages == 0


def test_translation_page_without_regions_scores_zero() -> None:
    two_pages = IngestArtifact(
        series="S",
        chapter="C",
        strip_width=1000,
        strip_height=6000,
        files=[
            SourceFile(index=0, name="01.jpg", sha256="x", width=1000, height=3000, y0=0, y1=3000),
            SourceFile(index=1, name="02.jpg", sha256="x", width=1000, height=3000, y0=3000, y1=6000),
        ],
    )
    final = FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id="r0001", text="Hello, world!", decision="pick"),
            FinalLine(region_id="r0002", text="Goodbye", decision="pick"),
        ],
    )
    report = score_chapter(
        "S",
        "C",
        two_pages,
        s1_regions(),
        final,
        s1_truth(),
        {1: "hello world goodbye", 2: "reference text"},
        TruthStats(pages=2, pages_without_truth=0, dropped=0),
    )
    assert report.chrf_pages == 2
    assert report.chrf_mean == 0.5


def test_to_json_shape() -> None:
    report = score(s1_truth(), s1_regions())
    data = json.loads(report.to_json())
    assert data["series"] == "S"
    assert data["recall"] == 2 / 3
    assert data["recall_chars"] == 7 / 11
    assert data["approx_boxes"] == 0
    assert data["ocr_chrf_pages"] == 1
    assert 0.0 <= data["ocr_chrf_mean"] <= 1.0
    assert data["missed"][0]["bbox"] == [100, 1000, 500, 1200]
    assert data["worst_cer"][0]["cer"] == 0.2
    empty = json.loads(score([], s1_regions()).to_json())
    assert empty["truth_boxes"] == 0
    assert empty["recall"] is None
    assert empty["recall_chars"] is None
    assert empty["precision"] is None
    assert empty["ocr_chrf_mean"] is None
    assert empty["ocr_chrf_pages"] == 0
    assert empty["approx_boxes"] == 0
    assert empty["cer_macro"] is None
    assert empty["missed"] == []


def test_missed_is_limited_and_sorted_by_chars_then_area() -> None:
    truth = [truth_box(1, 0, i * 300, 10 * (15 - i), i * 300 + 50, f"box{i}") for i in range(1, 15)]
    report = score(truth, s1_regions())
    assert len(report.missed) == 10
    # box10..14 have 5 usable characters each (largest areas first), then the 4-character boxes
    assert [box.text for box in report.missed] == [
        "box10",
        "box11",
        "box12",
        "box13",
        "box14",
        "box1",
        "box2",
        "box3",
        "box4",
        "box5",
    ]
    assert all(isinstance(r, BoxResult) and not r.detected for r in report.missed)


def test_missed_ties_break_by_area_then_page_then_position() -> None:
    truth = [
        truth_box(2, 500, 100, 600, 200, "같은글자열"),
        truth_box(1, 500, 100, 600, 200, "같은글자열"),
        truth_box(1, 0, 100, 100, 200, "같은글자열"),
        truth_box(1, 0, 400, 300, 600, "같은글자열"),
    ]
    report = score(truth, RegionsArtifact(regions=[]))
    assert [(box.page, box.bbox.x0, box.bbox.y0) for box in report.missed] == [
        (1, 0, 400),  # same text, largest area first
        (1, 0, 100),  # then lowest (page, y0, x0)
        (1, 500, 100),
        (2, 500, 100),
    ]


# ---------------------------------------------------------------- character-weighted recall


def test_recall_chars_weights_by_text_length() -> None:
    truth = [
        truth_box(1, 0, 0, 500, 200, "가나다라마바사아자차"),  # 10 usable characters
        truth_box(1, 600, 0, 900, 200, "안녕"),  # 2
        truth_box(1, 0, 400, 500, 600, "어"),  # 1
    ]
    regions = RegionsArtifact(regions=[region("r0001", 100, 50, 400, 150, "가나다라마바사아자차")])
    report = score(truth, regions)
    assert report.recall == 1 / 3
    assert report.recall_chars == 10 / 13
    assert report.precision == 1.0


def test_no_usable_boxes_makes_recall_chars_and_precision_none() -> None:
    report = score([truth_box(1, 0, 0, 100, 100, "...")], s1_regions())
    assert report.truth_boxes == 0
    assert report.ignored_boxes == 1
    assert report.recall is None
    assert report.recall_chars is None
    assert report.precision is None  # regions exist but there is no truth to assign them to


def test_approx_boxes_are_counted() -> None:
    truth = [
        truth_box(1, 0, 0, 500, 200, "안녕"),
        TruthBox(page=1, bbox=BBox(x0=600, y0=0, x1=900, y1=200), lines=("미안",), approx=True),
    ]
    report = score(truth, RegionsArtifact(regions=[]))
    assert report.truth_boxes == 2
    assert report.approx_boxes == 1


# ---------------------------------------------------------------- page-level OCR chrF


def _three_page_ingest() -> IngestArtifact:
    return IngestArtifact(
        series="S",
        chapter="C",
        strip_width=1000,
        strip_height=9000,
        files=[
            SourceFile(index=0, name="01.jpg", sha256="x", width=1000, height=3000, y0=0, y1=3000),
            SourceFile(index=1, name="02.jpg", sha256="x", width=1000, height=3000, y0=3000, y1=6000),
            SourceFile(index=2, name="03.jpg", sha256="x", width=1000, height=3000, y0=6000, y1=9000),
        ],
    )


def test_ocr_page_chrf_scores_every_truth_page() -> None:
    truth = [truth_box(1, 100, 100, 500, 300, "안녕하세요"), truth_box(2, 100, 3100, 500, 3300, "잘 가")]
    regions = RegionsArtifact(regions=[region("r0001", 150, 150, 450, 250, "안녕하세요")])
    report = score_chapter(
        "S",
        "C",
        _three_page_ingest(),
        regions,
        None,
        truth,
        {},
        TruthStats(pages=3, pages_without_truth=0, dropped=0),
    )
    assert report.ocr_chrf_pages == 2  # page 3 has a region but no truth: not counted
    assert report.ocr_chrf_mean == 0.5  # page 1 exact (1.0), page 2 truth without region (0.0)


def test_ocr_page_chrf_ignores_watermark_regions() -> None:
    truth = [truth_box(1, 100, 100, 500, 300, "안녕하세요")]
    regions = RegionsArtifact(
        regions=[
            region("r0001", 150, 150, 450, 250, "안녕하세요"),
            region("r0002", 150, 400, 450, 500, "watermark junk", kind="watermark"),
        ]
    )
    report = score_chapter("S", "C", s1_ingest(), regions, None, truth, {}, STATS)
    assert report.ocr_chrf_pages == 1
    assert report.ocr_chrf_mean == 1.0


def test_ocr_hypothesis_joins_regions_in_reading_order() -> None:
    regions = [region("r0001", 400, 200, 500, 260, "둘"), region("r0002", 100, 100, 200, 160, "하나")]
    assert _ocr_hypothesis(regions, 0, 3000) == "하나 둘"


def test_ocr_page_chrf_depends_on_the_hypothesis_order() -> None:
    regions = RegionsArtifact(
        regions=[region("r0001", 400, 200, 500, 260, "둘"), region("r0002", 100, 100, 200, 160, "하나")]
    )
    right = score_chapter(
        "S", "C", s1_ingest(), regions, None, [truth_box(1, 0, 0, 1000, 1000, "하나 둘")], {}, STATS
    )
    swapped = score_chapter(
        "S", "C", s1_ingest(), regions, None, [truth_box(1, 0, 0, 1000, 1000, "둘 하나")], {}, STATS
    )
    assert right.ocr_chrf_mean == 1.0
    assert swapped.ocr_chrf_mean is not None
    assert 0.0 < swapped.ocr_chrf_mean < 1.0


# ---------------------------------------------------------------- gaps found by the director's mutation run


def test_precision_counts_every_region_assigned_to_a_box() -> None:
    truth = [truth_box(1, 0, 0, 1000, 1000, "하나 둘")]
    regions = RegionsArtifact(
        regions=[region("r0001", 100, 100, 200, 160, "하나"), region("r0002", 400, 200, 500, 260, "둘")]
    )
    report = score(truth, regions)
    assert report.detected_boxes == 1
    assert report.assigned_regions == 2
    assert report.precision == 1.0  # both regions lie in the one box; precision is about regions, not boxes


def test_a_region_centred_on_the_page_end_belongs_to_the_next_page() -> None:
    final = FinalArtifact(
        judge_model="judge", lines=[FinalLine(region_id="r0001", text="hello", decision="pick")]
    )
    regions = RegionsArtifact(regions=[region("r0001", 0, 2990, 100, 3010, "안녕")])  # center row 3000 = y1
    report = score_chapter("S", "C", s1_ingest(), regions, final, [], {1: "hello"}, STATS)
    assert report.chrf_pages == 1
    assert report.chrf_mean == 0.0  # the region is not part of page 1 (rows [0, 3000))
