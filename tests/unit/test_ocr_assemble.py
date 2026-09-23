"""Assembling readings into regions: field copying, box rounding, dropped lines (card C4a),
text-colour sampling from the strip (director fix, no card -- see docs/CHECKPOINT.md)."""

from __future__ import annotations

import torch

from omniscan.core.schemas import RGB, BBox, OcrLine, Region
from omniscan.ocr.assemble import build_ocr_regions
from omniscan.ocr.lines import LineBox

ENGINE = "korean_PP-OCRv5_mobile_rec_safetensors"


def region(rid: str, box: tuple[int, int, int, int], *, text_color: RGB | None = (10, 20, 30)) -> Region:
    return Region(
        id=rid,
        slice_index=2,
        kind="bubble_text",
        bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
        bubble_bbox=BBox(x0=box[0] - 10, y0=box[1] - 10, x1=box[2] + 10, y1=box[3] + 10),
        polygon=[(0, 0), (5, 5)],
        reading_order=3,
        text_color=text_color,
        stroke_color=(200, 200, 200),
        mask_ref="masks/0.npz",
    )


def blank_strip(width: int, height: int, color: tuple[int, int, int]) -> torch.Tensor:
    """A uint8 [3, height, width] strip filled with one flat colour."""
    strip = torch.empty((3, height, width), dtype=torch.uint8)
    for channel, value in enumerate(color):
        strip[channel] = value
    return strip


def paint(strip: torch.Tensor, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    """Paint a rectangle of `strip` in place (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    for channel, value in enumerate(color):
        strip[channel, y0:y1, x0:x1] = value


def test_assemble_fills_lines_text_and_confidence() -> None:
    input_region = region("r0001", (5, 0, 200, 100))
    first = LineBox(box=(9.2, 4.6, 110.1, 24.9), score=0.99)
    second = LineBox(box=(9.2, 30.0, 110.0, 50.2), score=0.90)

    out = build_ocr_regions(
        [input_region],
        {"r0001": [first, second]},
        {("r0001", 0): ("안녕", 0.99), ("r0001", 1): ("하세요", 0.90)},
        blank_strip(200, 100, (0, 0, 0)),
        engine=ENGINE,
        lang="ko",
        strip_width=200,
        strip_height=100,
    )

    assert len(out) == 1
    built = out[0]
    assert built.id == "r0001" and built.slice_index == 2 and built.kind == "bubble_text"
    assert built.bbox == input_region.bbox and built.bubble_bbox == input_region.bubble_bbox
    assert built.polygon == [(0, 0), (5, 5)] and built.reading_order == 3
    assert built.text_color == (10, 20, 30) and built.stroke_color == (200, 200, 200)
    assert built.mask_ref == "masks/0.npz"
    assert built.text == "안녕\n하세요"
    assert built.confidence == 0.9  # the minimum line score
    assert built.lang == "ko" and built.orientation == "h" and built.ocr_alt is None
    assert [line.text for line in built.lines] == ["안녕", "하세요"]
    assert [line.score for line in built.lines] == [0.99, 0.90]
    assert all(line.engine == ENGINE and line.polygon is None for line in built.lines)
    assert [line.bbox for line in built.lines] == [
        BBox(x0=9, y0=4, x1=111, y1=25),
        BBox(x0=9, y0=30, x1=110, y1=51),
    ]  # rounded outwards
    # the input region is untouched
    assert input_region.text == "" and input_region.lines == [] and input_region.confidence == 0.0


def test_assemble_clamps_line_boxes_to_the_strip() -> None:
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 20))],
        {"r0001": [LineBox(box=(-4.2, -3.7, 500.9, 800.1), score=0.9)]},
        {("r0001", 0): ("텍스트", 0.9)},
        blank_strip(200, 100, (0, 0, 0)),
        engine=ENGINE,
        lang="ko",
        strip_width=200,
        strip_height=100,
    )
    assert out[0].lines[0].bbox == BBox(x0=0, y0=0, x1=200, y1=100)


def test_assemble_drops_empty_readings_and_keeps_the_order() -> None:
    first = region("r0001", (0, 0, 60, 20))
    second = region("r0002", (0, 30, 60, 50))
    blank = LineBox(box=(1.0, 2.0, 40.0, 18.0), score=0.99)

    out = build_ocr_regions(
        [first, second],
        {"r0001": [blank], "r0002": []},
        {("r0001", 0): ("  ", 0.99)},
        blank_strip(200, 100, (0, 0, 0)),
        engine=ENGINE,
        lang="ko",
        strip_width=200,
        strip_height=100,
    )

    assert [r.id for r in out] == ["r0001", "r0002"]  # input order
    assert out[0].lines == [] and out[0].text == "" and out[0].confidence == 0.0
    assert out[1].lines == [] and out[1].text == "" and out[1].confidence == 0.0


def test_assemble_confidence_is_the_minimum_over_surviving_lines() -> None:
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 60))],
        {
            "r0001": [
                LineBox(box=(1.0, 2.0, 40.0, 18.0), score=0.9),
                LineBox(box=(1.0, 22.0, 40.0, 38.0), score=0.1),
            ]
        },
        {("r0001", 0): ("안녕", 0.99), ("r0001", 1): ("잘가", 0.42)},
        blank_strip(200, 100, (0, 0, 0)),
        engine=ENGINE,
        lang="ko",
        strip_width=200,
        strip_height=100,
    )
    assert [line.text for line in out[0].lines] == ["안녕", "잘가"]
    assert out[0].confidence == 0.42
    assert isinstance(out[0].lines[0], OcrLine)


def test_text_color_is_sampled_from_the_minority_cluster_bright_on_dark() -> None:
    strip = blank_strip(60, 60, (30, 40, 70))
    paint(strip, (0, 0, 20, 20), (30, 40, 70))
    paint(strip, (5, 5, 15, 15), (230, 230, 230))  # 100 of 400 px: a clear minority
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 60), text_color=None)],
        {"r0001": [LineBox(box=(0.0, 0.0, 20.0, 20.0), score=0.9)]},
        {("r0001", 0): ("x", 0.9)},
        strip,
        engine=ENGINE,
        lang="ko",
        strip_width=60,
        strip_height=60,
    )
    assert out[0].text_color == (230, 230, 230)


def test_text_color_sampling_is_not_reversed_for_dark_ink_on_light_background() -> None:
    """The minority cluster is the ink regardless of whether it's the brighter or darker one."""
    strip = blank_strip(60, 60, (230, 230, 230))
    paint(strip, (5, 5, 15, 15), (10, 10, 10))
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 60), text_color=None)],
        {"r0001": [LineBox(box=(0.0, 0.0, 20.0, 20.0), score=0.9)]},
        {("r0001", 0): ("x", 0.9)},
        strip,
        engine=ENGINE,
        lang="ko",
        strip_width=60,
        strip_height=60,
    )
    assert out[0].text_color == (10, 10, 10)


def test_text_color_keeps_a_pre_set_value_even_though_the_strip_disagrees() -> None:
    strip = blank_strip(60, 60, (30, 40, 70))
    paint(strip, (5, 5, 15, 15), (230, 230, 230))
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 60), text_color=(1, 2, 3))],
        {"r0001": [LineBox(box=(0.0, 0.0, 20.0, 20.0), score=0.9)]},
        {("r0001", 0): ("x", 0.9)},
        strip,
        engine=ENGINE,
        lang="ko",
        strip_width=60,
        strip_height=60,
    )
    assert out[0].text_color == (1, 2, 3)


def test_text_color_is_none_for_a_uniform_crop() -> None:
    """Nothing to separate: the whole line box is one flat colour."""
    strip = blank_strip(60, 60, (30, 40, 70))
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 60), text_color=None)],
        {"r0001": [LineBox(box=(0.0, 0.0, 20.0, 20.0), score=0.9)]},
        {("r0001", 0): ("x", 0.9)},
        strip,
        engine=ENGINE,
        lang="ko",
        strip_width=60,
        strip_height=60,
    )
    assert out[0].text_color is None


def test_text_color_is_none_below_the_minimum_sample_size() -> None:
    strip = blank_strip(60, 60, (30, 40, 70))
    paint(strip, (0, 0, 3, 3), (230, 230, 230))  # 9 px total, under _MIN_INK_PX
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 60), text_color=None)],
        {"r0001": [LineBox(box=(0.0, 0.0, 3.0, 3.0), score=0.9)]},
        {("r0001", 0): ("x", 0.9)},
        strip,
        engine=ENGINE,
        lang="ko",
        strip_width=60,
        strip_height=60,
    )
    assert out[0].text_color is None


def test_text_color_combines_pixels_from_every_line_of_the_region() -> None:
    strip = blank_strip(60, 60, (30, 40, 70))
    paint(strip, (5, 5, 15, 15), (230, 230, 230))
    paint(strip, (35, 35, 45, 45), (230, 230, 230))
    out = build_ocr_regions(
        [region("r0001", (0, 0, 60, 60), text_color=None)],
        {
            "r0001": [
                LineBox(box=(0.0, 0.0, 20.0, 20.0), score=0.9),
                LineBox(box=(30.0, 30.0, 50.0, 50.0), score=0.9),
            ]
        },
        {("r0001", 0): ("x", 0.9), ("r0001", 1): ("y", 0.9)},
        strip,
        engine=ENGINE,
        lang="ko",
        strip_width=60,
        strip_height=60,
    )
    assert out[0].text_color == (230, 230, 230)
