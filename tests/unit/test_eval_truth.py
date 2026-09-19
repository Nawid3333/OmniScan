"""Tests for omniscan.eval.truth."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.schemas import BBox, IngestArtifact, SourceFile
from omniscan.eval.truth import (
    TruthBox,
    TruthStats,
    _page_number,
    load_english_pages,
    load_truth,
    parse_svg,
)

SVG2 = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="3000">'
    '<g><flowRoot transform="matrix(1,0,0,1,100,200)"><flowRegion>'
    '<rect x="0" y="0" width="400" height="300"/></flowRegion>'
    "<flowPara>첫째 줄</flowPara><flowPara>둘째 줄</flowPara></flowRoot></g>"
    '<flowRoot transform="translate(1000,1500) scale(0.5)"><flowRegion>'
    '<rect x="10" y="20" width="200" height="100"/></flowRegion>'
    "<flowPara>하나</flowPara></flowRoot>"
    '<flowRoot transform="translate(300,100) rotate(90)"><flowRegion>'
    '<rect x="0" y="0" width="100" height="50"/></flowRegion>'
    "<flowPara>셋</flowPara></flowRoot></svg>"
)

SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'
SODI_NS = 'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"'


def _svg(*body: str, width: int = 2000, height: int = 3000) -> str:
    return f'<svg {SVG_NS} {SODI_NS} width="{width}" height="{height}">{"".join(body)}</svg>'


def _text_golden() -> str:
    """Card golden 1: middle anchor, two role-line tspans, element transform."""
    return _svg(
        '<text style="font-size:20px;text-anchor:middle" x="1000" y="500" transform="translate(100,50)">'
        '<tspan sodipodi:role="line" x="1000" y="500" style="font-size:40px">안녕하세요</tspan>'
        '<tspan sodipodi:role="line" x="1000" y="540" style="font-size:40px">ab</tspan></text>'
    )


def test_parses_transforms_and_geometry() -> None:
    boxes, dropped = parse_svg(SVG2, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert [box.bbox for box in boxes] == [
        BBox(x0=50, y0=80, x1=250, y1=230),
        BBox(x0=502, y0=735, x1=553, y1=760),
        BBox(x0=125, y0=30, x1=150, y1=80),
    ]
    assert [box.lines for box in boxes] == [("첫째 줄", "둘째 줄"), ("하나",), ("셋",)]
    assert boxes[0].text == "첫째 줄 둘째 줄"


def test_maps_to_strip_space_with_offset_and_scale() -> None:
    boxes, dropped = parse_svg(SVG2, file_width=1000, file_height=1480, file_y0=3000, scale=2.0)
    assert dropped == 0
    assert [box.bbox for box in boxes] == [
        BBox(x0=100, y0=3160, x1=500, y1=3460),
        BBox(x0=1005, y0=4470, x1=1105, y1=4520),
        BBox(x0=250, y0=3060, x1=300, y1=3160),
    ]


def test_drops_box_completely_outside_the_page() -> None:
    svg = SVG2.replace('transform="matrix(1,0,0,1,100,200)"', 'transform="matrix(1,0,0,1,-2000,200)"')
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 1
    assert [box.lines for box in boxes] == [("하나",), ("셋",)]


def test_group_transform_composes_with_the_flow_roots() -> None:
    svg = SVG2.replace("<g><flowRoot", '<g transform="translate(100,0)"><flowRoot')
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert boxes[0].bbox == BBox(x0=100, y0=80, x1=300, y1=230)


def test_unknown_transform_falls_back_to_identity() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">'
        '<flowRoot transform="skewX(30)"><flowRegion>'
        '<rect x="10" y="20" width="100" height="50"/></flowRegion>'
        "<flowPara>글자</flowPara></flowRoot></svg>"
    )
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1000, file_y0=0, scale=1.0)
    assert dropped == 0
    assert boxes[0].bbox == BBox(x0=10, y0=20, x1=110, y1=70)


def test_broken_transform_does_not_crash() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">'
        '<flowRoot transform="nonsense"><flowRegion>'
        '<rect x="10" y="20" width="100" height="50"/></flowRegion>'
        "<flowPara>글자</flowPara></flowRoot></svg>"
    )
    boxes, _ = parse_svg(svg, file_width=1000, file_height=1000, file_y0=0, scale=1.0)
    assert boxes[0].bbox == BBox(x0=10, y0=20, x1=110, y1=70)


def test_svg_without_flow_roots() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">'
        '<rect x="0" y="0" width="10" height="10"/></svg>'
    )
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1000, file_y0=0, scale=1.0)
    assert boxes == []
    assert dropped == 0


def test_skips_empty_flow_roots_without_counting_them() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">'
        '<flowRoot><flowRegion><rect x="0" y="0" width="100" height="50"/></flowRegion>'
        "<flowPara></flowPara></flowRoot>"
        "<flowRoot><flowPara>글자</flowPara></flowRoot></svg>"
    )
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1000, file_y0=0, scale=1.0)
    assert boxes == []
    assert dropped == 0


def _source(index: int, name: str, *, y0: int, filtered: bool = False) -> SourceFile:
    return SourceFile(
        index=index,
        name=name,
        sha256="x",
        width=1000,
        height=1480,
        y0=y0,
        y1=y0 + 1480,
        scale=1.0,
        filtered=filtered,
    )


def _ingest() -> IngestArtifact:
    return IngestArtifact(
        series="PepperCarrotKR",
        chapter="Episode 06",
        strip_width=1000,
        strip_height=7400,
        files=[
            _source(0, "00.jpg", y0=0),
            _source(1, "01.jpg", y0=1480),
            _source(2, "03.jpg", y0=2960),
            _source(3, "09.jpg", y0=4440, filtered=True),
            _source(4, "cover.jpg", y0=5920),
        ],
    )


@pytest.fixture
def check_dir(tmp_path: Path) -> Path:
    """.../Episode 06/truth with the svg2 pages for kr and a joined-lines page for en."""
    truth = tmp_path / "translated-check" / "PepperCarrotKR" / "Episode 06" / "truth"
    kr = truth / "kr"
    kr.mkdir(parents=True)
    (kr / "E06P01.svg").write_text(SVG2, encoding="utf-8")
    (kr / "E06P03.svg").write_text(SVG2, encoding="utf-8")
    en = truth / "en"
    en.mkdir()
    (en / "E06P01.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1480">'
        '<flowRoot><flowRegion><rect x="0" y="0" width="100" height="50"/></flowRegion>'
        "<flowPara>Hello</flowPara><flowPara>there.</flowPara></flowRoot>"
        '<flowRoot><flowRegion><rect x="0" y="100" width="100" height="50"/></flowRegion>'
        "<flowPara>Bye</flowPara></flowRoot></svg>",
        encoding="utf-8",
    )
    return truth


def test_load_truth_maps_pages_to_strip_offsets(check_dir: Path) -> None:
    boxes, stats = load_truth(check_dir, "kr", _ingest())
    assert stats.pages == 3
    assert stats.pages_without_truth == 1
    assert stats.dropped == 0
    assert [(box.page, box.bbox) for box in boxes] == [
        (1, BBox(x0=50, y0=1560, x1=250, y1=1710)),
        (1, BBox(x0=502, y0=2215, x1=553, y1=2240)),
        (1, BBox(x0=125, y0=1510, x1=150, y1=1560)),
        (3, BBox(x0=50, y0=3040, x1=250, y1=3190)),
        (3, BBox(x0=502, y0=3695, x1=553, y1=3720)),
        (3, BBox(x0=125, y0=2990, x1=150, y1=3040)),
    ]


def test_load_truth_counts_pages_without_truth(check_dir: Path) -> None:
    boxes, stats = load_truth(check_dir, "en", _ingest())
    assert stats == TruthStats(pages=3, pages_without_truth=2, dropped=0)
    assert [(box.page, box.bbox, box.lines) for box in boxes] == [
        (1, BBox(x0=0, y0=1480, x1=100, y1=1530), ("Hello", "there.")),
        (1, BBox(x0=0, y0=1580, x1=100, y1=1630), ("Bye",)),
    ]


def test_load_truth_counts_every_page_when_language_is_absent(check_dir: Path) -> None:
    boxes, stats = load_truth(check_dir, "ja", _ingest())
    assert boxes == []
    assert stats == TruthStats(pages=3, pages_without_truth=3, dropped=0)


def test_load_english_pages_joins_box_lines(check_dir: Path) -> None:
    assert load_english_pages(check_dir, _ingest()) == {1: "Hello there. Bye"}


def test_truth_box_text_joins_lines() -> None:
    box = TruthBox(page=1, bbox=BBox(x0=0, y0=0, x1=1, y1=1), lines=("a", "b"))
    assert box.text == "a b"


def test_load_truth_of_empty_ingest(check_dir: Path) -> None:
    ingest = IngestArtifact(series="S", chapter="C", strip_width=1000, strip_height=0, files=[])
    boxes, stats = load_truth(check_dir, "kr", ingest)
    assert boxes == []
    assert stats == TruthStats(pages=0, pages_without_truth=0, dropped=0)


@pytest.mark.parametrize(
    "name, expected", [("01.jpg", 1), ("001.jpg", 1), ("10.jpg", 10), ("cover.jpg", None)]
)
def test_page_number(name: str, expected: int | None) -> None:
    assert _page_number(name) == expected


# ---------------------------------------------------------------- <text> elements (approximate boxes)


def test_text_element_golden_middle_anchor() -> None:
    boxes, dropped = parse_svg(_text_golden(), file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert len(boxes) == 1
    assert boxes[0].bbox == BBox(x0=500, y0=235, x1=600, y1=280)
    assert boxes[0].lines == ("안녕하세요", "ab")
    assert boxes[0].approx is True


def test_text_element_start_anchor() -> None:
    svg = _svg(
        '<text x="100" y="200" style="font-size:30px">'
        '<tspan sodipodi:role="line" x="100" y="200">가나</tspan></text>'
    )
    boxes, _ = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert [(box.bbox, box.lines, box.approx) for box in boxes] == [
        (BBox(x0=50, y0=65, x1=80, y1=84), ("가나",), True)
    ]


def test_text_element_end_anchor() -> None:
    svg = _svg(
        '<text style="text-anchor:end" x="300" y="100">'
        '<tspan sodipodi:role="line" x="300" y="100" style="font-size:10px">abcd</tspan></text>'
    )
    boxes, _ = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert boxes[0].bbox == BBox(x0=140, y0=25, x1=150, y1=32)


def test_text_anchor_on_the_tspan_only() -> None:
    svg = _svg(
        '<text x="1000" y="500" style="font-size:20px">'
        '<tspan sodipodi:role="line" x="1000" y="500" style="font-size:40px;text-anchor:end">abcd</tspan></text>'
    )
    boxes, _ = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert boxes[0].bbox == BBox(x0=460, y0=210, x1=500, y1=235)  # width 80: 4 half-width glyphs at 40 px


def test_tspan_without_xy_falls_back_to_the_element() -> None:
    svg = _svg('<text x="100" y="300" style="font-size:20px"><tspan sodipodi:role="line">가나</tspan></text>')
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert boxes[0].bbox == BBox(x0=50, y0=120, x1=70, y1=133)


def test_font_size_without_px() -> None:
    svg = _svg(
        '<text x="200" y="300" style="font-size:24">'
        '<tspan sodipodi:role="line" x="200" y="300">ab</tspan></text>'
    )
    boxes, _ = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert boxes[0].bbox == BBox(x0=100, y0=118, x1=112, y1=133)  # width 24: 2 half-width glyphs at 24 px


def test_list_valued_x_uses_the_first_number() -> None:
    svg = _svg(
        '<text x="10 20 30" y="300" style="font-size:20px">'
        '<tspan sodipodi:role="line" x="10 20 30" y="300">가</tspan></text>'
    )
    boxes, _ = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert boxes[0].bbox == BBox(x0=5, y0=120, x1=15, y1=133)


def test_group_transform_shifts_a_text_box() -> None:
    svg = _svg(
        '<g transform="translate(100,0)">'
        '<text x="1000" y="500" style="font-size:20px">'
        '<tspan sodipodi:role="line" x="1000" y="500" style="font-size:40px">가</tspan></text></g>'
    )
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert boxes[0].bbox == BBox(x0=550, y0=210, x1=570, y1=235)


def test_text_without_tspans_yields_one_box() -> None:
    svg = _svg('<text x="10" y="50" style="font-size:20px">가</text>', width=1000, height=1000)
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert [(box.bbox, box.lines, box.approx) for box in boxes] == [
        (BBox(x0=10, y0=30, x1=30, y1=55), ("가",), True)
    ]


def test_text_with_only_empty_lines_is_skipped() -> None:
    svg = _svg('<text x="10" y="50" style="font-size:20px"> </text>', width=1000, height=1000)
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert boxes == []
    assert dropped == 0


def test_unpositioned_text_lines_are_skipped() -> None:
    svg = _svg('<text style="font-size:20px"><tspan sodipodi:role="line">가</tspan></text>')
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert boxes == []
    assert dropped == 0


def test_text_inside_a_flow_root_is_not_a_second_box() -> None:
    svg = _svg(
        '<flowRoot><flowRegion><rect x="0" y="0" width="100" height="50"/></flowRegion>'
        '<flowPara>바깥<text x="10" y="20">안</text></flowPara></flowRoot>',
        width=1000,
        height=1000,
    )
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert [(box.bbox, box.lines, box.approx) for box in boxes] == [
        (BBox(x0=0, y0=0, x1=100, y1=50), ("바깥안",), False)
    ]


def test_mixed_svg_keeps_file_order() -> None:
    svg = _svg(
        '<flowRoot><flowRegion><rect x="0" y="0" width="100" height="50"/></flowRegion>'
        "<flowPara>첫</flowPara></flowRoot>"
        '<text x="1000" y="500" style="font-size:20px">'
        '<tspan sodipodi:role="line" x="1000" y="500" style="font-size:40px">둘</tspan></text>'
    )
    boxes, dropped = parse_svg(svg, file_width=1000, file_height=1480, file_y0=0, scale=1.0)
    assert dropped == 0
    assert [(box.lines, box.approx) for box in boxes] == [(("첫",), False), (("둘",), True)]
    assert boxes[1].bbox == BBox(x0=500, y0=210, x1=520, y1=235)


def test_load_english_pages_includes_text_element_lines(tmp_path: Path) -> None:
    en = tmp_path / "truth" / "en"
    en.mkdir(parents=True)
    (en / "E01P01.svg").write_text(
        f'<svg {SVG_NS} {SODI_NS} width="1000" height="1480">'
        '<flowRoot><flowRegion><rect x="0" y="0" width="100" height="50"/></flowRegion>'
        "<flowPara>Hello</flowPara></flowRoot>"
        '<text x="500" y="300" style="font-size:20px">'
        '<tspan sodipodi:role="line" x="500" y="300">world</tspan></text></svg>',
        encoding="utf-8",
    )
    ingest = IngestArtifact(
        series="S",
        chapter="C",
        strip_width=1000,
        strip_height=1480,
        files=[_source(0, "01.jpg", y0=0)],
    )
    assert load_english_pages(tmp_path / "truth", ingest) == {1: "Hello world"}
