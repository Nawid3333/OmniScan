"""Unit tests for omniscan.typeset.fit (CPU only; deterministic fakes plus real font metrics)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from omniscan.core.schemas import BBox
from omniscan.typeset.fit import (
    Fit,
    fit_text,
    inscribed_box,
    layout_region,
    line_height,
    wrap_words,
)
from omniscan.typeset.fonts import default_font_path, fonts_dir, load_font

# ---------------------------------------------------------------- fake fonts (card spec)


class Fake:
    """Advance width = len(text) * size * 0.5."""

    def __init__(self, size: int) -> None:
        self.size = size

    def getlength(self, text: str) -> float:
        return len(text) * self.size * 0.5


class Fake10:
    """Advance width = len(text) * 10, independent of any size."""

    def getlength(self, text: str) -> float:
        return len(text) * 10.0


def fake_factory(path: Path, size: int) -> Fake:
    return Fake(size)


# ---------------------------------------------------------------- wrapping


def test_wrap_words_greedy() -> None:
    assert wrap_words("aa bb cc dd", Fake10(), 55) == (["aa bb", "cc dd"], False)


def test_wrap_words_line_exactly_max_width_fits() -> None:
    assert wrap_words("aa bb", Fake10(), 50) == (["aa bb"], False)


def test_wrap_words_single_word_too_wide() -> None:
    assert wrap_words("abcdefghijkl mm", Fake10(), 50) == (["abcdefghijkl", "mm"], True)


def test_wrap_words_empty_and_whitespace() -> None:
    assert wrap_words("", Fake10(), 50) == ([], False)
    assert wrap_words("   \t \n ", Fake10(), 50) == ([], False)


def test_wrap_words_tabs_and_newlines_are_spaces() -> None:
    assert wrap_words("aa\tbb\ncc\r\ndd", Fake10(), 55) == (["aa bb", "cc dd"], False)


def test_line_height() -> None:
    assert line_height(40, 1.0) == 40
    assert line_height(30, 1.15) == 34  # 34.5 rounds to 34
    assert line_height(1, 0.1) == 1


# ---------------------------------------------------------------- fitting (fake font)

SENTENCE = "hello world"


def fake_fit(text: str, max_w: int, max_h: int) -> Fit:
    """fit_text with the card's fake font and fixed size range, for the deterministic cases."""
    return fit_text(
        text,
        max_w,
        max_h,
        Path("fake.ttf"),
        min_px=10,
        max_px=40,
        line_spacing=1.0,
        font_factory=fake_factory,
    )


def test_fit_text_picks_max_size() -> None:
    assert fake_fit(SENTENCE, 100, 100) == Fit(
        size_px=40, lines=["hello", "world"], width=100, height=80, overflow=False
    )


def test_fit_text_height_constrained() -> None:
    assert fake_fit(SENTENCE, 100, 60) == Fit(
        size_px=30, lines=["hello", "world"], width=75, height=60, overflow=False
    )


def test_fit_text_overflow_returns_min_size() -> None:
    assert fake_fit(SENTENCE, 20, 100) == Fit(
        size_px=10, lines=["hello", "world"], width=25, height=20, overflow=True
    )


def test_fit_text_empty_and_normalised() -> None:
    empty = Fit(size_px=40, lines=[], width=0, height=0, overflow=False)
    assert fake_fit("", 100, 100) == empty
    assert fake_fit("  \t\n ", 100, 100) == empty
    assert fake_fit("  hello \n world  ", 100, 100) == fake_fit("hello world", 100, 100)


def test_fit_text_validation() -> None:
    with pytest.raises(ValueError):
        fit_text(SENTENCE, 100, 100, Path("fake.ttf"), min_px=0, max_px=40, font_factory=fake_factory)
    with pytest.raises(ValueError):
        fit_text(SENTENCE, 100, 100, Path("fake.ttf"), min_px=40, max_px=10, font_factory=fake_factory)
    with pytest.raises(ValueError):
        fit_text(SENTENCE, 0, 100, Path("fake.ttf"), font_factory=fake_factory)
    with pytest.raises(ValueError):
        fit_text(SENTENCE, 100, 0, Path("fake.ttf"), font_factory=fake_factory)
    with pytest.raises(ValueError):
        fit_text(SENTENCE, 100, 100, Path("fake.ttf"), line_spacing=0, font_factory=fake_factory)


def test_fit_text_generous_box_returns_max_px() -> None:
    fit = fake_fit("hi there", 500, 500)
    assert fit.size_px == 40
    assert fit.overflow is False


# ---------------------------------------------------------------- fitting (real fonts)

SENTENCES = [
    "Are you okay? The dungeon just opened!",
    "I never asked for this power.",
    "Wait, that gate wasn't there yesterday.",
    "He said he would return by dawn, and he did.",
    "This is the deepest floor anyone has ever reached.",
    "Don't move. It can't see us if we don't move.",
    "The guild master summons you immediately!",
    "One day I will surpass every one of them.",
]
BOXES = [(60, 30), (80, 50), (100, 70), (150, 100), (250, 150), (400, 200)]
FONT_NAMES = ["ComicNeue-Bold.ttf", "Bangers-Regular.ttf"]


@pytest.mark.parametrize("font_name", FONT_NAMES)
@pytest.mark.parametrize("max_w,max_h", BOXES)
@pytest.mark.parametrize("sentence", SENTENCES)
def test_fit_real_font_measurements_agree(font_name: str, max_w: int, max_h: int, sentence: str) -> None:
    path = fonts_dir() / font_name
    fit = fit_text(sentence, max_w, max_h, path)
    normalized = " ".join(sentence.split())
    assert " ".join(fit.lines) == normalized
    if fit.overflow:
        return  # overflow behaviour is covered by the monotonic test below
    font = load_font(path, fit.size_px)
    assert fit.lines
    for text_line in fit.lines:
        assert font.getlength(text_line) <= max_w
    assert fit.height == len(fit.lines) * line_height(fit.size_px, 1.15)
    assert fit.height <= max_h
    assert fit.width == math.ceil(max(font.getlength(text_line) for text_line in fit.lines))
    if fit.size_px < 48:  # the size is optimal: one step larger does not fit
        larger = load_font(path, fit.size_px + 1)
        up_lines, up_too_wide = wrap_words(normalized, larger, max_w)
        assert up_too_wide or len(up_lines) * line_height(fit.size_px + 1, 1.15) > max_h


@pytest.mark.parametrize("font_name", FONT_NAMES)
@pytest.mark.parametrize("max_w,max_h", BOXES)
@pytest.mark.parametrize("sentence", SENTENCES)
def test_fit_real_font_monotonic_and_overflow(font_name: str, max_w: int, max_h: int, sentence: str) -> None:
    path = fonts_dir() / font_name
    small = fit_text(sentence, max_w, max_h, path)
    larger_box = fit_text(sentence, max_w + 60, max_h + 60, path)
    assert larger_box.size_px >= small.size_px
    if small.overflow:
        assert small.size_px == 14


# ---------------------------------------------------------------- inscribed box


def test_inscribed_box_without_polygon() -> None:
    assert inscribed_box(BBox(x0=100, y0=100, x1=300, y1=200), None, margin_px=6) == BBox(
        x0=106, y0=106, x1=294, y1=194
    )


def test_inscribed_box_without_polygon_tiny_bubble_degenerates() -> None:
    assert inscribed_box(BBox(x0=10, y0=10, x1=14, y1=14), None, margin_px=6) == BBox(
        x0=12, y0=12, x1=12, y1=12
    )


def _ellipse_polygon(points: int = 48) -> list[tuple[int, int]]:
    return [
        (
            round(200 + 200 * math.cos(2 * math.pi * i / points)),
            round(100 + 100 * math.sin(2 * math.pi * i / points)),
        )
        for i in range(points)
    ]


def test_inscribed_box_ellipse() -> None:
    bubble = BBox(x0=0, y0=0, x1=400, y1=200)
    polygon = _ellipse_polygon()
    box = inscribed_box(bubble, polygon, margin_px=0)
    assert 0.66 * 400 <= box.width <= 0.72 * 400
    assert 0.66 * 200 <= box.height <= 0.72 * 200
    assert abs((box.x0 + box.x1) / 2 - 200) <= 1
    assert abs((box.y0 + box.y1) / 2 - 100) <= 1
    with_margin = inscribed_box(bubble, polygon, margin_px=10)
    assert abs((box.width - with_margin.width) - 20) <= 1
    assert abs((box.height - with_margin.height) - 20) <= 1


def _test_point_inside(px: float, py: float, polygon: list[tuple[int, int]]) -> bool:
    """Test-local even-odd ray cast (independent of the implementation)."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > py) != (yj > py) and xi + (py - yi) * (xj - xi) / (yj - yi) > px:
            inside = not inside
        j = i
    return inside


def _box_samples(box: BBox) -> list[tuple[float, float]]:
    """The 24 boundary sample points of a box: 4 corners + 5 evenly spaced interior points per side."""
    x0, y0, x1, y1 = float(box.x0), float(box.y0), float(box.x1), float(box.y1)
    samples = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for ax, ay, bx, by in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        samples.extend((ax + (bx - ax) * k / 6, ay + (by - ay) * k / 6) for k in range(1, 6))
    return samples


def test_inscribed_box_rectangular_polygon_keeps_the_box() -> None:
    bubble = BBox(x0=0, y0=0, x1=400, y1=200)
    polygon = [(0, 0), (400, 0), (400, 200), (0, 200)]
    box = inscribed_box(bubble, polygon, margin_px=0)
    assert box.width >= 0.98 * 400
    assert box.height >= 0.98 * 200
    assert all(_test_point_inside(px, py, polygon) for px, py in _box_samples(box))


def test_inscribed_box_concave_l_polygon() -> None:
    bubble = BBox(x0=0, y0=0, x1=400, y1=200)
    polygon = [(0, 0), (400, 0), (400, 200), (250, 200), (250, 140), (0, 140)]
    box = inscribed_box(bubble, polygon, margin_px=0)
    assert all(_test_point_inside(px, py, polygon) for px, py in _box_samples(box))


# ---------------------------------------------------------------- layout items


def test_layout_region_dialogue() -> None:
    item = layout_region(
        "r0001", "Are you okay? The dungeon just opened!", BBox(x0=100, y0=100, x1=300, y1=200)
    )
    assert item.font == "ComicNeue-Bold.ttf"
    assert item.font_role == "dialogue"
    assert 14 <= item.size_px <= 48
    assert item.lines
    assert abs((item.box.x0 + item.box.x1) / 2 - 200) <= 1
    assert abs((item.box.y0 + item.box.y1) / 2 - 150) <= 1
    assert item.overflow is False


def test_layout_region_shout_uses_bangers() -> None:
    item = layout_region("r0002", "NO WAY!", BBox(x0=0, y0=0, x1=200, y1=100), role="shout")
    assert item.font == "Bangers-Regular.ttf"
    assert item.font_role == "shout"


def test_layout_region_style_passthrough() -> None:
    item = layout_region(
        "r0003",
        "Hello",
        BBox(x0=0, y0=0, x1=100, y1=50),
        align="left",
        color=(255, 0, 0),
        stroke_px=2,
        stroke_color=(0, 0, 255),
    )
    assert item.align == "left"
    assert item.color == (255, 0, 0)
    assert item.stroke_px == 2
    assert item.stroke_color == (0, 0, 255)


def test_layout_region_explicit_font_path() -> None:
    path = default_font_path("narration")
    item = layout_region("r0004", "Once upon a time.", BBox(x0=0, y0=0, x1=200, y1=60), font_path=path)
    assert item.font == "ComicNeue-Regular.ttf"


def test_layout_region_overflow_box_not_clamped() -> None:
    long_text = "This sentence is far too long to ever fit inside such a tiny target box, no matter what."
    item = layout_region("r0005", long_text, BBox(x0=500, y0=500, x1=510, y1=510))
    assert item.overflow is True
    assert item.size_px == 14
    assert item.box.width > 10
    assert item.box.height > 10
