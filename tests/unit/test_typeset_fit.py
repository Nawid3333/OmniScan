"""Unit tests for the layout engine omniscan.typeset.fit (CPU only; deterministic fakes plus real fonts)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from omniscan.core.schemas import BBox
from omniscan.typeset.fit import Fit, Shape, fit_shape, line_height
from omniscan.typeset.fonts import fonts_dir, load_font

# ---------------------------------------------------------------- fake font


class Fake:
    """Advance width = len(text) * size * 0.5."""

    def __init__(self, size: int) -> None:
        self.size = size

    def getlength(self, text: str) -> float:
        return len(text) * self.size * 0.5


def fake_factory(path: Path, size: int) -> Fake:
    return Fake(size)


FAKE = Path("fake.ttf")
REAL = fonts_dir() / "Mali-SemiBold.ttf"


def fit(text: str, shape: Shape, **kwargs: object) -> Fit:
    return fit_shape(text, shape, FAKE, font_factory=fake_factory, **kwargs)  # type: ignore[arg-type]


def rect(w: int, h: int) -> Shape:
    return Shape("rect", BBox(x0=0, y0=0, x1=w, y1=h))


def ellipse(w: int, h: int) -> Shape:
    return Shape("ellipse", BBox(x0=0, y0=0, x1=w, y1=h))


# ---------------------------------------------------------------- line_height / Shape


def test_line_height() -> None:
    assert line_height(20, 1.15) == 23
    assert line_height(1, 0.1) == 1


def test_rect_lines_are_the_full_width_while_the_block_fits() -> None:
    assert rect(200, 100).line_widths(4, 20, 25) == [200.0] * 4
    assert rect(200, 100).line_widths(5, 20, 25) == [0.0] * 5  # 125 px of block in 100 px


def test_ellipse_lines_are_widest_in_the_middle() -> None:
    widths = ellipse(400, 200).line_widths(5, 20, 24)
    assert widths[2] == max(widths)
    assert widths[0] == pytest.approx(widths[4]) and widths[1] == pytest.approx(widths[3])
    assert widths[0] < widths[1] < widths[2] < 400


def test_ellipse_line_width_is_the_chord_at_the_letters_outer_edge() -> None:
    (width,) = ellipse(400, 200).line_widths(1, 50, 60)
    dy = 0.8 * 50 / 2  # a lone line: the letters reach 40 % of the size above and below the centre
    assert width == pytest.approx(400 * math.sqrt(1 - (dy / 100) ** 2))


def test_ellipse_lines_outside_the_ellipse_have_no_room() -> None:
    assert ellipse(400, 100).line_widths(5, 20, 30)[0] == 0.0


# ---------------------------------------------------------------- fit_shape: sizes


def test_fit_picks_the_largest_size() -> None:
    result = fit("abcd", rect(100, 100), max_px=48)
    assert result.size_px == 48 and result.lines == ["abcd"] and result.overflow is False
    assert result.best_px == 48 and result.width == 96


def test_fit_is_height_constrained() -> None:
    result = fit("aa aa aa aa", rect(1000, 30))
    assert len(result.lines) == 1 and result.height <= 30


def test_size_cap_lowers_the_size_but_not_best_px() -> None:
    free = fit("hello there friend", ellipse(400, 200))
    capped = fit("hello there friend", ellipse(400, 200), size_cap=20)
    assert capped.size_px == 20 < free.size_px
    assert capped.best_px == free.best_px


def test_empty_text_has_no_lines() -> None:
    result = fit("   \n\t ", rect(100, 100))
    assert result.lines == [] and result.overflow is False and result.width == 0


def test_invalid_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="min_px"):
        fit("x", rect(10, 10), min_px=0)
    with pytest.raises(ValueError, match="min_px"):
        fit("x", rect(10, 10), min_px=20, max_px=10)


def test_every_line_fits_its_ellipse_width() -> None:
    shape = ellipse(360, 240)
    text = "Are you okay? The dungeon just opened and the monsters are coming out of the gate!"
    result = fit_shape(text, shape, REAL)
    font = load_font(REAL, result.size_px)
    pitch = line_height(result.size_px, 1.15)
    widths = shape.line_widths(len(result.lines), result.size_px, pitch)
    assert not result.overflow
    for line, width in zip(result.lines, widths, strict=True):
        assert font.getlength(line) <= width + 1e-6, line
    assert " ".join(result.lines) == text  # no word lost, reordered or split


def test_the_middle_lines_are_the_longest_in_an_ellipse() -> None:
    text = "I can't believe you actually came all the way here just to see me again."
    result = fit_shape(text, ellipse(320, 320), REAL)
    font = load_font(REAL, result.size_px)
    lengths = [font.getlength(line) for line in result.lines]
    middle = len(lengths) // 2
    assert max(lengths) in lengths[max(0, middle - 1) : middle + 2]
    assert lengths[0] < max(lengths) and lengths[-1] < max(lengths)


def test_the_balloon_layout_is_larger_than_its_inscribed_rectangle() -> None:
    text = "Hunter Association emergency announcement: evacuate the area immediately."
    in_ellipse = fit_shape(text, ellipse(420, 170), REAL)
    inscribed = fit_shape(text, Shape("rect", BBox(x0=0, y0=0, x1=297, y1=120)), REAL)  # 1/sqrt(2)
    assert in_ellipse.size_px > inscribed.size_px


# ---------------------------------------------------------------- fit_shape: phrasing


def test_lines_do_not_end_on_an_article_when_avoidable() -> None:
    result = fit_shape("We have to leave the city before the gate opens tonight.", ellipse(300, 220), REAL)
    assert len(result.lines) > 1
    for line in result.lines[:-1]:
        assert line.split()[-1].lower() not in {"the", "a", "to", "of", "and"}, result.lines


def test_breaks_prefer_the_end_of_a_sentence() -> None:
    result = fit_shape(
        "Are you okay? The dungeon just opened and the monsters are coming out!", ellipse(360, 230), REAL
    )
    assert result.lines[0].endswith("okay?"), result.lines


def test_balanced_lines_beat_greedy_ones_in_a_rectangle() -> None:
    result = fit("aa aa aa aa aa aa aa", rect(100, 100), min_px=10, max_px=10)
    lengths = [len(line) for line in result.lines]
    assert max(lengths) - min(lengths) <= 3  # greedy would leave a lone last word


# ---------------------------------------------------------------- hyphenation and overflow


def test_a_word_too_long_for_any_line_is_hyphenated() -> None:
    result = fit("Unbelievably", rect(60, 200), min_px=14, max_px=14)
    assert result.overflow is False
    assert len(result.lines) >= 2 and result.lines[0].endswith("-")
    assert "".join(line.rstrip("-") for line in result.lines) == "Unbelievably"
    assert all(len(line) * 7 <= 60 for line in result.lines)


def test_hyphen_pieces_on_one_line_are_rejoined() -> None:
    result = fit("go Unbelievably far", rect(90, 300), min_px=14, max_px=14)
    assert "".join(result.lines).replace("-", "").replace(" ", "") == "goUnbelievablyfar"
    assert all(" -" not in line and "- " not in line for line in result.lines)


def test_without_hyphenation_a_long_word_overflows_at_min_px() -> None:
    result = fit("Unbelievably", rect(60, 200), min_px=14, max_px=14, hyphenate=False)
    assert result.overflow is True and result.size_px == 14 and result.best_px == 0
    assert result.lines == ["Unbelievably"]


def test_text_too_long_for_the_shape_overflows_at_min_px() -> None:
    text = "This sentence is far too long to ever fit inside such a tiny box, no matter what."
    result = fit(text, ellipse(100, 60))
    assert result.overflow is True and result.size_px == 14
    assert " ".join(result.lines).replace("- ", "").replace("-", "").replace(" ", "") == text.replace(" ", "")


# ---------------------------------------------------------------- monotonic behaviour on real fonts


@pytest.mark.parametrize(("w", "h"), [(200, 120), (300, 200), (420, 260)])
def test_a_larger_balloon_never_gets_smaller_lettering(w: int, h: int) -> None:
    text = "The gate is closing! Everyone, get out now!"
    small = fit_shape(text, ellipse(w, h), REAL)
    large = fit_shape(text, ellipse(w + 60, h + 40), REAL)
    assert large.best_px >= small.best_px
