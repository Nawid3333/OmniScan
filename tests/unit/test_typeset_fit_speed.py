"""G2 fit-loop speed-up: identical layouts with far fewer measurements, plus the cross-call font cache."""

from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path
from typing import Any

import pytest
from PIL import ImageFont

from omniscan.typeset import fit as fit_module
from omniscan.typeset.fit import Fit, FontFactory, Measurable, fit_text, line_height, wrap_words
from omniscan.typeset.fonts import fonts_dir, load_font

# ---------------------------------------------------------------- pre-G2 reference (card test 3)


def old_wrap_words(text: str, font: Measurable, max_width: float) -> tuple[list[str], bool]:
    """The pre-G2 wrap_words, verbatim: the new code must return exactly these results."""
    words = text.split()
    if not words:
        return [], False
    lines: list[str] = []
    line = ""
    too_wide = False
    for word in words:
        if not line:
            if font.getlength(word) <= max_width:
                line = word
            else:  # a single word wider than the box: its own line, no hyphenation
                lines.append(word)
                too_wide = True
            continue
        candidate = f"{line} {word}"
        if font.getlength(candidate) <= max_width:
            line = candidate
            continue
        lines.append(line)
        if font.getlength(word) <= max_width:
            line = word
        else:
            lines.append(word)
            too_wide = True
            line = ""
    if line:
        lines.append(line)
    return lines, too_wide


def old_fit_text(
    text: str,
    max_w: int,
    max_h: int,
    font_path: Path,
    *,
    min_px: int = 14,
    max_px: int = 48,
    line_spacing: float = 1.15,
    font_factory: FontFactory = load_font,
) -> Fit:
    """The pre-G2 fit_text, verbatim: the new code must return exactly these results."""
    if min_px < 1 or max_px < min_px:
        raise ValueError(f"need 1 <= min_px <= max_px, got {min_px}..{max_px}")
    if max_w < 1 or max_h < 1:
        raise ValueError(f"need max_w >= 1 and max_h >= 1, got {max_w}x{max_h}")
    if line_spacing <= 0:
        raise ValueError(f"line_spacing must be > 0, got {line_spacing}")
    normalized = " ".join(text.split())
    if not normalized:
        return Fit(size_px=max_px, lines=[], width=0, height=0, overflow=False)
    fallback: Fit | None = None
    for size in range(max_px, min_px - 1, -1):
        font = font_factory(font_path, size)
        lines, too_wide = old_wrap_words(normalized, font, max_w)
        height = len(lines) * line_height(size, line_spacing)
        width = math.ceil(max((font.getlength(line) for line in lines), default=0.0))
        result = Fit(size_px=size, lines=lines, width=width, height=height, overflow=False)
        if not too_wide and height <= max_h:
            return result
        if size == min_px:
            fallback = result
    assert fallback is not None  # the loop always reaches min_px
    return dataclasses.replace(fallback, overflow=True)


# ---------------------------------------------------------------- table fake fonts (card spec)


ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def make_table(seed: int, count: int = 48) -> dict[str, float]:
    """Seeded per-word advances at reference size 100: 80..480, a fifth of the words 2.5x (long)."""
    rng = random.Random(seed)
    table: dict[str, float] = {}
    while len(table) < count:
        word = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12)))
        width = rng.uniform(80.0, 480.0)
        if rng.random() < 0.2:
            width *= 2.5
        table[word] = width
    return table


TABLE = make_table(seed=1234)
WORDS = list(TABLE)


class TableFont:
    """Fake font: per-word advances from TABLE scaled linearly with the size, spaces added per gap."""

    SPACE = 50.0  # advance of one space at reference size 100

    def __init__(self, size: int) -> None:
        self._scale = size / 100.0

    def getlength(self, text: str) -> float:
        words = text.split(" ")
        width = sum(TABLE.get(word, 40.0) for word in words)
        return (width + self.SPACE * (len(words) - 1)) * self._scale


class CountingTableFont:
    """TableFont that counts every getlength call in a one-element counter list."""

    def __init__(self, size: int, counter: list[int]) -> None:
        self._font = TableFont(size)
        self._counter = counter

    def getlength(self, text: str) -> float:
        self._counter[0] += 1
        return self._font.getlength(text)


class LinearFont:
    """Advance width = 7 * len(text), independent of the size (the card's simple spy metrics)."""

    def getlength(self, text: str) -> float:
        return 7.0 * len(text)


class DummyFont:
    """A Measurable placeholder for filling the font cache."""

    def getlength(self, text: str) -> float:
        return 0.0


def table_factory(counter: list[int] | None = None) -> FontFactory:
    def factory(path: Path, size: int) -> TableFont | CountingTableFont:
        return TableFont(size) if counter is None else CountingTableFont(size, counter)

    return factory


def linear_factory(path: Path, size: int) -> LinearFont:
    return LinearFont()


# ---------------------------------------------------------------- identical layouts (card test 3)


def random_case(rng: random.Random) -> tuple[str, int, int, int, int, float]:
    """A (text, max_w, max_h, min_px, max_px, line_spacing) case: 0-40 words (empty, one-word,
    repeated and box-wider words included), boxes 40x30..600x400, min_px == max_px sometimes."""
    kind = rng.random()
    if kind < 0.08:
        text = rng.choice(["", "   ", " \t \n "])
    elif kind < 0.16:
        text = rng.choice(WORDS)
    else:
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 40)))
    max_w = rng.randint(40, 600)
    max_h = rng.randint(30, 400)
    if rng.random() < 0.15:
        min_px = max_px = rng.randint(1, 48)
    else:
        min_px = rng.randint(1, 30)
        max_px = min_px + rng.randint(0, 40)
    return text, max_w, max_h, min_px, max_px, rng.uniform(0.9, 1.6)


def test_random_cases_return_identical_layouts() -> None:
    rng = random.Random(7)
    for i in range(300):
        text, max_w, max_h, min_px, max_px, spacing = random_case(rng)
        factory = table_factory()
        old = old_fit_text(
            text,
            max_w,
            max_h,
            Path("fake.ttf"),
            min_px=min_px,
            max_px=max_px,
            line_spacing=spacing,
            font_factory=factory,
        )
        new = fit_text(
            text,
            max_w,
            max_h,
            Path("fake.ttf"),
            min_px=min_px,
            max_px=max_px,
            line_spacing=spacing,
            font_factory=factory,
        )
        assert old == new, f"case {i}: {text!r} {max_w}x{max_h} {min_px}..{max_px} {spacing:.3f}"


HAND_CASES = [
    # empty text: no wrapping at all
    ("", 100, 100, 10, 20, 1.0, Fit(size_px=20, lines=[], width=0, height=0, overflow=False)),
    # nothing fits: the word is wider than the box at every size -> fallback at min_px
    ("hello", 20, 100, 10, 20, 1.0, Fit(size_px=10, lines=["hello"], width=35, height=10, overflow=True)),
    # fits only at min_px: the word fits the width, the height only allows size 10
    ("hello", 36, 10, 10, 20, 1.0, Fit(size_px=10, lines=["hello"], width=35, height=10, overflow=False)),
    # exactly fits at the only size
    ("ab cd", 35, 10, 10, 10, 1.0, Fit(size_px=10, lines=["ab cd"], width=35, height=10, overflow=False)),
]


@pytest.mark.parametrize("case", HAND_CASES)
def test_hand_made_cases_return_identical_layouts(
    case: tuple[str, int, int, int, int, float, Fit],
) -> None:
    text, max_w, max_h, min_px, max_px, spacing, expected = case
    old = old_fit_text(
        text,
        max_w,
        max_h,
        Path("fake.ttf"),
        min_px=min_px,
        max_px=max_px,
        line_spacing=spacing,
        font_factory=linear_factory,
    )
    new = fit_text(
        text,
        max_w,
        max_h,
        Path("fake.ttf"),
        min_px=min_px,
        max_px=max_px,
        line_spacing=spacing,
        font_factory=linear_factory,
    )
    assert old == expected
    assert new == expected


def test_wrap_words_identical_on_random_inputs() -> None:
    rng = random.Random(11)
    for i in range(200):
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 25)))
        font = TableFont(rng.randint(1, 60))
        max_width = rng.uniform(1.0, 900.0)
        assert wrap_words(text, font, max_width) == old_wrap_words(text, font, max_width), f"case {i}"


# ---------------------------------------------------------------- fewer measurements (card test 4)


def test_chapter_fit_uses_at_most_40_percent_of_the_old_measurements() -> None:
    rng = random.Random(5)
    chapter = [
        (
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 30))),
            rng.randint(120, 400),
            rng.randint(80, 250),
        )
        for _ in range(60)
    ]
    old_calls, new_calls = [0], [0]
    old_fits = [
        old_fit_text(text, w, h, Path("fake.ttf"), font_factory=table_factory(old_calls))
        for text, w, h in chapter
    ]
    new_fits = [
        fit_text(text, w, h, Path("fake.ttf"), font_factory=table_factory(new_calls))
        for text, w, h in chapter
    ]
    ratio = new_calls[0] / old_calls[0]
    print(
        f"\nfit_text getlength calls on 60 regions: old={old_calls[0]} new={new_calls[0]} ratio={ratio:.3f}"
    )
    assert old_fits == new_fits
    assert ratio <= 0.4, f"new code made {new_calls[0]} calls vs {old_calls[0]} ({ratio:.1%})"


# ---------------------------------------------------------------- cross-call cache (card test 5)


def test_cross_call_cache_for_the_default_font(monkeypatch: pytest.MonkeyPatch) -> None:
    path = fonts_dir() / "ComicNeue-Regular.ttf"
    text = "The dungeon gate opens at dawn and the guild master is waiting for you."
    calls = [0]
    real_getlength = ImageFont.FreeTypeFont.getlength

    def counting(self: ImageFont.FreeTypeFont, *args: Any, **kwargs: Any) -> float:
        calls[0] += 1
        return real_getlength(self, *args, **kwargs)

    monkeypatch.setattr(ImageFont.FreeTypeFont, "getlength", counting)
    fit_module._measure_cache.clear()
    first = fit_text(text, 200, 120, path)
    after_first = calls[0]
    assert after_first > 0
    assert fit_text(text, 200, 120, path) == first
    assert calls[0] == after_first  # the second call measured nothing through the real font

    fit_text(text, 200, 120, fonts_dir() / "ComicNeue-Bold.ttf")
    assert calls[0] > after_first  # a different font file never reuses the entries


def test_measure_cache_is_bounded_and_stays_correct() -> None:
    path = fonts_dir() / "ComicNeue-Regular.ttf"
    fit_module._measure_cache.clear()
    fit_module._measure_cache.update(
        {(f"fake-{i}.ttf", 10, "x"): 1.0 for i in range(fit_module._MEASURE_CACHE_MAX)}
    )
    assert len(fit_module._measure_cache) >= fit_module._MEASURE_CACHE_MAX
    filled = fit_text("goodbye cruel world", 200, 120, path)  # first miss clears the cache
    assert len(fit_module._measure_cache) < fit_module._MEASURE_CACHE_MAX
    fit_module._measure_cache.clear()
    cold = fit_text("goodbye cruel world", 200, 120, path)
    assert filled == cold


def test_font_cache_is_bounded_and_stays_correct() -> None:
    path = fonts_dir() / "ComicNeue-Regular.ttf"
    fit_module._font_cache.clear()
    fit_module._font_cache.update(
        {(f"fake-{i}.ttf", 10): DummyFont() for i in range(fit_module._FONT_CACHE_MAX)}
    )
    assert len(fit_module._font_cache) >= fit_module._FONT_CACHE_MAX
    filled = fit_text("hello world", 200, 120, path, min_px=60, max_px=60)  # miss: clears, then loads
    assert len(fit_module._font_cache) < fit_module._FONT_CACHE_MAX
    fit_module._font_cache.clear()
    cold = fit_text("hello world", 200, 120, path, min_px=60, max_px=60)
    assert filled == cold
