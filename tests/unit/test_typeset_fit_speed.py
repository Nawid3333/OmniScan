"""Layout engine cost: bounded measurements per balloon, plus the process-wide font and measurement caches."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from PIL import ImageFont

from omniscan.core.schemas import BBox
from omniscan.typeset import fit as fit_module
from omniscan.typeset.fit import Shape, fit_shape
from omniscan.typeset.fonts import fonts_dir

WORDS = [
    "the",
    "gate",
    "hunter",
    "dungeon",
    "monster",
    "system",
    "level",
    "quest",
    "guild",
    "shadow",
    "sword",
    "raid",
    "boss",
    "mana",
    "rank",
    "player",
    "awakened",
    "monarch",
    "I",
    "you",
    "we",
    "can't",
    "won't",
    "really",
    "never",
    "always",
]
FONT = fonts_dir() / "Mali-SemiBold.ttf"
SHAPE = Shape("ellipse", BBox(x0=0, y0=0, x1=300, y1=200))


class CountingFont:
    """Advance width proportional to length and size; counts every measurement."""

    def __init__(self, size: int, counter: list[int]) -> None:
        self.size = size
        self.counter = counter

    def getlength(self, text: str) -> float:
        self.counter[0] += 1
        return len(text) * self.size * 0.52


class DummyFont:
    """Stand-in cache entry that is never used for real measurements."""

    def getlength(self, text: str) -> float:
        return 0.0


def test_a_chapter_of_balloons_stays_within_the_measurement_budget() -> None:
    rng = random.Random(5)
    calls = [0]
    for _ in range(60):
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 30)))
        box = BBox(x0=0, y0=0, x1=rng.randint(120, 400), y1=rng.randint(80, 250))
        fit_shape(
            text, Shape("ellipse", box), Path("fake.ttf"), font_factory=lambda _p, s: CountingFont(s, calls)
        )
    per_balloon = calls[0] / 60
    print(f"\nfit_shape getlength calls: {per_balloon:.0f} per balloon")
    assert per_balloon <= 1200


def test_cross_call_cache_for_the_default_font(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "The dungeon gate opens at dawn and the guild master is waiting for you."
    calls = [0]
    real_getlength = ImageFont.FreeTypeFont.getlength

    def counting(self: ImageFont.FreeTypeFont, *args: Any, **kwargs: Any) -> float:
        calls[0] += 1
        return real_getlength(self, *args, **kwargs)

    monkeypatch.setattr(ImageFont.FreeTypeFont, "getlength", counting)
    fit_module._measure_cache.clear()
    first = fit_shape(text, SHAPE, FONT)
    after_first = calls[0]
    assert after_first > 0
    assert fit_shape(text, SHAPE, FONT) == first
    assert calls[0] == after_first  # the second call measured nothing through the real font

    fit_shape(text, SHAPE, fonts_dir() / "Mali-Bold.ttf")
    assert calls[0] > after_first  # a different font file never reuses the entries


def test_measure_cache_is_bounded_and_stays_correct() -> None:
    fit_module._measure_cache.clear()
    fit_module._measure_cache.update(
        {(f"fake-{i}.ttf", 10, "x"): 1.0 for i in range(fit_module._MEASURE_CACHE_MAX)}
    )
    filled = fit_shape("goodbye cruel world", SHAPE, FONT)  # the first miss clears the cache
    assert len(fit_module._measure_cache) < fit_module._MEASURE_CACHE_MAX
    fit_module._measure_cache.clear()
    assert fit_shape("goodbye cruel world", SHAPE, FONT) == filled


def test_font_cache_is_bounded_and_stays_correct() -> None:
    fit_module._font_cache.clear()
    fit_module._font_cache.update(
        {(f"fake-{i}.ttf", 10): DummyFont() for i in range(fit_module._FONT_CACHE_MAX)}
    )
    filled = fit_shape("hello world", SHAPE, FONT, min_px=30, max_px=30)  # a miss: clears, then loads
    assert len(fit_module._font_cache) < fit_module._FONT_CACHE_MAX
    fit_module._font_cache.clear()
    assert fit_shape("hello world", SHAPE, FONT, min_px=30, max_px=30) == filled


def test_fakes_never_touch_the_shared_caches() -> None:
    fit_module._measure_cache.clear()
    fit_module._font_cache.clear()
    fit_shape("hello world", SHAPE, Path("fake.ttf"), font_factory=lambda _p, s: CountingFont(s, [0]))
    assert not fit_module._measure_cache and not fit_module._font_cache
