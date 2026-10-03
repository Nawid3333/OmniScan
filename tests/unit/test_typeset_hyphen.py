"""Line-breaking rules per release language (typeset/hyphen.py) and how the fitter uses them (#43)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.schemas import BBox
from omniscan.typeset.fit import Shape, fit_shape
from omniscan.typeset.fonts import SFX_FONTS, STYLE_FONTS, fonts_dir
from omniscan.typeset.hyphen import dangling_words, hyphen_points


def _split(word: str) -> str:
    """`word` with a hyphen at every Spanish hyphenation point."""
    points = hyphen_points(word, "es")
    assert points is not None
    parts, last = [], 0
    for point in points:
        parts.append(word[last:point])
        last = point
    return "-".join([*parts, word[last:]])


@pytest.mark.parametrize(
    ("word", "syllables"),
    [
        ("casa", "ca-sa"),  # one consonant starts the next syllable
        ("canto", "can-to"),  # two are split ...
        ("hablar", "ha-blar"),  # ... unless they start a syllable together
        ("mucho", "mu-cho"),  # ch, ll, rr are one sound
        ("perro", "pe-rro"),
        ("instrucción", "ins-truc-ción"),  # three or more: the last cluster stays together
        ("obstáculo", "obs-tá-cu-lo"),
        ("leo", "le-o"),  # two strong vowels are two syllables
        ("día", "dí-a"),  # an accented i or u breaks the diphthong
        ("increíble", "in-cre-í-ble"),
        ("cielo", "cie-lo"),  # a diphthong stays together
        ("guerra", "gue-rra"),
        ("ahora", "a-ho-ra"),
        ("ayer", "a-yer"),  # y before a vowel is a consonant
        ("muy", "muy"),
        ("¡EXTRAORDINARIO!", "¡EX-TRA-OR-DI-NA-RIO!"),  # capitals and punctuation around the word
    ],
)
def test_spanish_words_break_between_syllables(word: str, syllables: str) -> None:
    assert _split(word) == syllables


def test_other_languages_break_anywhere_and_hyphenated_words_stay_whole() -> None:
    assert hyphen_points("Unbelievably", "en") is None
    assert hyphen_points("Donaudampfschiff", "de") is None
    assert hyphen_points("Jinwoo-ssi", "es") == []


def test_the_little_words_a_line_should_not_end_on() -> None:
    assert {"the", "of", "and"} <= dangling_words("en")
    assert {"el", "la", "de", "que", "y"} <= dangling_words("es") and "the" not in dangling_words("es")
    assert {"der", "und", "mit"} <= dangling_words("de")
    assert dangling_words("xx") == dangling_words("en")


class Fake:
    """Advance width = len(text) * size * 0.5."""

    def __init__(self, size: int) -> None:
        self.size = size

    def getlength(self, text: str) -> float:
        return len(text) * self.size * 0.5


def _fit(text: str, width: int, height: int, lang: str) -> list[str]:
    shape = Shape("rect", BBox(x0=0, y0=0, x1=width, y1=height))
    result = fit_shape(
        text,
        shape,
        Path("fake.ttf"),
        min_px=14,
        max_px=14,
        font_factory=lambda _p, size: Fake(size),
        lang=lang,
    )
    assert not result.overflow
    return result.lines


def test_a_long_spanish_word_is_hyphenated_at_a_syllable() -> None:
    # 80 px: at most 11 characters a line, hyphen included
    assert _fit("Extraordinariamente", 80, 300, "es") == ["Extraordi-", "nariamente"]  # ex-tra-or-di-|na...
    assert _fit("Extraordinariamente", 80, 300, "en") == ["Extraordin-", "ariamente"]  # wherever it fits


def test_a_spanish_line_does_not_end_on_an_article_or_preposition() -> None:
    assert _fit("Vamos a la casa de mi madre ahora mismo", 170, 300, "es") == [
        "Vamos a la casa",
        "de mi madre ahora mismo",
    ]
    assert _fit("Ella dijo que la casa de los padres era grande", 170, 300, "es") == [
        "Ella dijo que la casa",
        "de los padres era grande",
    ]
    # English rules do not know "de" and "los"
    assert _fit("Vamos a la casa de mi madre ahora mismo", 170, 300, "en")[0] == "Vamos a la casa de"


@pytest.mark.parametrize(
    "font",
    sorted({*STYLE_FONTS["webtoon"].values(), *STYLE_FONTS["manga"].values(), *SFX_FONTS.values()}),
)
def test_every_lettering_preset_has_the_spanish_and_german_letters(font: str) -> None:
    face = ImageFont.truetype(str(fonts_dir() / font), 40)

    def drawn(char: str) -> bytes:
        image = Image.new("L", (90, 90))
        ImageDraw.Draw(image).text((20, 10), char, font=face, fill=255)
        return image.tobytes()

    missing = drawn("\U0010fffd")  # a code point no font has: its .notdef box
    assert [c for c in "¿¡ñÑáéíóúÁÉÍÓÚüÜäöÄÖß«»" if drawn(c) == missing] == []
