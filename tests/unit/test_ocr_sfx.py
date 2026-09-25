"""Tests for sound-effect detection and style measurement after OCR (ocr/sfx.py): CPU, rendered lettering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.config import SfxConfig
from omniscan.core.schemas import BBox, Lang, OcrLine, Region, RegionKind
from omniscan.ocr.sfx import (
    default_sfx_text_paths,
    dialogue_glyph_size,
    ink_angle,
    is_sfx,
    levelled,
    load_sfx_lexicon,
    made_of_words,
    measure_lettering_style,
    normalise,
    reclassify_sfx_regions,
    thinnest_tilt,
)
from omniscan.typeset.sfx import weight_class
from tests.fixtures.korean_pages import FONTS_DIR

LEXICON = load_sfx_lexicon(default_sfx_text_paths())
KO = LEXICON["ko"]


def region(
    rid: str,
    text: str,
    box: BBox,
    *,
    kind: RegionKind = "free_text",
    lang: Lang = "ko",
) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind=kind,
        bbox=box,
        text=text,
        lang=lang,
        lines=[OcrLine(bbox=box, text=text, score=0.95, engine="test")],
    )


def dialogue(i: int) -> Region:
    """A bubble with 12 letters in a 240 x 60 box: letter size sqrt(240*60/12) ~ 35 px."""
    return region(
        f"d{i}",
        "괜찮아요 던전이 열렸어요",
        BBox(x0=0, y0=i * 100, x1=240, y1=i * 100 + 60),
        kind="bubble_text",
    )


DIALOGUE = [dialogue(i) for i in range(3)]


# ---------------------------------------------------------------- text normalisation and the lexicon


@pytest.mark.parametrize(
    ("text", "core"),
    [
        ("쿠구구구궁!!", "쿠구궁"),
        ("쾅 쾅 쾅!", "쾅"),
        ("ドドドドド", "ド"),
        ("ガッ", "ガ"),
        ("ゴゴゴゴー", "ゴ"),
        ("...!?", ""),
    ],
)
def test_normalise(text: str, core: str) -> None:
    assert normalise(text) == core


def test_made_of_words_accepts_repeats_and_compounds() -> None:
    assert made_of_words(normalise("두근두근"), KO)
    assert made_of_words(normalise("쾅쾅쾅"), KO)
    assert made_of_words(normalise("쿵쾅"), KO)
    assert not made_of_words(normalise("그래요"), KO)
    assert not made_of_words("", KO)


def test_the_shipped_lexicon_covers_the_three_source_languages() -> None:
    assert {"ko", "ja", "zh"} <= set(LEXICON)
    assert "쾅" in KO and "ド" in LEXICON["ja"] and "砰" in LEXICON["zh"]


def test_lexicon_files_are_concatenated_and_validated(tmp_path: Path) -> None:
    user = tmp_path / "sfx_text.toml"
    user.write_text('[sfx_text]\nko = ["뿌앙"]\n', encoding="utf-8")
    merged = load_sfx_lexicon([*default_sfx_text_paths(), user, tmp_path / "missing.toml"])
    assert "뿌앙" in merged["ko"] and "쾅" in merged["ko"]
    user.write_text("[sfx_text]\nko = [1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"sfx_text\.ko"):
        load_sfx_lexicon([user])
    user.write_text("not toml [[", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid TOML"):
        load_sfx_lexicon([user])


# ---------------------------------------------------------------- is_sfx / reclassification


def test_a_lexicon_effect_in_free_text_is_an_sfx() -> None:
    big = region("r", "쾅!!", BBox(x0=0, y0=0, x1=160, y1=160))
    assert is_sfx(big, KO, dialogue_glyph_size(DIALOGUE), SfxConfig())
    assert is_sfx(big, KO, None, SfxConfig())  # no dialogue to compare with: the lexicon decides


def test_a_lexicon_word_lettered_smaller_than_dialogue_is_not_an_sfx() -> None:
    small = region("r", "헉", BBox(x0=0, y0=0, x1=20, y1=20))
    assert not is_sfx(small, KO, dialogue_glyph_size(DIALOGUE), SfxConfig())


def test_huge_short_lettering_is_an_sfx_without_the_lexicon() -> None:
    huge = region("r", "뿌앙", BBox(x0=0, y0=0, x1=300, y1=150))  # ~150 px letters vs ~35 px dialogue
    reference = dialogue_glyph_size(DIALOGUE)
    by_size = SfxConfig(size_ratio=2.0)
    assert is_sfx(huge, KO, reference, by_size)
    assert not is_sfx(huge, KO, None, by_size)  # size alone needs a reference
    assert not is_sfx(huge, KO, reference, SfxConfig(size_ratio=10.0))
    assert not is_sfx(huge, KO, reference, SfxConfig())  # off by default: a big sign would pass too


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("쾅", "bubble_text"),  # in a balloon: dialogue lettering, not an effect
        ("그럴 리가 없어", "free_text"),  # not onomatopoeia, normal size
        ("쾅쾅쾅쾅쾅쾅쾅쾅쾅", "free_text"),  # too many letters
        ("BOOM", "free_text"),  # already English
        ("1화", "free_text"),  # a chapter title
    ],
)
def test_not_sfx(text: str, kind: RegionKind) -> None:
    candidate = region("r", text, BBox(x0=0, y0=0, x1=300, y1=150), kind=kind)
    assert not is_sfx(candidate, KO, dialogue_glyph_size(DIALOGUE), SfxConfig())


def test_reclassify_changes_only_effects_and_keeps_the_rest_identical() -> None:
    boom = region("s", "쿠구구궁", BBox(x0=0, y0=500, x1=400, y1=640))
    doki = region("j", "ドキドキ", BBox(x0=0, y0=700, x1=300, y1=800), lang="ja")
    talk = region("t", "그럴 리가 없어", BBox(x0=0, y0=900, x1=240, y1=960))
    out = reclassify_sfx_regions([*DIALOGUE, boom, doki, talk], LEXICON, SfxConfig())
    assert [r.kind for r in out[3:]] == ["sfx", "sfx", "free_text"]
    assert all(a is b for a, b in zip(out[:3], DIALOGUE, strict=True)) and out[5] is talk


# ---------------------------------------------------------------- style measurement


def render_sfx(
    text: str,
    *,
    font: str = "NanumGothic-Bold.ttf",
    size: int = 110,
    fill: tuple[int, int, int] = (220, 40, 40),
    outline: tuple[int, int, int] | None = (255, 255, 255),
    stroke: int = 6,
    background: tuple[int, int, int] = (90, 110, 130),
    angle: float = 0.0,
) -> tuple[torch.Tensor, BBox]:
    """An effect lettered on a plain page: uint8 [3, H, W] strip and its ink box."""
    typeface = ImageFont.truetype(str(FONTS_DIR / font), size)
    layer = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (450, 300),
        text,
        font=typeface,
        anchor="mm",
        fill=(*fill, 255),
        stroke_width=stroke if outline else 0,
        stroke_fill=(*(outline or fill), 255),
    )
    if angle:
        layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(450, 300))
    page = Image.new("RGBA", (900, 600), (*background, 255))
    page.alpha_composite(layer)
    x0, y0, x1, y1 = layer.getchannel("A").point(lambda v: 255 if v > 40 else 0).getbbox()  # type: ignore[misc]
    strip = torch.from_numpy(np.array(page.convert("RGB"))).permute(2, 0, 1).contiguous()
    return strip, BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def close(a: tuple[int, int, int] | None, b: tuple[int, int, int], tol: int = 40) -> bool:
    return a is not None and max(abs(x - y) for x, y in zip(a, b, strict=True)) <= tol


def test_fill_and_outline_colours_are_measured() -> None:
    strip, box = render_sfx("쾅!")
    styled = measure_lettering_style(strip, region("s", "쾅!", box, kind="sfx"))
    assert close(styled.text_color, (220, 40, 40)), styled.text_color
    assert close(styled.stroke_color, (255, 255, 255)), styled.stroke_color


def test_plain_lettering_has_no_outline() -> None:
    strip, box = render_sfx("두근두근", fill=(20, 20, 20), outline=None, background=(235, 235, 235))
    styled = measure_lettering_style(strip, region("s", "두근두근", box, kind="sfx"))
    assert close(styled.text_color, (20, 20, 20)) and styled.stroke_color is None
    assert styled.angle == 0.0


@pytest.mark.parametrize("angle", [-20.0, 15.0, 25.0])
def test_the_tilt_of_the_lettering_is_measured(angle: float) -> None:
    strip, box = render_sfx("두근두근", outline=None, angle=angle, background=(240, 240, 240))
    styled = measure_lettering_style(strip, region("s", "두근두근", box, kind="sfx"))
    assert styled.angle == pytest.approx(angle, abs=4.0)


def test_ink_angle_ignores_blobs_and_vertical_lettering() -> None:
    blob = torch.zeros((50, 50), dtype=torch.bool)
    blob[10:40, 10:40] = True
    assert ink_angle(blob) == 0.0
    column = torch.zeros((200, 40), dtype=torch.bool)
    column[10:190, 15:25] = True
    assert ink_angle(column) == 0.0


def test_bolder_lettering_weighs_more() -> None:
    bold_strip, bold_box = render_sfx(
        "쾅쾅", font="NanumGothic-Bold.ttf", outline=None, background=(240, 240, 240), fill=(0, 0, 0)
    )
    thin_strip, thin_box = render_sfx(
        "쾅쾅", font="NanumGothic-Regular.ttf", outline=None, background=(240, 240, 240), fill=(0, 0, 0)
    )
    bold = measure_lettering_style(bold_strip, region("s", "쾅쾅", bold_box, kind="sfx"))
    thin = measure_lettering_style(thin_strip, region("s", "쾅쾅", thin_box, kind="sfx"))
    assert bold.weight is not None and thin.weight is not None
    assert bold.weight > thin.weight
    assert weight_class(bold.weight) == "bold" and weight_class(thin.weight) == "light"


def test_a_white_caption_with_a_black_outline_is_not_black_text() -> None:
    strip, box = render_sfx(
        "돌아올 거야", size=60, fill=(255, 255, 255), outline=(0, 0, 0), stroke=4, background=(150, 120, 170)
    )
    styled = measure_lettering_style(strip, region("f", "돌아올 거야", box))
    assert close(styled.text_color, (255, 255, 255)) and close(styled.stroke_color, (0, 0, 0))
    assert styled.angle == 0.0 and styled.weight is None  # tilt and weight are measured on effects only


def test_unreadable_art_leaves_the_region_unchanged() -> None:
    strip = torch.full((3, 200, 200), 128, dtype=torch.uint8)
    original = region("s", "쾅", BBox(x0=50, y0=50, x1=150, y1=150), kind="sfx")
    assert measure_lettering_style(strip, original) is original


@pytest.mark.parametrize("outline", [None, (255, 255, 255)])
@pytest.mark.parametrize("angle", [0.0, 15.0, -25.0])
def test_the_weight_class_survives_tilt_and_outline(
    outline: tuple[int, int, int] | None, angle: float
) -> None:
    for font, expected in (("NanumGothic-Bold.ttf", "bold"), ("NanumGothic-Regular.ttf", "light")):
        strip, box = render_sfx(
            "쾅!",
            font=font,
            outline=outline,
            stroke=8,
            size=150,
            fill=(20, 20, 20),
            background=(200, 170, 190),
            angle=angle,
        )
        styled = measure_lettering_style(strip, region("s", "쾅!", box, kind="sfx"))
        assert weight_class(styled.weight) == expected, (font, styled.weight)


def test_an_outline_lining_the_counters_is_not_a_fill() -> None:
    # a thin "ㅇ" with a thick white outline: its counter is white, which is the outline, not a fill
    strip, box = render_sfx(
        "쾅!",
        font="NanumGothic-Regular.ttf",
        outline=(255, 255, 255),
        stroke=8,
        size=150,
        fill=(20, 20, 20),
        background=(200, 170, 190),
    )
    styled = measure_lettering_style(strip, region("s", "쾅!", box, kind="sfx"))
    assert close(styled.text_color, (20, 20, 20)) and close(styled.stroke_color, (255, 255, 255))


def test_levelled_undoes_a_tilt() -> None:
    bar = torch.zeros((200, 300), dtype=torch.bool)
    for x in range(60, 240):
        y = 100 - round((x - 150) * 0.364)  # rising 20 degrees to the right
        bar[y - 4 : y + 4, x] = True
    assert thinnest_tilt(bar) == pytest.approx(20.0, abs=1.0)
    assert thinnest_tilt(levelled(bar, thinnest_tilt(bar))) == pytest.approx(0.0, abs=1.0)


def render_on_art(
    text: str, *, font: str, fill: tuple[int, int, int], outline: tuple[int, int, int] | None, angle: float
) -> tuple[torch.Tensor, BBox, torch.Tensor]:
    """Lettering over a panel crossed by dark ink lines: strip, lettering box, truth mask of the visible
    lettering (coverage above 16/255: fainter anti-aliasing changes a pixel by less than that)."""
    page = Image.new("RGBA", (900, 600), (205, 180, 195, 255))
    art = ImageDraw.Draw(page)
    for k in range(9):  # dark ink lines of the art, several running under the lettering
        art.line([(60 + 90 * k, 80), (300 + 60 * k, 560)], fill=(25, 25, 35, 255), width=3 + k % 3)
    layer = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (450, 300),
        text,
        font=ImageFont.truetype(str(FONTS_DIR / font), 130),
        anchor="mm",
        fill=(*fill, 255),
        stroke_width=9 if outline else 0,
        stroke_fill=(*(outline or fill), 255),
    )
    if angle:
        layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(450, 300))
    alpha = np.array(layer.getchannel("A"))
    page.alpha_composite(layer)
    x0, y0, x1, y1 = layer.getchannel("A").point(lambda v: 255 if v > 40 else 0).getbbox()  # type: ignore[misc]
    strip = torch.from_numpy(np.array(page.convert("RGB"))).permute(2, 0, 1).contiguous()
    return strip, BBox(x0=x0, y0=y0, x1=x1, y1=y1), torch.from_numpy(alpha > 16)


def test_thin_coloured_lettering_over_ink_lines_keeps_its_weight_and_tilt() -> None:
    strip, box, _ = render_on_art(
        "휘익", font="NanumGothic-Regular.ttf", fill=(40, 190, 90), outline=None, angle=-18
    )
    styled = measure_lettering_style(strip, region("s", "휘익", box, kind="sfx"))
    assert close(styled.text_color, (40, 190, 90))
    assert styled.angle == pytest.approx(-18.0, abs=4.0)
    assert weight_class(styled.weight) == "light"  # the black art lines are not part of the letters


def test_the_weight_does_not_depend_on_the_tilt() -> None:
    level_strip, level_box = render_sfx(
        "두근두근", outline=None, fill=(20, 20, 20), background=(230, 225, 235)
    )
    tilted_strip, tilted_box = render_sfx(
        "두근두근", outline=None, fill=(20, 20, 20), background=(230, 225, 235), angle=-25
    )
    level = measure_lettering_style(level_strip, region("s", "두근두근", level_box, kind="sfx")).weight
    tilted = measure_lettering_style(tilted_strip, region("s", "두근두근", tilted_box, kind="sfx")).weight
    assert level is not None and tilted is not None
    assert tilted == pytest.approx(level, rel=0.1)  # strokes are measured levelled
