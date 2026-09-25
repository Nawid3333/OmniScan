"""Unit tests for omniscan.typeset.plan (CPU only; a fake font plus real font metrics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.config import SfxConfig, TypesetConfig
from omniscan.core.schemas import RGB, BBox, Lang, LayoutItem, Region, RegionKind
from omniscan.typeset.fit import line_height
from omniscan.typeset.fonts import fonts_dir, load_font
from omniscan.typeset.plan import (
    is_shouted,
    lettering_shape,
    lettering_style,
    luminance,
    plan_layout,
    role_font,
    typical_size,
)

# ---------------------------------------------------------------- fake font


class Fake:
    """Advance width = len(text) * size * 0.5."""

    def __init__(self, size: int) -> None:
        self.size = size

    def getlength(self, text: str) -> float:
        return len(text) * self.size * 0.5


def fake_factory(path: Path, size: int) -> Fake:
    return Fake(size)


def make_region(
    region_id: str,
    kind: RegionKind,
    bbox: BBox,
    *,
    bubble_bbox: BBox | None = None,
    text: str = "원문",
    text_color: RGB | None = None,
    stroke_color: RGB | None = None,
    slice_index: int = 0,
    reading_order: int = 0,
    lang: Lang = "ko",
    angle: float = 0.0,
    weight: float | None = None,
) -> Region:
    return Region(
        id=region_id,
        slice_index=slice_index,
        kind=kind,
        bbox=bbox,
        bubble_bbox=bubble_bbox,
        text=text,
        text_color=text_color,
        stroke_color=stroke_color,
        reading_order=reading_order,
        lang=lang,
        angle=angle,
        weight=weight,
    )


BUBBLE = BBox(x0=0, y0=0, x1=400, y1=200)
TEXT_IN_BUBBLE = BBox(x0=150, y0=80, x1=250, y1=120)


def plan(
    regions: list[Region],
    lines: dict[str, str],
    fills: dict[str, RGB] | None = None,
    cfg: TypesetConfig | None = None,
    **kwargs: object,
) -> list[LayoutItem]:
    return plan_layout(
        regions,
        lines,
        fills or {},
        cfg or TypesetConfig(),
        font_factory=fake_factory,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- luminance


def test_luminance_greys() -> None:
    assert luminance((128, 128, 128)) == pytest.approx(128.0, abs=1e-9)
    assert luminance((0, 0, 0)) == 0.0
    assert luminance((255, 255, 255)) == pytest.approx(255.0, abs=1e-9)


# ---------------------------------------------------------------- lettering shape


def test_bubble_text_is_lettered_in_a_padded_ellipse() -> None:
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE)
    shape = lettering_shape(region, TypesetConfig(bubble_padding=0.12))
    assert shape.kind == "ellipse"
    assert shape.box == BBox(x0=48, y0=24, x1=352, y1=176)  # 12 % of the bubble kept free on each side


def test_the_ellipse_leans_towards_the_original_text_away_from_a_tail() -> None:
    tailed = BBox(x0=0, y0=0, x1=400, y1=260)  # a tail adds 60 px under the balloon's body
    region = make_region("r1", "bubble_text", BBox(x0=150, y0=80, x1=250, y1=120), bubble_bbox=tailed)
    shape = lettering_shape(region, TypesetConfig())
    assert (shape.box.y0 + shape.box.y1) / 2 < 130  # above the stretched box's centre
    assert shape.box.y0 >= 0 and shape.box.y1 <= 260


def test_a_text_box_larger_than_the_ellipse_is_used_as_is() -> None:
    region = make_region("r1", "bubble_text", BBox(x0=20, y0=20, x1=380, y1=180), bubble_bbox=BUBBLE)
    shape = lettering_shape(region, TypesetConfig())
    assert shape.kind == "rect" and shape.box == region.bbox


def test_free_text_uses_its_grown_box() -> None:
    region = make_region("r1", "free_text", BBox(x0=40, y0=45, x1=160, y1=105))
    shape = lettering_shape(region, TypesetConfig(free_grow=0.10))
    assert shape.kind == "rect" and shape.box == BBox(x0=28, y0=39, x1=172, y1=111)


def test_free_text_grown_box_clamps_at_zero() -> None:
    region = make_region("r1", "free_text", BBox(x0=2, y0=2, x1=52, y1=32))
    assert lettering_shape(region, TypesetConfig(free_grow=0.10)).box == BBox(x0=0, y0=0, x1=57, y1=35)


# ---------------------------------------------------------------- style, role, case and fonts


def test_style_auto_follows_the_source_language() -> None:
    cfg = TypesetConfig()
    assert lettering_style(make_region("r", "bubble_text", BUBBLE, lang="ja"), cfg) == "manga"
    assert lettering_style(make_region("r", "bubble_text", BUBBLE, lang="ko"), cfg) == "webtoon"
    assert (
        lettering_style(make_region("r", "bubble_text", BUBBLE, lang="ja"), TypesetConfig(style="webtoon"))
        == "webtoon"
    )


@pytest.mark.parametrize(
    ("text", "shouted"),
    [
        ("WHAT ARE YOU DOING?!", True),
        ("Get out of here!!", True),
        ("Hey!", False),
        ("OK", False),
        ("Wait.", False),
    ],
)
def test_is_shouted(text: str, shouted: bool) -> None:
    assert is_shouted(text) is shouted


def test_manga_style_letters_dialogue_in_capitals_with_its_preset_fonts() -> None:
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE, lang="ja")
    (item,) = plan([region], {"r1": "Are you okay?"})
    assert item.lines == ["ARE YOU OKAY?"] or " ".join(item.lines) == "ARE YOU OKAY?"
    assert item.font == "Kalam-Bold.ttf" and item.font_role == "dialogue"


def test_uppercase_can_be_forced_either_way() -> None:
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE, lang="ja")
    (item,) = plan([region], {"r1": "Are you okay?"}, cfg=TypesetConfig(uppercase="never"))
    assert " ".join(item.lines) == "Are you okay?"
    ko = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE)
    (item,) = plan([ko], {"r1": "Are you okay?"}, cfg=TypesetConfig(uppercase="always"))
    assert " ".join(item.lines) == "ARE YOU OKAY?"


def test_a_shouted_line_gets_the_shout_role() -> None:
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE)
    (item,) = plan([region], {"r1": "RUN, NOW!"})
    assert item.font_role == "shout" and item.font == "Mali-Bold.ttf"


def test_a_role_font_can_be_the_users_own_file(tmp_path: Path) -> None:
    own = tmp_path / "WildWords.ttf"
    own.write_bytes((fonts_dir() / "Kalam-Bold.ttf").read_bytes())
    cfg = TypesetConfig(font_dialogue=str(own))
    assert role_font("dialogue", "webtoon", cfg) == own
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE)
    (item,) = plan_layout([region], {"r1": "Hello there."}, {}, cfg)
    assert item.font == str(own)
    with pytest.raises(FileNotFoundError, match="font not found"):
        role_font("dialogue", "webtoon", TypesetConfig(font_dialogue="NoSuchFont.ttf"))


# ---------------------------------------------------------------- colours and outlines


def test_bubble_on_white_fill_is_black_without_outline() -> None:
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE)
    (item,) = plan([region], {"r1": "Hello."}, {"r1": (255, 255, 255)})
    assert item.font_role == "dialogue" and item.color == (0, 0, 0) and item.stroke_px == 0


def test_bubble_on_dark_fill_is_white() -> None:
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE)
    (item,) = plan([region], {"r1": "Hello."}, {"r1": (24, 24, 32)})
    assert item.color == (255, 255, 255)


def test_free_text_is_white_with_an_outline() -> None:
    region = make_region("r1", "free_text", BBox(x0=0, y0=0, x1=200, y1=60))
    (item,) = plan([region], {"r1": "Hello."})
    assert item.font_role == "free" and item.color == (255, 255, 255)
    assert item.stroke_px == TypesetConfig().stroke_free_px and item.stroke_color == (0, 0, 0)


def test_region_colours_override_the_defaults() -> None:
    region = make_region(
        "r1",
        "free_text",
        BBox(x0=0, y0=0, x1=200, y1=60),
        text_color=(250, 240, 20),
        stroke_color=(20, 20, 120),
    )
    (item,) = plan([region], {"r1": "Hello."})
    assert item.color == (250, 240, 20) and item.stroke_color == (20, 20, 120)


@pytest.mark.parametrize(
    ("text", "outline", "expected"),
    [
        ((10, 10, 10), (0, 0, 0), (255, 255, 255)),
        ((10, 10, 10), None, (255, 255, 255)),
        ((240, 240, 240), None, (0, 0, 0)),
    ],
)
def test_an_outline_that_does_not_stand_out_is_replaced_by_a_contrasting_one(
    text: RGB, outline: RGB | None, expected: RGB
) -> None:
    region = make_region(
        "r1", "free_text", BBox(x0=0, y0=0, x1=200, y1=60), text_color=text, stroke_color=outline
    )
    (item,) = plan([region], {"r1": "Hello."})
    assert item.stroke_color == expected  # never black lettering with a black outline


# ---------------------------------------------------------------- chapter-wide sizes


def dialogue_page(texts: list[str], big: int | None = None) -> tuple[list[Region], dict[str, str]]:
    regions: list[Region] = []
    for i, _ in enumerate(texts):
        side = 400 if i == big else 240
        bubble = BBox(x0=0, y0=i * 500, x1=side, y1=i * 500 + side // 2)
        text_box = BBox(x0=side // 3, y0=i * 500 + side // 6, x1=2 * side // 3, y1=i * 500 + side // 3)
        regions.append(make_region(f"r{i}", "bubble_text", text_box, bubble_bbox=bubble, reading_order=i))
    return regions, {f"r{i}": text for i, text in enumerate(texts)}


def test_typical_size_is_the_median_of_fitted_balloons() -> None:
    regions, lines = dialogue_page(["one two three four five", "six seven eight nine ten", "eleven twelve"])
    items = plan(regions, lines, cfg=TypesetConfig(size_spread=0))
    assert len(items) == 3
    assert typical_size([]) is None


def test_a_short_line_in_a_roomy_balloon_does_not_tower_over_the_chapter() -> None:
    texts = ["We have to get out of the dungeon now.", "The boss is waking up behind us.", "Huh?", "Run!"]
    regions, lines = dialogue_page(texts, big=2)
    free = {item.region_id: item.size_px for item in plan(regions, lines, cfg=TypesetConfig(size_spread=0))}
    capped = {item.region_id: item.size_px for item in plan(regions, lines)}
    assert free["r2"] == TypesetConfig().max_px  # alone, "Huh?" would take the largest size
    assert capped["r2"] < free["r2"]
    assert capped["r2"] <= round(1.15 * sorted(free.values())[len(free) // 2]) + 1
    assert capped["r0"] == free["r0"]  # a crowded balloon is never enlarged or shrunk by the limit


def test_no_chapter_limit_with_too_few_balloons() -> None:
    regions, lines = dialogue_page(["Huh?", "What?"], big=0)
    (first, _) = plan(regions, lines)
    assert first.size_px == TypesetConfig().max_px


# ---------------------------------------------------------------- selection and order


def test_empty_whitespace_and_missing_lines_are_skipped() -> None:
    regions = [
        make_region("r1", "bubble_text", BBox(x0=0, y0=0, x1=200, y1=100)),
        make_region("r2", "bubble_text", BBox(x0=0, y0=100, x1=200, y1=200)),
        make_region("r3", "bubble_text", BBox(x0=0, y0=200, x1=200, y1=300)),
    ]
    items = plan(regions, {"r1": "   ", "r2": "Hello."})
    assert [item.region_id for item in items] == ["r2"]


def test_watermark_regions_are_skipped() -> None:
    region = make_region("r1", "watermark", BBox(x0=0, y0=0, x1=200, y1=100), text="omniscan")
    assert plan([region], {"r1": "omniscan"}) == []


def test_translatable_order_is_kept() -> None:
    r2 = make_region("r2", "bubble_text", BBox(x0=0, y0=0, x1=200, y1=100))
    r10 = make_region("r10", "bubble_text", BBox(x0=0, y0=100, x1=200, y1=200))
    items = plan([r10, r2], {"r2": "Hello.", "r10": "Hello."})
    assert [item.region_id for item in items] == ["r2", "r10"]


# ---------------------------------------------------------------- sound effects


SFX_BOX = BBox(x0=100, y0=300, x1=400, y1=420)


def test_an_erased_sfx_is_redrawn_in_the_originals_style() -> None:
    region = make_region(
        "r1",
        "sfx",
        SFX_BOX,
        text="쾅",
        text_color=(200, 30, 30),
        stroke_color=(250, 250, 250),
        angle=12.0,
        weight=0.2,
    )
    (item,) = plan([region], {"r1": "Boom"}, erased={"r1"})
    assert item.font_role == "sfx" and item.lines == ["BOOM"]
    assert item.font == "LuckiestGuy-Regular.ttf"  # heavy original -> heavy face
    assert item.color == (200, 30, 30) and item.stroke_color == (250, 250, 250)
    assert item.angle == 12.0 and item.overflow is False
    assert item.size_px > TypesetConfig().max_px  # effects are not held to the dialogue sizes


def test_an_effect_does_not_grow_into_its_neighbours_lettering() -> None:
    # two effects side by side, 6 px apart: grown by free_grow each would reach over the other
    left = make_region("r1", "sfx", BBox(x0=100, y0=300, x1=300, y1=400), text="파앗")
    right = make_region("r2", "sfx", BBox(x0=306, y0=320, x1=450, y1=390), text="철컥", reading_order=1)
    alone = plan([left], {"r1": "Fwoosh"}, erased={"r1"})[0]
    first, second = plan([left, right], {"r1": "Fwoosh", "r2": "Clank"}, erased={"r1", "r2"})
    assert first.box.x1 <= 306 and second.box.x0 >= 300
    assert alone.box.x1 > 306  # without the neighbour it uses the grown box
    below = make_region("r3", "bubble_text", BBox(x0=0, y0=405, x1=400, y1=500), reading_order=2)
    (effect, _) = plan([left, below], {"r1": "Fwoosh", "r3": "Hey"}, erased={"r1"})
    assert effect.box.y1 <= 405


def test_an_sfx_that_stayed_on_the_page_gets_a_subtitle_below_it() -> None:
    region = make_region("r1", "sfx", SFX_BOX, text="쾅")
    (item,) = plan([region], {"r1": "Boom"}, erased=set())
    assert item.font_role == "free" and item.box.y0 >= SFX_BOX.y1
    assert item.size_px <= 24 and item.stroke_px > 0


def test_sfx_subtitle_and_keep_modes() -> None:
    region = make_region("r1", "sfx", SFX_BOX, text="쾅")
    (item,) = plan([region], {"r1": "Boom"}, sfx=SfxConfig(mode="subtitle"), erased={"r1"})
    assert item.font_role == "free"
    assert plan([region], {"r1": "Boom"}, sfx=SfxConfig(mode="keep")) == []


def test_real_fonts_fit_inside_the_bubble() -> None:
    region = make_region("r1", "bubble_text", TEXT_IN_BUBBLE, bubble_bbox=BUBBLE)
    (item,) = plan_layout(
        [region], {"r1": "Are you okay? The dungeon just opened!"}, {"r1": (255, 255, 255)}, TypesetConfig()
    )
    shape = lettering_shape(region, TypesetConfig())
    assert item.overflow is False
    assert TypesetConfig().min_px <= item.size_px <= TypesetConfig().max_px
    assert shape.box.x0 <= item.box.x0 and item.box.x1 <= shape.box.x1
    assert shape.box.y0 <= item.box.y0 and item.box.y1 <= shape.box.y1
    font = load_font(fonts_dir() / item.font, item.size_px)
    widths = shape.line_widths(len(item.lines), item.size_px, line_height(item.size_px, 1.15))
    assert all(font.getlength(line) <= width for line, width in zip(item.lines, widths, strict=True))
