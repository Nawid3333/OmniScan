"""The whole-page lettering sweep (ocr/sweep.py): grouping, straightened crops, classification — CPU."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from omniscan.core.schemas import BBox, Region, Slice
from omniscan.detect.tiles import plan_tiles
from omniscan.ocr.lines import LineBox
from omniscan.ocr.sfx import default_sfx_text_paths, load_sfx_lexicon, thinnest_tilt
from omniscan.ocr.sweep import (
    Candidate,
    SweepRules,
    group_words,
    outside_regions,
    reading_crops,
    sweep,
    sweep_regions,
)
from tests.fixtures.korean_pages import FONTS_DIR

KO = load_sfx_lexicon(default_sfx_text_paths())["ko"]
RULES = SweepRules(
    words=KO,
    watermark_patterns=("구글검색",),
    min_score=0.5,
    max_chars=8,
    engine="test+craft",
    lang="ko",
)
SLICES = [Slice(index=0, y0=0, y1=500), Slice(index=1, y0=500, y1=1000)]


def word(x0: float, y0: float, x1: float, y1: float, score: float = 0.9) -> LineBox:
    return LineBox((x0, y0, x1, y1), score)


def detected(rid: str, box: BBox, *, bubble: BBox | None = None, slice_index: int = 0) -> Region:
    return Region(id=rid, slice_index=slice_index, kind="bubble_text", bbox=box, bubble_bbox=bubble)


# ---------------------------------------------------------------- which words are new


def test_words_inside_a_region_or_its_balloon_belong_to_it() -> None:
    regions = [
        detected("r0001", BBox(x0=100, y0=100, x1=300, y1=160), bubble=BBox(x0=60, y0=60, x1=340, y1=220))
    ]
    inside_text = word(120, 110, 200, 150)
    inside_balloon = word(70, 180, 150, 210)  # the detector's text box missed this line
    half_out = word(310, 150, 420, 200)  # a sliver over the balloon's edge
    art = word(400, 400, 500, 460)
    kept = outside_regions([inside_text, inside_balloon, half_out, art], regions)
    assert kept == [half_out, art]


# ---------------------------------------------------------------- grouping


def test_words_side_by_side_stay_separate_effects() -> None:
    # CRAFT links the letters of a word itself; two words on one line are two effects (번쩍 스윽)
    candidates = group_words([word(10, 10, 90, 60), word(100, 12, 170, 58)], min_px=20)
    assert [c.box for c in candidates] == [(10, 10, 90, 60), (100, 12, 170, 58)]
    assert not any(c.vertical for c in candidates) and candidates[0].letter_px == 50


def test_stacked_boxes_of_other_widths_or_shapes_stay_apart() -> None:
    wide = group_words([word(50, 60, 110, 120), word(40, 125, 160, 185)], min_px=20)  # 60 vs 120 wide
    assert len(wide) == 2
    lines = group_words(
        [word(10, 10, 200, 60), word(10, 65, 200, 115)], min_px=20
    )  # two lines, not syllables
    assert len(lines) == 2


def test_stacked_syllables_join_into_a_column() -> None:
    (candidate,) = group_words([word(50, 130, 110, 190), word(52, 60, 108, 122)], min_px=20)
    assert candidate.vertical
    assert candidate.words == ((52, 60, 108, 122), (50, 130, 110, 190))  # top to bottom
    assert candidate.letter_px == 60


def test_a_tall_single_word_is_a_column_and_small_lettering_is_dropped() -> None:
    (tall,) = group_words([word(10, 10, 60, 130)], min_px=20)
    assert tall.vertical and tall.letter_px == 50
    assert group_words([word(10, 10, 90, 25)], min_px=20) == []  # 15 px letters


# ---------------------------------------------------------------- what the recogniser reads


def rendered(text: str, *, size: int, tilt: float = 0.0, vertical: bool = False) -> tuple[torch.Tensor, BBox]:
    """Black Korean lettering on a white 600x600 strip, optionally tilted; the strip and its ink box."""
    image = Image.new("L", (600, 600), 255)
    layer = Image.new("L", (600, 600), 0)
    font = ImageFont.truetype(str(FONTS_DIR / "NanumGothic-Bold.ttf"), size)
    rows = list(text) if vertical else [text]
    draw = ImageDraw.Draw(layer)
    for k, row in enumerate(rows):
        draw.text((300, 300 + (k - (len(rows) - 1) / 2) * size * 1.05), row, font=font, anchor="mm", fill=255)
    if tilt:
        layer = layer.rotate(tilt, resample=Image.Resampling.BICUBIC, center=(300, 300))
    image.paste(0, mask=layer)
    x0, y0, x1, y1 = layer.point(lambda v: 255 if v > 127 else 0).getbbox()  # type: ignore[misc]
    strip = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).contiguous()
    return strip, BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def as_candidate(box: BBox, *, vertical: bool = False) -> Candidate:
    b = (float(box.x0), float(box.y0), float(box.x1), float(box.y1))
    return Candidate(box=b, words=(b,), vertical=vertical)


def test_level_lettering_is_read_as_cropped_and_as_bare_letters() -> None:
    strip, box = rendered("쾅쾅쾅", size=60)
    crop, letters = reading_crops(strip, as_candidate(box))
    pad = round(0.06 * box.height) + 4
    assert abs(crop.shape[1] - (box.height + 2 * pad)) <= 3
    assert abs(crop.shape[2] - (box.width + 2 * pad)) <= 3
    assert letters.shape == crop.shape
    assert set(letters.unique().tolist()) == {0, 255}  # black letters on white, nothing else
    black, ink = letters[0] == 0, crop.float().mean(dim=0) < 128
    assert int((black & ink).sum()) >= 0.9 * int((black | ink).sum())  # the ink, anti-aliasing aside


def test_bare_letters_leave_the_art_behind() -> None:
    strip, box = rendered("쾅쾅쾅", size=60)
    _, clean = reading_crops(strip, as_candidate(box))
    middle = (box.x0 + box.x1) // 2
    strip[:, :, middle : middle + 2] = 0  # a thin ink line of the art crossing the lettering
    crop, letters = reading_crops(strip, as_candidate(box))
    line = torch.nonzero((crop[0] == 0).all(dim=0)).flatten()  # columns dark from top to bottom
    assert line.numel() == 2 and letters.shape == clean.shape  # the crop has the line
    assert torch.equal(letters[:, :, line], clean[:, :, line])  # the bare letters do not


def test_tilted_lettering_is_turned_level_before_reading() -> None:
    strip, box = rendered("쾅쾅쾅", size=60, tilt=20)
    crop, letters = reading_crops(strip, as_candidate(box))
    assert abs(thinnest_tilt(crop.float().mean(dim=0) < 128)) <= 3.0
    assert abs(thinnest_tilt(letters[0] == 0)) <= 3.0
    assert crop.shape[2] / crop.shape[1] > 1.3 * box.width / box.height  # a level line, not the tilted box


def test_a_column_is_laid_out_as_a_row() -> None:
    strip, box = rendered("쿵쿵", size=70, vertical=True)
    for row in reading_crops(strip, as_candidate(box, vertical=True)):
        assert row.shape[2] > 1.5 * row.shape[1]
        cols = torch.nonzero((row.float().mean(dim=0) < 128).any(dim=0)).flatten()
        assert int(cols[-1] - cols[0]) > 1.5 * box.width  # two syllables side by side


# ---------------------------------------------------------------- what becomes a region


def cand(x0: float, y0: float, x1: float, y1: float, *, vertical: bool = False) -> Candidate:
    return Candidate(box=(x0, y0, x1, y1), words=((x0, y0, x1, y1),), vertical=vertical)


def test_known_effects_and_watermarks_become_regions() -> None:
    regions = [detected("r0001", BBox(x0=0, y0=0, x1=50, y1=50))]
    candidates = [
        cand(100, 100, 200, 160),
        cand(300, 100, 350, 220, vertical=True),
        cand(100, 600, 260, 650),
        cand(100, 300, 200, 360),
    ]
    readings = [
        [("번찍!", 0.8), ("벤찍", 0.9)],  # the closer reading wins over the surer one
        [("쿨굼", 0.6), ("쿵쿵", 0.7)],
        [("구글검색 사이트", 0.9), ("", 0.0)],
        [("나무", 0.9), ("나문", 0.95)],
    ]
    out, counts = sweep_regions(regions, candidates, readings, SLICES, RULES, (600, 1000))
    assert counts == {"sweep_sfx": 2.0, "sweep_watermarks": 1.0, "sweep_unknown": 1.0}
    assert [(r.id, r.kind, r.slice_index, r.reading_order) for r in out[1:]] == [
        ("r0002", "sfx", 0, 1),
        ("r0003", "sfx", 0, 2),
        ("r0004", "watermark", 1, 0),
    ]
    flash, thuds, stamp = out[1:]
    assert flash.text == "번쩍!" and flash.ocr_alt == "번찍!"  # respelled, punctuation kept
    assert flash.bbox == BBox(x0=100, y0=100, x1=200, y1=160) and flash.confidence == 0.8
    assert flash.lines[0].text == "번쩍!" and flash.lines[0].engine == "test+craft"
    assert thuds.text == "쿵쿵" and thuds.ocr_alt is None and thuds.orientation == "v"
    assert stamp.text == "구글검색 사이트"


def test_unsure_or_foreign_readings_and_hidden_slices_add_nothing() -> None:
    hidden = [Slice(index=0, y0=0, y1=500, blank=True), Slice(index=1, y0=500, y1=1000, filtered=True)]
    two = [cand(10, 10, 90, 60), cand(10, 600, 90, 660)]
    out, _ = sweep_regions([], two, [[("쾅", 0.9)]] * 2, hidden, RULES, (600, 1000))
    assert out == []
    readings = [[("쾅", 0.3)], [("BOOM", 0.9)], [("", 0.9)], [("쾅쾅쾅쾅쾅쾅쾅쾅쾅", 0.9)]]
    out, counts = sweep_regions([], [cand(10, 10, 90, 60)] * 4, readings, SLICES, RULES, (600, 1000))
    assert out == [] and counts["sweep_unknown"] == 2.0  # the Latin and the too-long reading


class FakeWords:
    def __init__(self, words: list[tuple[tuple[float, float, float, float], float]]) -> None:
        self.words = words

    def detect(
        self, tiles: Sequence[torch.Tensor]
    ) -> list[list[tuple[tuple[float, float, float, float], float]]]:
        return [self.words if i == 0 else [] for i in range(len(tiles))]


class FakeReader:
    def __init__(self, reading: tuple[str, float]) -> None:
        self.reading = reading
        self.crops: list[torch.Tensor] = []

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]:
        self.crops.extend(crops)
        return [self.reading for _ in crops]


def test_sweep_reads_only_what_the_regions_do_not_cover() -> None:
    strip, box = rendered("쾅", size=90)
    regions = [detected("r0001", BBox(x0=0, y0=0, x1=120, y1=80))]
    words = FakeWords([((10.0, 10.0, 100.0, 60.0), 0.9), ((box.x0, box.y0, box.x1, box.y1), 0.95)])
    reader = FakeReader(("쾅", 0.9))
    tiles = plan_tiles(600, 600, 1280, 0.25)
    out, metrics = sweep(strip, regions, SLICES[:1], tiles, words, reader, RULES, min_px=24)
    assert metrics == {
        "sweep_words": 1.0,
        "sweep_candidates": 1.0,
        "sweep_sfx": 1.0,
        "sweep_watermarks": 0.0,
        "sweep_unknown": 0.0,
    }
    assert len(reader.crops) == 2  # the crop and the bare letters
    assert [r.kind for r in out] == ["bubble_text", "sfx"]


def test_sweep_without_candidates_reads_nothing() -> None:
    strip = torch.full((3, 200, 200), 255, dtype=torch.uint8)
    reader = FakeReader(("쾅", 0.9))
    tiles = plan_tiles(200, 200, 1280, 0.25)
    out, metrics = sweep(strip, [], SLICES[:1], tiles, FakeWords([]), reader, RULES, min_px=24)
    assert out == [] and reader.crops == [] and metrics["sweep_candidates"] == 0.0
