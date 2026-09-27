"""Tests for BallonsTranslator projects and manga-image-translator text files in and out of a chapter (torch-free)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import omniscan.interchange.cli as interchange_cli
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    IngestArtifact,
    InpaintArtifact,
    InpaintItem,
    LayoutArtifact,
    LayoutItem,
    Region,
    RegionKind,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.edits import store
from omniscan.interchange import ballons, mit
from omniscan.interchange.blocks import Block, join_lines, match_blocks, overlap, region_for, strip_box


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def region(rid: str, bbox: BBox, *, kind: RegionKind = "bubble_text", bubble: BBox | None = None) -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=bbox, bubble_bbox=bubble, text=f"src {rid}")


# two 800x1200 pages in a 400-wide strip (scale 0.5): page 1 is strip rows 0-600, page 2 rows 600-1200
INGEST = IngestArtifact(
    series="S",
    chapter="Chapter 1",
    strip_width=400,
    strip_height=1200,
    files=[
        SourceFile(index=0, name="001.jpg", sha256="a", width=800, height=1200, y0=0, y1=600, scale=0.5),
        SourceFile(index=1, name="002.jpg", sha256="b", width=800, height=1200, y0=600, y1=1200, scale=0.5),
    ],
)
REGIONS = [
    region("r0001", box(20, 10, 120, 50), bubble=box(0, 0, 160, 80)),
    region("r0002", box(180, 520, 220, 560), kind="sfx"),
    region("r0003", box(100, 900, 300, 960), bubble=box(80, 880, 320, 1000)),
    region("r0004", box(0, 0, 400, 8), kind="watermark"),
]

# one block as BallonsTranslator saves it (ProjImgTrans.to_dict), trimmed of the fields nothing here reads
BT_BLOCK = {
    "xyxy": [44, 24, 236, 96],
    "lines": [[[44, 24], [236, 24], [236, 58], [44, 58]], [[60, 62], [220, 62], [220, 96], [60, 96]]],
    "language": "unknown",
    "vec": [0.0, 34.0],
    "angle": 0,
    "text": ["어디", "가?"],
    "translation": "Where are\nyou going?",
    "src_is_vertical": False,
    "_detected_font_size": 34.0,
    "fontformat": {"font_family": "Arial", "font_size": 30.0, "frgb": [0, 0, 0], "alignment": 1},
}

# manga-image-translator's --save-text output for two images in one file (--save-text-file), NumPy 2 and 1
MIT_TEXT = """
[C:\\manga\\raw\\001.jpg]

-- 1 --
color: #1: black (fg, bg: #000000 #ffffff)
text:  어디 가?
trans: Where are you going?
coords: [np.int64(44), np.int64(24), np.int64(236), np.int64(24), np.int64(236), np.int64(58), np.int64(44), np.int64(58)]
coords: [np.int64(60), np.int64(62), np.int64(220), np.int64(62), np.int64(220), np.int64(96), np.int64(60), np.int64(96)]

-- 2 --
color: #1: black (fg, bg: #000000 #ffffff)
text:  쾅
trans: BOOM
coords: [360, 1040, 440, 1040, 440, 1120, 360, 1120]


[/home/me/raw/002.jpg]

-- 1 --
color: #2: white (fg, bg: #ffffff #000000)
text:  뛰어!
trans: Run!
[louder]
coords: [np.float32(200.5), np.float32(600.0), np.float32(600.0), np.float32(600.0), np.float32(600.0), np.float32(720.0), np.float32(200.5), np.float32(720.0)]

"""


# ---------------------------------------------------------------- blocks


def test_join_lines_spaces_only_between_words_that_take_spaces() -> None:
    assert join_lines(["어디", "가?"]) == "어디 가?"
    assert join_lines(["どこへ", "行く？", ""]) == "どこへ行く？"
    assert join_lines(["Where are", " you going? "]) == "Where are you going?"
    assert join_lines(["你好", "world"]) == "你好world" and join_lines([]) == ""


def test_blocks_scale_into_the_strip_and_find_their_region() -> None:
    page1, page2 = INGEST.files
    assert strip_box(Block("001.jpg", 44, 24, 236, 96), page1, 400) == box(22, 12, 118, 48)
    assert strip_box(Block("002.jpg", 700, 1100, 900, 1300), page2, 400) == box(350, 1150, 400, 1200)
    assert strip_box(Block("001.jpg", 0, 1300, 10, 1400), page1, 400) is None  # below the page
    assert overlap(box(0, 0, 10, 10), box(5, 5, 100, 100)) == 0.25
    assert region_for(REGIONS, box(22, 12, 118, 48)).id == "r0001"  # type: ignore[union-attr]
    assert region_for(REGIONS, box(130, 60, 150, 76)).id == "r0001"  # type: ignore[union-attr]  # balloon only
    # a block holding the whole sfx box, its centre outside it: the overlap decides, not the centre
    assert region_for(REGIONS, box(150, 500, 300, 570)).id == "r0002"  # type: ignore[union-attr]
    assert region_for(REGIONS, box(250, 0, 350, 6)) is None  # the watermark never takes a block
    assert region_for(REGIONS, box(340, 300, 380, 340)) is None


def test_match_blocks_joins_reports_and_skips_empty_ones() -> None:
    blocks = [
        Block("001.jpg", 44, 24, 236, 96, "어디", "Where"),
        Block("001.png", 50, 30, 200, 90, "가?", "to?"),  # renamed by the group: matched by its stem
        Block("001.jpg", 600, 300, 700, 400, "", "lost"),
        Block("001.jpg", 360, 1040, 440, 1120),  # nothing in it
        Block("002.jpg", 200, 600, 600, 720, "뛰어!", "Run!"),
        Block("099.jpg", 0, 0, 10, 10, "x", "nowhere"),
    ]
    found = match_blocks(blocks, INGEST, REGIONS)
    assert {rid: [b.translation for b in bs] for rid, bs in found.matched.items()} == {
        "r0001": ["Where", "to?"],
        "r0003": ["Run!"],
    }
    assert [(b.translation, strip) for b, strip in found.unmatched] == [("lost", box(300, 150, 350, 200))]
    assert found.unknown_pages == ["099.jpg"]


# ---------------------------------------------------------------- BallonsTranslator


def test_read_project_takes_boxes_and_texts() -> None:
    data = {"directory": "D:/raw", "pages": {"001.jpg": [BT_BLOCK], "002.jpg": []}, "current_img": "001.jpg"}
    assert ballons.read_project(data) == [
        Block("001.jpg", 44.0, 24.0, 236.0, 96.0, "어디 가?", "Where are\nyou going?")
    ]
    old = {"pages": {"1.png": [{"xyxy": [1, 2, 3, 4], "text": "one\ntwo", "translation": None}]}}
    assert ballons.read_project(old) == [Block("1.png", 1.0, 2.0, 3.0, 4.0, "one two", "")]
    for bad, message in (
        ([], 'no "pages" object'),
        ({"pages": {"1.png": {}}}, "not a list"),
        ({"pages": {"1.png": [{"xyxy": [1, 2, 3]}]}}, "block 1: no xyxy box"),
        ({"pages": {"1.png": [{"xyxy": [1, 2, 3, True]}]}}, "block 1: no xyxy box"),
    ):
        with pytest.raises(ValueError, match=message):
            ballons.read_project(bad)


def test_a_written_project_reads_back_to_the_chapter() -> None:
    item = LayoutItem(
        region_id="r0003",
        font_role="dialogue",
        font="ComicNeue-Bold.ttf",
        size_px=20,
        lines=["Run!"],
        box=box(100, 900, 300, 960),
        color=(250, 250, 250),
        stroke_px=2,
        stroke_color=(0, 0, 0),
    )
    english = {"r0001": "Where to?", "r0002": "BOOM", "r0003": "Run!", "r0004": "site.com"}
    blocks = ballons.page_blocks(INGEST, REGIONS, english, [item])
    project = ballons.write_project(Path("/p/Chapter 1"), INGEST, blocks)
    assert project["current_img"] == "001.jpg" and project["directory"] == str(Path("/p/Chapter 1"))
    assert project["image_info"] == {n: {"width": 800, "height": 1200} for n in ("001.jpg", "002.jpg")}
    run = blocks["002.jpg"][0]
    assert run["xyxy"] == [200, 600, 600, 720] and run["fontformat"] == {
        "font_size": 40.0,
        "frgb": [250, 250, 250],
        "srgb": [0, 0, 0],
        "stroke_width": 0.1,
        "alignment": 1,
        "vertical": False,
    }
    back = ballons.read_project(json.loads(json.dumps(project)))
    assert [(b.page, b.translation) for b in back] == [
        ("001.jpg", "Where to?"),
        ("001.jpg", "BOOM"),
        ("002.jpg", "Run!"),
    ]  # the watermark is never a block
    found = match_blocks(back, INGEST, REGIONS)
    assert {rid: bs[0].text for rid, bs in found.matched.items()} == {
        "r0001": "src r0001",
        "r0002": "src r0002",
        "r0003": "src r0003",
    }
    assert ballons.page_name(INGEST.files[0].model_copy(update={"name": "001.GIF"})) == "001.jpg"
    assert ballons.project_file(Path("/p/Chapter 1")) == Path("/p/Chapter 1/imgtrans_Chapter 1.json")


# ---------------------------------------------------------------- manga-image-translator


def test_mit_text_files_parse_as_written() -> None:
    assert mit.parse(MIT_TEXT) == [
        Block("001.jpg", 44.0, 24.0, 236.0, 96.0, "어디 가?", "Where are you going?"),
        Block("001.jpg", 360.0, 1040.0, 440.0, 1120.0, "쾅", "BOOM"),
        Block("002.jpg", 200.5, 600.0, 600.0, 720.0, "뛰어!", "Run!\n[louder]"),  # a two-line translation
    ]
    with pytest.raises(ValueError, match="no \\[image path\\] line"):
        mit.parse("just notes\n")
    with pytest.raises(ValueError, match=r"line 4: region 1 of '1\.png' has no coords"):
        mit.parse("\n[1.png]\n\n-- 1 --\ntext:  a\ntrans: b\n")
    with pytest.raises(ValueError, match="line 6: coords"):
        mit.parse("\n[1.png]\n\n-- 1 --\ntext:  a\ncoords: [1, 2, 3]\n")


# ---------------------------------------------------------------- the CLI


runner = CliRunner()


def png(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def chapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChapterPaths:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    monkeypatch.setattr(interchange_cli, "get_config", lambda: cfg)
    paths = SeriesPaths.from_config(cfg, "S").chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    INGEST.save(paths.artifact("ingest.json"))
    SlicesArtifact(strip_width=400, strip_height=1200, bands=[], slices=[Slice(index=0, y0=0, y1=1200)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(regions=REGIONS).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id="r0001", text="Where are you going?", decision="pick"),
            FinalLine(region_id="r0003", text="Run!", decision="pick"),
        ],
    ).save(paths.artifact("final.json"))
    return paths


def test_ballons_import_takes_translations_sources_and_new_regions(
    chapter: ChapterPaths, tmp_path: Path
) -> None:
    folder = tmp_path / "bt" / "ch1"
    folder.mkdir(parents=True)
    lost = {"xyxy": [600, 300, 700, 400], "text": ["어?"], "translation": "Huh?"}
    boom = {"xyxy": [360, 1040, 440, 1120], "text": ["쾅!"], "translation": "KABOOM"}
    project = {"pages": {"001.jpg": [BT_BLOCK, boom, lost], "002.jpg": []}}
    ballons.project_file(folder).write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
    dry = runner.invoke(interchange_cli.ballons_app, ["import", "S", "Chapter 1", str(folder), "--dry-run"])
    assert dry.output.splitlines() == [
        "ballons: would import 1 English line(s); 1 line(s) already the same",
        "ballons: 001.jpg block at 300,150-350,200 (strip) is over no region",
    ]
    assert not chapter.artifact("edits.json").exists()
    args = ["import", "S", "Chapter 1", str(ballons.project_file(folder)), "--source", "--add"]
    result = runner.invoke(interchange_cli.ballons_app, args)
    assert result.output.splitlines() == [
        "ballons: imported 1 English line(s), 2 source text(s); 1 line(s) already the same",
        "ballons: 001.jpg block at 300,150-350,200 (strip) is over no region — added as m0001",
    ]
    assert store.history_steps(chapter) == (1, 0)  # the whole import is one undo step
    regions = {r.id: r for r in store.current_regions(chapter)}
    assert (regions["r0001"].text, regions["r0002"].text, regions["m0001"].text) == ("어디 가?", "쾅!", "어?")
    assert regions["m0001"].bbox == box(300, 150, 350, 200)
    lines = {line.region_id: line.text for line in FinalArtifact.load(chapter.artifact("final.json")).lines}
    assert (lines["r0001"], lines["r0002"], lines["m0001"]) == ("Where are you going?", "KABOOM", "Huh?")
    again = runner.invoke(interchange_cli.ballons_app, ["import", "S", "Chapter 1", str(folder)])
    assert again.output.startswith("ballons: imported 0 English line(s); 3 line(s) already the same")
    (folder / "broken.json").write_text("{", encoding="utf-8")
    refused = runner.invoke(
        interchange_cli.ballons_app, ["import", "S", "Chapter 1", str(folder / "broken.json")]
    )
    assert refused.exit_code == 2 and "broken.json" in refused.output


def test_ballons_export_writes_a_project_folder(chapter: ChapterPaths, tmp_path: Path) -> None:
    for name, colour in (("001.jpg", (200, 10, 10)), ("002.jpg", (10, 10, 200))):
        (chapter.raw_dir / name).write_bytes(png(800, 1200, colour))  # a PNG named .jpg: Pillow reads either
    target = tmp_path / "out" / "S" / "_ballons" / "Chapter 1"
    first = runner.invoke(interchange_cli.ballons_app, ["export", "S", "Chapter 1"])
    assert (
        first.output
        == f"ballons: 3 text block(s) on 2 page(s) (not cleaned yet: no inpainted pages) -> {target}\n"
    )
    assert sorted(p.name for p in target.iterdir()) == ["001.jpg", "002.jpg", "imgtrans_Chapter 1.json"]
    InpaintArtifact(items=[InpaintItem(region_id="r0001", box=box(20, 10, 120, 50), method="flat")]).save(
        chapter.artifact("inpaint.json")
    )
    white = np.full((40, 100, 3), 255, np.uint8)
    np.savez(
        chapter.artifact("patches.npz"), **{"r0001.pixels": white, "r0001.mask": np.ones((40, 100), bool)}
    )
    LayoutArtifact(
        items=[
            LayoutItem(
                region_id="r0001",
                font_role="dialogue",
                font="ComicNeue-Bold.ttf",
                size_px=15,
                lines=["Where are you going?"],
                box=box(20, 10, 120, 50),
            )
        ]
    ).save(chapter.artifact("layout.json"))
    out = tmp_path / "mine"
    second = runner.invoke(interchange_cli.ballons_app, ["export", "S", "Chapter 1", "--out", str(out)])
    assert second.output == f"ballons: 3 text block(s) on 2 page(s) -> {out}\n"
    with Image.open(out / "inpainted" / "001.png") as cleaned:
        pixels = np.asarray(cleaned.convert("RGB"))
    assert pixels.shape == (1200, 800, 3)  # back at the page's own size
    assert pixels[60, 200].tolist() == [255, 255, 255] and pixels[300, 400].tolist() == [200, 10, 10]
    project = json.loads(ballons.project_file(out).read_text(encoding="utf-8"))
    assert project["directory"] == str(out.resolve())
    first_block = project["pages"]["001.jpg"][0]
    assert (
        first_block["translation"] == "Where are you going?"
        and first_block["fontformat"]["font_size"] == 30.0
    )
    back = ballons.read_project(project)
    assert [b.text for b in back] == ["src r0001", "src r0002", "src r0003"]


def test_mit_import_reads_several_files(chapter: ChapterPaths, tmp_path: Path) -> None:
    first, second = MIT_TEXT.split("\n\n[/home/me/raw/002.jpg]")
    a, b = tmp_path / "001_translations.txt", tmp_path / "002_translations.txt"
    a.write_text(first, encoding="utf-8")
    b.write_text("\n[/home/me/raw/002.jpg]" + second, encoding="utf-8")
    result = runner.invoke(interchange_cli.mit_app, ["import", "S", "Chapter 1", str(a), str(b)])
    assert result.output.splitlines() == ["mit: imported 2 English line(s); 1 line(s) already the same"]
    lines = {line.region_id: line.text for line in FinalArtifact.load(chapter.artifact("final.json")).lines}
    assert (lines["r0002"], lines["r0003"]) == ("BOOM", "Run! [louder]")
    bad = tmp_path / "notes.txt"
    bad.write_text("not a text file", encoding="utf-8")
    refused = runner.invoke(interchange_cli.mit_app, ["import", "S", "Chapter 1", str(bad)])
    assert refused.exit_code == 2 and "no [image path] line" in refused.output
