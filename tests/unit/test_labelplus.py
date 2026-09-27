"""Tests for omniscan.interchange — LabelPlus files in and out of a chapter (torch-free)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.interchange.cli as labelplus_cli
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    IngestArtifact,
    Region,
    RegionKind,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.edits import store
from omniscan.interchange.labelplus import (
    Label,
    LabelFile,
    export_labels,
    match_labels,
    parse,
    region_at,
    write,
)

# a file as LabelPlus writes it: BOM, CRLF, its default header, a two-line label, a label outside a balloon
LABELPLUS = (
    "﻿1,0\r\n-\r\n框内\r\n框外\r\n-\r\nDefault Comment\r\nYou can edit me\r\n\r\n\r\n"
    ">>>>>>>>[001.jpg]<<<<<<<<\r\n"
    "----------------[1]----------------[0.075,0.042,1]\r\nWhere am I?\r\nWho are you?\r\n\r\n"
    "----------------[2]----------------[0.5,0.9,2]\r\nBOOM\r\n\r\n"
    ">>>>>>>>[002.jpg]<<<<<<<<\r\n\r\n"
)


def region(rid: str, box: BBox, *, kind: RegionKind = "bubble_text", bubble: BBox | None = None) -> Region:
    return Region(id=rid, slice_index=0, kind=kind, bbox=box, bubble_bbox=bubble, text=f"src {rid}")


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


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
    region("r0001", box(20, 10, 40, 40), bubble=box(0, 0, 100, 60)),
    region("r0002", box(180, 520, 220, 560), kind="sfx"),
    region("r0003", box(100, 900, 300, 960), bubble=box(80, 880, 320, 1000)),
    region("r0004", box(0, 0, 400, 20), kind="watermark"),
]


def test_parse_reads_labelplus_as_it_writes_files() -> None:
    doc = parse(LABELPLUS)
    assert doc.groups == ["框内", "框外"] and doc.comment == "Default Comment\nYou can edit me"
    assert doc.pages == {
        "001.jpg": [
            Label(number=1, x=0.075, y=0.042, group=1, text="Where am I?\nWho are you?"),
            Label(number=2, x=0.5, y=0.9, group=2, text="BOOM"),
        ],
        "002.jpg": [],
    }


def test_write_then_parse_gives_the_same_labels() -> None:
    doc = LabelFile(
        comment="from OmniScan", pages={"001.jpg": [Label(1, 0.1234, 0.5, 2, "Two\nlines")], "2.png": []}
    )
    again = parse(write(doc))
    assert again.comment == "from OmniScan" and again.groups == ["框内", "框外"]
    assert again.pages == {"001.jpg": [Label(1, 0.123, 0.5, 2, "Two\nlines")], "2.png": []}


def test_parse_refuses_what_is_not_labelplus() -> None:
    with pytest.raises(ValueError, match="no page marker"):
        parse("just some notes\n")
    with pytest.raises(ValueError, match="line 3: label position"):
        parse(">>>>>>>>[1.jpg]<<<<<<<<\n\n----------------[1]----------------[left,top]\nhi\n")


def test_export_places_labels_at_the_text_centres_on_their_pages() -> None:
    texts = {"r0001": "Hi", "r0002": "BOOM", "r0003": "Run!", "r0004": "site.com"}
    doc = export_labels(INGEST, REGIONS, texts, comment="c")
    assert doc.pages == {
        "001.jpg": [Label(1, 30 / 400, 25 / 600, 1, "Hi"), Label(2, 200 / 400, 540 / 600, 2, "BOOM")],
        "002.jpg": [Label(1, 200 / 400, 330 / 600, 1, "Run!")],  # the watermark is never a label
    }
    assert export_labels(INGEST, REGIONS, {}).pages == {"001.jpg": [], "002.jpg": []}


def test_labels_point_into_text_boxes_then_balloons() -> None:
    assert region_at(REGIONS, 30, 25).id == "r0001"  # type: ignore[union-attr]
    assert region_at(REGIONS, 90, 55).id == "r0001"  # type: ignore[union-attr]  # in the balloon only
    assert region_at(REGIONS, 350, 300) is None
    assert region_at(REGIONS, 200, 10) is None  # the watermark never takes a label
    nested = [region("big", box(0, 0, 100, 100)), region("small", box(40, 40, 60, 60))]
    assert region_at(nested, 50, 50).id == "small"  # type: ignore[union-attr]


def test_match_joins_labels_of_one_region_and_reports_the_rest() -> None:
    doc = LabelFile(
        pages={
            "001.png": [  # renamed by the group: matched by its stem
                Label(1, 0.075, 0.042, 1, "Where am I?\nWho are you?"),
                Label(2, 0.08, 0.05, 1, "Hello?"),
                Label(3, 0.875, 0.5, 1, "lost"),
                Label(4, 0.5, 0.5, 1, "   "),
            ],
            "002.jpg": [Label(1, 0.5, 0.55, 1, "Run!")],
            "099.jpg": [Label(1, 0.5, 0.5, 1, "nowhere")],
        }
    )
    found = match_labels(doc, INGEST, REGIONS)
    assert found.texts == {"r0001": "Where am I? Who are you? Hello?", "r0003": "Run!"}
    assert found.joined == 1
    assert [(page, label.number) for page, label in found.unmatched] == [("001.png", 3)]
    assert found.unknown_pages == ["099.jpg"]


def test_exported_labels_come_back_to_their_regions() -> None:
    texts = {"r0001": "Hi", "r0002": "BOOM", "r0003": "Run!"}
    found = match_labels(parse(write(export_labels(INGEST, REGIONS, texts))), INGEST, REGIONS)
    assert found.texts == texts and not found.unmatched


# ---------------------------------------------------------------- the CLI


runner = CliRunner()


@pytest.fixture
def chapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChapterPaths:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    monkeypatch.setattr(labelplus_cli, "get_config", lambda: cfg)
    series = SeriesPaths.from_config(cfg, "S")
    (series.library_dir / "Chapter 1").mkdir(parents=True)
    paths = series.chapter("Chapter 1")
    INGEST.save(paths.artifact("ingest.json"))
    SlicesArtifact(strip_width=400, strip_height=1200, bands=[], slices=[Slice(index=0, y0=0, y1=1200)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(regions=REGIONS).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id="r0001", text="Hi", decision="pick"),
            FinalLine(region_id="r0003", text="Run!", decision="pick"),
        ],
    ).save(paths.artifact("final.json"))
    return paths


def test_cli_export_writes_a_file_labelplus_reads(chapter: ChapterPaths, tmp_path: Path) -> None:
    result = runner.invoke(labelplus_cli.labelplus_app, ["export", "S", "Chapter 1"])
    target = tmp_path / "out" / "S" / "_labelplus" / "Chapter 1.txt"
    assert result.output == f"labelplus: 2 label(s) on 2 page(s) -> {target}\n"
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # a BOM, like LabelPlus' own files
    doc = parse(raw.decode("utf-8"))
    assert [label.text for labels in doc.pages.values() for label in labels] == ["Hi", "Run!"]
    source = tmp_path / "source.txt"
    runner.invoke(
        labelplus_cli.labelplus_app, ["export", "S", "Chapter 1", "--text", "source", "--out", str(source)]
    )
    assert [
        label.text for labels in parse(source.read_text("utf-8-sig")).pages.values() for label in labels
    ] == [
        "src r0001",
        "src r0002",
        "src r0003",
    ]


def test_cli_import_writes_hand_lines_and_skips_unchanged_ones(chapter: ChapterPaths, tmp_path: Path) -> None:
    doc = export_labels(INGEST, REGIONS, {"r0001": "Hi", "r0002": "KABOOM", "r0003": "Run for it!"})
    doc.pages["001.jpg"].append(Label(3, 0.875, 0.5, 1, "lost"))
    file = tmp_path / "proofread.txt"
    file.write_text(write(doc), encoding="utf-8-sig")
    dry = runner.invoke(labelplus_cli.labelplus_app, ["import", "S", "Chapter 1", str(file), "--dry-run"])
    assert dry.output.startswith("labelplus: would import 2 line(s); 1 already the same\n")
    assert "001.jpg label 3 (0.875, 0.500) is outside every region" in dry.output
    assert not chapter.artifact("edits.json").exists()
    runner.invoke(labelplus_cli.labelplus_app, ["import", "S", "Chapter 1", str(file)])
    lines = {line.region_id: line for line in FinalArtifact.load(chapter.artifact("final.json")).lines}
    assert (lines["r0002"].text, lines["r0003"].text, lines["r0003"].decision) == (
        "KABOOM",
        "Run for it!",
        "manual",
    )
    assert sorted(edit.region_id for edit in store.load_edits(chapter).translations) == ["r0002", "r0003"]
    bad = tmp_path / "notes.txt"
    bad.write_text("not labelplus", encoding="utf-8")
    refused = runner.invoke(labelplus_cli.labelplus_app, ["import", "S", "Chapter 1", str(bad)])
    assert refused.exit_code == 2 and "no page marker" in refused.output
