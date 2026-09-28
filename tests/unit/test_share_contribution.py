"""Tests for contribution archives (share/contribution.py) and `omniscan contribute export` (torch-free)."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import omniscan.share.cli as share_cli
from omniscan.core.config import Config, PathsConfig, ShareConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    GlossaryEntry,
    IngestArtifact,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.edits import store
from omniscan.glossary.store import GlossaryStore
from omniscan.share.contribution import (
    CONTRIBUTION_FILE,
    SALT_FILE,
    ShareOptOutError,
    build,
    digest,
    install_salt,
    read_archive,
    summarize,
    write_archive,
)

WIDTH, PAGE_H = 200, 300
SECRET_CAMERA = "SecretCam 3000"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


def page_image(shade: int) -> Image.Image:
    """A page with a gradient and a dark block, so a wrong crop or a shifted page shows."""
    pixels = np.full((PAGE_H, WIDTH, 3), shade, np.uint8)
    pixels[:, :, 1] = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
    pixels[40:80, 30:90] = 10
    return Image.fromarray(pixels)


def make_chapter(cfg: Config, name: str) -> ChapterPaths:
    """Two raw JPEG pages carrying camera EXIF, stitched as a 600-row strip with four read and judged regions."""
    paths = SeriesPaths.from_config(cfg, "S").chapter(name)
    paths.raw_dir.mkdir(parents=True)
    exif = Image.Exif()
    exif[0x010F] = SECRET_CAMERA  # Make
    for i, shade in enumerate((200, 120)):
        page_image(shade).save(paths.raw_dir / f"scan-{i + 1}.jpg", quality=95, exif=exif)
    IngestArtifact(
        series="S",
        chapter=name,
        strip_width=WIDTH,
        strip_height=2 * PAGE_H,
        files=[
            SourceFile(
                index=i,
                name=f"scan-{i + 1}.jpg",
                sha256="0" * 64,
                width=WIDTH,
                height=PAGE_H,
                y0=i * PAGE_H,
                y1=(i + 1) * PAGE_H,
            )
            for i in range(2)
        ],
    ).save(paths.artifact("ingest.json"))
    SlicesArtifact(
        strip_width=WIDTH, strip_height=2 * PAGE_H, bands=[], slices=[Slice(index=0, y0=0, y1=2 * PAGE_H)]
    ).save(paths.artifact("slices.json"))
    boxes = {"r0001": 20, "r0002": 120, "r0003": 320, "r0004": 420}
    regions = [
        Region(
            id=rid,
            slice_index=0,
            kind="bubble_text",
            bbox=BBox(x0=20, y0=y0, x1=120, y1=y0 + 40),
            reading_order=i,
            text=f"원문 {i + 1}",
        )
        for i, (rid, y0) in enumerate(boxes.items())
    ]
    RegionsArtifact(regions=regions).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id=rid, text=f"Line {i + 1}", decision="pick") for i, rid in enumerate(boxes)
        ],
    ).save(paths.artifact("final.json"))
    return paths


@pytest.fixture
def series(cfg: Config) -> SeriesPaths:
    """One kind of correction per page, so each one alone must make its page count. Chapter 1, page 0: an OCR
    fix, a region type changed, a box added; page 1: a line typed by hand, a suggestion kept. Chapter 2, page 0:
    a box deleted; page 1: lettering set with a font given as a full path. Chapter 3: no edits. Chapter 4: a
    speaker set on page 0, page 1 untouched. A locked and a proposed glossary term."""
    one, two = make_chapter(cfg, "Chapter 1"), make_chapter(cfg, "Chapter 2")
    make_chapter(cfg, "Chapter 3")
    four = make_chapter(cfg, "Chapter 4")
    store.update_region(four, "r0001", direction="ltr", speaker="Jinwoo")
    store.update_region(one, "r0001", direction="ltr", text="원문 하나")
    store.update_region(one, "r0002", direction="ltr", kind="sfx")
    store.add_region(one, BBox(x0=130, y0=200, x1=190, y1=240), direction="ltr", text="추가")
    store.set_translation(one, "r0003", "Typed by hand", direction="ltr")
    store.set_translation(one, "r0004", "Suggested line", direction="ltr", suggested_by="gemma")
    store.delete_region(two, "r0002", direction="ltr")
    font = "C:\\Users\\limex\\fonts\\Mine.ttf"
    store.set_layout(two, "r0003", {"font": font, "size_px": 22, "box": BBox(x0=10, y0=310, x1=150, y1=370)})
    auto = RegionsArtifact.load(one.artifact(store.OCR_AUTO_FILE))  # a re-run whose OCR learned the fix:
    auto.regions[0] = auto.regions[0].model_copy(
        update={"text": "원문 하나"}
    )  # the reading at the edit counts
    auto.save(one.artifact(store.OCR_AUTO_FILE))
    paths = SeriesPaths.from_config(cfg, "S")
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    with GlossaryStore(paths.db) as db:
        db.add(
            GlossaryEntry(source="진우", target="Jinwoo", type="person", status="locked", aliases=["성진우"])
        )
        db.add(GlossaryEntry(source="게이트", target="Gate", status="proposed"))
    return paths


def test_only_corrected_pages_with_the_pipeline_output_next_to_the_edits(
    series: SeriesPaths, cfg: Config
) -> None:
    contribution, pages = build(series, cfg)
    salt = install_salt(cfg.paths.work_root)
    assert contribution.series_id == digest("S", salt)
    one, two, four = contribution.chapters  # Chapter 3 has no edits
    assert (one.id, one.order, two.order, four.order) == (digest("S\0Chapter 1", salt), 0, 1, 3)
    assert [page.index for page in one.pages] == [0, 1] and [page.index for page in two.pages] == [0, 1]
    assert [page.index for page in four.pages] == [0]  # its untouched page 1 stays home
    assert [page.member for page in pages] == [page.image for c in (one, two, four) for page in c.pages]
    speaker = four.pages[0].regions[0]
    assert (speaker.speaker, speaker.edited, speaker.ocr_text, speaker.text) == (
        "Jinwoo",
        True,
        "원문 1",
        "원문 1",
    )
    first = {region.id: region for region in one.pages[0].regions}
    fixed = first["r0001"]
    assert (fixed.ocr_text, fixed.text, fixed.edited) == ("원문 1", "원문 하나", True)
    assert (fixed.english_from, fixed.english, fixed.machine_english) == ("machine", "Line 1", "Line 1")
    assert (first["r0002"].auto_kind, first["r0002"].kind, first["r0002"].edited) == (
        "bubble_text",
        "sfx",
        True,
    )
    added = first["m0001"]
    assert (added.added, added.ocr_text, added.auto_kind, added.text, added.english_from) == (
        True,
        None,
        None,
        "추가",
        None,
    )
    typed, kept = one.pages[1].regions
    assert (typed.id, typed.machine_english, typed.english, typed.english_from, typed.edited) == (
        "r0003",
        "Line 3",
        "Typed by hand",
        "typed",
        False,  # only its English line was written by hand
    )
    assert typed.box == BBox(x0=20, y0=20, x1=120, y1=60)  # page pixels: strip row 320 is row 20 of page 1
    assert (kept.id, kept.english, kept.english_from) == ("r0004", "Suggested line", "suggestion")
    assert [(t.source, t.target, t.aliases) for t in contribution.glossary] == [
        ("진우", "Jinwoo", ["성진우"])
    ]


def test_deleted_boxes_and_lettering(series: SeriesPaths, cfg: Config) -> None:
    contribution, _ = build(series, cfg, ["Chapter 2"])
    (chapter,) = contribution.chapters
    untouched, deleted = chapter.pages[0].regions
    assert (untouched.id, untouched.edited, untouched.english_from) == ("r0001", False, "machine")
    assert (deleted.id, deleted.deleted, deleted.text, deleted.machine_english, deleted.english) == (
        "r0002",
        True,
        "원문 2",
        "Line 2",
        None,
    )
    lettered = chapter.pages[1].regions[0]
    assert lettered.lettering is not None and not lettered.edited
    assert (lettered.lettering.font, lettered.lettering.size_px, lettered.lettering.box) == (
        "Mine.ttf",
        22,
        BBox(x0=10, y0=10, x1=150, y1=70),
    )
    summary = summarize(contribution)
    assert (summary.pages, summary.regions, summary.deleted, summary.lettering) == (2, 4, 1, 1)
    assert (summary.added, summary.ocr_fixes, summary.english, summary.other) == (0, 0, 0, 0)


def test_archive_holds_the_pages_without_names_paths_or_metadata(
    series: SeriesPaths, cfg: Config, tmp_path: Path
) -> None:
    contribution, pages = build(series, cfg)
    archive_path = tmp_path / "share" / "c.zip"
    size = write_archive(archive_path, contribution, pages)
    assert size == archive_path.stat().st_size and not list(archive_path.parent.glob("*.tmp"))
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert names == [CONTRIBUTION_FILE, *(page.member for page in pages)]
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        text = archive.read(CONTRIBUTION_FILE).decode()
        with Image.open(io.BytesIO(archive.read(pages[1].member))) as image:
            assert image.size == (WIDTH, PAGE_H) and "exif" not in image.info and not image.getexif()
            shown = np.asarray(image.convert("RGB")).astype(int)
    today = datetime.now(UTC).date().isoformat()
    for private in ("Chapter", '"S"', str(tmp_path), "limex", "scan-", SECRET_CAMERA, today, '"created"'):
        assert private not in text and all(private not in name for name in names)
    assert (cfg.paths.work_root / SALT_FILE).read_text(encoding="ascii") not in text
    expected = np.asarray(page_image(120)).astype(int)  # pages[1] is Chapter 1's second page
    assert np.abs(shown - expected).mean() < 3
    assert read_archive(archive_path) == contribution


def test_opting_out_per_machine_or_per_series(series: SeriesPaths, cfg: Config) -> None:
    machine_off = cfg.model_copy(update={"share": ShareConfig(enabled=False)})
    with pytest.raises(ShareOptOutError, match="this machine is opted out"):
        build(series, machine_off)
    (series.library_dir / "series.toml").write_text("[share]\nenabled = true\n", encoding="utf-8")
    with pytest.raises(ShareOptOutError, match="this machine is opted out"):
        build(series, machine_off)  # a series cannot opt back in when the machine opted out
    (series.library_dir / "series.toml").write_text("[share]\nenabled = false\n", encoding="utf-8")
    with pytest.raises(ShareOptOutError, match="S is opted out"):
        build(series, cfg)


def test_ids_are_salted_per_install(series: SeriesPaths, cfg: Config, tmp_path: Path) -> None:
    first, _ = build(series, cfg, ["Chapter 1"])
    again, _ = build(series, cfg, ["Chapter 1"])
    assert (first.series_id, first.chapters[0].id) == (again.series_id, again.chapters[0].id)  # stable
    assert first.series_id != hashlib.sha256(b"S").hexdigest()[:16]  # not found by hashing known titles
    other_install = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"work_root": tmp_path / "other-work"})}
    )
    elsewhere = install_salt(other_install.paths.work_root)
    assert elsewhere != install_salt(cfg.paths.work_root) and len(elsewhere) == 32
    assert digest("S", elsewhere) != first.series_id
    (cfg.paths.work_root / SALT_FILE).write_text("not hex", encoding="ascii")
    with pytest.raises(ValueError, match="damaged"):
        build(series, cfg)


def test_redrawn_boxes_cleared_lines_and_repeated_chapters(cfg: Config) -> None:
    five = make_chapter(cfg, "Chapter 5")
    new = store.add_region(five, BBox(x0=22, y0=22, x1=122, y1=64), direction="ltr", text="다시")
    store.update_region(five, "r0002", direction="ltr", kind="sfx")  # an edit claims it: a box drawn
    beside = store.add_region(five, BBox(x0=21, y0=121, x1=121, y1=161), direction="ltr")  # over it adds one
    store.set_translation(five, "r0003", "", direction="ltr")  # the judge's line cleared by hand
    contribution, pages = build(SeriesPaths.from_config(cfg, "S"), cfg, ["Chapter 5", "Chapter 5"])
    (chapter,) = contribution.chapters  # asked twice, built once
    assert len(pages) == len(chapter.pages) == 2
    ids = [region.id for page in chapter.pages for region in page.regions]
    assert ids.count("r0002") == 1 and beside.id in ids
    by_id = {region.id: region for page in chapter.pages for region in page.regions}
    assert not by_id["r0002"].deleted and by_id["r0002"].kind == "sfx"
    replaced = by_id["r0001"]
    assert (replaced.deleted, replaced.replaced_by, replaced.edited, replaced.text) == (
        True,
        new.id,
        True,
        "원문 1",
    )
    assert by_id[new.id].added and not by_id[new.id].deleted
    cleared = by_id["r0003"]
    assert (cleared.english, cleared.english_from, cleared.machine_english) == (None, "typed", "Line 3")
    summary = summarize(contribution)
    assert (summary.redrawn, summary.deleted, summary.added, summary.english) == (1, 0, 2, 1)


def test_a_salt_another_export_just_made_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made = install_salt(tmp_path)
    read = Path.read_text
    raced: list[Path] = []

    def racing(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == SALT_FILE and not raced:  # not there yet when read, written by another export
            raced.append(self)  # right after: it must be read, never replaced
            raise FileNotFoundError(self)
        return read(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", racing)
    assert install_salt(tmp_path) == made and raced


def test_contribute_export_on_the_command_line(
    series: SeriesPaths, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(share_cli, "get_config", lambda: cfg)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    dry = runner.invoke(share_cli.contribute_app, ["export", "S", "--dry-run"])
    assert dry.output == (
        "contribute: would write 3 chapter(s), 5 page(s), 11 region(s) and 1 glossary term(s); corrections:"
        " 1 OCR text, 1 region type, 1 added box, 1 deleted box, 2 English lines, 1 lettering,"
        " 1 other region edit\n"
    )
    assert not list(tmp_path.glob("*.zip"))
    done = runner.invoke(share_cli.contribute_app, ["export", "S", "-c", "Chapter 1"])
    default = tmp_path / f"omniscan-contribution-{digest('S', install_salt(cfg.paths.work_root))}.zip"
    assert done.exit_code == 0 and done.output.startswith("contribute: wrote 1 chapter(s), 2 page(s)")
    assert f"to {default} (" in done.output and len(read_archive(default).chapters) == 1
    as_json = json.loads(
        runner.invoke(share_cli.contribute_app, ["export", "S", "-o", "x.zip", "--json"]).output
    )
    assert (as_json["pages"], as_json["path"]) == (5, "x.zip") and as_json["bytes"] == Path(
        "x.zip"
    ).stat().st_size
    nothing = runner.invoke(share_cli.contribute_app, ["export", "S", "-c", "Chapter 3"])
    assert nothing.output == "contribute: S has no hand corrections to share; nothing exported\n"
    unknown = runner.invoke(share_cli.contribute_app, ["export", "S", "-c", "Chapter 9"])
    assert unknown.exit_code == 2 and "no chapter 'Chapter 9' in S" in unknown.output
    typo = runner.invoke(share_cli.contribute_app, ["export", "Sx"])
    assert typo.exit_code == 2 and "no series 'Sx' in the library" in typo.output
    (tmp_path / "folder.zip").mkdir()
    blocked = runner.invoke(share_cli.contribute_app, ["export", "S", "-o", "folder.zip"])
    assert blocked.exit_code == 2 and "nothing exported" in blocked.output
    assert blocked.exception is None or isinstance(blocked.exception, SystemExit)  # a message, no traceback
    (series.library_dir / "series.toml").write_text("[share]\nenabled = false\n", encoding="utf-8")
    out = runner.invoke(share_cli.contribute_app, ["export", "S", "-o", "y.zip"])
    assert out.exit_code == 2 and "opted out of sharing" in out.output and not Path("y.zip").exists()
