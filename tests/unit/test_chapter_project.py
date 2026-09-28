"""Tests for chapter project archives (interchange/project.py) and `omniscan project` (torch-free)."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.interchange.cli as interchange_cli
import omniscan.interchange.project as project_mod
from omniscan.core.config import Config, PathsConfig, series_config
from omniscan.core.manifest import hash_inputs
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    ChapterProject,
    FinalArtifact,
    FinalLine,
    IngestArtifact,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.edits import store
from omniscan.interchange.project import PROJECT_FILE, ProjectError, pack, show, unpack


def make_cfg(root: Path) -> Config:
    return Config(
        paths=PathsConfig(library_root=root / "lib", work_root=root / "work", output_root=root / "out")
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_cfg(tmp_path / "sender")


@pytest.fixture
def chapter(cfg: Config) -> ChapterPaths:
    """A chapter with raw pages, stage artifacts, a hand edit with its undo history, a nested work file, a temp and
    a hidden file (both left out), finished pages and the series' settings files."""
    series = SeriesPaths.from_config(cfg, "S")
    paths = series.chapter("Chapter 1")
    paths.raw_dir.mkdir(parents=True)
    for name in ("001.jpg", "002.jpg"):
        (paths.raw_dir / name).write_bytes(b"\xff\xd8" + name.encode() * 50)
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=200,
        strip_height=400,
        files=[
            SourceFile(
                index=i, name=name, sha256="0" * 64, width=200, height=200, y0=200 * i, y1=200 * (i + 1)
            )
            for i, name in enumerate(("001.jpg", "002.jpg"))
        ],
    ).save(paths.artifact("ingest.json"))
    SlicesArtifact(strip_width=200, strip_height=400, bands=[], slices=[Slice(index=0, y0=0, y1=400)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=10, y0=10, x1=90, y1=60),
                text="안녕",
            )
        ]
    ).save(paths.artifact("ocr.json"))
    FinalArtifact(judge_model="judge", lines=[FinalLine(region_id="r0001", text="Hi", decision="pick")]).save(
        paths.artifact("final.json")
    )
    store.set_translation(paths, "r0001", "Hello!", direction="ltr")
    (paths.work_dir / "translations").mkdir()
    (paths.work_dir / "translations" / "gemma.json").write_text('{"x": 1}', encoding="utf-8")
    (paths.work_dir / "ocr.json.tmp").write_text("half-written", encoding="utf-8")
    (paths.work_dir / ".DS_Store").write_bytes(b"junk")
    paths.output_dir.mkdir(parents=True)
    (paths.output_dir / "0001.jpg").write_bytes(b"\xff\xd8finished")
    (series.library_dir / "series.toml").write_text("[ocr]\nlang = 'ko'\n", encoding="utf-8")
    (series.library_dir / "voices.toml").write_text("[[character]]\nname = 'Jinwoo'\n", encoding="utf-8")
    return paths


def tree(root: Path) -> dict[str, str]:
    """Relative path -> sha256 of every file under `root`."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_a_chapter_travels_with_all_its_work(chapter: ChapterPaths, cfg: Config, tmp_path: Path) -> None:
    archive = tmp_path / "ch1.omniscan"
    packed = pack(SeriesPaths.from_config(cfg, "S"), "Chapter 1", archive)
    paths_in = sorted(file.path for file in packed.files)
    assert "work/translations/gemma.json" in paths_in and "work/edits_history.json" in paths_in
    assert not [p for p in paths_in if p.endswith(".tmp") or "/." in p or p.startswith("output/")]
    assert {p for p in paths_in if p.startswith("series/")} == {"series/series.toml", "series/voices.toml"}
    assert show(archive) == packed
    receiver = make_cfg(tmp_path / "receiver")
    done = unpack(archive, receiver)
    got = done.paths
    assert (got.series, got.chapter, done.replaced) == ("S", "Chapter 1", False)
    assert sorted(done.series_files) == ["series.toml", "voices.toml"] and done.kept_series_files == []
    assert tree(got.raw_dir) == tree(chapter.raw_dir)
    sent = {
        k: v for k, v in tree(chapter.work_dir).items() if not k.endswith(".tmp") and not k.startswith(".")
    }
    assert tree(got.work_dir) == sent
    assert store.load_edits(got) == store.load_edits(chapter) and store.history_steps(got) == (1, 0)
    assert not got.output_dir.exists()  # finished pages only with with_output
    raw = sorted(got.raw_dir.iterdir())
    assert hash_inputs(raw) == hash_inputs(sorted(chapter.raw_dir.iterdir()))  # the stages stay done
    assert not list(got.work_dir.parent.glob(".*"))  # no staging folder left behind


def test_finished_pages_renaming_and_replacing(chapter: ChapterPaths, cfg: Config, tmp_path: Path) -> None:
    archive = tmp_path / "ch1.omniscan"
    pack(SeriesPaths.from_config(cfg, "S"), "Chapter 1", archive, with_output=True)
    receiver = make_cfg(tmp_path / "receiver")
    renamed = unpack(archive, receiver, series="T", chapter="Ep 1")
    assert (renamed.paths.series, renamed.paths.chapter) == ("T", "Ep 1")
    assert (renamed.paths.output_dir / "0001.jpg").read_bytes() == b"\xff\xd8finished"
    with pytest.raises(FileExistsError, match="exists already"):
        unpack(archive, receiver, series="T", chapter="Ep 1")
    (renamed.paths.work_dir / "stale.json").write_text("{}", encoding="utf-8")
    voices = renamed.paths.raw_dir.parent / "voices.toml"
    voices.write_text("# the receiver's own voices\n", encoding="utf-8")
    again = unpack(archive, receiver, series="T", chapter="Ep 1", force=True)
    assert again.replaced and not (renamed.paths.work_dir / "stale.json").exists()
    assert again.kept_series_files == ["series.toml", "voices.toml"] and again.series_files == []
    assert voices.read_text(encoding="utf-8") == "# the receiver's own voices\n"  # never overwritten
    assert not list(renamed.paths.work_dir.parent.glob(".*"))


def craft(
    path: Path, members: dict[str, bytes], compress_type: int = zipfile.ZIP_STORED, **project: object
) -> Path:
    """An archive with `members` and a project.json listing them (fields overridable, e.g. files=…)."""
    files = [
        {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in members.items()
    ]
    meta = {"app_version": "0.0.0", "series": "S", "chapter": "Chapter 1", "files": files, **project}
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data, compress_type=compress_type)
        archive.writestr(PROJECT_FILE, json.dumps(meta))
    return path


RAW = {"raw/001.jpg": b"page"}


def named(key: str, field: str | None, name: str) -> bytes:
    """An artifact whose list `key` names the file `name` (in `field` of each item, or as the item)."""
    return json.dumps({key: [{field: name} if field else name]}).encode()


@pytest.mark.parametrize(
    ("members", "project", "message"),
    [
        ({**RAW, "work/../../evil.txt": b"x"}, {}, "does not write"),
        ({**RAW, "/etc/passwd": b"x"}, {}, "does not write"),
        ({**RAW, "raw\\..\\evil.txt": b"x"}, {}, "does not write"),
        ({**RAW, "raw/./x.jpg": b"x"}, {}, "does not write"),
        ({**RAW, "series/evil.toml": b"x"}, {}, "does not write"),
        ({**RAW, "raw/a:b.jpg": b"x"}, {}, "does not write"),  # no valid file name on Windows
        ({**RAW, "raw/NUL": b"x"}, {}, "does not write"),  # Windows device names, any case or extension
        ({**RAW, "raw/com1.jpg": b"x"}, {}, "does not write"),
        ({**RAW, "work/Aux .json": b"x"}, {}, "does not write"),
        ({**RAW, "output/LPT\u00b9.jpg": b"x"}, {}, "does not write"),
        ({**RAW, "raw/a": b"1", "raw/a/b.jpg": b"2"}, {}, "also lists a folder"),
        ({**RAW, "raw/A": b"1", "raw/a/b.jpg": b"2"}, {}, "also lists a folder"),
        ({**RAW, "series/series.toml": b"[ocr\n"}, {}, "not valid TOML"),
        ({**RAW, "series/series.toml": b"\xff\xfe"}, {}, "not UTF-8"),
        ({**RAW, "series/voices.toml": b"character = 1\n"}, {}, "voices.toml"),
        ({**RAW, "work/ingest.json": named("files", "name", "../../secret.png")}, {}, "outside the chapter"),
        ({**RAW, "work/ingest.json": named("filtered_files", None, "..\\x.jpg")}, {}, "outside the chapter"),
        ({**RAW, "work/export.json": named("files", "name", "/etc/x.jpg")}, {}, "outside the chapter"),
        (
            {**RAW, "output/omniscan-chapter.json": named("pages", "file", "NUL.jpg")},
            {},
            "outside the chapter",
        ),
        ({**RAW, "work/export.json": b"{"}, {}, "not valid JSON"),
        ({**RAW, "work/export.json": b'{"files": 3}'}, {}, "export.json is not valid"),
        ({**RAW, "work/a.json": b"1", "work/A.json": b"2"}, {}, "twice"),
        ({"work/ocr.json": b"{}"}, {}, "no raw pages"),
        (RAW, {"series": "../x"}, "cannot be a folder name"),
        (RAW, {"chapter": "_filtered"}, "cannot be a folder name"),
        (RAW, {"series": "CON"}, "cannot be a folder name"),
        (RAW, {"chapter": "nul.txt"}, "cannot be a folder name"),
        (RAW, {"files": []}, "do not match"),
        (RAW, {"files": [{"path": "raw/001.jpg", "sha256": "0" * 64, "bytes": 4}]}, "damaged"),
        (RAW, {"files": [{"path": "raw/001.jpg", "sha256": "0" * 64, "bytes": 2}]}, "larger than"),
        (
            RAW,
            {"files": [{"path": "raw/001.jpg", "sha256": "0" * 64, "bytes": 17 * 1024**3}]},
            "more than 16 GB",
        ),
    ],
)
def test_unsafe_or_damaged_archives_are_refused(
    tmp_path: Path, members: dict[str, bytes], project: dict[str, Any], message: str
) -> None:
    receiver = make_cfg(tmp_path / "receiver")
    with pytest.raises(ProjectError, match=message):
        unpack(craft(tmp_path / "bad.omniscan", members, **project), receiver)
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "bad.omniscan"]
    assert written == []  # nothing lands anywhere, not even half an unpack


def test_someone_elses_series_toml_gives_only_numbers_switches_and_choices(tmp_path: Path) -> None:
    hostile = (
        "[inpaint]\n"
        "lama_url = 'https://attacker.example/x.bat'\n"
        "lama_sha256 = '" + "a" * 64 + "'\n"
        "lama_file = '../../../Startup/x.bat'\n"
        "pad_px = 4\n"
        "glyph_mask = false\n"
        "[ocr]\n"
        "lang = 'ja'\n"
        "rec_repo = 'attacker/model'\n"
        "rec_batch_size = 100000\n"
        "drop_conf = 'high'\n"
        "[typeset]\n"
        "font_dialogue = '/home/someone/secret.ttf'\n"
        "style = 'manga'\n"
        "[paths]\n"
        "models_dir = '/tmp/evil'\n"
        "[detect]\n"
        "no_such_key = 1\n"
    )
    receiver = make_cfg(tmp_path / "receiver")
    done = unpack(craft(tmp_path / "p.omniscan", {**RAW, "series/series.toml": hostile.encode()}), receiver)
    assert done.series_files == ["series.toml"]
    assert done.dropped_settings == [
        "inpaint.lama_url",
        "inpaint.lama_sha256",
        "inpaint.lama_file",
        "ocr.rec_repo",
        "ocr.rec_batch_size",
        "ocr.drop_conf",
        "typeset.font_dialogue",
        "paths.models_dir",
        "detect.no_such_key",
    ]
    got = series_config(receiver, done.paths.raw_dir.parent)
    default = Config()
    assert (got.inpaint.pad_px, got.inpaint.glyph_mask, got.ocr.lang, got.typeset.style) == (
        4,
        False,
        "ja",
        "manga",
    )
    assert got.inpaint.lama_url == default.inpaint.lama_url
    assert got.inpaint.lama_file == default.inpaint.lama_file
    assert got.ocr.rec_batch_size == default.ocr.rec_batch_size
    assert got.paths == receiver.paths


def test_series_files_without_anything_to_take(tmp_path: Path) -> None:
    receiver = make_cfg(tmp_path / "receiver")
    members = {**RAW, "series/series.toml": b"[inpaint]\nlama_url = 'https://attacker.example'\n"}
    done = unpack(craft(tmp_path / "p.omniscan", members), receiver)
    assert (done.series_files, done.dropped_settings) == ([], ["inpaint.lama_url"])
    assert not (done.paths.raw_dir.parent / "series.toml").exists()
    own = "# kept as sent\n[ocr]\nlang = 'zh'\n"
    again = unpack(
        craft(tmp_path / "q.omniscan", {**RAW, "series/series.toml": own.encode()}, series="T"), receiver
    )
    assert (again.paths.raw_dir.parent / "series.toml").read_text(encoding="utf-8") == own
    other = unpack(craft(tmp_path / "r.omniscan", members, series="T", chapter="Chapter 2"), receiver)
    assert (other.kept_series_files, other.dropped_settings) == (["series.toml"], [])  # nothing was taken


def test_archive_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receiver = make_cfg(tmp_path / "receiver")
    archive = craft(tmp_path / "p.omniscan", {**RAW, "raw/002.jpg": b"page 2"})
    monkeypatch.setattr(project_mod, "MAX_FILES", 1)
    with pytest.raises(ProjectError, match="more than 1 files"):
        show(archive)
    monkeypatch.setattr(project_mod, "MAX_FILES", 10)
    monkeypatch.setattr(project_mod, "MAX_PROJECT_JSON_BYTES", 100)
    with pytest.raises(ProjectError, match=r"project\.json is too large"):
        unpack(archive, receiver)
    assert not receiver.paths.library_root.exists()


def corrupt(path: Path, how: str, member: str) -> Path:
    """Damage the archive at `path`: flip bytes of `member`'s data, or mark its first member encrypted."""
    data = bytearray(path.read_bytes())
    if how == "encrypted":
        for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):  # local header, central entry
            data[data.find(signature) + offset] |= 0x1
    else:
        start = data.find(member.encode()) + len(member)
        data[start + 4 : start + 12] = bytes(255 - b for b in data[start + 4 : start + 12])
    path.write_bytes(bytes(data))
    return path


@pytest.mark.parametrize(
    ("how", "compress_type", "member"),
    [
        ("flipped", zipfile.ZIP_STORED, "raw/001.jpg"),
        ("flipped", zipfile.ZIP_DEFLATED, "raw/001.jpg"),
        ("flipped", zipfile.ZIP_STORED, PROJECT_FILE),
        ("encrypted", zipfile.ZIP_STORED, "raw/001.jpg"),
    ],
)
def test_damaged_archives_are_refused(tmp_path: Path, how: str, compress_type: int, member: str) -> None:
    receiver = make_cfg(tmp_path / "receiver")
    members = {"raw/001.jpg": bytes(range(256)) * 8}
    archive = corrupt(craft(tmp_path / "p.omniscan", members, compress_type=compress_type), how, member)
    with pytest.raises(ProjectError, match="damaged"):
        unpack(archive, receiver)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [archive]


def test_a_failed_replace_puts_the_old_chapter_back(
    chapter: ChapterPaths, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "ch1.omniscan"
    pack(SeriesPaths.from_config(cfg, "S"), "Chapter 1", archive, with_output=True)
    receiver = make_cfg(tmp_path / "receiver")
    got = unpack(archive, receiver).paths
    store.set_translation(got, "r0001", "The receiver's line", direction="ltr")
    before = {root: tree(root) for root in (got.raw_dir, got.work_dir, got.output_dir)}
    rename = Path.rename

    def locked(self: Path, target: Path) -> Path:
        if self.name == ".Chapter 1.unpacking" and receiver.paths.output_root in self.parents:
            raise PermissionError("a file is open in another program")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", locked)
    with pytest.raises(PermissionError):
        unpack(archive, receiver, force=True)
    assert {root: tree(root) for root in before} == before  # raw and work moved back, output never left
    for root in (got.raw_dir, got.work_dir, got.output_dir):
        assert not list(root.parent.glob(".*"))
    monkeypatch.setattr(Path, "rename", rename)
    leftover = got.work_dir.with_name(".Chapter 1.replaced")
    leftover.mkdir()
    with pytest.raises(ProjectError, match="left from an earlier unpack"):
        unpack(archive, receiver, force=True)  # it may be the only copy: never deleted
    assert leftover.is_dir() and {root: tree(root) for root in before} == before


def test_finished_pages_count_as_the_chapter(chapter: ChapterPaths, cfg: Config, tmp_path: Path) -> None:
    archive = tmp_path / "ch1.omniscan"
    pack(SeriesPaths.from_config(cfg, "S"), "Chapter 1", archive)  # no finished pages
    receiver = make_cfg(tmp_path / "receiver")
    stale = SeriesPaths.from_config(receiver, "S").chapter("Chapter 1").output_dir / "0001.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"finished before")
    with pytest.raises(FileExistsError, match="exists already"):
        unpack(archive, receiver)
    assert stale.read_bytes() == b"finished before"
    unpack(archive, receiver, force=True)
    assert not stale.exists()  # pages lettered from the old chapter's work would not match the new work


def test_pack_refuses_what_unpack_refuses_and_skips_links(
    chapter: ChapterPaths, cfg: Config, tmp_path: Path
) -> None:
    drafts = SeriesPaths.from_config(cfg, "_drafts")
    drafts.chapter("Chapter 1").raw_dir.mkdir(parents=True)
    with pytest.raises(ProjectError, match="cannot be a folder name"):
        pack(drafts, "Chapter 1", tmp_path / "d.omniscan")
    secret = tmp_path / "secret.txt"
    secret.write_text("not the chapter's", encoding="utf-8")
    try:
        (chapter.work_dir / "link.json").symlink_to(secret)
    except OSError:
        pytest.skip("this system does not let tests create symbolic links")
    packed = pack(SeriesPaths.from_config(cfg, "S"), "Chapter 1", tmp_path / "ch1.omniscan")
    assert "work/link.json" not in {file.path for file in packed.files}


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot create the file")
def test_pack_refuses_a_file_name_windows_cannot_create(
    chapter: ChapterPaths, cfg: Config, tmp_path: Path
) -> None:
    (chapter.work_dir / "nul.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProjectError, match=r"nul\.json cannot be packed"):
        pack(SeriesPaths.from_config(cfg, "S"), "Chapter 1", tmp_path / "ch1.omniscan")
    assert list(tmp_path.glob("*.omniscan*")) == []


def test_files_that_are_not_projects(tmp_path: Path) -> None:
    (tmp_path / "x.omniscan").write_text("not a zip", encoding="utf-8")
    with pytest.raises(ProjectError, match="not a zip"):
        show(tmp_path / "x.omniscan")
    with zipfile.ZipFile(tmp_path / "y.omniscan", "w") as archive:
        archive.writestr("raw/001.jpg", b"page")
    with pytest.raises(ProjectError, match=r"no project\.json"):
        show(tmp_path / "y.omniscan")


def test_project_commands(
    chapter: ChapterPaths, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(interchange_cli, "get_config", lambda: cfg)
    packed = runner.invoke(interchange_cli.project_app, ["pack", "S", "Chapter 1"])
    default = tmp_path / "S - Chapter 1.omniscan"
    assert packed.exit_code == 0 and default.is_file()
    assert packed.output.startswith("project: 2 raw page file(s), ")
    assert "work file(s), 2 series file(s) -> " in packed.output
    shown = runner.invoke(interchange_cli.project_app, ["show", str(default)])
    assert shown.output.startswith("project: S/Chapter 1 (OmniScan ") and "unpacked" in shown.output
    as_json = ChapterProject.model_validate_json(
        runner.invoke(interchange_cli.project_app, ["show", str(default), "--json"]).output
    )
    assert as_json.chapter == "Chapter 1"
    monkeypatch.setattr(interchange_cli, "get_config", lambda: make_cfg(tmp_path / "receiver"))
    first = runner.invoke(interchange_cli.project_app, ["unpack", str(default)])
    assert first.output == "project: unpacked S/Chapter 1 (added series.toml, voices.toml)\n"
    refused = runner.invoke(interchange_cli.project_app, ["unpack", str(default)])
    assert refused.exit_code == 2 and "exists already (force replaces it)" in refused.output
    forced = runner.invoke(interchange_cli.project_app, ["unpack", str(default), "--force"])
    assert forced.output == "project: replaced S/Chapter 1 (kept your series.toml, voices.toml)\n"
    unknown = runner.invoke(interchange_cli.project_app, ["pack", "S", "Chapter 9"])
    assert unknown.exit_code == 2 and "no chapter 'Chapter 9' in S" in unknown.output
    (tmp_path / "bad.omniscan").write_text("nope", encoding="utf-8")
    bad = runner.invoke(interchange_cli.project_app, ["unpack", str(tmp_path / "bad.omniscan")])
    assert bad.exit_code == 2 and "not a zip file" in bad.output
    members = {**RAW, "series/series.toml": b"[inpaint]\npad_px = 4\nlama_file = 'x/../y.pt'\n"}
    craft(tmp_path / "t.omniscan", members, series="T")
    noted = runner.invoke(interchange_cli.project_app, ["unpack", str(tmp_path / "t.omniscan")])
    assert noted.output == "project: unpacked T/Chapter 1 (added series.toml; left out inpaint.lama_file)\n"
    monkeypatch.setattr(interchange_cli, "get_config", lambda: cfg)
    SeriesPaths.from_config(cfg, "_x").chapter("c").raw_dir.mkdir(parents=True)
    refused_pack = runner.invoke(interchange_cli.project_app, ["pack", "_x", "c"])
    assert refused_pack.exit_code == 2 and "cannot be a folder name" in refused_pack.output
