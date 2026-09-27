"""Tests for chapter project archives (interchange/project.py) and `omniscan project` (torch-free)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.interchange.cli as interchange_cli
from omniscan.core.config import Config, PathsConfig
from omniscan.core.manifest import hash_inputs
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    ChapterProject,
    FinalArtifact,
    FinalLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
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


def craft(path: Path, members: dict[str, bytes], **project: object) -> Path:
    """An archive with `members` and a project.json listing them (fields overridable, e.g. files=…)."""
    files = [
        {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in members.items()
    ]
    meta = {"app_version": "0.0.0", "series": "S", "chapter": "Chapter 1", "files": files, **project}
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr(PROJECT_FILE, json.dumps(meta))
    return path


RAW = {"raw/001.jpg": b"page"}


@pytest.mark.parametrize(
    ("members", "project", "message"),
    [
        ({**RAW, "work/../../evil.txt": b"x"}, {}, "does not write"),
        ({**RAW, "/etc/passwd": b"x"}, {}, "does not write"),
        ({**RAW, "raw\\..\\evil.txt": b"x"}, {}, "does not write"),
        ({**RAW, "raw/./x.jpg": b"x"}, {}, "does not write"),
        ({**RAW, "series/evil.toml": b"x"}, {}, "does not write"),
        ({**RAW, "raw/a:b.jpg": b"x"}, {}, "does not write"),  # no valid file name on Windows
        ({**RAW, "work/a.json": b"1", "work/A.json": b"2"}, {}, "twice"),
        ({"work/ocr.json": b"{}"}, {}, "no raw pages"),
        (RAW, {"series": "../x"}, "cannot be a folder name"),
        (RAW, {"chapter": "_filtered"}, "cannot be a folder name"),
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
    tmp_path: Path, members: dict[str, bytes], project: dict[str, object], message: str
) -> None:
    receiver = make_cfg(tmp_path / "receiver")
    with pytest.raises(ProjectError, match=message):
        unpack(craft(tmp_path / "bad.omniscan", members, **project), receiver)
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "bad.omniscan"]
    assert written == []  # nothing lands anywhere, not even half an unpack


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
