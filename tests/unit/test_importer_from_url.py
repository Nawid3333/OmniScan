"""Tests for omniscan.importer.from_url and `omniscan import --from-url` (issue #71).

The downloader is a real subprocess here: `tests/fixtures/fake_mangadl.py`, which follows the documented `--json`
contract (folders under `--out`, one result object on stdout, exit 0 / 2 / 1) for the scenario named in the URL.
No network, no real downloader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, ImporterConfig, PathsConfig
from omniscan.importer.from_url import (
    DOWNLOADS_DIR,
    DownloaderError,
    DownloadResult,
    discard_download,
    download,
    download_dir,
    downloader_command,
    parse_result,
    plan_download,
)
from omniscan.importer.plan import ImportPlanError
from tests.fixtures import fake_mangadl

runner = CliRunner()


@pytest.fixture
def mangadl() -> list[str]:
    """The command that runs the fake downloader (tests/fixtures/fake_mangadl.py)."""
    return [sys.executable, str(Path(fake_mangadl.__file__))]


@pytest.fixture
def cfg(tmp_path: Path, mangadl: list[str], monkeypatch: pytest.MonkeyPatch) -> Config:
    """Config with all paths under tmp_path and the fake downloader configured; the CLI reads it."""
    config = Config(
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
        importer=ImporterConfig(downloader=mangadl),
    )
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: config)
    return config


def _library_chapters(cfg: Config, series: str) -> dict[str, list[str]]:
    """Chapter folder -> page file names, as imported."""
    root = cfg.paths.library_root / series
    return (
        {d.name: sorted(p.name for p in d.iterdir()) for d in sorted(root.iterdir())} if root.is_dir() else {}
    )


# ---- the subprocess contract -------------------------------------------------------------------------------------


def test_download_passes_the_documented_flags_and_reads_the_result(
    tmp_path: Path, mangadl: list[str]
) -> None:
    dest = tmp_path / "dl"
    result = download("https://fake.test/complete", dest, command=mangadl, chapters="1-2")

    assert result == DownloadResult(
        out_dir=dest,
        series="solo-leveling",
        complete=["num1_Chapter 1", "num2_Chapter 2"],
        incomplete=[],
        finished=True,
    )
    argv = json.loads((tmp_path / "dl.argv.json").read_text(encoding="utf-8"))
    assert argv == [
        f"--out={dest}",
        "--yes",
        "--json",
        "--no-convert",
        "--chapters=1-2",
        "--",
        "https://fake.test/complete",
    ]


def test_a_url_or_range_starting_with_a_dash_stays_a_value(tmp_path: Path, mangadl: list[str]) -> None:
    """`--from-url "--out=/elsewhere"` must not become one of the downloader's own options."""
    dest = tmp_path / "dl"
    download("--out=/elsewhere/unknown", dest, command=mangadl, chapters="-3")
    argv = json.loads((tmp_path / "dl.argv.json").read_text(encoding="utf-8"))
    assert argv[-2:] == ["--", "--out=/elsewhere/unknown"] and "--chapters=-3" in argv
    assert next(arg for arg in argv if arg.startswith("--out=")) == f"--out={dest}"


def test_messages_go_to_on_line_when_given(tmp_path: Path, mangadl: list[str]) -> None:
    lines: list[str] = []
    download("https://fake.test/complete", tmp_path / "dl", command=mangadl, on_line=lines.append)
    assert lines == ["Site: fake (complete)"]


def test_exit_2_is_a_partial_result_not_an_error(tmp_path: Path, mangadl: list[str]) -> None:
    result = download("https://fake.test/partial", tmp_path / "dl", command=mangadl)
    assert result.complete == ["num1_Chapter 1"]
    assert result.incomplete == ["num2_Chapter 2"]
    assert not result.finished


def test_a_failed_run_reports_the_downloader_error(tmp_path: Path, mangadl: list[str]) -> None:
    with pytest.raises(DownloaderError, match="the download failed: Unsupported site"):
        download("https://fake.test/error", tmp_path / "dl", command=mangadl)


def test_no_result_object_is_an_error_carrying_the_last_messages(tmp_path: Path, mangadl: list[str]) -> None:
    with pytest.raises(DownloaderError, match="without printing a result") as exc:
        download("https://fake.test/silent", tmp_path / "dl", command=mangadl, on_line=lambda _line: None)
    assert "RuntimeError: boom" in str(exc.value)


def test_an_unknown_schema_is_refused(tmp_path: Path, mangadl: list[str]) -> None:
    with pytest.raises(DownloaderError, match=r"schema 9.*update OmniScan"):
        download("https://fake.test/schema9", tmp_path / "dl", command=mangadl)


def test_a_result_without_schema_reads_as_downloader_1_4(tmp_path: Path) -> None:
    stdout = json.dumps({"site": "x", "chapters": 1, "incomplete_chapters": []})
    result = parse_result(stdout, 0, tmp_path)
    assert result.complete is None and result.series is None and result.finished


def test_only_the_last_stdout_line_is_the_result(tmp_path: Path) -> None:
    """An older downloader could print a registry warning on stdout before the object."""
    stdout = "[registry] failed to load site module src.sites.mangago\n" + json.dumps(
        {"schema": 1, "series": "s", "complete_chapters": ["num1_Chapter 1"], "incomplete_chapters": []}
    )
    assert parse_result(stdout, 0, tmp_path).complete == ["num1_Chapter 1"]


def test_unexpected_field_types_are_ignored(tmp_path: Path) -> None:
    stdout = json.dumps(
        {"schema": 1, "series": 7, "complete_chapters": "all", "incomplete_chapters": [1, "a"]}
    )
    result = parse_result(stdout, 2, tmp_path)
    assert result.series is None and result.complete is None and result.incomplete == ["a"]


def test_a_missing_downloader_says_how_to_install_it() -> None:
    with pytest.raises(DownloaderError, match="install manhwa-manga-downloader"):
        downloader_command("definitely-not-a-real-mangadl-executable")
    with pytest.raises(DownloaderError, match="empty"):
        downloader_command([])


def test_a_command_list_keeps_its_arguments(mangadl: list[str]) -> None:
    command = downloader_command(mangadl)
    assert Path(command[0]).resolve() == Path(sys.executable).resolve() and command[1:] == mangadl[1:]


def test_the_download_folder_is_the_same_for_every_run_of_a_url(tmp_path: Path) -> None:
    first = download_dir(tmp_path, "https://fake.test/a")
    assert first == download_dir(tmp_path, " https://fake.test/a ")
    assert first != download_dir(tmp_path, "https://fake.test/b")
    assert first.parent == tmp_path / DOWNLOADS_DIR


# ---- planning ------------------------------------------------------------------------------------------------------


def test_plan_names_the_series_after_the_downloader(tmp_path: Path, mangadl: list[str]) -> None:
    result = download("https://fake.test/complete", tmp_path / "dl", command=mangadl)
    plan = plan_download(result, series=None)
    assert plan.series == "solo-leveling"
    assert [(item.chapter, len(item.files)) for item in plan.items] == [("Chapter 1", 2), ("Chapter 2", 1)]


def test_a_site_id_series_needs_series(tmp_path: Path, mangadl: list[str]) -> None:
    result = download("https://fake.test/id", tmp_path / "dl", command=mangadl)
    with pytest.raises(ImportPlanError, match=r"site's id \(1234\).*--series"):
        plan_download(result, series=None)
    assert plan_download(result, series="Solo Leveling").series == "Solo Leveling"


def test_a_downloader_that_cannot_name_the_series_needs_series(tmp_path: Path, mangadl: list[str]) -> None:
    result = download("https://fake.test/legacy", tmp_path / "dl", command=mangadl)
    with pytest.raises(ImportPlanError, match="pass --series"):
        plan_download(result, series=None)
    assert len(plan_download(result, series="S").items) == 2  # no complete_chapters: every finished folder


def test_only_chapters_this_run_finished_are_planned(tmp_path: Path, mangadl: list[str]) -> None:
    """A chapter left in the folder by an earlier run of the same URL is not part of this run."""
    dest = tmp_path / "dl"
    earlier = dest / "num7_Chapter 7"
    earlier.mkdir(parents=True)
    (earlier / "0001.jpg").write_bytes(b"old")
    plan = plan_download(download("https://fake.test/complete", dest, command=mangadl), series=None)
    assert [item.chapter for item in plan.items] == ["Chapter 1", "Chapter 2"]


def test_incomplete_chapters_are_left_out_with_a_warning(tmp_path: Path, mangadl: list[str]) -> None:
    plan = plan_download(download("https://fake.test/partial", tmp_path / "dl", command=mangadl), series=None)
    assert [item.chapter for item in plan.items] == ["Chapter 1"]
    assert any("run the same command again" in warning for warning in plan.warnings)


def test_a_run_that_finished_no_chapter_cannot_be_planned(tmp_path: Path, mangadl: list[str]) -> None:
    result = download("https://fake.test/none-finished", tmp_path / "dl", command=mangadl)
    with pytest.raises(ImportPlanError, match=r"finished no chapter \(incomplete: num1_Chapter 1\)"):
        plan_download(result, series=None)


def test_discard_removes_the_download_folder(tmp_path: Path, mangadl: list[str]) -> None:
    result = download("https://fake.test/complete", tmp_path / "dl", command=mangadl)
    discard_download(result)
    assert not result.out_dir.exists()


# ---- `omniscan import --from-url` ------------------------------------------------------------------------------------


def test_cli_imports_a_complete_download_and_removes_its_folder(cfg: Config) -> None:
    url = "https://fake.test/complete"
    outcome = runner.invoke(app, ["import", "--from-url", url])
    assert outcome.exit_code == 0, outcome.output
    assert _library_chapters(cfg, "solo-leveling") == {
        "Chapter 1": ["0001.jpg", "0002.jpg"],
        "Chapter 2": ["0001.jpg"],
    }
    assert not download_dir(cfg.paths.work_root, url).exists()


def test_cli_imports_what_finished_and_keeps_the_rest_for_a_rerun(cfg: Config) -> None:
    url = "https://fake.test/partial"
    outcome = runner.invoke(app, ["import", "--from-url", url, "--series", "Solo Leveling"])
    assert outcome.exit_code == 0, outcome.output
    assert list(_library_chapters(cfg, "Solo Leveling")) == ["Chapter 1"]
    assert "run the same command again" in outcome.output
    assert download_dir(cfg.paths.work_root, url).is_dir()


def test_cli_dry_run_downloads_but_writes_nothing_to_the_library(cfg: Config) -> None:
    url = "https://fake.test/complete"
    outcome = runner.invoke(app, ["import", "--from-url", url, "--dry-run"])
    assert outcome.exit_code == 0, outcome.output
    assert "plan for series 'solo-leveling' — 2 chapter(s)" in outcome.output
    assert not cfg.paths.library_root.exists()
    assert download_dir(cfg.paths.work_root, url).is_dir()


def test_cli_reports_a_failed_download_with_exit_2(cfg: Config) -> None:
    outcome = runner.invoke(app, ["import", "--from-url", "https://fake.test/error"])
    assert outcome.exit_code == 2
    assert "the download failed: Unsupported site" in outcome.output


def test_cli_reports_a_missing_downloader(cfg: Config) -> None:
    cfg.importer.downloader = "definitely-not-a-real-mangadl-executable"
    outcome = runner.invoke(app, ["import", "--from-url", "https://fake.test/complete"])
    assert outcome.exit_code == 2
    assert "install manhwa-manga-downloader" in outcome.output


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["import"], "give a folder/archive, or --from-url URL"),
        (["import", ".", "--from-url", "https://fake.test/complete"], "not both"),
        (["import", "--from-url", "https://fake.test/complete", "--move"], "--move does not apply"),
        (["import", ".", "--chapters", "1-3"], "--chapters only applies to --from-url"),
    ],
)
def test_cli_refuses_contradicting_options(cfg: Config, argv: list[str], message: str) -> None:
    outcome = runner.invoke(app, argv)
    assert outcome.exit_code == 2
    assert message in outcome.output


def test_importer_downloader_accepts_a_name_or_a_command_list() -> None:
    assert Config().importer.downloader == "mangadl"
    command = ["C:/mmd/.venv/Scripts/python.exe", "C:/mmd/main.py"]
    assert Config(importer=ImporterConfig(downloader=command)).importer.downloader == command
