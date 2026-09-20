"""Tests for the `omniscan acquire` CLI (plan/run/check): fakes only, no network, no secrets file."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

import omniscan.acquire.cli as acquire_cli
from omniscan.acquire.extractor import ExtractedImage, Extraction
from omniscan.acquire.plan import write_accepted
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from tests.unit.test_acquire_run import (
    FakeExtractor,
    IMAGE_URL,
    PAGE_URL,
    extraction,
)
from tests.unit.test_acquire_download import Server, noise_image

runner = CliRunner()

TEMPLATE = "https://site.test/chapter-{n}"


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A config with every path under tmp_path, patched into the acquire CLI."""
    config = Config(
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )
    monkeypatch.setattr(acquire_cli, "get_config", lambda: config)
    return config


@pytest.fixture
def acquire_env(tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Wire a fake extractor, a fake key and a mock image transport into the CLI; return the fakes."""
    del tmp_path
    server = Server()
    extractor = FakeExtractor()
    keys: list[str] = []
    monkeypatch.setattr(
        acquire_cli, "make_extractor", lambda key: (keys.append(key), extractor)[1]
    )
    monkeypatch.setattr(
        acquire_cli,
        "get_secrets",
        lambda: SimpleNamespace(extractpics_api_key=SecretStr("test-key")),
    )
    monkeypatch.setattr(acquire_cli, "make_image_client", server.client)
    return server, extractor, keys


def write_sources(cfg: Config, series: str, entries: list[tuple[str, str]]) -> None:
    """A sources.toml for the series with one [[chapter]] entry per (name, url)."""
    path = cfg.paths.library_root / series / "sources.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"[[chapter]]\nname = {name!r}\nurl = {url!r}\n\n" for name, url in entries),
        encoding="utf-8",
    )


def script_chapters(server: Server, extractor: FakeExtractor, first: int, last: int) -> None:
    """Script the fake extractor and mock transport for chapters first..last of 5 pages each."""
    for n in range(first, last + 1):
        extractor.script(PAGE_URL.format(n=n), extraction(n))
        for i in range(1, 6):
            server.script(IMAGE_URL.format(n=n, i=i), noise_image("JPEG", seed=n * 10 + i))


def script_pages(server: Server, extractor: FakeExtractor, number: int, pages: int) -> None:
    """Script chapter `number` with exactly `pages` real page images (no extras)."""
    extractor.script(
        PAGE_URL.format(n=number),
        Extraction(
            id=f"e{number}",
            page_url=PAGE_URL.format(n=number),
            images=tuple(ExtractedImage(url=IMAGE_URL.format(n=number, i=i)) for i in range(1, pages + 1)),
            credits=1,
        ),
    )
    for i in range(1, pages + 1):
        server.script(IMAGE_URL.format(n=number, i=i), noise_image("JPEG", seed=number * 10 + i))


# ---------------------------------------------------------------- plan


def test_plan_template_select_prints_rows_and_estimate(cfg: Config) -> None:
    result = runner.invoke(
        app,
        ["acquire", "plan", "S", "--template", TEMPLATE, "--first", "1", "--last", "5", "--select", "2-4"],
    )
    assert result.exit_code == 0
    assert len([line for line in result.output.splitlines() if "https://site.test/chapter-" in line]) == 3
    assert "3 chapter(s): 3 to do, 0 done — estimated credits: 3 (mode basic)" in result.output


def test_plan_json(cfg: Config) -> None:
    result = runner.invoke(
        app, ["acquire", "plan", "S", "--template", TEMPLATE, "--first", "1", "--last", "2", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["series"] == "S"
    assert payload["mode"] == "basic"
    assert payload["credits_total"] == 2
    assert payload["todo"] == 2 and payload["done"] == 0
    assert [item["name"] for item in payload["items"]] == ["Chapter 1", "Chapter 2"]
    assert payload["items"][0] == {
        "name": "Chapter 1",
        "url": "https://site.test/chapter-1",
        "state": "todo",
        "credits": 1,
    }


def test_plan_mode_advanced_costs_two_per_chapter(cfg: Config) -> None:
    result = runner.invoke(
        app,
        ["acquire", "plan", "S", "--template", TEMPLATE, "--first", "1", "--last", "5",
         "--select", "2-4", "--mode", "advanced"],
    )
    assert result.exit_code == 0
    assert "3 chapter(s): 3 to do, 0 done — estimated credits: 6 (mode advanced)" in result.output


def test_plan_urls_file_with_comments_and_named_lines(tmp_path: Path, cfg: Config) -> None:
    urls = tmp_path / "urls.txt"
    urls.write_text(
        "# demo chapter list\n\nChapter 1 A | https://site.test/a/1\nhttps://site.test/a/2\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["acquire", "plan", "S", "--urls", str(urls)])
    assert result.exit_code == 0
    names = [line.split()[1] for line in result.output.splitlines() if "https://site.test/a/" in line]
    assert names == ["Chapter", "Chapter"]  # names column of the two rows
    assert "Chapter 1 A" in result.output
    assert "2 chapter(s): 2 to do, 0 done — estimated credits: 2 (mode basic)" in result.output


def test_plan_urls_stdin(cfg: Config) -> None:
    result = runner.invoke(
        app, ["acquire", "plan", "S", "--urls", "-"], input="https://site.test/s/1\n"
    )
    assert result.exit_code == 0
    assert "Chapter 1" in result.output
    assert "1 chapter(s): 1 to do, 0 done — estimated credits: 1 (mode basic)" in result.output


def test_plan_link_twice_and_first_number(cfg: Config) -> None:
    result = runner.invoke(
        app,
        ["acquire", "plan", "S", "--link", "https://site.test/a/1", "--link", "https://site.test/a/2",
         "--first-number", "5"],
    )
    assert result.exit_code == 0
    assert "Chapter 5" in result.output and "Chapter 6" in result.output
    assert "2 chapter(s): 2 to do, 0 done — estimated credits: 2 (mode basic)" in result.output


def test_plan_two_source_groups_is_refused(tmp_path: Path, cfg: Config) -> None:
    urls = tmp_path / "urls.txt"
    urls.write_text("https://site.test/a/1\n", encoding="utf-8")
    result = runner.invoke(
        app, ["acquire", "plan", "S", "--urls", str(urls), "--link", "https://site.test/a/2"]
    )
    assert result.exit_code == 2
    assert "give at most one of --urls, --link, --template" in result.output


def test_plan_without_sources_prints_the_message(cfg: Config) -> None:
    result = runner.invoke(app, ["acquire", "plan", "S"])
    assert result.exit_code == 2
    expected = cfg.paths.library_root / "S" / "sources.toml"
    assert f"no chapter sources: give --urls, --link, --template or create {expected}" in result.output


def test_plan_done_chapter_costs_zero(cfg: Config) -> None:
    chapter = cfg.paths.library_root / "S" / "Chapter 1"
    chapter.mkdir(parents=True)
    write_accepted(chapter, "ok", [], 4, source_url="https://site.test/chapter-1")
    result = runner.invoke(
        app, ["acquire", "plan", "S", "--template", TEMPLATE, "--first", "1", "--last", "2"]
    )
    assert result.exit_code == 0
    assert "2 chapter(s): 1 to do, 1 done — estimated credits: 1 (mode basic)" in result.output
    done_row = next(line for line in result.output.splitlines() if "Chapter 1" in line)
    assert "done" in done_row and " 0 " in done_row


def test_plan_drm_url_is_refused(cfg: Config) -> None:
    result = runner.invoke(app, ["acquire", "plan", "S", "--link", "https://comic.naver.com/webtoon/1"])
    assert result.exit_code == 2
    assert "Naver Webtoon" in result.output


def test_plan_select_that_matches_nothing_exits_2(cfg: Config) -> None:
    result = runner.invoke(
        app, ["acquire", "plan", "S", "--template", TEMPLATE, "--first", "1", "--last", "2",
              "--select", "99"]
    )
    assert result.exit_code == 2
    assert "nothing matches '99'" in result.output


# ---------------------------------------------------------------- run


def test_run_happy_path_with_yes(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    script_chapters(server, extractor, 1, 3)
    result = runner.invoke(
        app, ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "3", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert (
        "You are responsible for having the right to download and process this content." in result.output
    )
    assert "3 chapter(s): 3 to do, 0 done — estimated credits: 3 (mode basic)" in result.output
    assert "✓ Chapter 1: 5 pages" in result.output
    assert "✓ Chapter 3: 5 pages" in result.output
    assert "ok 3, review 0, failed 0, skipped 0 — credits used 3" in result.output
    assert keys == ["test-key"]
    assert "test-key" not in result.output and "test-key" not in result.stderr
    assert (cfg.paths.library_root / "S" / "Chapter 1" / "001.jpg").exists()
    assert (cfg.paths.library_root / "S" / "Chapter 1" / "accepted.json").exists()


def test_run_confirmation_declined(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    script_chapters(server, extractor, 1, 3)
    result = runner.invoke(
        app, ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "3"],
        input="n\n",
    )
    assert result.exit_code == 2
    assert "Spend up to 3 credit(s)?" in result.output
    assert "aborted" in result.output
    assert extractor.calls == []
    assert keys == []


def test_run_confirmation_eof_is_an_abort(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    result = runner.invoke(
        app, ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "1"]
    )
    assert result.exit_code == 2
    assert "aborted" in result.output
    assert extractor.calls == []


def test_run_confirmation_accepted(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    script_chapters(server, extractor, 1, 1)
    result = runner.invoke(
        app, ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "1"],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert extractor.calls == ["https://site.test/chapter-1"]


def test_run_missing_api_key_refuses(cfg: Config, acquire_env, monkeypatch: pytest.MonkeyPatch) -> None:
    server, extractor, keys = acquire_env
    monkeypatch.setattr(acquire_cli, "get_secrets", lambda: SimpleNamespace(extractpics_api_key=None))
    result = runner.invoke(
        app, ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "1", "--yes"]
    )
    assert result.exit_code == 2
    assert "EXTRACTPICS_API_KEY is not set; put it in ~/.config/omniscan/secrets.env" in result.output
    assert extractor.calls == []
    assert keys == []


def test_run_failed_chapter_exits_1_with_the_summary(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    script_chapters(server, extractor, 1, 3)
    server.script(IMAGE_URL.format(n=2, i=2), 404)
    result = runner.invoke(
        app, ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "3", "--yes"]
    )
    assert result.exit_code == 1
    assert "✗ Chapter 2: 1 of 5 image(s) failed: #2 HTTP 404" in result.output
    assert "ok 2, review 0, failed 1, skipped 0 — credits used 3" in result.output


def test_run_max_credits_skips_the_rest(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    script_chapters(server, extractor, 1, 3)
    result = runner.invoke(
        app,
        ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "3",
         "--yes", "--max-credits", "1"],
    )
    assert result.exit_code == 0
    assert "- Chapter 2: skipped (credit budget reached)" in result.output
    assert "ok 1, review 0, failed 0, skipped 2 — credits used 1" in result.output
    assert extractor.calls == ["https://site.test/chapter-1"]


def test_run_review_event_line(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    library = cfg.paths.library_root / "S"
    for n in (1, 3, 4):
        chapter = library / f"Chapter {n}"
        chapter.mkdir(parents=True)
        write_accepted(chapter, "ok", [], 20, source_url=f"https://site.test/chapter-{n}")
    script_pages(server, extractor, 2, 2)
    result = runner.invoke(
        app,
        ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "4", "--yes",
         "--max-credits", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "? Chapter 2: 2 pages (review: few_pages)" in result.output
    assert "- Chapter 1: skipped (already acquired)" in result.output
    assert "ok 0, review 1, failed 0, skipped 3 — credits used 1" in result.output


def test_run_json_prints_the_result_only_on_stdout(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    script_chapters(server, extractor, 1, 2)
    result = runner.invoke(
        app,
        ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "2", "--yes", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["series"] == "S"
    assert payload["mode"] == "basic"
    assert [outcome["status"] for outcome in payload["outcomes"]] == ["ok", "ok"]
    assert [outcome["name"] for outcome in payload["outcomes"]] == ["Chapter 1", "Chapter 2"]
    assert [outcome["pages"] for outcome in payload["outcomes"]] == [5, 5]
    assert [outcome["credits"] for outcome in payload["outcomes"]] == [1, 1]
    assert payload["outcomes"][0]["error"] is None
    assert payload["outcomes"][0]["findings"] == []
    assert payload["credits_used"] == 2
    # progress went to stderr; stdout stays the final JSON alone
    assert "✓ Chapter 1: 5 pages" in result.stderr
    assert "You are responsible for having the right" in result.stderr
    assert "ok 2" not in result.stdout


def test_run_nothing_to_do(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    chapter = cfg.paths.library_root / "S" / "Chapter 1"
    chapter.mkdir(parents=True)
    write_accepted(chapter, "ok", [], 5, source_url="https://site.test/chapter-1")
    result = runner.invoke(
        app, ["acquire", "run", "S", "--template", TEMPLATE, "--first", "1", "--last", "1", "--yes"]
    )
    assert result.exit_code == 0
    assert "nothing to do" in result.output
    assert extractor.calls == []


def test_run_uses_sources_toml_when_no_source_options_are_given(cfg: Config, acquire_env) -> None:
    server, extractor, keys = acquire_env
    write_sources(cfg, "S", [("Chapter 1", "https://site.test/chapter-1")])
    script_chapters(server, extractor, 1, 1)
    result = runner.invoke(app, ["acquire", "run", "S", "--yes"])
    assert result.exit_code == 0, result.output
    assert "1 chapter(s): 1 to do, 0 done — estimated credits: 1 (mode basic)" in result.output
    assert "✓ Chapter 1: 5 pages" in result.output


# ---------------------------------------------------------------- check


def build_library(cfg: Config) -> None:
    """Chapters 1-3 with 20 pages each (3 also holding a text file named .jpg) and chapter 5 with 5."""
    for name, pages, extra in (("Chapter 1", 20, False), ("Chapter 2", 20, False),
                               ("Chapter 3", 20, True), ("Chapter 5", 5, False)):
        chapter_dir = cfg.paths.library_root / "S" / name
        chapter_dir.mkdir(parents=True)
        for i in range(1, pages + 1):
            (chapter_dir / f"{i:04d}.jpg").write_bytes(noise_image("JPEG", seed=i))
        if extra:
            (chapter_dir / "bad.jpg").write_text("not an image", encoding="utf-8")


def test_check_reports_review_failed_and_numbering(cfg: Config) -> None:
    build_library(cfg)
    result = runner.invoke(app, ["acquire", "check", "S"])
    assert result.exit_code == 1
    assert "Chapter 1: 20 pages — ok" in result.output
    assert "Chapter 3: 20 pages — failed" in result.output
    assert "error unreadable: bad.jpg: cannot be decoded" in result.output
    assert "Chapter 5: 5 pages — review" in result.output
    assert "few_pages" in result.output
    assert "chapter 4 is missing" in result.output


def test_check_json(cfg: Config) -> None:
    build_library(cfg)
    result = runner.invoke(app, ["acquire", "check", "S", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["series"] == "S"
    assert [chapter["name"] for chapter in payload["chapters"]] == [
        "Chapter 1", "Chapter 2", "Chapter 3", "Chapter 5",
    ]
    assert payload["chapters"][0]["verdict"] == "ok"
    assert payload["chapters"][2]["verdict"] == "failed"
    assert payload["chapters"][2]["findings"] == [
        {"level": "error", "code": "unreadable", "message": "bad.jpg: cannot be decoded"}
    ]
    assert payload["chapters"][3]["verdict"] == "review"
    assert payload["series_findings"] == [
        {"level": "warn", "code": "numbering_gap", "message": "chapter 4 is missing"}
    ]


def test_check_unknown_series_exits_1(cfg: Config) -> None:
    result = runner.invoke(app, ["acquire", "check", "Nope"])
    assert result.exit_code == 1
    assert "no chapters" in result.output


# ---------------------------------------------------------------- registration


def test_help_lists_the_acquire_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "acquire" in result.output


def test_reference_is_still_a_stub() -> None:
    result = runner.invoke(app, ["reference"])
    assert result.exit_code == 2
    assert "reference: not implemented yet" in result.output