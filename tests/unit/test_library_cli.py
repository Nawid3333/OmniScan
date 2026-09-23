"""Tests for the `omniscan library` command group (card L1)."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from typer.testing import CliRunner

import omniscan.cli
import omniscan.library.cli
from omniscan.core.config import Config, PathsConfig

runner = CliRunner()

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ANILIST_FIXTURE = FIXTURES / "metadata_anilist_solo_leveling.json"
MANGADEX_FIXTURE = FIXTURES / "metadata_mangadex_solo_leveling.json"
JIKAN_FIXTURE = FIXTURES / "metadata_jikan_solo_leveling.json"


def image_bytes(fmt: str) -> bytes:
    """A real image of the given Pillow format as bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (400, 600)).save(buffer, fmt)
    return buffer.getvalue()


JPEG_BYTES = image_bytes("JPEG")


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A config with every path under tmp_path, patched into the CLI."""
    config = Config(
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )
    monkeypatch.setattr(omniscan.library.cli, "get_config", lambda: config)
    return config


def patch_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Request]:
    """Route the library command's HTTP through a mock transport and record every request."""
    seen: list[httpx.Request] = []

    def fake_client() -> httpx.Client:
        def recording(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        return httpx.Client(transport=httpx.MockTransport(recording), follow_redirects=True, timeout=5.0)

    monkeypatch.setattr(omniscan.library.cli, "make_http_client", fake_client)
    return seen


def fixtures_handler(request: httpx.Request) -> httpx.Response:
    """The three provider fixtures, and a JPEG for every cover host."""
    host = request.url.host
    if host == "graphql.anilist.co":
        return httpx.Response(200, content=ANILIST_FIXTURE.read_bytes())
    if host == "api.mangadex.org":
        return httpx.Response(200, content=MANGADEX_FIXTURE.read_bytes())
    if host == "api.jikan.moe":
        return httpx.Response(200, content=JIKAN_FIXTURE.read_bytes())
    return httpx.Response(200, content=JPEG_BYTES)


def meta_of(cfg: Config, series: str) -> dict[str, Any]:
    return json.loads((cfg.paths.library_root / series / "_meta" / "series.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- cover


def test_cover_picks_the_best_match_and_writes_the_files(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_http(monkeypatch, fixtures_handler)
    result = runner.invoke(omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling"])
    assert result.exit_code == 0, result.output
    assert "1. anilist" in result.output and "Solo Leveling" in result.output
    assert "cover saved:" in result.output
    cover = cfg.paths.library_root / "S" / "_meta" / "cover.jpg"
    assert cover.is_file()
    meta = meta_of(cfg, "S")
    assert meta["provider"] == "anilist"
    assert meta["title"] == "Solo Leveling"
    assert meta["cover_file"] == "cover.jpg"
    assert meta["credit"] == "Cover and metadata: AniList"


def test_cover_pick_2_chooses_the_second_candidate(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, fixtures_handler)
    result = runner.invoke(
        omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling", "--pick", "2"]
    )
    assert result.exit_code == 0, result.output
    meta = meta_of(cfg, "S")
    assert meta["id"] == "201652"  # the second AniList candidate


def test_cover_no_confident_match_exits_1_and_writes_nothing(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_http(monkeypatch, fixtures_handler)
    result = runner.invoke(
        omniscan.cli.app, ["library", "cover", "S", "--title", "Completely Different Title"]
    )
    assert result.exit_code == 1
    assert "no confident match" in result.output
    assert not (cfg.paths.library_root / "S").exists()


def test_cover_json_payload(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, fixtures_handler)
    result = runner.invoke(omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"series", "query", "candidates", "errors", "chosen", "cover"}
    assert payload["series"] == "S" and payload["query"] == "Solo Leveling"
    assert payload["errors"] == {}
    assert [candidate["provider"] for candidate in payload["candidates"]] == [
        "anilist",
        "anilist",
        "anilist",
        "mangadex",
        "mangadex",
        "jikan",
        "jikan",
    ]
    assert payload["candidates"][0] == {
        "provider": "anilist",
        "id": "105398",
        "title": "Solo Leveling",
        "year": 2018,
        "country": "KR",
        "score": 1.0,
        "cover_url": "https://s4.anilist.co/file/anilistcdn/media/manga/cover/large/bx105398-b673Vt5ZSuz3.jpg",
    }
    assert payload["chosen"]["id"] == "105398"
    assert payload["cover"].endswith("cover.jpg")


def test_cover_json_without_a_confident_match(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, fixtures_handler)
    result = runner.invoke(
        omniscan.cli.app,
        ["library", "cover", "S", "--title", "Completely Different Title", "--json"],
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["chosen"] is None and payload["cover"] is None


def test_cover_provider_restriction(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = patch_http(monkeypatch, fixtures_handler)
    result = runner.invoke(
        omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling", "--provider", "mangadex"]
    )
    assert result.exit_code == 0, result.output
    assert all(request.url.host in ("api.mangadex.org", "uploads.mangadex.org") for request in seen)
    assert any(request.url.host == "api.mangadex.org" for request in seen)
    meta = meta_of(cfg, "S")
    assert meta["provider"] == "mangadex"


def test_cover_reports_provider_errors_and_still_succeeds(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.jikan.moe":
            return httpx.Response(504)
        return fixtures_handler(request)

    patch_http(monkeypatch, handler)
    result = runner.invoke(omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling"])
    assert result.exit_code == 0, result.output
    assert "jikan: HTTP 504" in result.output
    assert meta_of(cfg, "S")["provider"] == "anilist"


def test_cover_with_a_candidate_without_cover_fails_cleanly(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload: Any = json.loads(MANGADEX_FIXTURE.read_text(encoding="utf-8"))
    for item in payload["data"]:
        item["relationships"] = [r for r in item["relationships"] if r.get("type") != "cover_art"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.mangadex.org":
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    patch_http(monkeypatch, handler)
    result = runner.invoke(
        omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling", "--provider", "mangadex"]
    )
    assert result.exit_code == 1
    assert "candidate has no cover" in result.output
    assert not (cfg.paths.library_root / "S").exists()


def test_cover_file_sets_a_user_cover_without_any_http_call(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def booby_trapped() -> httpx.Client:
        raise AssertionError("no HTTP expected with --file")

    monkeypatch.setattr(omniscan.library.cli, "make_http_client", booby_trapped)
    source = tmp_path / "mine.png"
    source.write_bytes(image_bytes("PNG"))
    result = runner.invoke(omniscan.cli.app, ["library", "cover", "S", "--file", str(source)])
    assert result.exit_code == 0, result.output
    assert f"cover set from {source}" in result.output
    assert (cfg.paths.library_root / "S" / "_meta" / "cover.png").is_file()
    assert meta_of(cfg, "S")["cover_source"] == "user"


def test_cover_unknown_file_exits_2(cfg: Config) -> None:
    result = runner.invoke(omniscan.cli.app, ["library", "cover", "S", "--file", "does-not-exist.png"])
    assert result.exit_code == 2


def test_cover_existing_text_file_exits_2(cfg: Config, tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("hello", encoding="utf-8")
    result = runner.invoke(omniscan.cli.app, ["library", "cover", "S", "--file", str(source)])
    assert result.exit_code == 2


def test_cover_pick_out_of_range_exits_2(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, fixtures_handler)
    result = runner.invoke(
        omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling", "--pick", "99"]
    )
    assert result.exit_code == 2
    assert not (cfg.paths.library_root / "S").exists()


# ---------------------------------------------------------------- info


def test_info_without_metadata(cfg: Config) -> None:
    result = runner.invoke(omniscan.cli.app, ["library", "info", "S"])
    assert result.exit_code == 1
    assert "no metadata for S" in result.output


def test_info_after_a_cover_run(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, fixtures_handler)
    runner.invoke(omniscan.cli.app, ["library", "cover", "S", "--title", "Solo Leveling"])
    result = runner.invoke(omniscan.cli.app, ["library", "info", "S"])
    assert result.exit_code == 0, result.output
    assert "title: Solo Leveling" in result.output
    assert "provider: anilist" in result.output
    assert "credit: Cover and metadata: AniList" in result.output
    assert "cover:" in result.output

    as_json = runner.invoke(omniscan.cli.app, ["library", "info", "S", "--json"])
    assert as_json.exit_code == 0
    meta = json.loads(as_json.output)
    assert meta["provider"] == "anilist"
    assert meta["cover_file"] == "cover.jpg"
