"""Tests for omniscan.library.metadata (card L1): the three providers, matching, covers, files."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from omniscan.library.metadata import (
    ANILIST_QUERY,
    Candidate,
    MetadataClient,
    best_match,
    download_cover,
    find_cover,
    match_score,
    meta_dir,
    read_series_meta,
    set_cover_from_file,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ANILIST_FIXTURE = FIXTURES / "metadata_anilist_solo_leveling.json"
MANGADEX_FIXTURE = FIXTURES / "metadata_mangadex_solo_leveling.json"
JIKAN_FIXTURE = FIXTURES / "metadata_jikan_solo_leveling.json"

MANGADEX_ID = "32d76d19-8a05-4db0-9fc2-e0b0648fe9d0"
MANGADEX_FILE = "e90bdc47-c8b9-4df7-b2c0-17641b645ee1.jpg"

SOLO_LEVELING_TITLES = ("Solo Leveling", "Na Honjaman Level Up", "나 혼자만 레벨업")
ANILIST_CANDIDATE = Candidate(
    provider="anilist",
    id="105398",
    title="Solo Leveling",
    titles=SOLO_LEVELING_TITLES,
    year=2018,
    country="KR",
    status="finished",
    cover_url="https://s4.anilist.co/file/anilistcdn/media/manga/cover/large/bx105398-b673Vt5ZSuz3.jpg",
    url="https://anilist.co/manga/105398",
)
MANGADEX_CANDIDATE = Candidate(
    provider="mangadex",
    id=MANGADEX_ID,
    title="Solo Leveling",
    titles=("Solo Leveling", "Na Honjaman Level-Up", "나 혼자만 레벨업"),
    year=2018,
    country="KR",
    status="completed",
    cover_url=f"https://uploads.mangadex.org/covers/{MANGADEX_ID}/{MANGADEX_FILE}.512.jpg",
    url=f"https://mangadex.org/title/{MANGADEX_ID}",
)
JIKAN_CANDIDATE = Candidate(
    provider="jikan",
    id="121496",
    title="Solo Leveling",
    titles=("Solo Leveling", "Na Honjaman Level-up", "나 혼자만 레벨업"),
    year=2016,
    country="KR",
    status="finished",
    cover_url="https://cdn.myanimelist.net/images/manga/3/222295l.jpg",
    url="https://myanimelist.net/manga/121496/Na_Honjaman_Level-up",
)


def image_bytes(fmt: str, size: tuple[int, int] = (400, 600)) -> bytes:
    """A real image of the given Pillow format as bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, fmt)
    return buffer.getvalue()


JPEG_BYTES = image_bytes("JPEG")


def json_response(path: Path) -> httpx.Response:
    return httpx.Response(200, content=path.read_bytes())


def serving_fixtures(request: httpx.Request) -> httpx.Response:
    host = request.url.host
    if host == "graphql.anilist.co":
        return json_response(ANILIST_FIXTURE)
    if host == "api.mangadex.org":
        return json_response(MANGADEX_FIXTURE)
    if host == "api.jikan.moe":
        return json_response(JIKAN_FIXTURE)
    raise AssertionError(f"unexpected URL {request.url}")


def client_of(
    handler: Any, *, max_retries: int = 2
) -> tuple[MetadataClient, list[httpx.Request], list[float]]:
    """A MetadataClient over a mock transport, recording every request and every sleep."""
    seen: list[httpx.Request] = []
    sleeps: list[float] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = MetadataClient(
        client=httpx.Client(transport=httpx.MockTransport(recording)),
        sleep=sleeps.append,
        max_retries=max_retries,
    )
    return client, seen, sleeps


def http_of(handler: Any) -> tuple[httpx.Client, list[httpx.Request]]:
    """An httpx client over a mock transport that records every request."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(recording)), seen


def all_solo_candidates() -> tuple[Candidate, ...]:
    client, _seen, _sleeps = client_of(serving_fixtures)
    return client.search("Solo Leveling").candidates


# ---------------------------------------------------------------- providers


def test_anilist_fixture_yields_three_candidates() -> None:
    client, _seen, sleeps = client_of(lambda request: json_response(ANILIST_FIXTURE))
    result = client.search("Solo Leveling", providers=("anilist",))
    assert sleeps == []
    assert result.errors == {}
    assert [c.title for c in result.candidates] == [
        "Solo Leveling",
        "The Privilege of the Second Life is Power Leveling",
        "Solo Leveling: Ragnarok",
    ]
    assert result.candidates[0] == ANILIST_CANDIDATE


def test_anilist_request_carries_query_variables_and_headers() -> None:
    client, seen, _sleeps = client_of(lambda request: json_response(ANILIST_FIXTURE))
    client.search("Solo Leveling", providers=("anilist",), limit=5)
    request = seen[0]
    assert str(request.url) == "https://graphql.anilist.co"
    assert json.loads(request.read()) == {
        "query": ANILIST_QUERY,
        "variables": {"search": "Solo Leveling", "perPage": 5},
    }
    assert "OmniScan/" in request.headers["User-Agent"]
    assert request.headers["Accept"] == "application/json"


def test_mangadex_fixture_yields_two_candidates() -> None:
    client, seen, _sleeps = client_of(lambda request: json_response(MANGADEX_FIXTURE))
    result = client.search("Solo Leveling", providers=("mangadex",))
    assert len(result.candidates) == 2
    first = result.candidates[0]
    assert first.title == "Solo Leveling"  # from the altTitles 'en' entry
    assert "나 혼자만 레벨업" in first.titles
    assert "Na Honjaman Level-Up" in first.titles and "Na Honjaman Lebel-eob" in first.titles
    assert first.country == "KR"
    assert first.year == 2018
    assert first.status == "completed"
    assert first.cover_url == f"https://uploads.mangadex.org/covers/{MANGADEX_ID}/{MANGADEX_FILE}.512.jpg"
    assert first.url == f"https://mangadex.org/title/{MANGADEX_ID}"
    params = seen[0].url.params
    assert params["title"] == "Solo Leveling"
    assert params.get_list("includes[]") == ["cover_art"]
    assert params["order[relevance]"] == "desc"
    assert params["limit"] == "5"


def test_mangadex_candidate_without_cover_relationship() -> None:
    payload: Any = json.loads(MANGADEX_FIXTURE.read_text(encoding="utf-8"))
    payload["data"].append(
        {
            "id": "no-cover-id",
            "type": "manga",
            "attributes": {
                "title": {"en": "No Cover Series"},
                "altTitles": [],
                "originalLanguage": "ko",
                "year": None,
                "status": "completed",
            },
            "relationships": [{"id": "a", "type": "author"}],
        }
    )
    client, _seen, _sleeps = client_of(lambda request: httpx.Response(200, json=payload))
    candidates = client.search("Solo Leveling", providers=("mangadex",)).candidates
    assert candidates[-1].cover_url is None
    assert candidates[-1].title == "No Cover Series"


def test_jikan_fixture_yields_two_candidates() -> None:
    client, _seen, _sleeps = client_of(lambda request: json_response(JIKAN_FIXTURE))
    result = client.search("Solo Leveling", providers=("jikan",))
    assert len(result.candidates) == 2
    assert result.candidates[0] == JIKAN_CANDIDATE
    second = result.candidates[1]
    assert second.title == "Solo Leveling: Ragnarok"
    assert second.cover_url == "https://cdn.myanimelist.net/images/manga/1/999999.jpg"  # large is null
    assert second.id == "156239"


def test_jikan_all_null_images_give_no_cover() -> None:
    payload: Any = json.loads(JIKAN_FIXTURE.read_text(encoding="utf-8"))
    payload["data"].append(
        {
            "mal_id": 1,
            "url": "https://myanimelist.net/manga/1/Null",
            "images": {"jpg": {"image_url": None, "small_image_url": None, "large_image_url": None}},
            "title": "Null Images",
            "title_english": None,
            "title_japanese": None,
            "titles": [],
            "type": "Manga",
            "status": "Finished",
            "published": {"prop": {"from": {"year": 2000}}},
        }
    )
    client, _seen, _sleeps = client_of(lambda request: httpx.Response(200, json=payload))
    candidates = client.search("Solo Leveling", providers=("jikan",)).candidates
    assert candidates[-1].cover_url is None
    assert candidates[-1].country == "JP"
    assert candidates[-1].year == 2000


# ---------------------------------------------------------------- errors never raise


def test_429_waits_retry_after_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return json_response(ANILIST_FIXTURE)

    client, seen, sleeps = client_of(handler)
    result = client.search("Solo Leveling", providers=("anilist",))
    assert sleeps == [3.0]
    assert len(seen) == 2
    assert len(result.candidates) == 3
    assert result.errors == {}


def test_429_retry_after_is_capped_at_60() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "900"})
        return json_response(ANILIST_FIXTURE)

    client, _seen, sleeps = client_of(handler)
    result = client.search("Solo Leveling", providers=("anilist",))
    assert sleeps == [60.0]
    assert result.errors == {}


def test_429_without_or_with_bad_retry_after_defaults_to_5() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            header = "900" if calls["n"] == 1 else "soon"
            return httpx.Response(429, headers={"Retry-After": header})
        return json_response(ANILIST_FIXTURE)

    client, _seen, sleeps = client_of(handler)
    client.search("Solo Leveling", providers=("anilist",))
    assert sleeps == [60.0, 5.0]


def test_500_retries_with_backoff_then_reports_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.mangadex.org":
            return httpx.Response(500)
        return json_response(ANILIST_FIXTURE)

    client, seen, sleeps = client_of(handler)
    result = client.search("Solo Leveling", providers=("anilist", "mangadex"))
    assert result.errors == {"mangadex": "HTTP 500"}
    assert [c.provider for c in result.candidates] == ["anilist", "anilist", "anilist"]
    assert sleeps == [0.5, 1.0]
    assert len(seen) == 4  # 1 anilist request + 3 mangadex attempts


def test_jikan_504_forever_reports_error() -> None:
    client, seen, _sleeps = client_of(lambda request: httpx.Response(504))
    result = client.search("Solo Leveling", providers=("jikan",))
    assert result.errors == {"jikan": "HTTP 504"}
    assert result.candidates == ()
    assert len(seen) == 3


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (httpx.ConnectError("boom"), "connection error"),
        (httpx.ReadTimeout("too slow"), "timeout"),
    ],
)
def test_transport_errors_become_error_texts(exception: Exception, expected: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception

    client, _seen, _sleeps = client_of(handler)
    result = client.search("Solo Leveling", providers=("anilist",))
    assert result.errors == {"anilist": expected}
    assert result.candidates == ()


@pytest.mark.parametrize(
    "responder",
    [
        lambda request: httpx.Response(200, json={}),
        lambda request: httpx.Response(200, text="<html>not json</html>"),
    ],
    ids=["wrong-shape", "non-json"],
)
def test_wrong_body_shapes_are_unexpected_response(responder: Any) -> None:
    client, _seen, _sleeps = client_of(responder)
    for provider in ("anilist", "mangadex", "jikan"):
        result = client.search("Solo Leveling", providers=(provider,))  # type: ignore[arg-type]
        assert result.errors == {provider: "unexpected response"}
        assert result.candidates == ()


def test_empty_query_raises() -> None:
    client, seen, _sleeps = client_of(serving_fixtures)
    with pytest.raises(ValueError):
        client.search("   ")
    assert seen == []


def test_single_provider_restriction() -> None:
    client, seen, _sleeps = client_of(serving_fixtures)
    result = client.search("Solo Leveling", providers=("jikan",))
    assert all("api.jikan.moe" in str(request.url) for request in seen)
    assert all(candidate.provider == "jikan" for candidate in result.candidates)


# ---------------------------------------------------------------- matching


def test_match_score_exact_and_variants() -> None:
    solo = ANILIST_CANDIDATE
    assert match_score("Solo Leveling", solo) == 1.0
    assert match_score("solo-leveling", solo) == 1.0
    assert match_score("나 혼자만 레벨업", solo) == 1.0


def test_match_score_typo_is_close() -> None:
    assert match_score("Solo Levelling", ANILIST_CANDIDATE) >= 0.9


def test_match_score_prefix_rule() -> None:
    ragnarok = next(c for c in all_solo_candidates() if c.title == "Solo Leveling: Ragnarok")
    score = match_score("Solo Leveling", ragnarok)
    assert 0.85 <= score < 1.0  # the prefix rule, but below the exact match


def test_match_score_unrelated_query_is_low() -> None:
    assert match_score("Naruto", ANILIST_CANDIDATE) < 0.4


def test_match_score_empty_normalised_query() -> None:
    assert match_score("!!!", ANILIST_CANDIDATE) == 0.0


def test_best_match_prefers_the_anilist_exact_candidate_on_ties() -> None:
    best = best_match("Solo Leveling", all_solo_candidates())
    assert best is not None
    candidate, score = best
    assert candidate == ANILIST_CANDIDATE  # tie at 1.0 -> the earlier provider wins
    assert score == 1.0


def test_best_match_threshold() -> None:
    candidates = all_solo_candidates()
    assert best_match("Completely Different Title", candidates) is None
    assert best_match("Naruto", candidates) is None  # below the default 0.75
    low = best_match("Naruto", candidates, min_score=0.1)
    assert low is not None


# ---------------------------------------------------------------- cover download


def test_download_cover_writes_cover_and_series_json(tmp_path: Path) -> None:
    mdir = tmp_path / "S" / "_meta"
    mdir.mkdir(parents=True)
    (mdir / "cover.png").write_bytes(b"old png")
    http, seen = http_of(lambda request: httpx.Response(200, content=JPEG_BYTES))
    cover = download_cover(ANILIST_CANDIDATE, mdir, client=http)
    assert cover == mdir / "cover.jpg"
    assert cover.read_bytes() == JPEG_BYTES
    assert not (mdir / "cover.png").exists()
    assert not list(mdir.glob("*.part"))
    assert "User-Agent" in seen[0].headers and "OmniScan/" in seen[0].headers["User-Agent"]
    meta = read_series_meta(mdir)
    assert meta is not None
    datetime.fromisoformat(meta.pop("fetched_at"))
    assert meta == {
        "provider": "anilist",
        "id": "105398",
        "title": "Solo Leveling",
        "titles": list(SOLO_LEVELING_TITLES),
        "year": 2018,
        "country": "KR",
        "status": "finished",
        "url": "https://anilist.co/manga/105398",
        "cover_url": ANILIST_CANDIDATE.cover_url,
        "cover_file": "cover.jpg",
        "cover_source": "anilist",
        "credit": "Cover and metadata: AniList",
    }


def test_download_cover_creates_the_meta_dir(tmp_path: Path) -> None:
    mdir = tmp_path / "fresh" / "_meta"
    http, _seen = http_of(lambda request: httpx.Response(200, content=JPEG_BYTES))
    cover = download_cover(JIKAN_CANDIDATE, mdir, client=http)
    assert cover == mdir / "cover.jpg"
    meta = read_series_meta(mdir)
    assert meta is not None
    assert meta["credit"] == "Cover and metadata: MyAnimeList (via Jikan)"
    assert meta["cover_source"] == "jikan"


@pytest.mark.parametrize(("fmt", "name"), [("PNG", "cover.png"), ("WEBP", "cover.webp")])
def test_download_cover_extension_follows_the_format(fmt: str, name: str, tmp_path: Path) -> None:
    http, _seen = http_of(lambda request: httpx.Response(200, content=image_bytes(fmt)))
    cover = download_cover(MANGADEX_CANDIDATE, tmp_path, client=http)
    assert cover.name == name
    meta = read_series_meta(tmp_path)
    assert meta is not None
    assert meta["credit"] == "Cover and metadata: MangaDex"


def test_download_cover_small_size_requests_the_small_url(tmp_path: Path) -> None:
    http, seen = http_of(lambda request: httpx.Response(200, content=JPEG_BYTES))
    download_cover(MANGADEX_CANDIDATE, tmp_path, client=http, size="small")
    assert seen[0].url.path.endswith(f"/{MANGADEX_FILE}.256.jpg")


def test_download_cover_rejects_non_image_and_writes_nothing(tmp_path: Path) -> None:
    mdir = tmp_path / "S" / "_meta"
    mdir.mkdir(parents=True)
    (mdir / "cover.jpg").write_bytes(JPEG_BYTES)
    http, _seen = http_of(lambda request: httpx.Response(200, content=b"definitely not an image"))
    with pytest.raises(ValueError):
        download_cover(ANILIST_CANDIDATE, mdir, client=http)
    assert (mdir / "cover.jpg").read_bytes() == JPEG_BYTES  # the existing cover stays
    assert read_series_meta(mdir) is None
    assert not list(mdir.glob("*.part"))


def test_download_cover_rejects_oversized_bodies(tmp_path: Path) -> None:
    http, _seen = http_of(lambda request: httpx.Response(200, content=b"x" * (10 * 1024 * 1024 + 1)))
    with pytest.raises(ValueError):
        download_cover(ANILIST_CANDIDATE, tmp_path, client=http)
    assert list(tmp_path.iterdir()) == []


def test_download_cover_rejects_404(tmp_path: Path) -> None:
    http, _seen = http_of(lambda request: httpx.Response(404))
    with pytest.raises(ValueError, match="404"):
        download_cover(ANILIST_CANDIDATE, tmp_path, client=http)


def test_download_cover_rejects_missing_cover_url(tmp_path: Path) -> None:
    naked = Candidate(
        provider="anilist",
        id="1",
        title="No Cover",
        titles=("No Cover",),
        year=None,
        country=None,
        status=None,
        cover_url=None,
        url=None,
    )
    with pytest.raises(ValueError, match="no cover"):
        download_cover(naked, tmp_path)


# ---------------------------------------------------------------- user cover


def test_set_cover_from_file_updates_series_json_and_keeps_keys(tmp_path: Path) -> None:
    mdir = tmp_path / "S" / "_meta"
    mdir.mkdir(parents=True)
    (mdir / "series.json").write_text(json.dumps({"provider": "anilist", "title": "Solo Leveling"}), "utf-8")
    (mdir / "cover.jpg").write_bytes(JPEG_BYTES)
    source = tmp_path / "mine.png"
    source.write_bytes(image_bytes("PNG"))
    cover = set_cover_from_file(mdir, source)
    assert cover == mdir / "cover.png"
    assert cover.read_bytes() == source.read_bytes()
    assert not (mdir / "cover.jpg").exists()
    meta = read_series_meta(mdir)
    assert meta is not None
    assert meta["cover_source"] == "user"
    assert meta["cover_file"] == "cover.png"
    assert meta["provider"] == "anilist"  # the other keys are kept
    datetime.fromisoformat(meta["fetched_at"])


def test_set_cover_from_file_creates_series_json(tmp_path: Path) -> None:
    source = tmp_path / "mine.jpg"
    source.write_bytes(JPEG_BYTES)
    cover = set_cover_from_file(tmp_path / "new" / "_meta", source)
    assert cover.name == "cover.jpg"
    meta = read_series_meta(tmp_path / "new" / "_meta")
    assert meta is not None
    assert meta["cover_source"] == "user"


def test_set_cover_from_file_rejects_text(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        set_cover_from_file(tmp_path / "_meta", source)


def test_find_cover_and_read_series_meta(tmp_path: Path) -> None:
    assert find_cover(tmp_path) is None
    assert read_series_meta(tmp_path) is None
    (tmp_path / "series.json").write_text("{not json", encoding="utf-8")
    assert read_series_meta(tmp_path) is None
    (tmp_path / "cover.webp").write_bytes(image_bytes("WEBP"))
    assert find_cover(tmp_path) == tmp_path / "cover.webp"


def test_meta_dir_layout() -> None:
    assert meta_dir(Path("/lib"), "My Series") == Path("/lib/My Series/_meta")
