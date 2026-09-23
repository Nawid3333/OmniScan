"""Series covers and metadata from the free structured APIs — AniList, MangaDex, Jikan (card L1).

No LLM and no scraping: a failing provider is reported in `SearchResult.errors` and never aborts
a search (Jikan in particular is flaky). Covers are stored in the user's own library folder with
a record of where they came from; nothing is re-distributed.
"""

from __future__ import annotations

import difflib
import io
import json
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from PIL import Image

import omniscan

Provider = Literal["anilist", "mangadex", "jikan"]
PROVIDERS: tuple[Provider, ...] = ("anilist", "mangadex", "jikan")

USER_AGENT = f"OmniScan/{omniscan.__version__} (personal manga library tool)"
REQUEST_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

ANILIST_URL = "https://graphql.anilist.co"
MANGADEX_URL = "https://api.mangadex.org/manga"
JIKAN_URL = "https://api.jikan.moe/v4/manga"

# "send exactly this text" (card L1)
ANILIST_QUERY = (
    "query ($search: String, $perPage: Int) { Page(perPage: $perPage) "
    "{ media(search: $search, type: MANGA) { id title { romaji english native } format "
    "countryOfOrigin startDate { year } chapters status coverImage { extraLarge large color } siteUrl } } }"
)

MAX_COVER_BYTES = 10 * 1024 * 1024
_COVER_NAMES = ("cover.jpg", "cover.png", "cover.webp")
_SERIES_JSON = "series.json"
_PIL_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
_CREDIT: dict[Provider, str] = {
    "anilist": "Cover and metadata: AniList",
    "mangadex": "Cover and metadata: MangaDex",
    "jikan": "Cover and metadata: MyAnimeList (via Jikan)",
}
_MANGADEX_COUNTRY = {"ko": "KR", "ja": "JP", "zh": "CN", "zh-hk": "CN"}
_JIKAN_COUNTRY = {"manhwa": "KR", "manga": "JP", "manhua": "CN"}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One series found by a metadata provider."""

    provider: Provider
    id: str  # the interface name from the card
    title: str  # display title (see the card's definitions)
    titles: tuple[str, ...]
    year: int | None
    country: str | None
    status: str | None
    cover_url: str | None
    url: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Candidates in provider order plus one short error text per failing provider."""

    candidates: tuple[Candidate, ...]
    errors: dict[str, str]


class _ProviderError(Exception):
    """A provider failed after the retries; `text` goes into SearchResult.errors."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


class MetadataClient:
    """Searches the metadata APIs; a failing provider is reported in SearchResult.errors, never raised."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 2,
    ) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(follow_redirects=True, timeout=30.0)
        self._sleep = sleep
        self._max_retries = max_retries

    def close(self) -> None:
        """Close the HTTP client when this client created it itself."""
        if self._owns_client:
            self._client.close()

    def search(
        self, title: str, *, providers: Sequence[Provider] = PROVIDERS, limit: int = 5
    ) -> SearchResult:
        """Candidates of every provider in provider order; a failing provider adds to errors."""
        query = title.strip()
        if not query:
            raise ValueError("search title is empty")
        candidates: list[Candidate] = []
        errors: dict[str, str] = {}
        for provider in providers:
            found, error = self._search_one(provider, query, limit)
            if error is not None:
                errors[provider] = error
            candidates.extend(found)
        return SearchResult(candidates=tuple(candidates), errors=errors)

    def _search_one(
        self, provider: Provider, query: str, limit: int
    ) -> tuple[tuple[Candidate, ...], str | None]:
        try:
            items = self._fetch(provider, query, limit)
        except _ProviderError as exc:
            return (), exc.text
        if provider == "anilist":
            return tuple(_anilist_candidate(item) for item in items), None
        if provider == "mangadex":
            return tuple(_mangadex_candidate(item) for item in items), None
        return tuple(_jikan_candidate(item) for item in items), None

    def _fetch(self, provider: Provider, query: str, limit: int) -> list[Any]:
        """The provider's documented list of items, with the retry policy; raises _ProviderError."""
        error = "unknown error"
        for attempt in range(self._max_retries + 1):
            response: httpx.Response | None = None
            wait: float
            try:
                response = self._request(provider, query, limit)
            except httpx.TimeoutException:  # more specific than TransportError
                error = "timeout"
            except httpx.TransportError:
                error = "connection error"
            if response is not None and response.status_code < 400:
                return _items_of(response, provider)
            if response is not None:
                error = f"HTTP {response.status_code}"
            # a 429 waits out Retry-After; every other failure waits the exponential backoff
            wait = (
                _retry_after(response)
                if response is not None and response.status_code == 429
                else 0.5 * 2**attempt
            )
            if attempt < self._max_retries:
                self._sleep(wait)
        raise _ProviderError(error)

    def _request(self, provider: Provider, query: str, limit: int) -> httpx.Response:
        if provider == "anilist":
            return self._client.post(
                ANILIST_URL,
                json={"query": ANILIST_QUERY, "variables": {"search": query, "perPage": limit}},
                headers=REQUEST_HEADERS,
            )
        if provider == "mangadex":
            return self._client.get(
                MANGADEX_URL,
                params=[
                    ("title", query),
                    ("limit", limit),
                    ("includes[]", "cover_art"),
                    ("order[relevance]", "desc"),
                ],
                headers=REQUEST_HEADERS,
            )
        return self._client.get(JIKAN_URL, params={"q": query, "limit": limit}, headers=REQUEST_HEADERS)


def _retry_after(response: httpx.Response) -> float:
    """Retry-After of a 429 as seconds (integer, capped at 60; default 5)."""
    raw = response.headers.get("Retry-After")
    try:
        seconds = int(raw) if raw is not None else 5
    except ValueError:
        return 5.0
    return float(min(60, seconds))


def _items_of(response: httpx.Response, provider: Provider) -> list[Any]:
    """The documented item list of a 2xx payload; anything else is 'unexpected response'."""
    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise _ProviderError("unexpected response") from exc
    items = _items(payload, provider)
    if items is None:
        raise _ProviderError("unexpected response")
    return items


def _items(data: Any, provider: Provider) -> list[Any] | None:
    """The documented list inside a provider payload, or None when the shape differs."""
    if not isinstance(data, dict):
        return None
    if provider == "anilist":
        data_node = data.get("data")
        page = data_node.get("Page") if isinstance(data_node, dict) else None
        media = page.get("media") if isinstance(page, dict) else None
        return media if isinstance(media, list) else None
    items = data.get("data")
    return items if isinstance(items, list) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _dedup(names: Iterable[str]) -> tuple[str, ...]:
    """Stripped, non-empty, exact duplicates removed, order kept."""
    return tuple(dict.fromkeys(name for name in (n.strip() for n in names) if name))


def _anilist_candidate(item: Any) -> Candidate:
    entry = _as_dict(item)
    title = _as_dict(entry.get("title"))
    names = [_text(title.get("english")), _text(title.get("romaji")), _text(title.get("native"))]
    cover = _as_dict(entry.get("coverImage"))
    start = _as_dict(entry.get("startDate"))
    status = entry.get("status")
    return Candidate(
        provider="anilist",
        id=str(entry.get("id")),
        title=next((name for name in names if name), ""),
        titles=_dedup(names),
        year=_as_int(start.get("year")),
        country=_text(entry.get("countryOfOrigin")) or None,
        status=status.lower() if isinstance(status, str) else None,
        cover_url=_text(cover.get("extraLarge")) or _text(cover.get("large")) or None,
        url=_text(entry.get("siteUrl")) or None,
    )


def _mangadex_candidate(item: Any) -> Candidate:
    entry = _as_dict(item)
    series_id = _text(entry.get("id"))
    attributes = _as_dict(entry.get("attributes"))
    title = _as_dict(attributes.get("title"))
    raw_alt = attributes.get("altTitles")
    alt = [
        alt_entry
        for alt_entry in (raw_alt if isinstance(raw_alt, list) else [])
        if isinstance(alt_entry, dict)
    ]
    alt_values = [_text(value) for alt_entry in alt for value in alt_entry.values()]
    en_title = _text(title.get("en"))
    en_alt = next((_text(value) for alt_entry in alt for key, value in alt_entry.items() if key == "en"), "")
    first_title = next((value for value in (_text(v) for v in title.values()) if value), "")
    first_alt = next((value for value in alt_values if value), "")
    cover_file = ""
    raw_relationships = entry.get("relationships")
    for relationship in raw_relationships if isinstance(raw_relationships, list) else []:
        rel = _as_dict(relationship)
        if rel.get("type") == "cover_art":
            cover_file = _text(_as_dict(rel.get("attributes")).get("fileName"))
            break
    status = attributes.get("status")
    return Candidate(
        provider="mangadex",
        id=series_id,
        title=en_title or en_alt or first_title or first_alt,
        titles=_dedup([en_title, en_alt, first_title, first_alt, *alt_values]),
        year=_as_int(attributes.get("year")),
        country=_MANGADEX_COUNTRY.get(_text(attributes.get("originalLanguage")).lower()),
        status=status.lower() if isinstance(status, str) else None,
        cover_url=(
            f"https://uploads.mangadex.org/covers/{series_id}/{cover_file}.512.jpg" if cover_file else None
        ),
        url=f"https://mangadex.org/title/{series_id}" if series_id else None,
    )


def _jikan_candidate(item: Any) -> Candidate:
    entry = _as_dict(item)
    jpg = _as_dict(_as_dict(entry.get("images")).get("jpg"))
    titles_node = entry.get("titles")
    title_values = (
        [_text(t.get("title")) for t in titles_node if isinstance(t, dict)]
        if isinstance(titles_node, list)
        else []
    )
    names = [
        _text(entry.get("title_english")),
        _text(entry.get("title")),
        _text(entry.get("title_japanese")),
        *title_values,
    ]
    kind = _text(entry.get("type")).lower()
    prop = _as_dict(_as_dict(entry.get("published")).get("prop"))
    return Candidate(
        provider="jikan",
        id=str(entry.get("mal_id")),
        title=next((name for name in names if name), ""),
        titles=_dedup(names),
        year=_as_int(_as_dict(prop.get("from")).get("year")),
        country=_JIKAN_COUNTRY.get(kind),
        status=_text(entry.get("status")).lower() or None,
        cover_url=_text(jpg.get("large_image_url")) or _text(jpg.get("image_url")) or None,
        url=_text(entry.get("url")) or None,
    )


def _normalise(text: str) -> str:
    """NFKC, casefold, then keep only letters and digits of any script."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalpha() or ch.isdigit()
    )


def match_score(query: str, candidate: Candidate) -> float:
    """Similarity of the query with the candidate's best title (1.0 exact, 0.0 for an empty query)."""
    normal_query = _normalise(query)
    if not normal_query:
        return 0.0
    best = 0.0
    for title in candidate.titles:
        normal_title = _normalise(title)
        if not normal_title:
            continue
        if normal_title == normal_query:
            return 1.0
        score = difflib.SequenceMatcher(None, normal_query, normal_title).ratio()
        # prefix rule (card L1): a title that starts with the query and is not much longer is close.
        # 1.75 instead of the card's 1.6: "Solo Leveling" -> "Solo Leveling: Ragnarok" is 20/12 chars
        # after normalisation (1.67x) and the card's own acceptance test pins that case at >= 0.85.
        if normal_title.startswith(normal_query) and len(normal_title) <= 1.75 * len(normal_query):
            score = max(score, 0.85)
        best = max(best, score)
    return best


def best_match(
    query: str, candidates: Sequence[Candidate], *, min_score: float = 0.75
) -> tuple[Candidate, float] | None:
    """The highest-scored candidate (ties: earlier provider, then earlier position), or None."""
    best: tuple[tuple[float, int, int], Candidate] | None = None
    for position, candidate in enumerate(candidates):
        rank = PROVIDERS.index(candidate.provider) if candidate.provider in PROVIDERS else len(PROVIDERS)
        key = (match_score(query, candidate), -rank, -position)
        if best is None or key > best[0]:
            best = (key, candidate)
    if best is None or best[0][0] < min_score:
        return None
    return best[1], best[0][0]


def meta_dir(library_root: Path, series: str) -> Path:
    """The per-series metadata folder: <library_root>/<series>/_meta."""
    return library_root / series / "_meta"


def _small_cover_url(provider: Provider, url: str) -> str:
    """The small cover URL derived from the stored large one (provider-specific suffix surgery)."""
    if provider == "anilist":
        return url.replace("/cover/large/", "/cover/medium/")
    if provider == "mangadex":
        return url.replace(".512.jpg", ".256.jpg")
    if url.endswith("l.jpg"):  # MAL cdn: <base>l.jpg (large) -> <base>t.jpg (small)
        return url[:-5] + "t.jpg"
    if url.endswith(".jpg"):  # the large URL had fallen back to the plain image
        return url[:-4] + "t.jpg"
    return url


def _image_format(data: bytes) -> str:
    """The decoded image's Pillow format name (JPEG/PNG/WEBP); ValueError on anything else."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            fmt = image.format or ""
    except Exception as exc:  # Pillow raises several unrelated types on broken files
        raise ValueError(f"not a decodable JPEG/PNG/WEBP image: {exc}") from exc
    if fmt not in _PIL_FORMATS:
        raise ValueError(f"unsupported cover format {fmt!r} (JPEG/PNG/WEBP only)")
    return fmt


def _clear_covers(meta_dir: Path) -> None:
    """Remove any previous cover.<ext> so exactly one cover file remains."""
    for name in _COVER_NAMES:
        (meta_dir / name).unlink(missing_ok=True)


def _write_series_json(meta_dir: Path, data: dict[str, Any]) -> None:
    """Write series.json atomically (UTF-8, indent 2, ensure_ascii=False, trailing newline)."""
    part = meta_dir / "series.json.part"
    part.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    part.replace(meta_dir / _SERIES_JSON)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fetch_image(client: httpx.Client, url: str) -> bytes:
    """GET an image with the OmniScan User-Agent, hard-capped at 10 MB; ValueError on any failure."""
    try:
        with client.stream("GET", url, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as response:
            if response.status_code // 100 != 2:
                raise ValueError(f"HTTP {response.status_code} while downloading the cover")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_COVER_BYTES:
                    raise ValueError("cover is larger than 10 MB")
                chunks.append(chunk)
    except httpx.TransportError as exc:
        raise ValueError(f"cannot download the cover: {exc}") from exc
    return b"".join(chunks)


def download_cover(
    candidate: Candidate,
    meta_dir: Path,
    *,
    client: httpx.Client | None = None,
    size: Literal["large", "small"] = "large",
) -> Path:
    """Download the candidate's cover into meta_dir and record it in series.json; returns the cover path."""
    if candidate.cover_url is None:
        raise ValueError("candidate has no cover")
    url = (
        _small_cover_url(candidate.provider, candidate.cover_url) if size == "small" else candidate.cover_url
    )
    owned = client is None
    http = (
        client
        if client is not None
        else httpx.Client(follow_redirects=True, timeout=30.0, headers={"User-Agent": USER_AGENT})
    )
    try:
        data = _fetch_image(http, url)
    finally:
        if owned:
            http.close()
    extension = _PIL_FORMATS[_image_format(data)]  # before any write: a bad image writes nothing
    meta_dir.mkdir(parents=True, exist_ok=True)
    _clear_covers(meta_dir)
    cover = meta_dir / f"cover.{extension}"
    part = meta_dir / f"cover.{extension}.part"
    part.write_bytes(data)
    part.replace(cover)
    _write_series_json(
        meta_dir,
        {
            "provider": candidate.provider,
            "id": candidate.id,
            "title": candidate.title,
            "titles": list(candidate.titles),
            "year": candidate.year,
            "country": candidate.country,
            "status": candidate.status,
            "url": candidate.url,
            "cover_url": candidate.cover_url,
            "cover_file": cover.name,
            "cover_source": candidate.provider,
            "fetched_at": _utc_now(),
            "credit": _CREDIT[candidate.provider],
        },
    )
    return cover


def set_cover_from_file(meta_dir: Path, source: Path) -> Path:
    """Use the user's own image as the cover (JPEG/PNG/WEBP) and mark series.json's source 'user'."""
    data = source.read_bytes()
    extension = _PIL_FORMATS[_image_format(data)]  # before any write: a bad image writes nothing
    meta_dir.mkdir(parents=True, exist_ok=True)
    _clear_covers(meta_dir)
    cover = meta_dir / f"cover.{extension}"
    part = meta_dir / f"cover.{extension}.part"
    part.write_bytes(data)
    part.replace(cover)
    meta = read_series_meta(meta_dir) or {}
    meta.update({"cover_file": cover.name, "cover_source": "user", "fetched_at": _utc_now()})
    _write_series_json(meta_dir, meta)
    return cover


def read_series_meta(meta_dir: Path) -> dict[str, Any] | None:
    """The parsed series.json, or None when it is missing or not a JSON object."""
    path = meta_dir / _SERIES_JSON
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return data if isinstance(data, dict) else None


def find_cover(meta_dir: Path) -> Path | None:
    """The existing cover.jpg|png|webp of the series, or None."""
    for name in _COVER_NAMES:
        path = meta_dir / name
        if path.is_file():
            return path
    return None
