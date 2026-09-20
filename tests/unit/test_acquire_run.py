"""Tests for omniscan.acquire.run (no network; scripted extractor fake + httpx.MockTransport)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx

from omniscan.acquire.drm import DrmPlatformError
from omniscan.acquire.extractor import (
    AuthError,
    ExtractError,
    Extraction,
    ExtractedImage,
    Mode,
    QuotaError,
    credit_cost,
)
from omniscan.acquire.plan import AcquirePlan, build_plan, read_accepted, write_accepted
from omniscan.acquire.run import SELECTION_NAME, ChapterOutcome, acquire_chapters
from omniscan.acquire.sources import ChapterSource, named_chapters
from tests.unit.test_acquire_download import Server, noise_image

PAGE_URL = "https://site.test/chapter-{n}"
IMAGE_URL = "https://cdn.test/c{n}/{i:03d}.jpg"
EXTRAS = ("/logo.svg", "/icon.png", "/thumb_120x120.jpg")
DROPPED_REASONS = ("not a page format (.svg)", "url contains 'icon'", "url contains 'thumb'")


class FakeExtractor:
    """Scripted Extractor fake: per-URL outcome (Extraction or exception); records every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._scripted: dict[str, Extraction | Exception] = {}

    def script(self, url: str, outcome: Extraction | Exception) -> None:
        self._scripted[url] = outcome

    def extract(self, url: str, *, mode: Mode = "basic") -> Extraction:
        self.calls.append(url)
        outcome = self._scripted[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def extraction(chapter: int, mode: Mode = "basic", extras: bool = True) -> Extraction:
    """One page's extraction: 5 chapter pages plus (optionally) logo, icon and thumbnail URLs."""
    urls = [IMAGE_URL.format(n=chapter, i=i) for i in range(1, 6)]
    if extras:
        urls += [f"https://cdn.test/c{chapter}{extra}" for extra in EXTRAS]
    return Extraction(
        id=f"e{chapter}",
        page_url=PAGE_URL.format(n=chapter),
        images=tuple(ExtractedImage(url=url) for url in urls),
        credits=credit_cost(1, mode),
    )


def scripted_chapters(
    count: int, *, mode: Mode = "basic"
) -> tuple[Server, httpx.Client, FakeExtractor]:
    """A mock server + client + fake extractor scripted for `count` chapters of 5 pages each."""
    server = Server()
    extractor = FakeExtractor()
    for n in range(1, count + 1):
        extractor.script(PAGE_URL.format(n=n), extraction(n, mode=mode))
        for i in range(1, 6):
            server.script(IMAGE_URL.format(n=n, i=i), noise_image("JPEG", seed=n * 10 + i))
    return server, server.client(), extractor


def acquire(
    series_dir: Path,
    plan: AcquirePlan,
    extractor: FakeExtractor,
    client: httpx.Client | None = None,
    *,
    max_credits: int | None = None,
    on_event: Callable[[ChapterOutcome], None] | None = None,
) -> list[ChapterOutcome]:
    """acquire_chapters over the plan with the card's defaults."""
    return acquire_chapters(
        plan, series_dir, extractor, client=client, max_credits=max_credits, on_event=on_event
    )


def run_one(series_dir: Path, name: str, page_count: int, *, seed: int = 99):
    """Run one target chapter that downloads `page_count` real pages from a fake CDN."""
    server = Server()
    for i in range(1, page_count + 1):
        server.script(
            f"https://cdn.test/x/{i:03d}.jpg",
            noise_image("JPEG", seed=seed + i, width=800, height=1200),
        )
    extractor = FakeExtractor()
    extractor.script(
        f"https://site.test/{name}",
        Extraction(
            id="e",
            page_url=f"https://site.test/{name}",
            images=tuple(
                ExtractedImage(url=f"https://cdn.test/x/{i:03d}.jpg") for i in range(1, page_count + 1)
            ),
            credits=1,
        ),
    )
    sources = named_chapters([(name, f"https://site.test/{name}")])
    plan = build_plan(sources, series_dir)
    return acquire(series_dir, plan, extractor, server.client())


def accepted_dir(tmp_path: Path, name: str, pages: int) -> None:
    """A chapter folder that only carries an accepted.json recording `pages` pages."""
    chapter_dir = tmp_path / name
    chapter_dir.mkdir()
    write_accepted(chapter_dir, "ok", [], pages, source_url=f"https://site.test/{name}")


# ------------------------------------------------------------------ 1. happy path


def test_happy_path_downloads_pages_in_order_with_a_selection_record(tmp_path: Path) -> None:
    server, client, extractor = scripted_chapters(3)
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])
    plan = build_plan(sources, tmp_path)
    events: list[ChapterOutcome] = []

    outcomes = acquire(tmp_path, plan, extractor, client, on_event=events.append)

    assert [outcome.status for outcome in outcomes] == ["ok", "ok", "ok"]
    assert [outcome.pages for outcome in outcomes] == [5, 5, 5]
    assert [outcome.credits for outcome in outcomes] == [1, 1, 1]
    assert [outcome.error for outcome in outcomes] == [None, None, None]
    assert events == outcomes  # on_event fired once per chapter, in plan order
    for n in range(1, 4):
        chapter_dir = tmp_path / f"Chapter {n}"
        selection = json.loads((chapter_dir / SELECTION_NAME).read_text(encoding="utf-8"))
        assert selection == {
            "page_url": PAGE_URL.format(n=n),
            "kept": [IMAGE_URL.format(n=n, i=i) for i in range(1, 6)],
            "dropped": [
                {"url": f"https://cdn.test/c{n}{extra}", "reason": reason}
                for extra, reason in zip(EXTRAS, DROPPED_REASONS, strict=True)
            ],
        }
        assert [p.name for p in sorted(chapter_dir.iterdir())] == [
            "001.jpg",
            "002.jpg",
            "003.jpg",
            "004.jpg",
            "005.jpg",
            "accepted.json",
            "acquire.json",
            SELECTION_NAME,
        ]
        accepted = read_accepted(chapter_dir)
        assert accepted is not None and accepted["status"] == "ok"
        assert accepted["pages"] == 5
        assert accepted["source_url"] == PAGE_URL.format(n=n)
    # every image request carried its chapter page URL as Referer
    for request in server.requests:
        chapter = int(request.url.path.split("/")[1].removeprefix("c"))
        assert request.headers["Referer"] == PAGE_URL.format(n=chapter)


# ------------------------------------------------------------------ 2. done chapters


def test_done_chapters_are_skipped_and_never_extracted(tmp_path: Path) -> None:
    accepted_dir(tmp_path, "Chapter 1", 5)
    server, client, extractor = scripted_chapters(3)
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])

    outcomes = acquire(tmp_path, build_plan(sources, tmp_path), extractor, client)

    assert outcomes[0] == ChapterOutcome("Chapter 1", "skipped", 5, 0, (), "already acquired")
    assert [outcome.status for outcome in outcomes] == ["skipped", "ok", "ok"]
    assert extractor.calls == [PAGE_URL.format(n=2), PAGE_URL.format(n=3)]


def test_force_reruns_done_chapters(tmp_path: Path) -> None:
    accepted_dir(tmp_path, "Chapter 1", 5)  # status ok → done; a re-run without --force would skip it
    server, client, extractor = scripted_chapters(3)
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])

    outcomes = acquire(tmp_path, build_plan(sources, tmp_path, force=True), extractor, client)

    assert [outcome.status for outcome in outcomes] == ["ok", "ok", "ok"]
    assert extractor.calls == [PAGE_URL.format(n=1), PAGE_URL.format(n=2), PAGE_URL.format(n=3)]


# ------------------------------------------------------------------ 3. credit budget


def test_credit_budget_skips_the_rest_in_basic_mode(tmp_path: Path) -> None:
    server, client, extractor = scripted_chapters(4)
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 5)])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client, max_credits=2)

    assert [outcome.status for outcome in outcomes] == ["ok", "ok", "skipped", "skipped"]
    assert [outcome.credits for outcome in outcomes] == [1, 1, 0, 0]
    assert outcomes[2].error == "credit budget reached"
    assert outcomes[3].error == "credit budget reached"
    assert extractor.calls == [PAGE_URL.format(n=1), PAGE_URL.format(n=2)]


def test_credit_budget_in_advanced_mode_runs_one_chapter(tmp_path: Path) -> None:
    server, client, extractor = scripted_chapters(4, mode="advanced")
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 5)])
    plan = build_plan(sources, tmp_path, mode="advanced")

    outcomes = acquire(tmp_path, plan, extractor, client, max_credits=3)

    assert [outcome.status for outcome in outcomes] == ["ok", "skipped", "skipped", "skipped"]
    assert [outcome.credits for outcome in outcomes] == [2, 0, 0, 0]
    assert extractor.calls == [PAGE_URL.format(n=1)]


# ------------------------------------------------------------------ 4. failures


def test_extract_error_fails_only_that_chapter(tmp_path: Path) -> None:
    server, client, extractor = scripted_chapters(3)
    extractor.script(PAGE_URL.format(n=2), ExtractError("extraction ended with status 'failed'"))
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client)

    assert [outcome.status for outcome in outcomes] == ["ok", "failed", "ok"]
    assert outcomes[1].credits == 0
    assert "failed" in outcomes[1].error


def test_auth_error_stops_the_run(tmp_path: Path) -> None:
    server, client, extractor = scripted_chapters(3)
    extractor.script(PAGE_URL.format(n=2), AuthError("extract.pics rejected the API key"))
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client)

    assert [outcome.status for outcome in outcomes] == ["ok", "failed", "skipped"]
    assert outcomes[1].error == "extract.pics rejected the API key"
    assert outcomes[2].error == "stopped: extract.pics rejected the API key"
    assert extractor.calls == [PAGE_URL.format(n=1), PAGE_URL.format(n=2)]


def test_quota_error_stops_the_run(tmp_path: Path) -> None:
    server, client, extractor = scripted_chapters(3)
    extractor.script(PAGE_URL.format(n=2), QuotaError("extract.pics reports no credits left"))
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client)

    assert [outcome.status for outcome in outcomes] == ["ok", "failed", "skipped"]
    assert outcomes[2].error == "stopped: extract.pics reports no credits left"


def test_drm_page_url_fails_only_that_chapter(tmp_path: Path) -> None:
    """The extract back-end refuses DRM pages (like ExtractPicsClient does); run records a failure."""
    server, client, extractor = scripted_chapters(3)
    extractor.script(
        PAGE_URL.format(n=2),
        DrmPlatformError(
            "Naver Webtoon protects its pages against extraction; OmniScan does not acquire"
            " from paid DRM platforms"
        ),
    )
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client)

    assert [outcome.status for outcome in outcomes] == ["ok", "failed", "ok"]
    assert "Naver Webtoon" in outcomes[1].error


def test_drm_image_url_is_caught_by_the_downloader(tmp_path: Path) -> None:
    """The safety net: pages whose image URLs sit on a DRM platform fail that chapter only."""
    server, client, extractor = scripted_chapters(3)
    urls = [f"https://comic.naver.com/img/{i:03d}.jpg" for i in range(1, 6)]
    extractor.script(
        PAGE_URL.format(n=2),
        Extraction(
            id="e2",
            page_url=PAGE_URL.format(n=2),
            images=tuple(ExtractedImage(url=url) for url in urls),
            credits=1,
        ),
    )
    sources = named_chapters([(f"Chapter {n}", PAGE_URL.format(n=n)) for n in range(1, 4)])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client)

    assert [outcome.status for outcome in outcomes] == ["ok", "failed", "ok"]
    assert "Naver Webtoon" in outcomes[1].error


def test_page_list_filtered_to_nothing_fails_with_the_counts(tmp_path: Path) -> None:
    server = Server()
    extractor = FakeExtractor()
    urls = [f"https://cdn.test/c1{extra}" for extra in EXTRAS] + ["https://cdn.test/c1/hero_300x300.jpg"]
    extractor.script(
        PAGE_URL.format(n=1),
        Extraction(
            id="e1",
            page_url=PAGE_URL.format(n=1),
            images=tuple(ExtractedImage(url=url) for url in urls),
            credits=1,
        ),
    )
    sources = named_chapters([("Chapter 1", PAGE_URL.format(n=1))])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, server.client())

    assert [outcome.status for outcome in outcomes] == ["failed"]
    assert outcomes[0].error == "no page images found (4 image(s) on the page were filtered out)"
    assert outcomes[0].credits == 1  # the extraction was charged
    selection = json.loads((tmp_path / "Chapter 1" / SELECTION_NAME).read_text(encoding="utf-8"))
    assert selection["kept"] == []
    assert len(selection["dropped"]) == 4


def test_image_404_fails_the_chapter_and_keeps_partials(tmp_path: Path) -> None:
    server, client, extractor = scripted_chapters(1)
    server.script(IMAGE_URL.format(n=1, i=2), 404)
    sources = named_chapters([("Chapter 1", PAGE_URL.format(n=1))])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client)

    assert [outcome.status for outcome in outcomes] == ["failed"]
    assert "1 of 5 image(s) failed: #2 HTTP 404" in outcomes[0].error
    chapter_dir = tmp_path / "Chapter 1"
    assert (chapter_dir / "001.jpg").exists()
    assert (chapter_dir / "acquire.json").exists()
    assert (chapter_dir / SELECTION_NAME).exists()
    assert read_accepted(chapter_dir) is None  # no acceptance record after a failed download

    # a re-run after the transport is fixed completes the chapter without re-downloading the files
    re_server = Server()
    for i in range(1, 6):
        re_server.script(IMAGE_URL.format(n=1, i=i), noise_image("JPEG", seed=10 + i))
    re_extractor = FakeExtractor()
    re_extractor.script(PAGE_URL.format(n=1), extraction(1))

    re_outcomes = acquire(
        tmp_path, build_plan(sources, tmp_path), re_extractor, re_server.client()
    )

    assert [outcome.status for outcome in re_outcomes] == ["ok"]
    assert [str(request.url) for request in re_server.requests] == [IMAGE_URL.format(n=1, i=2)]
    assert read_accepted(chapter_dir)["status"] == "ok"


# ------------------------------------------------------------------ 5. completeness


def test_few_pages_when_three_accepted_chapters_have_more(tmp_path: Path) -> None:
    accepted_dir(tmp_path, "Chapter 1", 20)
    accepted_dir(tmp_path, "Chapter 2", 20)
    accepted_dir(tmp_path, "Chapter 3", 20)

    outcomes = run_one(tmp_path, "Chapter 4", 4)

    assert [outcome.status for outcome in outcomes] == ["review"]
    assert [finding.code for finding in outcomes[0].findings] == ["few_pages"]
    assert read_accepted(tmp_path / "Chapter 4")["status"] == "review"


def test_median_needs_three_accepted_chapters(tmp_path: Path) -> None:
    accepted_dir(tmp_path, "Chapter 2", 20)
    accepted_dir(tmp_path, "Chapter 3", 20)

    outcomes = run_one(tmp_path, "Chapter 1", 4)

    assert [outcome.status for outcome in outcomes] == ["ok"]
    assert outcomes[0].findings == ()


def test_previous_is_the_nearest_earlier_accepted_chapter(tmp_path: Path) -> None:
    accepted_dir(tmp_path, "Chapter 1", 20)
    (tmp_path / "Chapter 2").mkdir()  # in the library but not accepted

    outcomes = run_one(tmp_path, "Chapter 3", 4)

    assert [outcome.status for outcome in outcomes] == ["review"]
    assert [finding.code for finding in outcomes[0].findings] == ["few_pages"]
    assert "the previous chapter had 20" in outcomes[0].findings[0].message


def test_zero_valid_images_fails_the_chapter(tmp_path: Path) -> None:
    """Every image downloads but the content filter rejects them all → the chapter has no pages."""
    server = Server()
    urls = [f"https://cdn.test/x/{i:03d}.jpg" for i in range(1, 6)]
    for url in urls:
        server.script(url, noise_image("JPEG", seed=1, width=200, height=500))
    extractor = FakeExtractor()
    extractor.script(
        PAGE_URL.format(n=1),
        Extraction(
            id="e",
            page_url=PAGE_URL.format(n=1),
            images=tuple(ExtractedImage(url=url) for url in urls),
            credits=1,
        ),
    )
    sources = named_chapters([("Chapter 1", PAGE_URL.format(n=1))])
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, server.client())

    assert [outcome.status for outcome in outcomes] == ["failed"]
    assert [finding.code for finding in outcomes[0].findings] == ["no_pages"]
    assert outcomes[0].error == "no pages found"
    assert read_accepted(tmp_path / "Chapter 1") is None


def test_chapter_source_type_is_accepted(tmp_path: Path) -> None:
    """ChapterSource objects built directly (not via named_chapters) run the same way."""
    server, client, extractor = scripted_chapters(1)
    sources = [ChapterSource(name="Chapter 1", url=PAGE_URL.format(n=1))]
    plan = build_plan(sources, tmp_path)

    outcomes = acquire(tmp_path, plan, extractor, client)

    assert [outcome.status for outcome in outcomes] == ["ok"]