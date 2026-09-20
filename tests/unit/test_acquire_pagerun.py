"""Tests for omniscan.acquire.pagerun (CPU only; the real extract.pics image list as the golden fixture)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from omniscan.acquire.pagerun import PageSelection, select_pages

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "extractpics_peppercarrot_ep06.json"


def test_golden_peppercarrot_ep06() -> None:
    """The real 43-URL page reduces to exactly the 11 chapter pages with the documented reasons."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    images: list[str] = fixture["images"]

    selection = select_pages(images)

    assert selection.kept == tuple(images[i] for i in fixture["expected_pages"])
    assert len(selection.dropped) == 32
    assert Counter(reason for _, reason in selection.dropped) == Counter(
        {
            "not a page format (.svg)": 25,
            "not part of the page run": 4,
            "thumbnail size in name": 2,
            "url contains 'icon'": 1,
        }
    )
    assert all(reason for _, reason in selection.dropped)


def test_duplicates() -> None:
    a = "https://cdn.test/ch/001.jpg"
    run = (a, "https://cdn.test/ch/002.jpg", "https://cdn.test/ch/003.jpg", "https://cdn.test/ch/004.jpg")

    selection = select_pages([a, *run])

    assert selection.kept == run
    assert selection.dropped == ((a, "duplicate"),)


def test_hashed_names_have_no_run() -> None:
    urls = tuple(f"https://cdn.test/img/{name}" for name in ("3fa9c2.jpg", "91bb07.jpg", "c0de11.jpg", "ab12cd.jpg"))

    selection = select_pages(urls)

    assert selection.kept == urls
    assert selection.dropped == ()


def test_min_share_gates_the_run() -> None:
    run = tuple(f"https://cdn.test/run/00{i}.jpg" for i in (1, 2, 3))
    others = tuple(f"https://cdn.test/other-{i}/photo.jpg" for i in range(5))
    urls = run + others

    assert select_pages(urls).kept == urls  # share 3/8 < 0.5: no dominant run, everything kept
    selection = select_pages(urls, min_share=0.3)
    assert selection.kept == run
    assert selection.dropped == tuple((url, "not part of the page run") for url in others)


def test_tie_prefers_the_group_that_starts_first() -> None:
    first = tuple(f"https://cdn.test/a/00{i}.jpg" for i in (1, 2, 3))
    second = tuple(f"https://cdn.test/b/00{i}.jpg" for i in (1, 2, 3))

    assert select_pages(first + second).kept == first
    assert select_pages(second + first).kept == second


def test_extensionless_query_and_upper_case_suffixes() -> None:
    extensionless = "https://cdn.test/x/jpeg"
    assert select_pages([extensionless]).kept == (extensionless,)

    run = ("https://cdn.test/a/001.jpg", "https://cdn.test/a/001.jpg?w=800", "https://cdn.test/a/002.jpg")
    selection = select_pages(run)  # grouped by path only: one run of 3
    assert selection.kept == run
    assert selection.dropped == ()

    upper = "https://cdn.test/a/001.JPG"
    assert select_pages([upper]).kept == (upper,)


def test_thumbnail_size_in_name() -> None:
    small = "https://cdn.test/x/photo-150x150.jpg"
    large = "https://cdn.test/x/page-1080x1920.jpg"

    selection = select_pages([small, large])

    assert selection.dropped == ((small, "thumbnail size in name"),)
    assert selection.kept == (large,)


def test_empty_input() -> None:
    assert select_pages([]) == PageSelection(kept=(), dropped=())


def _is_subsequence(sub: tuple[str, ...], full: list[str]) -> bool:
    """True when `sub` is a subsequence of `full` (greedy left-to-right match)."""
    position = 0
    for item in sub:
        while position < len(full) and full[position] != item:
            position += 1
        if position == len(full):
            return False
        position += 1
    return True


def test_every_input_appears_exactly_once_in_document_order() -> None:
    pool = [
        "https://cdn.test/a/001.jpg",
        "https://cdn.test/a/002.jpg",
        "https://cdn.test/a/003.jpg",
        "https://cdn.test/b/logo.png",
        "https://cdn.test/c/menu.svg",
        "https://cdn.test/d/photo-150x150.jpg",
        "https://cdn.test/e/jpeg",
    ]
    for cut in range(2, len(pool) + 1):
        urls = pool[:cut]
        selection = select_pages(urls)

        counts = Counter(selection.kept) + Counter(url for url, _ in selection.dropped)
        assert counts == Counter(urls)
        assert _is_subsequence(selection.kept, urls)
        assert _is_subsequence(tuple(url for url, _ in selection.dropped), urls)
        assert all(reason for _, reason in selection.dropped)