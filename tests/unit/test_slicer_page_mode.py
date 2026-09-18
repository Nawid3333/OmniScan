"""Unit tests for slice_by_pages."""

import pytest

from omniscan.core.schemas import SourceFile
from omniscan.slicer import slice_by_pages

W = 800


def _page(index: int, name: str, y0: int, y1: int) -> SourceFile:
    return SourceFile(index=index, name=name, sha256=f"{index:064x}", width=W, height=y1 - y0, y0=y0, y1=y1)


def _three() -> list[SourceFile]:
    return [
        _page(0, "p01.jpg", 0, 1000),
        _page(1, "p02.jpg", 1000, 2200),
        _page(2, "p03.jpg", 2200, 3000),
    ]


def test_one_slice_per_file_matching_boundaries() -> None:
    arti = slice_by_pages(_three(), W, 3000)
    assert len(arti.slices) == 3
    for i, (sl, f) in enumerate(zip(arti.slices, _three(), strict=True)):
        assert (sl.index, sl.y0, sl.y1) == (i, f.y0, f.y1)
        assert sl.blank is False
        assert sl.forced_cut is False
        assert sl.source_files == [f.index]


def test_no_bands_and_page_params() -> None:
    arti = slice_by_pages(_three(), W, 3000)
    assert arti.bands == []
    assert arti.params == {"mode": "page"}


def test_strip_dimensions_copied_to_artifact() -> None:
    arti = slice_by_pages(_three(), 1234, 3000)
    assert arti.strip_width == 1234
    assert arti.strip_height == 3000


def test_gap_raises_naming_both_files_and_size() -> None:
    files = [_page(0, "p01.jpg", 0, 1000), _page(1, "p02.jpg", 1005, 3000)]
    with pytest.raises(ValueError, match=r"gap of 5px.*p01\.jpg.*p02\.jpg"):
        slice_by_pages(files, W, 3000)


def test_overlap_raises_naming_both_files() -> None:
    files = [_page(0, "p01.jpg", 0, 1000), _page(1, "p02.jpg", 900, 3000)]
    with pytest.raises(ValueError, match=r"overlap of 100px.*p01\.jpg.*p02\.jpg"):
        slice_by_pages(files, W, 3000)


def test_first_page_not_at_zero_raises() -> None:
    files = [_page(0, "p01.jpg", 10, 1000), _page(1, "p02.jpg", 1000, 3000)]
    with pytest.raises(ValueError, match="y0=10, expected 0"):
        slice_by_pages(files, W, 3000)


def test_last_page_not_at_strip_height_raises() -> None:
    files = [_page(0, "p01.jpg", 0, 1000), _page(1, "p02.jpg", 1000, 2990)]
    with pytest.raises(ValueError, match=r"p02\.jpg.*y1=2990.*strip_height=3000"):
        slice_by_pages(files, W, 3000)


def test_out_of_order_files_rejected_even_if_tiling() -> None:
    files = [
        _page(1, "p02.jpg", 1000, 2000),
        _page(0, "p01.jpg", 0, 1000),
        _page(2, "p03.jpg", 2000, 3000),
    ]
    with pytest.raises(ValueError, match=r"out of order.*p02\.jpg.*p01\.jpg"):
        slice_by_pages(files, W, 3000)


def test_empty_files_with_zero_height_is_valid() -> None:
    arti = slice_by_pages([], W, 0)
    assert arti.slices == []
    assert arti.bands == []
    assert arti.strip_width == W
    assert arti.strip_height == 0
    assert arti.params == {"mode": "page"}


def test_empty_files_with_nonzero_height_raises() -> None:
    with pytest.raises(ValueError, match="strip_height=3000"):
        slice_by_pages([], W, 3000)


def test_single_file_spanning_whole_strip() -> None:
    arti = slice_by_pages([_page(0, "p01.jpg", 0, 4000)], W, 4000)
    assert len(arti.slices) == 1
    assert (arti.slices[0].y0, arti.slices[0].y1) == (0, 4000)
    assert arti.slices[0].source_files == [0]


def test_source_file_indices_not_starting_at_zero() -> None:
    files = [
        _page(3, "p04.jpg", 0, 1000),
        _page(4, "p05.jpg", 1000, 2200),
        _page(5, "p06.jpg", 2200, 3000),
    ]
    arti = slice_by_pages(files, W, 3000)
    assert [sl.source_files for sl in arti.slices] == [[3], [4], [5]]
    assert [sl.index for sl in arti.slices] == [0, 1, 2]
