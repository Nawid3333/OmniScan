"""Tests for page fingerprinting: reading-order dhashes, similarity matrices, block-wise batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omniscan.filter.hashing import dhash, similarity
from omniscan.match.pages import BITS, chapter_hashes, similarity_blocks, similarity_matrix
from tests.fixtures.chapter_sets import write_chapter, write_page


def test_chapter_hashes_follow_reading_order(tmp_path: Path) -> None:
    write_page(tmp_path / "2.jpg", 11)
    write_page(tmp_path / "10.jpg", 22)  # natural sort: 2 before 10, not lexicographic
    write_page(tmp_path / "1.jpg", 33)
    hashes = chapter_hashes(tmp_path)
    assert hashes == [
        dhash_of(tmp_path / "1.jpg"),
        dhash_of(tmp_path / "2.jpg"),
        dhash_of(tmp_path / "10.jpg"),
    ]


def dhash_of(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as img:
        return dhash(img)


def test_chapter_hashes_of_empty_folder_is_empty(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not an image")
    assert chapter_hashes(tmp_path) == []


def test_similarity_matrix_matches_the_shared_hashing_helpers(tmp_path: Path) -> None:
    write_page(tmp_path / "a.jpg", 1)
    write_page(tmp_path / "b.jpg", 2)
    hashes = chapter_hashes(tmp_path)
    matrix = similarity_matrix(hashes, hashes)
    assert matrix.shape == (2, 2)
    assert matrix.diagonal().min() == 1.0  # a page is identical to itself
    assert 0.0 <= matrix[0, 1] <= 1.0
    assert matrix[0, 1] == pytest.approx(similarity(hashes[0], hashes[1], bits=BITS))
    assert matrix[0, 1] < 0.7  # two different pages: dhash's ~0.5 random baseline, far below a same page


def test_similarity_blocks_agree_with_per_pair_matrices(tmp_path: Path) -> None:
    chapters_a = [
        write_chapter(tmp_path / "a" / name, list(range(1 + 10 * k, 10 * k + 4)))
        for k, name in enumerate(["one", "two", "three", "four"])
    ]
    chapters_b = [
        write_chapter(tmp_path / "b" / name, list(range(50 + 10 * k, 10 * k + 54)))
        for k, name in enumerate(["x", "y"])
    ]
    chapters_b.append(write_chapter(tmp_path / "b" / "empty", []))  # zero pages: padding must not leak
    hashes_a = [chapter_hashes(chapter) for chapter in chapters_a]
    hashes_b = [chapter_hashes(chapter) for chapter in chapters_b]
    for a_indices, block in similarity_blocks(hashes_a, hashes_b):
        for a_row, a_index in enumerate(a_indices):
            for b_index in range(len(hashes_b)):
                sliced = block[a_row, b_index][: len(hashes_a[a_index]), : len(hashes_b[b_index])]
                assert np.array_equal(sliced, similarity_matrix(hashes_a[a_index], hashes_b[b_index]))


def test_similarity_blocks_respects_the_block_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hashes_a = [[0xAAAA * k + 1, 0xBBBB * k + 2] for k in range(1, 6)]
    hashes_b = [[0xCCCC * k + 3] for k in range(1, 4)]
    # a cap below one chapter pair's XOR intermediate forces the matrix to be cut per A chapter
    monkeypatch.setattr("omniscan.match.pages.BLOCK_ELEMENTS", 100)
    blocks = list(similarity_blocks(hashes_a, hashes_b))
    assert len(blocks) > 1  # the cap, not the input size, decided how the matrix was cut
    covered: list[int] = []
    p_max = max(len(chapter) for chapter in hashes_a)
    q_max = max(len(chapter) for chapter in hashes_b)
    for a_indices, block in blocks:
        assert block.shape == (len(a_indices), len(hashes_b), p_max, q_max)
        covered += list(a_indices)
    assert covered == list(range(len(hashes_a)))
    for a_indices, block in blocks:
        for a_row, a_index in enumerate(a_indices):
            for b_index in range(len(hashes_b)):
                assert np.array_equal(
                    block[a_row, b_index], similarity_matrix(hashes_a[a_index], hashes_b[b_index])
                )


def test_similarity_blocks_of_empty_series_yields_nothing() -> None:
    assert list(similarity_blocks([], [[1, 2]])) == []
    assert list(similarity_blocks([[1, 2]], [])) == []
