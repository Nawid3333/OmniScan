"""Tests for omniscan.filter.apply: overrides, verdicts, fingerprint, example files, slice marking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from PIL import Image

from omniscan.core.schemas import FilterArtifact, FilterDecision, Slice, SlicesArtifact
from omniscan.filter.apply import (
    EMPTY_OVERRIDES,
    Overrides,
    apply_slice_filter,
    example_files,
    examples_fingerprint,
    file_verdict,
    load_overrides,
    slice_verdict,
)
from omniscan.filter.decide import ExampleHash
from omniscan.filter.hashing import dhash
from tests.fixtures import images

THRESHOLD = 0.90


def manual(
    target: Literal["file", "slice"],
    index: int,
    decision: Literal["keep", "filtered", "restored"],
    method: Literal["phash", "embed", "manual"] = "manual",
) -> FilterDecision:
    return FilterDecision(target=target, index=index, decision=decision, score=1.0, method=method)


def artifact_json(tmp_path: Path, decisions: list[FilterDecision]) -> Path:
    path = tmp_path / "filter.json"
    FilterArtifact(decisions=decisions).save(path)
    return path


# ---------------------------------------------------------------- load_overrides


def test_load_overrides_last_write_wins_both_targets(tmp_path: Path) -> None:
    path = artifact_json(
        tmp_path,
        [
            manual("file", 1, "filtered"),
            manual("file", 1, "restored"),
            manual("slice", 3, "restored"),
            manual("slice", 3, "filtered"),
            manual("file", 2, "filtered"),
        ],
    )
    overrides = load_overrides(path)
    assert overrides.files_restored == {1}
    assert overrides.files_forced == {2}
    assert overrides.slices_restored == frozenset()
    assert overrides.slices_forced == {3}


def test_load_overrides_ignores_automatic_entries(tmp_path: Path) -> None:
    path = artifact_json(
        tmp_path,
        [
            FilterDecision(target="file", index=0, decision="filtered", score=0.99, method="phash"),
            FilterDecision(target="slice", index=0, decision="restored", score=1.0),  # default method=phash
        ],
    )
    assert load_overrides(path) == EMPTY_OVERRIDES


def test_load_overrides_manual_keep_counts_as_neither(tmp_path: Path) -> None:
    overrides = load_overrides(artifact_json(tmp_path, [manual("file", 0, "keep")]))
    assert overrides.files_restored == frozenset()
    assert overrides.files_forced == frozenset()


def test_load_overrides_missing_or_corrupt_file_is_empty(tmp_path: Path) -> None:
    assert load_overrides(tmp_path / "absent.json") == EMPTY_OVERRIDES
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_overrides(corrupt) == EMPTY_OVERRIDES
    invalid = tmp_path / "invalid.json"
    payload = json.dumps({"decisions": [{"target": "file", "index": 0, "decision": "weird"}]})
    invalid.write_text(payload, encoding="utf-8")
    assert load_overrides(invalid) == EMPTY_OVERRIDES


# ---------------------------------------------------------------- verdicts


def overrides_of(
    *,
    file_forced: frozenset[int] = frozenset(),
    file_restored: frozenset[int] = frozenset(),
    slice_forced: frozenset[int] = frozenset(),
    slice_restored: frozenset[int] = frozenset(),
) -> Overrides:
    return Overrides(
        files_restored=file_restored,
        files_forced=file_forced,
        slices_restored=slice_restored,
        slices_forced=slice_forced,
    )


def test_file_verdict_threshold_boundary() -> None:
    overrides = EMPTY_OVERRIDES
    assert file_verdict(0, THRESHOLD, "global/x.jpg", THRESHOLD, overrides) == "filtered"
    assert file_verdict(0, THRESHOLD - 1e-9, "global/x.jpg", THRESHOLD, overrides) == "keep"
    assert file_verdict(0, 0.95, None, THRESHOLD, overrides) == "keep"  # no matched example, no verdict
    assert file_verdict(0, 0.0, None, THRESHOLD, overrides) == "keep"  # without examples


def test_file_verdict_forced_beats_non_match_and_restored_beats_match() -> None:
    forced = overrides_of(file_forced=frozenset({4}))
    assert file_verdict(4, 0.0, None, THRESHOLD, forced) == "filtered"
    restored = overrides_of(file_restored=frozenset({4}))
    assert file_verdict(4, 0.99, "global/x.jpg", THRESHOLD, restored) == "keep"


def test_slice_verdict_threshold_and_overrides() -> None:
    overrides = EMPTY_OVERRIDES
    assert slice_verdict(7, THRESHOLD, "global/x.jpg", THRESHOLD, overrides) == "filtered"
    assert slice_verdict(7, THRESHOLD - 1e-9, "global/x.jpg", THRESHOLD, overrides) == "keep"
    assert slice_verdict(7, 0.0, None, THRESHOLD, overrides_of(slice_forced=frozenset({7}))) == "filtered"
    assert (
        slice_verdict(7, 1.0, "global/x.jpg", THRESHOLD, overrides_of(slice_restored=frozenset({7})))
        == "keep"
    )


# ---------------------------------------------------------------- examples_fingerprint / example_files


def test_examples_fingerprint_changes_with_name_or_hash_and_is_stable() -> None:
    base = [ExampleHash(name="global/a.jpg", hash=12), ExampleHash(name="S/b.jpg", hash=34)]
    fingerprint = examples_fingerprint(base)
    assert examples_fingerprint(list(base)) == fingerprint
    assert examples_fingerprint([ExampleHash(name="global/renamed.jpg", hash=12), *base[1:]]) != fingerprint
    assert examples_fingerprint([base[0], ExampleHash(name="S/b.jpg", hash=56)]) != fingerprint
    assert examples_fingerprint([]) == examples_fingerprint([])


def test_example_files_lists_global_and_series(tmp_path: Path) -> None:
    (tmp_path / "global").mkdir()
    (tmp_path / "Series").mkdir()
    (tmp_path / "Series" / "sub").mkdir()
    second = images.plain_jpeg(tmp_path / "global" / "b.jpg")
    first = images.plain_jpeg(tmp_path / "global" / "a.png")
    series = images.plain_jpeg(tmp_path / "Series" / "c.jpeg")
    images.plain_jpeg(tmp_path / "Series" / "sub" / "d.jpg")  # nested: not picked up
    (tmp_path / "Series" / "notes.txt").write_text("ignored", encoding="utf-8")
    assert example_files(tmp_path, "Series") == [first, second, series]
    assert example_files(tmp_path, "Other") == [first, second]
    assert example_files(tmp_path / "missing", "Series") == []


# ---------------------------------------------------------------- apply_slice_filter


def solid_banner(height: int = 100, width: int = 4) -> torch.Tensor:
    """A solid white block (the dHash of any solid image is all-zero bits)."""
    return torch.full((3, height, width), 255, dtype=torch.uint8)


def example_hash_of_solid() -> ExampleHash:
    return ExampleHash(name="global/end.jpg", hash=dhash(Image.new("RGB", (64, 64), (255, 255, 255))))


def slice_artifact(*ranges: tuple[int, int], blanks: tuple[int, ...] = ()) -> SlicesArtifact:
    return SlicesArtifact(
        strip_width=4,
        strip_height=ranges[-1][1],
        bands=[],
        slices=[Slice(index=i, y0=y0, y1=y1, blank=i in blanks) for i, (y0, y1) in enumerate(ranges)],
    )


def test_apply_slice_filter_marks_matching_and_skips_blank() -> None:
    noise = torch.randint(0, 256, (3, 100, 4), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    strip = torch.cat([solid_banner(), solid_banner(), noise], dim=1)
    slices = slice_artifact((0, 100), (100, 200), (200, 300), blanks=(0,))

    result = apply_slice_filter(strip, slices, [example_hash_of_solid()], 0.90, EMPTY_OVERRIDES, "Chapter 1")
    assert [s.filtered for s in result.slices] == [False, True, False]
    assert [s.blank for s in result.slices] == [True, False, False]  # blank untouched
    assert slices.slices[1].filtered is False  # the input artifact is not mutated


def test_apply_slice_filter_saves_filtered_slice_jpeg(tmp_path: Path) -> None:
    dest = tmp_path / "_filtered" / "Chapter 1_slice_0000.jpg"
    apply_slice_filter(
        solid_banner(),
        slice_artifact((0, 100)),
        [example_hash_of_solid()],
        0.90,
        EMPTY_OVERRIDES,
        "Chapter 1",
        filtered_dir=dest.parent,
    )
    assert dest.is_file()
    with Image.open(dest) as saved:
        assert saved.format == "JPEG"
        assert saved.size == (4, 100)


def test_apply_slice_filter_overrides() -> None:
    noise = torch.randint(0, 256, (3, 100, 4), dtype=torch.uint8, generator=torch.Generator().manual_seed(2))
    strip = torch.cat([noise, solid_banner()], dim=1)
    slices = slice_artifact((0, 100), (100, 200))

    forced = Overrides(
        files_forced=frozenset(),
        files_restored=frozenset(),
        slices_forced=frozenset({1}),
        slices_restored=frozenset(),
    )
    result = apply_slice_filter(strip, slices, [], 0.90, forced, "Chapter 1")
    assert [s.filtered for s in result.slices] == [False, True]

    restored = Overrides(
        files_forced=frozenset(),
        files_restored=frozenset(),
        slices_forced=frozenset(),
        slices_restored=frozenset({1}),
    )
    result = apply_slice_filter(strip, slices, [example_hash_of_solid()], 0.90, restored, "Chapter 1")
    assert [s.filtered for s in result.slices] == [False, False]


@pytest.mark.parametrize("shape", [(3, 10, 4), (3, 9, 9)])
def test_apply_slice_filter_handles_short_slices(shape: tuple[int, int, int]) -> None:
    strip = torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(3))
    result = apply_slice_filter(strip, slice_artifact((0, shape[1])), [], 0.90, EMPTY_OVERRIDES, "Chapter 1")
    assert result.slices[0].filtered is False
