"""Tests for omniscan.filter.decide (fixtures written into tmp_path)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageStat

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    FilterArtifact,
    FilterDecision,
    IngestArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.filter import decide
from omniscan.filter.decide import (
    ExampleHash,
    best_match,
    decide_files,
    decide_slices,
    effective_decision,
    load_examples,
    restore,
)
from tests.fixtures import images


def make_paths(tmp_path: Path, series: str = "Series", chapter: str = "Chapter 1") -> ChapterPaths:
    return ChapterPaths(
        series=series,
        chapter=chapter,
        raw_dir=tmp_path / "raw",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        filtered_dir=tmp_path / "out" / "_filtered" / chapter,
    )


def source_file(index: int, name: str, height: int = 300) -> SourceFile:
    return SourceFile(index=index, name=name, sha256="0" * 64, width=400, height=height, y0=0, y1=height)


def ingest_of(*files: SourceFile) -> IngestArtifact:
    strip_height = sum(f.y1 - f.y0 for f in files)
    return IngestArtifact(
        series="Series", chapter="Chapter 1", strip_width=400, strip_height=strip_height, files=list(files)
    )


def slices_of(*ranges: tuple[int, int]) -> SlicesArtifact:
    slices = [Slice(index=i, y0=y0, y1=y1) for i, (y0, y1) in enumerate(ranges)]
    return SlicesArtifact(strip_width=400, strip_height=ranges[-1][1], bands=[], slices=slices)


def mean_channel_diff(a: Image.Image, b: Image.Image) -> list[float]:
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    return list(ImageStat.Stat(diff).mean)


# ---------------------------------------------------------------- load_examples / best_match


def test_load_examples_missing_dirs(tmp_path: Path) -> None:
    assert load_examples(tmp_path, "Series") == []


def test_load_examples_finds_global_and_series(tmp_path: Path) -> None:
    (tmp_path / "global").mkdir()
    (tmp_path / "Series").mkdir()
    images.gradient_jpeg(tmp_path / "global" / "end_card.jpg")
    images.plain_jpeg(tmp_path / "Series" / "promo.jpg", size=(300, 200))
    (tmp_path / "Series" / "notes.txt").write_text("ignored")

    examples = load_examples(tmp_path, "Series")
    assert [e.name for e in examples] == ["global/end_card.jpg", "Series/promo.jpg"]
    with Image.open(tmp_path / "global" / "end_card.jpg") as expected:
        assert examples[0].hash == decide.dhash(expected)
    with Image.open(tmp_path / "Series" / "promo.jpg") as expected:
        assert examples[1].hash == decide.dhash(expected)


def test_best_match_empty_and_closest(tmp_path: Path) -> None:
    assert best_match(0, []) is None

    far = ExampleHash(name="global/far.jpg", hash=0)
    near = ExampleHash(name="global/near.jpg", hash=0b11)  # 2 bits away from the query
    query = 0b1111
    match = best_match(query, [far, near])
    assert match is not None
    example, score = match
    assert example is near
    assert score == pytest.approx(1.0 - 2 / 64)


# ---------------------------------------------------------------- decide_files


def test_decide_files_filters_identical_file(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    paths.raw_dir.mkdir(parents=True)
    examples_dir = tmp_path / "promo" / "global"
    examples_dir.mkdir(parents=True)
    # invert=True: a falling ramp hashes to all-ones, maximally far from the all-zero hash of the
    # solid-color raws (dHash only fires on decreasing adjacent pairs).
    example = images.gradient_jpeg(examples_dir / "end_card.jpg", invert=True)

    images.plain_jpeg(paths.raw_dir / "001.jpg")
    shutil.copy2(example, paths.raw_dir / "002.jpg")
    images.plain_jpeg(paths.raw_dir / "003.jpg", color=(60, 60, 200))

    ingest = ingest_of(source_file(0, "001.jpg"), source_file(1, "002.jpg"), source_file(2, "003.jpg"))
    decisions = decide_files(paths, ingest, load_examples(tmp_path / "promo", "Series"))

    assert [(d.index, d.decision) for d in decisions] == [(0, "keep"), (1, "filtered"), (2, "keep")]
    filtered = decisions[1]
    assert filtered.matched_example == "global/end_card.jpg"
    assert filtered.score == pytest.approx(1.0)
    assert filtered.method == "phash"
    assert all(d.matched_example is None for d in decisions if d.decision == "keep")

    copied = paths.filtered_dir / "002.jpg"
    assert copied.is_file()
    assert copied.read_bytes() == (paths.raw_dir / "002.jpg").read_bytes()


def test_decide_files_empty_examples_keeps_all(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    paths.raw_dir.mkdir(parents=True)
    images.plain_jpeg(paths.raw_dir / "001.jpg")
    images.plain_jpeg(paths.raw_dir / "002.jpg", color=(60, 60, 200))

    ingest = ingest_of(source_file(0, "001.jpg"), source_file(1, "002.jpg"))
    decisions = decide_files(paths, ingest, [])

    assert [(d.decision, d.score, d.matched_example) for d in decisions] == [
        ("keep", 0.0, None),
        ("keep", 0.0, None),
    ]
    assert not paths.filtered_dir.exists()


# ---------------------------------------------------------------- decide_slices


def test_decide_slices_filters_middle_slice(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    examples_dir = tmp_path / "promo" / "global"
    examples_dir.mkdir(parents=True)
    example_path = images.gradient_jpeg(examples_dir / "end_card.jpg", size=(400, 100), invert=True)

    with Image.open(example_path) as img:
        promo = img.convert("RGB")
    strip = Image.new("RGB", (400, 300), (255, 255, 255))
    strip.paste(promo, (0, 100))
    slices = slices_of((0, 100), (100, 200), (200, 300))

    with Image.open(example_path) as img:
        examples = load_examples(tmp_path / "promo", "Series")
    decisions = decide_slices(paths, "Chapter 1", strip, slices, examples)

    assert [(d.index, d.decision) for d in decisions] == [(0, "keep"), (1, "filtered"), (2, "keep")]
    saved = paths.filtered_dir / "Chapter 1_slice_0001.jpg"
    assert saved.is_file()
    with Image.open(saved) as img:
        written = img.convert("RGB")
    source_crop = strip.crop((0, 100, 400, 200))
    assert max(mean_channel_diff(written, source_crop)) <= 3


def test_decide_slices_empty_examples_keeps_all(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    strip = Image.new("RGB", (400, 300))
    decisions = decide_slices(paths, "Chapter 1", strip, slices_of((0, 150), (150, 300)), [])
    assert [(d.decision, d.score) for d in decisions] == [("keep", 0.0), ("keep", 0.0)]
    assert not paths.filtered_dir.exists()


def test_threshold_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = make_paths(tmp_path)
    strip = Image.new("RGB", (400, 100))
    slices = slices_of((0, 100))
    # All-zero query hash; example 6 bits away → similarity lands exactly on the threshold.
    monkeypatch.setattr(decide, "dhash", lambda img, hash_size=8: 0)
    at_threshold = ExampleHash(name="global/exact.jpg", hash=0b111111)  # 1 - 6/64 == 0.90625
    decisions = decide_slices(paths, "Chapter 1", strip, slices, [at_threshold], threshold=0.90625)
    assert decisions[0].decision == "filtered"
    assert decisions[0].score == pytest.approx(0.90625)

    one_further = ExampleHash(name="global/far.jpg", hash=0b1111111)  # 1 - 7/64 == 0.890625
    decisions = decide_slices(paths, "Chapter 1", strip, slices, [one_further], threshold=0.90625)
    assert decisions[0].decision == "keep"


# ---------------------------------------------------------------- effective_decision / restore


def restored_artifact() -> FilterArtifact:
    return FilterArtifact(
        decisions=[
            FilterDecision(
                target="file", index=1, decision="filtered", score=0.99, matched_example="global/x.jpg"
            ),
            FilterDecision(target="slice", index=0, decision="filtered", score=0.97),
            FilterDecision(target="file", index=1, decision="restored", score=1.0, method="manual"),
        ]
    )


def test_effective_decision_last_wins() -> None:
    artifact = restored_artifact()
    assert effective_decision(artifact, "file", 1) == "restored"
    assert effective_decision(artifact, "slice", 0) == "filtered"


def test_effective_decision_no_match() -> None:
    assert effective_decision(restored_artifact(), "slice", 9) is None
    assert effective_decision(FilterArtifact(decisions=[]), "file", 0) is None


def test_restore_appends_and_persists(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    artifact_path = paths.artifact("filter.json")
    restored_artifact().save(artifact_path)

    decision = restore(paths, "slice", 0)
    assert decision.decision == "restored"
    assert decision.score == 1.0
    assert decision.method == "manual"

    reloaded = FilterArtifact.load(artifact_path)
    assert len(reloaded.decisions) == 4
    assert effective_decision(reloaded, "slice", 0) == "restored"


def test_restore_missing_artifact_raises(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"filter\.json"):
        restore(paths, "file", 0)


def test_filter_artifact_roundtrip(tmp_path: Path) -> None:
    artifact = restored_artifact()
    dest = tmp_path / "filter.json"
    artifact.save(dest)
    assert FilterArtifact.load(dest) == artifact
