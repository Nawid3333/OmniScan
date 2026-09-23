"""Tests added by the Q2 mutation review — each fails on a surviving mutant of filter/glossary/importer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import GlossaryEntry, IngestArtifact, Slice, SlicesArtifact, SourceFile
from omniscan.filter import decide
from omniscan.filter.decide import ExampleHash, decide_files, decide_slices, load_examples
from omniscan.filter.hashing import dhash
from omniscan.glossary.match import Match, find_terms
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml, import_yaml
from omniscan.importer.plan import ImportPlanError, plan_import
from tests.fixtures import images

# ---------------------------------------------------------------- shared helpers


def make_paths(tmp_path: Path, series: str = "Series", chapter: str = "Chapter 1") -> ChapterPaths:
    return ChapterPaths(
        series=series,
        chapter=chapter,
        raw_dir=tmp_path / "raw",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        filtered_dir=tmp_path / "out" / "_filtered" / chapter,
    )


def source_file(index: int, name: str) -> SourceFile:
    return SourceFile(index=index, name=name, sha256="0" * 64, width=400, height=300, y0=0, y1=300)


def ingest_of(*files: SourceFile) -> IngestArtifact:
    return IngestArtifact(
        series="Series",
        chapter="Chapter 1",
        strip_width=400,
        strip_height=300 * len(files),
        files=list(files),
    )


def slices_of(*ranges: tuple[int, int]) -> SlicesArtifact:
    slices = [Slice(index=i, y0=y0, y1=y1) for i, (y0, y1) in enumerate(ranges)]
    return SlicesArtifact(strip_width=400, strip_height=ranges[-1][1], bands=[], slices=slices)


def write_files(folder: Path, names: list[str]) -> None:
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(b"img")


# ---------------------------------------------------------------- filter (decide / hashing)


def test_dhash_pins_rising_and_falling_ramp_bit_patterns(tmp_path: Path) -> None:
    images.gradient_jpeg(tmp_path / "rising.jpg")
    images.gradient_jpeg(tmp_path / "falling.jpg", invert=True)
    with Image.open(tmp_path / "rising.jpg") as rising, Image.open(tmp_path / "falling.jpg") as falling:
        assert dhash(rising) == 0  # every adjacent pair increases → no bit set
        assert dhash(falling) == 2**64 - 1  # every pair decreases → all 64 bits set


def test_decide_files_threshold_inclusive_scores_exact_and_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = make_paths(tmp_path)
    paths.raw_dir.mkdir(parents=True)
    images.plain_jpeg(paths.raw_dir / "001.jpg")
    images.plain_jpeg(paths.raw_dir / "002.jpg")
    ingest = ingest_of(source_file(0, "001.jpg"), source_file(1, "002.jpg"))
    # All-zero query hash; example sits exactly 6 bits away → similarity lands exactly on the threshold.
    monkeypatch.setattr(decide, "dhash", lambda img, hash_size=8: 0)
    at = ExampleHash(name="global/at.jpg", hash=0b111111)  # 1 - 6/64 == 0.90625
    decisions = decide_files(paths, ingest, [at], threshold=0.90625)
    assert [(d.decision, d.score, d.matched_example) for d in decisions] == [
        ("filtered", pytest.approx(0.90625), "global/at.jpg"),
        ("filtered", pytest.approx(0.90625), "global/at.jpg"),
    ]

    one_further = ExampleHash(name="global/under.jpg", hash=0b1111111)  # 1 - 7/64 == 0.890625 < 0.90
    decisions = decide_files(paths, ingest, [one_further])  # default threshold, keep reports the best score
    assert [(d.decision, d.score, d.matched_example) for d in decisions] == [
        ("keep", pytest.approx(0.890625), None),
        ("keep", pytest.approx(0.890625), None),
    ]


def test_decide_slices_keep_score_reports_best_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = make_paths(tmp_path)
    strip = Image.new("RGB", (400, 100))
    monkeypatch.setattr(decide, "dhash", lambda img, hash_size=8: 0)
    under = ExampleHash(name="global/under.jpg", hash=0b1111111)  # 1 - 7/64 == 0.890625 < 0.90 default
    decisions = decide_slices(paths, "Chapter 1", strip, slices_of((0, 100)), [under])
    assert [(d.decision, d.score, d.matched_example) for d in decisions] == [
        ("keep", pytest.approx(0.890625), None)
    ]


def test_decide_slices_saved_crop_is_the_exact_box_and_names_the_example(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    examples_dir = tmp_path / "promo" / "global"
    examples_dir.mkdir(parents=True)
    example_path = images.gradient_jpeg(examples_dir / "end_card.jpg", size=(400, 100), invert=True)
    with Image.open(example_path) as img:
        promo = img.convert("RGB")
    strip = Image.new("RGB", (400, 300), (255, 255, 255))
    strip.paste(promo, (0, 100))
    with Image.open(example_path) as img:
        examples = load_examples(tmp_path / "promo", "Series")

    decisions = decide_slices(
        paths, "Chapter 1", strip, slices_of((0, 100), (100, 200), (200, 300)), examples
    )

    assert decisions[1].matched_example == "global/end_card.jpg"
    saved = paths.filtered_dir / "Chapter 1_slice_0001.jpg"
    with Image.open(saved) as img:
        assert img.size == (400, 100)  # exactly (y1 - y0) rows of the crop box


# ---------------------------------------------------------------- glossary (match / store / yaml_io)


def test_find_terms_consumes_every_listed_particle() -> None:
    entries = [GlossaryEntry(id=1, source="지훈", target="Jihoon")]
    particles = (
        "이가",
        "은는",
        "을를",
        "의",
        "아",
        "야",
        "도",
        "에게",
        "에서",
        "에",
        "이",
        "가",
        "은",
        "는",
        "을",
        "를",
    )
    for particle in particles:
        matches = find_terms("지훈" + particle + " 다음", entries, lang="ko")
        assert len(matches) == 1, particle
        assert matches[0].particle == particle, particle
        assert matches[0].end == 2 + len(particle), particle


def test_find_terms_scan_resumes_after_the_particle() -> None:
    # "에" is itself a glossary term here: resuming the scan at term_end instead of after the particle
    # would match it inside the particle region and produce an overlapping match.
    entries = [
        GlossaryEntry(id=1, source="학교", target="school"),
        GlossaryEntry(id=2, source="에", target="at"),
    ]
    matches = find_terms("학교에서 갔다", entries, lang="ko")
    assert matches == [Match(entry_id=1, source="학교", start=0, end=4, particle="에서")]


def test_find_terms_reaches_a_term_at_the_last_character() -> None:
    matches = find_terms("학교가", [GlossaryEntry(id=1, source="가", target="Ga")], lang="ko")
    assert matches == [Match(entry_id=1, source="가", start=2, end=3, particle=None)]


def test_find_by_source_lowest_id_wins_among_duplicate_sources(tmp_path: Path) -> None:
    with GlossaryStore(tmp_path / "series.db") as store:
        first = store.add(GlossaryEntry(source="하늘", target="Sky"))
        store.add(GlossaryEntry(source="하늘", target="Sky (second row)"))
        found = store.find_by_source("하늘")
        assert found == first


def test_export_yaml_creates_missing_parent_directory(tmp_path: Path) -> None:
    dest = tmp_path / "made" / "up" / "glossary.yaml"
    with GlossaryStore(tmp_path / "series.db") as store:
        store.add(GlossaryEntry(source="지훈", target="Jihoon"))
        export_yaml(store, dest)
    assert len(yaml.safe_load(dest.read_text(encoding="utf-8"))) == 1


def test_import_yaml_replace_writes_entries_in_file_order(tmp_path: Path) -> None:
    # export must write entries id-ascending (human-review contract) and a replace-import must re-assign
    # ids in file order, so the round trip preserves order.
    yaml_path = tmp_path / "glossary.yaml"
    with GlossaryStore(tmp_path / "series.db") as store:
        store.add(GlossaryEntry(source="지훈", target="Jihoon"))
        store.add(GlossaryEntry(source="학교", target="school"))
        export_yaml(store, yaml_path)
        assert import_yaml(store, yaml_path, mode="replace") == 2
        assert [e.source for e in store.list()] == ["지훈", "학교"]  # id order follows file order
        export_yaml(store, yaml_path)
    exported = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert [row["source"] for row in exported] == ["지훈", "학교"]


# ---------------------------------------------------------------- importer (plan / execute)


def test_case_c_groups_episode_marker_files(tmp_path: Path) -> None:
    src = tmp_path / "dump"
    write_files(src, ["ep3_01.jpg", "ep2_01.jpg"])
    plan = plan_import(src, series="Solo Leveling")
    assert [(item.chapter, [p.name for p in item.files]) for item in plan.items] == [
        ("Chapter 2", ["ep2_01.jpg"]),
        ("Chapter 3", ["ep3_01.jpg"]),
    ]


def test_empty_source_message_names_the_empty_folder(tmp_path: Path) -> None:
    src = tmp_path / "vacant"
    src.mkdir()
    with pytest.raises(ImportPlanError, match="source folder is empty"):
        plan_import(src, series="Solo Leveling")


def test_warnings_use_natural_order(tmp_path: Path) -> None:
    src = tmp_path / "Chapter 1"
    write_files(src, ["1.jpg", "10.txt", "2.txt"])
    plan = plan_import(src, series="Solo Leveling")
    assert plan.warnings == ["skipped non-image file: 2.txt", "skipped non-image file: 10.txt"]


def test_case_c_chapters_ascending_even_when_natural_order_disagrees(tmp_path: Path) -> None:
    # Natural file order puts "A_Ch10_01.jpg" before "B_Ch2_01.jpg"; the plan must still list chapters ascending.
    src = tmp_path / "dump"
    write_files(src, ["A_Ch10_01.jpg", "B_Ch2_01.jpg"])
    plan = plan_import(src, series="Solo Leveling")
    assert [(item.chapter, [p.name for p in item.files]) for item in plan.items] == [
        ("Chapter 2", ["B_Ch2_01.jpg"]),
        ("Chapter 10", ["A_Ch10_01.jpg"]),
    ]
