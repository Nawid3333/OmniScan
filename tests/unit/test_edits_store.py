"""Tests for omniscan.edits.store — edit operations on a hand-built chapter work dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    BBox,
    ChapterEdits,
    FinalArtifact,
    FinalLine,
    OcrLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.edits.apply import SOURCE_CHANGED


def box(x0: int, y0: int, x1: int, y1: int) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def region(rid: str, bbox: BBox, text: str, order: int) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind="bubble_text",
        bbox=bbox,
        reading_order=order,
        lines=[OcrLine(bbox=bbox, text=text, score=0.9, engine="ocr")],
        text=text,
        confidence=0.9,
    )


R1 = region("r0001", box(10, 10, 110, 60), "안녕", 0)
R2 = region("r0002", box(10, 200, 110, 260), "반가워", 1)


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    paths = ChapterPaths(
        series="S",
        chapter="Chapter 1",
        raw_dir=tmp_path / "library" / "S" / "Chapter 1",
        work_dir=tmp_path / "work" / "S" / "Chapter 1",
        output_dir=tmp_path / "output" / "S" / "Chapter 1",
        filtered_dir=tmp_path / "output" / "S" / "_filtered" / "Chapter 1",
    )
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(regions=[R1, R2]).save(paths.artifact("ocr.json"))
    FinalArtifact(
        judge_model="judge",
        lines=[
            FinalLine(region_id="r0001", text="Hi", decision="pick"),
            FinalLine(region_id="r0002", text="Glad", decision="pick"),
        ],
    ).save(paths.artifact("final.json"))
    return paths


def ocr(paths: ChapterPaths) -> list[Region]:
    return RegionsArtifact.load(paths.artifact("ocr.json")).regions


def final(paths: ChapterPaths) -> dict[str, FinalLine]:
    return {line.region_id: line for line in FinalArtifact.load(paths.artifact("final.json")).lines}


def test_first_edit_keeps_the_pipeline_output_as_auto_files(paths: ChapterPaths) -> None:
    store.update_region(paths, "r0001", direction="ltr", text="안녕하세요")
    assert RegionsArtifact.load(paths.artifact("ocr_auto.json")).regions == [R1, R2]
    assert FinalArtifact.load(paths.artifact("final_auto.json")).lines[0].text == "Hi"
    assert ocr(paths)[0].text == "안녕하세요"
    edits = store.load_edits(paths)
    assert [(e.region_id, e.anchor, e.text) for e in edits.regions] == [("r0001", R1.bbox, "안녕하세요")]


def test_repeated_edits_of_one_region_merge_into_one_edit(paths: ChapterPaths) -> None:
    store.update_region(paths, "r0001", direction="ltr", text="안녕하세요")
    moved = box(20, 300, 120, 350)
    region = store.update_region(paths, "r0001", direction="ltr", bbox=moved, kind="free_text")
    assert (region.bbox, region.kind, region.text) == (moved, "free_text", "안녕하세요")
    edits = store.load_edits(paths)
    assert len(edits.regions) == 1 and edits.regions[0].anchor == R1.bbox
    # moved below r0002: reading order follows
    assert [(r.id, r.reading_order) for r in ocr(paths)] == [("r0001", 1), ("r0002", 0)]
    # a third edit still finds the moved region's edit
    store.update_region(paths, "r0001", direction="ltr", text="안녕!")
    assert len(store.load_edits(paths).regions) == 1


def test_box_is_clamped_into_the_strip_and_empty_boxes_are_refused(paths: ChapterPaths) -> None:
    region = store.update_region(paths, "r0001", direction="ltr", bbox=box(-50, 580, 150, 700))
    assert region.bbox == box(0, 580, 150, 600)
    with pytest.raises(ValueError, match="empty inside the strip"):
        store.update_region(paths, "r0001", direction="ltr", bbox=box(250, 10, 300, 60))


def test_unknown_region_and_missing_slices_are_not_found(paths: ChapterPaths) -> None:
    with pytest.raises(store.EditNotFoundError, match=r"region 'r0009' not found in ocr\.json"):
        store.update_region(paths, "r0009", direction="ltr", text="x")
    paths.artifact("slices.json").unlink()
    with pytest.raises(store.EditNotFoundError, match=r"slices\.json not found"):
        store.update_region(paths, "r0001", direction="ltr", text="x")


def test_add_region_then_delete_forgets_it(paths: ChapterPaths) -> None:
    added = store.add_region(paths, box(10, 100, 110, 150), direction="ltr", text="툭", kind="sfx", lang="ko")
    assert added.id == "m0001" and added.kind == "sfx" and added.reading_order == 1
    assert [r.id for r in ocr(paths)] == ["r0001", "r0002", "m0001"]
    assert store.add_region(paths, box(10, 400, 110, 450), direction="ltr").id == "m0002"
    store.delete_region(paths, "m0001", direction="ltr")
    assert [r.id for r in ocr(paths)] == ["r0001", "r0002", "m0002"]
    assert [e.region_id for e in store.load_edits(paths).regions] == ["m0002"]


def test_delete_and_revert_a_pipeline_region(paths: ChapterPaths) -> None:
    store.delete_region(paths, "r0002", direction="ltr")
    assert [r.id for r in ocr(paths)] == ["r0001"]
    assert [r.id for r in store.deleted_regions(paths)] == ["r0002"]
    restored = store.revert_region(paths, "r0002", direction="ltr")
    assert restored == R2
    assert ocr(paths) == [R1, R2]
    assert store.load_edits(paths) == ChapterEdits()


def test_revert_of_an_added_region_removes_it_and_unedited_region_has_nothing_to_revert(
    paths: ChapterPaths,
) -> None:
    store.add_region(paths, box(10, 400, 110, 450), direction="ltr")
    assert store.revert_region(paths, "m0001", direction="ltr") is None
    assert [r.id for r in ocr(paths)] == ["r0001", "r0002"]
    with pytest.raises(store.EditNotFoundError, match="no edit to revert"):
        store.revert_region(paths, "r0001", direction="ltr")


def test_translation_edit_and_revert(paths: ChapterPaths) -> None:
    line = store.set_translation(paths, "r0002", "Nice to see you", direction="ltr")
    assert (line.text, line.decision, line.flags) == ("Nice to see you", "manual", [])
    assert final(paths)["r0002"].text == "Nice to see you"
    store.set_translation(paths, "r0002", "Good to see you", direction="ltr")
    assert len(store.load_edits(paths).translations) == 1
    restored = store.revert_translation(paths, "r0002", direction="ltr")
    assert restored is not None and restored.text == "Glad"
    assert final(paths)["r0002"].decision == "pick"
    with pytest.raises(store.EditNotFoundError, match="no hand-written line"):
        store.revert_translation(paths, "r0002", direction="ltr")


def test_hand_written_line_follows_its_region_when_moved_and_reverted(paths: ChapterPaths) -> None:
    """Found driving the studio: a moved box orphaned its line (matched by the old position)."""
    store.set_translation(paths, "r0002", "Nice to see you", direction="ltr")
    store.update_region(paths, "r0002", direction="ltr", bbox=box(60, 400, 160, 460))
    assert final(paths)["r0002"].text == "Nice to see you"
    store.update_region(paths, "r0002", direction="ltr", bbox=box(100, 500, 200, 560))
    assert final(paths)["r0002"].text == "Nice to see you"
    store.revert_region(paths, "r0002", direction="ltr")
    assert ocr(paths)[1].bbox == R2.bbox
    assert final(paths)["r0002"].text == "Nice to see you"
    assert store.load_edits(paths).translations[0].anchor == R2.bbox


def test_fixing_the_source_flags_the_hand_written_line(paths: ChapterPaths) -> None:
    store.set_translation(paths, "r0002", "Nice to see you", direction="ltr")
    store.update_region(paths, "r0002", direction="ltr", text="또 봐")
    assert final(paths)["r0002"].flags == [SOURCE_CHANGED]


def test_english_only_edit_leaves_ocr_json_bytes_alone(paths: ChapterPaths) -> None:
    """ocr.json is the translate stage's input: rewriting it would re-translate the chapter on the next run."""
    legacy = paths.artifact("ocr.json").read_text(encoding="utf-8").replace('"angle": 0.0,', "")
    paths.artifact("ocr.json").write_text(legacy, encoding="utf-8")  # as an older version wrote it
    before = paths.artifact("ocr.json").read_bytes()
    store.set_translation(paths, "r0001", "Hello", direction="ltr")
    assert paths.artifact("ocr.json").read_bytes() == before


def test_translation_without_a_judge_run_creates_final_json(paths: ChapterPaths) -> None:
    paths.artifact("final.json").unlink()
    store.set_translation(paths, "r0001", "Hello", direction="ltr")
    artifact = FinalArtifact.load(paths.artifact("final.json"))
    assert artifact.judge_model == store.MANUAL_JUDGE
    assert [(x.region_id, x.text) for x in artifact.lines] == [("r0001", "Hello")]
    # the empty judge output is kept, so reverting leaves no line behind
    assert store.revert_translation(paths, "r0001", direction="ltr") is None
    assert FinalArtifact.load(paths.artifact("final.json")).lines == []


def test_chapter_without_ocr_starts_from_hand_drawn_regions(paths: ChapterPaths) -> None:
    paths.artifact("ocr.json").unlink()
    paths.artifact("final.json").unlink()
    added = store.add_region(paths, box(10, 10, 110, 60), direction="ltr", text="안녕")
    assert [r.id for r in ocr(paths)] == [added.id]
    assert RegionsArtifact.load(paths.artifact("ocr_auto.json")).regions == []
    assert not paths.artifact("final.json").exists()  # no judge ran and no line was written


def test_hand_lettering_set_replace_revert_and_follow_a_moved_region(paths: ChapterPaths) -> None:
    edit = store.set_layout(paths, "r0001", {"color": (255, 0, 0), "size_px": 20})
    assert (edit.region_id, edit.anchor, edit.color, edit.size_px) == ("r0001", R1.bbox, (255, 0, 0), 20)
    store.set_layout(paths, "r0001", {"hidden": True})  # a second set replaces the first entirely
    layout = store.load_edits(paths).layout
    assert len(layout) == 1 and layout[0].hidden and layout[0].color is None
    assert store.hand_lettered_ids(paths) == ["r0001"]
    store.update_region(paths, "r0001", direction="ltr", bbox=box(20, 400, 120, 460))
    assert store.hand_lettered_ids(paths) == ["r0001"]  # the lettering followed the moved box
    store.revert_layout(paths, "r0001")
    assert store.load_edits(paths).layout == [] and store.hand_lettered_ids(paths) == []
    with pytest.raises(store.EditNotFoundError, match="no hand lettering"):
        store.revert_layout(paths, "r0001")
    with pytest.raises(ValueError):
        store.set_layout(paths, "r0002", {"size_px": 1})  # below LayoutEdit's minimum
