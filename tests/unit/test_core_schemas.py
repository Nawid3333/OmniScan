from pathlib import Path

import pytest
from pydantic import ValidationError

from omniscan.core.schemas import (
    Band,
    BBox,
    FinalArtifact,
    FinalLine,
    OcrLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)


def test_bbox_rejects_degenerate() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=10, y0=0, x1=5, y1=10)


def test_bbox_iou() -> None:
    a = BBox(x0=0, y0=0, x1=10, y1=10)
    assert a.iou(a) == 1.0
    assert a.iou(BBox(x0=5, y0=0, x1=15, y1=10)) == pytest.approx(50 / 150)
    assert a.iou(BBox(x0=20, y0=20, x1=30, y1=30)) == 0.0


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Slice(index=0, y0=0, y1=10, bogus=True)  # type: ignore[call-arg]


def test_slices_roundtrip_atomic(tmp_path: Path) -> None:
    art = SlicesArtifact(
        strip_width=800,
        strip_height=5000,
        bands=[Band(y0=100, y1=180, color=(255, 255, 255))],
        slices=[Slice(index=0, y0=0, y1=140), Slice(index=1, y0=140, y1=5000, blank=False)],
        params={"band_min_px": 50},
    )
    path = tmp_path / "w" / "slices.json"
    art.save(path)
    assert not path.with_suffix(".json.tmp").exists()
    assert SlicesArtifact.load(path) == art


def test_regions_and_final_roundtrip(tmp_path: Path) -> None:
    region = Region(
        id="r0001",
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=1, y0=2, x1=30, y1=40),
        lines=[OcrLine(bbox=BBox(x0=1, y0=2, x1=30, y1=20), text="민준 형", score=0.98, engine="ko-rec")],
        text="민준 형",
    )
    RegionsArtifact(regions=[region]).save(tmp_path / "ocr.json")
    assert RegionsArtifact.load(tmp_path / "ocr.json").regions[0].text == "민준 형"
    final = FinalArtifact(
        judge_model="gemma4:31b-cloud",
        lines=[FinalLine(region_id="r0001", text="Min-jun hyung", decision="pick")],
    )
    final.save(tmp_path / "final.json")
    assert FinalArtifact.load(tmp_path / "final.json").lines[0].decision == "pick"
