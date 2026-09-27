"""Tests for the web API's editing routes (src/omniscan/web/app.py) — torch-free fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from omniscan.core.config import Config, PathsConfig
from omniscan.core.schemas import (
    BBox,
    FinalArtifact,
    FinalLine,
    OcrLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.web.app import create_app

SERIES = "Solo Leveling"
CHAPTER = "Chapter 1"
BASE = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}"


def bbox(x0: int, y0: int, x1: int, y1: int) -> dict[str, int]:
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


@pytest.fixture
def work(tmp_path: Path) -> Path:
    (tmp_path / "lib" / SERIES / CHAPTER).mkdir(parents=True)
    work = tmp_path / "work" / SERIES / CHAPTER
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        work / "slices.json"
    )
    regions = [
        Region(
            id=rid,
            slice_index=0,
            kind="bubble_text",
            bbox=BBox(**box),
            reading_order=order,
            lines=[OcrLine(bbox=BBox(**box), text=text, score=0.9, engine="ocr")],
            text=text,
            confidence=0.9,
        )
        for rid, box, text, order in (
            ("r0001", bbox(10, 10, 110, 60), "안녕", 0),
            ("r0002", bbox(10, 200, 110, 260), "반가워", 1),
        )
    ]
    RegionsArtifact(regions=regions).save(work / "ocr.json")
    FinalArtifact(judge_model="judge", lines=[FinalLine(region_id="r0001", text="Hi", decision="pick")]).save(
        work / "final.json"
    )
    return work


@pytest.fixture
def client(tmp_path: Path, work: Path) -> TestClient:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    return TestClient(create_app(cfg))


def ocr_ids(work: Path) -> list[str]:
    return [r.id for r in RegionsArtifact.load(work / "ocr.json").regions]


def test_patch_region_fixes_the_text_and_returns_the_region(client: TestClient, work: Path) -> None:
    response = client.patch(f"{BASE}/regions/r0002", json={"text": "반가워요", "kind": "free_text"})
    assert response.status_code == 200
    body = response.json()
    assert (body["id"], body["text"], body["kind"], body["confidence"]) == (
        "r0002",
        "반가워요",
        "free_text",
        1.0,
    )
    assert RegionsArtifact.load(work / "ocr.json").regions[1].text == "반가워요"
    edits = client.get(f"{BASE}/edits").json()
    assert [e["region_id"] for e in edits["regions"]] == ["r0002"]
    assert edits["translations"] == [] and edits["deleted_regions"] == []


def test_add_region_uses_the_series_language_and_returns_201(client: TestClient, work: Path) -> None:
    library = work.parents[2] / "lib" / SERIES
    (library / "series.toml").write_text('[ocr]\nlang = "ja"\n', encoding="utf-8")
    response = client.post(f"{BASE}/regions", json={"bbox": bbox(10, 100, 110, 150), "text": "ドン"})
    assert response.status_code == 201
    body = response.json()
    assert (body["id"], body["lang"], body["text"], body["reading_order"]) == ("m0001", "ja", "ドン", 1)
    assert ocr_ids(work) == ["r0001", "r0002", "m0001"]


def test_delete_lists_the_region_as_deleted_and_revert_restores_it(client: TestClient, work: Path) -> None:
    assert client.delete(f"{BASE}/regions/r0001").json() == {"deleted": "r0001"}
    assert ocr_ids(work) == ["r0002"]
    assert [r["id"] for r in client.get(f"{BASE}/edits").json()["deleted_regions"]] == ["r0001"]
    response = client.post(f"{BASE}/regions/r0001/revert", json={})
    assert response.status_code == 200 and response.json()["region"]["text"] == "안녕"
    assert ocr_ids(work) == ["r0001", "r0002"]


def test_translation_edit_and_revert(client: TestClient, work: Path) -> None:
    response = client.put(f"{BASE}/final/r0002", json={"text": "Nice to see you"})
    assert response.status_code == 200 and response.json()["decision"] == "manual"
    lines = FinalArtifact.load(work / "final.json").lines
    assert [(x.region_id, x.text) for x in lines] == [("r0001", "Hi"), ("r0002", "Nice to see you")]
    response = client.post(f"{BASE}/final/r0002/revert", json={})
    assert response.json() == {"line": None}  # the judge never translated r0002
    assert [x.region_id for x in FinalArtifact.load(work / "final.json").lines] == ["r0001"]


def test_errors_map_to_404_422_and_415(client: TestClient, work: Path) -> None:
    before = (work / "ocr.json").read_bytes()
    response = client.patch(f"{BASE}/regions/r0009", json={"text": "x"})
    assert response.status_code == 404
    assert response.json() == {"detail": "region 'r0009' not found in ocr.json"}
    response = client.patch(f"{BASE}/regions/r0001", json={"bbox": bbox(300, 10, 400, 60)})
    assert response.status_code == 422 and "empty inside the strip" in response.json()["detail"]
    assert client.patch(f"{BASE}/regions/r0001", json={"kind": "speech"}).status_code == 422
    assert client.post(f"{BASE}/regions", json={"bbox": bbox(5, 5, 1, 1)}).status_code == 422
    response = client.patch(
        f"{BASE}/regions/r0001", content='{"text": "x"}', headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 415
    response = client.post(
        f"{BASE}/regions/r0001/revert", content="{}", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 415
    assert client.post(f"{BASE}/regions/r0001/revert", json={}).status_code == 404  # nothing to revert
    assert (work / "ocr.json").read_bytes() == before


def test_routes_cannot_escape_the_roots(client: TestClient) -> None:
    for url in (
        f"/api/series/..%2F..%2Fsecret/chapters/{quote(CHAPTER)}/regions/r0001",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2Fsecret/regions/r0001",
    ):
        assert client.patch(url, json={"text": "x"}).status_code == 404, url
        assert client.delete(url).status_code == 404, url


def test_cors_allows_the_editing_methods(client: TestClient) -> None:
    response = client.options(
        f"{BASE}/regions/r0001",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE"},
    )
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_web_app_import_is_torch_free() -> None:
    """`omniscan serve` (the editor's server) starts without loading torch; only the queue worker needs it."""
    code = "import sys, omniscan.web.app; raise SystemExit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_edits_report_which_regions_are_edited(client: TestClient) -> None:
    client.patch(f"{BASE}/regions/r0002", json={"text": "반가워요"})
    client.put(f"{BASE}/final/r0001", json={"text": "Hello"})
    added = client.post(f"{BASE}/regions", json={"bbox": bbox(10, 400, 110, 450)}).json()
    edits = client.get(f"{BASE}/edits").json()
    assert edits["edited_region_ids"] == ["r0002", added["id"]]
    assert edits["manual_translation_ids"] == ["r0001"]


def test_run_can_start_at_a_later_stage(client: TestClient) -> None:
    response = client.post(f"{BASE}/run", json={"start": "inpaint", "through": "export"})
    assert response.status_code == 202
    assert response.json()["stages"] == ["inpaint", "inpaint_lama", "typeset", "export"]
    assert client.post(f"{BASE}/run", json={"start": "export", "through": "ocr"}).status_code == 422
    assert client.post(f"{BASE}/run", json={"start": "paint", "through": "ocr"}).status_code == 422
