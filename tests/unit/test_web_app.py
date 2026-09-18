"""Tests for the web debug API (src/omniscan/web/app.py)."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import quote

from fastapi.testclient import TestClient
from PIL import Image

from omniscan.core.config import Config, PathsConfig
from omniscan.core.schemas import (
    Band,
    BBox,
    Candidate,
    CandidateRun,
    FinalArtifact,
    FinalLine,
    GlossaryEntry,
    IngestArtifact,
    OcrLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.glossary.store import GlossaryStore
from omniscan.web.app import create_app

SERIES = "Solo Leveling"
CHAPTER = "Chapter 1"


def make_client(tmp_path: Path) -> TestClient:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    return TestClient(create_app(cfg))


def make_jpeg(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (100, 150), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def write_chapter(tmp_path: Path) -> dict[str, bytes]:
    """One series/chapter with two raw JPEGs, a matching ingest.json and a slices.json."""
    raw_dir = tmp_path / "lib" / SERIES / CHAPTER
    work_dir = tmp_path / "work" / SERIES / CHAPTER
    raw_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    pages = {f"{i + 1:04d}.jpg": make_jpeg((40 * (i + 1), 10, 10)) for i in range(2)}
    for name, data in pages.items():
        (raw_dir / name).write_bytes(data)
    ingest = IngestArtifact(
        series=SERIES,
        chapter=CHAPTER,
        strip_width=100,
        strip_height=300,
        files=[
            SourceFile(index=0, name="0001.jpg", sha256="00" * 32, width=100, height=150, y0=0, y1=150),
            SourceFile(index=1, name="0002.jpg", sha256="11" * 32, width=100, height=150, y0=150, y1=300),
        ],
    )
    ingest.save(work_dir / "ingest.json")
    SlicesArtifact(
        strip_width=100,
        strip_height=300,
        bands=[Band(y0=70, y1=90, color=(255, 255, 255))],
        slices=[
            Slice(index=0, y0=0, y1=80, source_files=[0]),
            Slice(index=1, y0=80, y1=300, forced_cut=True, source_files=[0, 1]),
        ],
    ).save(work_dir / "slices.json")
    return pages


def test_series_empty_when_library_root_missing(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/api/series").status_code == 200
    assert client.get("/api/series").json() == []


def test_series_skips_underscore_directories(tmp_path: Path) -> None:
    for name in ("Solo Leveling", "_reference_en"):
        (tmp_path / "lib" / name).mkdir(parents=True)
    client = make_client(tmp_path)
    assert client.get("/api/series").json() == ["Solo Leveling"]


def test_chapters_in_reading_order(tmp_path: Path) -> None:
    series_dir = tmp_path / "lib" / SERIES
    for name in ("Chapter 2", "Chapter 10", "Chapter 1"):
        (series_dir / name).mkdir(parents=True)
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/chapters")
    assert response.status_code == 200
    assert response.json() == ["Chapter 1", "Chapter 2", "Chapter 10"]


def test_chapters_unknown_series_returns_empty(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/series/No%20Such%20Series/chapters")
    assert response.status_code == 200
    assert response.json() == []


def test_ingest_serves_exact_file_bytes(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/ingest")
    assert response.status_code == 200
    on_disk = (tmp_path / "work" / SERIES / CHAPTER / "ingest.json").read_bytes()
    assert response.content == on_disk


def test_ingest_missing_returns_404(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    (tmp_path / "work" / SERIES / CHAPTER / "ingest.json").unlink()
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/ingest")
    assert response.status_code == 404
    assert response.json() == {"detail": "ingest.json not found"}


def test_slices_present_and_absent(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/slices"
    response = client.get(url)
    assert response.status_code == 200
    on_disk = (tmp_path / "work" / SERIES / CHAPTER / "slices.json").read_bytes()
    assert response.content == on_disk
    (tmp_path / "work" / SERIES / CHAPTER / "slices.json").unlink()
    response = client.get(url)
    assert response.status_code == 404
    assert response.json() == {"detail": "slices.json not found"}


def test_page_serves_raw_bytes_with_content_type(tmp_path: Path) -> None:
    pages = write_chapter(tmp_path)
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/pages/1")
    assert response.status_code == 200
    assert response.content == pages["0002.jpg"]
    assert response.headers["content-type"].startswith("image/jpeg")


def test_page_out_of_range_or_missing_ingest_returns_404(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    client = make_client(tmp_path)
    base = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/pages"
    assert client.get(f"{base}/2").status_code == 404
    assert client.get(f"{base}/-1").status_code == 404
    (tmp_path / "work" / SERIES / CHAPTER / "ingest.json").unlink()
    assert client.get(f"{base}/0").status_code == 404


def test_traversal_cannot_escape_roots(tmp_path: Path) -> None:
    """`..` / absolute segments in series/chapter must yield a clean 404, never outside files."""
    write_chapter(tmp_path)
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"outside the roots")
    client = make_client(tmp_path)
    escapes = (
        "/api/series/..%2Fsecret/chapters/ingest/pages/0",
        f"/api/series/..%2F..%2Fsecret/chapters/{quote(CHAPTER)}/pages/0",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2F..%2Fsecret.json/ingest",
        f"/api/series/%2Fetc%2Fpasswd/chapters/{quote(CHAPTER)}/ingest",
        f"/api/series/{quote(SERIES)}/chapters/%2Fetc%2Fpasswd/slices",
        "/api/series/../../secret/chapters",
    )
    for url in escapes:
        response = client.get(url)
        assert response.status_code == 404, url
        assert b"outside the roots" not in response.content, url


def _ocr_artifact() -> RegionsArtifact:
    return RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=10, y0=20, x1=90, y1=60),
                reading_order=1,
                lang="ko",
                lines=[
                    OcrLine(
                        bbox=BBox(x0=10, y0=20, x1=90, y1=40),
                        text="첫 줄",
                        score=0.97,
                        engine="korean_PP-OCRv5_mobile_rec",
                    )
                ],
                text="첫 줄\n둘째 줄",
                confidence=0.93,
                ocr_alt="첫 줄\n둘째 즁",
            ),
            Region(
                id="r0002",
                slice_index=0,
                kind="sfx",
                bbox=BBox(x0=5, y0=100, x1=50, y1=140),
                reading_order=2,
                text="우웅",
                confidence=0.41,
            ),
        ]
    )


def test_ocr_serves_exact_file_bytes(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    _ocr_artifact().save(tmp_path / "work" / SERIES / CHAPTER / "ocr.json")
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/ocr")
    assert response.status_code == 200
    on_disk = (tmp_path / "work" / SERIES / CHAPTER / "ocr.json").read_bytes()
    assert response.content == on_disk


def test_ocr_missing_returns_404(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/ocr")
    assert response.status_code == 404
    assert response.json() == {"detail": "ocr.json not found"}


def test_ocr_traversal_cannot_escape_roots(tmp_path: Path) -> None:
    """`..` / absolute segments on the ocr route must yield a clean 404, never outside files."""
    write_chapter(tmp_path)
    secret = tmp_path / "secret.json"
    secret.write_bytes(b"outside the roots")
    client = make_client(tmp_path)
    escapes = (
        f"/api/series/..%2F..%2Fsecret/chapters/{quote(CHAPTER)}/ocr",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2F..%2Fsecret.json/ocr",
        f"/api/series/%2Fetc%2Fpasswd/chapters/{quote(CHAPTER)}/ocr",
        f"/api/series/{quote(SERIES)}/chapters/%2Fetc%2Fpasswd/ocr",
    )
    for url in escapes:
        response = client.get(url)
        assert response.status_code == 404, url
        assert b"outside the roots" not in response.content, url


# ---------------------------------------------------------------- translation view (B10)


def write_translation_chapter(tmp_path: Path) -> Path:
    """One chapter with ocr.json (two regions), translations/runA.json + runB.json and no final.json."""
    work = tmp_path / "work" / SERIES / CHAPTER
    work.mkdir(parents=True, exist_ok=True)
    RegionsArtifact(
        regions=[
            Region(
                id="r0001",
                slice_index=0,
                kind="bubble_text",
                bbox=BBox(x0=10, y0=20, x1=90, y1=60),
                reading_order=1,
                text="철수가 왔다",
                confidence=0.9,
            ),
            Region(
                id="r0002",
                slice_index=0,
                kind="sfx",
                bbox=BBox(x0=5, y0=100, x1=50, y1=140),
                reading_order=2,
                text="아무것도 없다",
                confidence=0.9,
            ),
        ]
    ).save(work / "ocr.json")
    for run_id, texts in (
        ("runA", {"r0001": "Cheolsu came", "r0002": "Whoosh"}),
        ("runB", {"r0001": "Cheolsu has come"}),
    ):
        CandidateRun(
            run_id=run_id,
            profile="fast",
            model="qwen3:8b",
            candidates=[Candidate(region_id=rid, text=text) for rid, text in texts.items()],
        ).save(work / "translations" / f"{run_id}.json")
    return work


def write_final(work: Path, lines: list[FinalLine]) -> None:
    FinalArtifact(judge_model="judge:8b", lines=lines).save(work / "final.json")


def add_glossary(work_root: Path) -> tuple[GlossaryEntry, GlossaryEntry]:
    """A locked `철수 -> Cheolsu` entry and a rejected one that would match r0002."""
    with GlossaryStore(work_root / "series.db") as store:
        locked = store.add(GlossaryEntry(source="철수", target="Cheolsu", status="locked"))
        rejected = store.add(GlossaryEntry(source="아무것도", target="Nothing", status="rejected"))
    return locked, rejected


def test_translations_lists_run_ids_in_natural_order(tmp_path: Path) -> None:
    work = write_translation_chapter(tmp_path)
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/translations"
    assert client.get(url).json() == ["runA", "runB"]
    (work / "translations" / "run2.json").write_bytes(b"{}")
    (work / "translations" / "run10.json").write_bytes(b"{}")
    (work / "translations" / ".hidden.json").write_bytes(b"{}")
    assert client.get(url).json() == ["run2", "run10", "runA", "runB"]


def test_translations_empty_when_directory_missing(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/translations"
    assert client.get(url).status_code == 200
    assert client.get(url).json() == []


def test_translation_run_serves_exact_bytes(tmp_path: Path) -> None:
    work = write_translation_chapter(tmp_path)
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/translations/runB"
    response = client.get(url)
    assert response.status_code == 200
    assert response.content == (work / "translations" / "runB.json").read_bytes()


def test_translation_run_missing_or_escapes_returns_404(tmp_path: Path) -> None:
    write_translation_chapter(tmp_path)
    secret = tmp_path / "secret.json"
    secret.write_bytes(b"outside the roots")
    client = make_client(tmp_path)
    base = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/translations"
    for run_id in ("nope", "..", "..%2F..%2Fsecret", "..%2Fsecret"):
        response = client.get(f"{base}/{run_id}")
        assert response.status_code == 404, run_id
        assert b"outside the roots" not in response.content, run_id


def test_final_serves_exact_bytes_and_404_when_absent(tmp_path: Path) -> None:
    work = write_translation_chapter(tmp_path)
    write_final(work, [FinalLine(region_id="r0001", text="Cheolsu came", decision="pick")])
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/final"
    response = client.get(url)
    assert response.status_code == 200
    assert response.content == (work / "final.json").read_bytes()
    (work / "final.json").unlink()
    assert client.get(url).status_code == 404
    assert client.get(url).json() == {"detail": "final.json not found"}


def test_glossary_returns_all_entries(tmp_path: Path) -> None:
    write_translation_chapter(tmp_path)
    work_root = tmp_path / "work" / SERIES
    locked, rejected = add_glossary(work_root)
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/glossary")
    assert response.status_code == 200
    entries = response.json()
    assert [e["id"] for e in entries] == [locked.id, rejected.id]
    assert [e["status"] for e in entries] == ["locked", "rejected"]
    assert entries[0]["source"] == "철수"
    assert entries[0]["target"] == "Cheolsu"


def test_glossary_empty_without_db_and_file_not_created(tmp_path: Path) -> None:
    write_translation_chapter(tmp_path)
    client = make_client(tmp_path)
    response = client.get(f"/api/series/{quote(SERIES)}/glossary")
    assert response.status_code == 200
    assert response.json() == []
    assert not (tmp_path / "work" / SERIES / "series.db").exists()


def test_glossary_hits_with_locked_target_in_final(tmp_path: Path) -> None:
    work = write_translation_chapter(tmp_path)
    add_glossary(work.parent)
    write_final(work, [FinalLine(region_id="r0001", text="Cheolsu came", decision="pick")])
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/glossary-hits"
    response = client.get(url)
    assert response.status_code == 200
    regions = response.json()["regions"]
    assert regions["r0001"] == [
        {
            "entry_id": 1,
            "source": "철수",
            "target": "Cheolsu",
            "status": "locked",
            "start": 0,
            "end": 3,
            "particle": "가",
            "target_in_final": True,
        }
    ]
    assert regions["r0002"] == []


def test_glossary_hits_target_missing_from_final(tmp_path: Path) -> None:
    work = write_translation_chapter(tmp_path)
    add_glossary(work.parent)
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/glossary-hits"

    # A final line whose text lacks the locked target -> false.
    write_final(work, [FinalLine(region_id="r0001", text="Chul-soo came", decision="rewrite")])
    assert client.get(url).json()["regions"]["r0001"][0]["target_in_final"] is False

    # No final.json at all -> null.
    (work / "final.json").unlink()
    assert client.get(url).json()["regions"]["r0001"][0]["target_in_final"] is None

    # A final.json without a line for r0001 -> null.
    write_final(work, [FinalLine(region_id="r0002", text="nothing", decision="pick")])
    assert client.get(url).json()["regions"]["r0001"][0]["target_in_final"] is None


def test_glossary_hits_invalid_final_treated_as_absent(tmp_path: Path) -> None:
    work = write_translation_chapter(tmp_path)
    add_glossary(work.parent)
    (work / "final.json").write_bytes(b"{not json")
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/glossary-hits"
    response = client.get(url)
    assert response.status_code == 200
    assert response.json()["regions"]["r0001"][0]["target_in_final"] is None


def test_glossary_hits_rejected_entry_and_missing_db(tmp_path: Path) -> None:
    work = write_translation_chapter(tmp_path)
    add_glossary(work.parent)  # the rejected `아무것도` entry must not hit r0002
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/glossary-hits"
    assert client.get(url).json()["regions"]["r0002"] == []

    # No series.db: every region maps to [] and no db file is created.
    (work.parent / "series.db").unlink()
    assert client.get(url).json()["regions"] == {"r0001": [], "r0002": []}
    assert not (work.parent / "series.db").exists()


def test_glossary_hits_missing_ocr_returns_404(tmp_path: Path) -> None:
    write_translation_chapter(tmp_path)
    (tmp_path / "work" / SERIES / CHAPTER / "ocr.json").unlink()
    client = make_client(tmp_path)
    url = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/glossary-hits"
    assert client.get(url).status_code == 404
    assert client.get(url).json() == {"detail": "ocr.json not found"}


def test_new_routes_traversal_cannot_escape_roots(tmp_path: Path) -> None:
    """`..` / absolute segments in series/chapter on the new routes must yield a clean 404."""
    write_translation_chapter(tmp_path)
    secret = tmp_path / "secret.json"
    secret.write_bytes(b"outside the roots")
    client = make_client(tmp_path)
    escapes = (
        f"/api/series/..%2F..%2Fsecret/chapters/{quote(CHAPTER)}/translations",
        f"/api/series/..%2F..%2Fsecret/chapters/{quote(CHAPTER)}/final",
        f"/api/series/..%2F..%2Fsecret/chapters/{quote(CHAPTER)}/glossary-hits",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2Fsecret/translations/runA",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2Fsecret/final",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2Fsecret/glossary-hits",
        "/api/series/..%2F..%2Fsecret/glossary",
    )
    for url in escapes:
        response = client.get(url)
        assert response.status_code == 404, url
        assert b"outside the roots" not in response.content, url


# ---------------------------------------------------------------- reader view (B12)


def make_png(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (80, 120), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_output(tmp_path: Path, names: dict[str, bytes]) -> Path:
    """Place finished output images into the chapter's output dir."""
    out_dir = tmp_path / "out" / SERIES / CHAPTER
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in names.items():
        (out_dir / name).write_bytes(data)
    return out_dir


def output_url(suffix: str = "") -> str:
    return f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/output{suffix}"


def test_output_lists_names_in_natural_order(tmp_path: Path) -> None:
    pages = {
        "1.jpg": make_jpeg((10, 10, 10)),
        "2.jpg": make_jpeg((20, 10, 10)),
        "10.jpg": make_jpeg((30, 10, 10)),
    }
    write_output(tmp_path, pages)
    (tmp_path / "out" / SERIES / CHAPTER / "notes.txt").write_bytes(b"not an image")
    (tmp_path / "out" / SERIES / CHAPTER / "extra").mkdir()
    (tmp_path / "out" / SERIES / CHAPTER / "extra" / "3.jpg").write_bytes(make_jpeg((40, 10, 10)))
    client = make_client(tmp_path)
    response = client.get(output_url())
    assert response.status_code == 200
    assert response.json() == ["1.jpg", "2.jpg", "10.jpg"]


def test_output_empty_when_directory_absent(tmp_path: Path) -> None:
    write_chapter(tmp_path)
    client = make_client(tmp_path)
    response = client.get(output_url())
    assert response.status_code == 200
    assert response.json() == []


def test_output_traversal_cannot_escape_roots(tmp_path: Path) -> None:
    """`..` / absolute segments in series/chapter on the output routes must yield a clean 404."""
    write_chapter(tmp_path)
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(make_jpeg((10, 10, 10)))
    client = make_client(tmp_path)
    escapes = (
        "/api/series/..%2F..%2Fsecret/chapters/x/output",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2Fsecret/output",
        "/api/series/..%2F..%2Fsecret/chapters/x/output/1.jpg",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2Fsecret/output/1.jpg",
        f"/api/series/{quote(SERIES)}/chapters/%2E%2E/output",
    )
    for url in escapes:
        response = client.get(url)
        assert response.status_code == 404, url
        assert b"secret" not in response.content, url


def test_output_image_serves_exact_bytes_with_content_type(tmp_path: Path) -> None:
    pages = {
        "1.jpg": make_jpeg((10, 10, 10)),
        "0002.png": make_png((20, 20, 20)),
    }
    write_output(tmp_path, pages)
    client = make_client(tmp_path)
    response = client.get(output_url("/1.jpg"))
    assert response.status_code == 200
    assert response.content == pages["1.jpg"]
    assert response.headers["content-type"].startswith("image/jpeg")
    response = client.get(output_url("/0002.png"))
    assert response.status_code == 200
    assert response.content == pages["0002.png"]
    assert response.headers["content-type"].startswith("image/png")


def test_output_image_404_on_bad_names(tmp_path: Path) -> None:
    write_output(tmp_path, {"1.jpg": make_jpeg((10, 10, 10)), "notes.txt": b"text"})
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(make_jpeg((30, 10, 10)))
    out_dir = tmp_path / "out" / SERIES / CHAPTER
    (out_dir / "escape.jpg").symlink_to(secret)
    client = make_client(tmp_path)
    missing = (
        "99.jpg",
        "notes.txt",
        "..%2F..%2Fsecret.jpg",
        "..%2Fsecret.jpg",
        "a%5Cb.jpg",
        "escape.jpg",
    )
    for name in missing:
        response = client.get(output_url(f"/{name}"))
        assert response.status_code == 404, name
        assert b"secret" not in response.content, name
