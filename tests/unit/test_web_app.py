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
    IngestArtifact,
    OcrLine,
    Region,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
    SourceFile,
)
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
