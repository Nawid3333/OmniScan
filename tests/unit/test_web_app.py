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
    FilterArtifact,
    FilterDecision,
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


def test_page_resolves_by_source_file_index_with_gaps(tmp_path: Path) -> None:
    """ingest.files may have gaps (promo-filtered raws keep their original index): pages/N must
    resolve by `SourceFile.index`, never by list position (regression for card F2a)."""
    write_chapter(tmp_path)
    work = tmp_path / "work" / SERIES / CHAPTER
    ingest = IngestArtifact.load(work / "ingest.json")
    gapped = IngestArtifact(
        series=ingest.series,
        chapter=ingest.chapter,
        strip_width=ingest.strip_width,
        strip_height=ingest.strip_height,
        files=[f.model_copy(update={"index": f.index * 2}) for f in ingest.files],  # indices 0 and 2
    )
    gapped.save(work / "ingest.json")
    client = make_client(tmp_path)
    base = f"/api/series/{quote(SERIES)}/chapters/{quote(CHAPTER)}/pages"
    assert client.get(f"{base}/0").status_code == 200
    assert client.get(f"{base}/2").status_code == 200
    assert client.get(f"{base}/1").status_code == 404  # the gap 404s, it does not return the next page


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
    missing = ["99.jpg", "notes.txt", "..%2F..%2Fsecret.jpg", "..%2Fsecret.jpg", "a%5Cb.jpg"]
    try:
        (out_dir / "escape.jpg").symlink_to(secret)
    except OSError:  # Windows without Developer Mode / admin cannot create symlinks; skip only that case
        pass
    else:
        missing.append("escape.jpg")
    client = make_client(tmp_path)
    for name in missing:
        response = client.get(output_url(f"/{name}"))
        assert response.status_code == 404, name
        assert b"secret" not in response.content, name


# ---------------------------------------------------------------- filtered view (B30)


def write_filter_fixtures(tmp_path: Path) -> Path:
    """Chapters 1-3: Ch1 with a mixed filter.json, Ch2 keep-only, Ch3 without filter.json.

    Every chapter gets an ingest.json (SourceFile indices 0..5) and a slices.json (slices 7 and 9)."""
    work_root = tmp_path / "work" / SERIES
    for chapter in ("Chapter 1", "Chapter 2", "Chapter 3"):
        (tmp_path / "lib" / SERIES / chapter).mkdir(parents=True, exist_ok=True)
        work = work_root / chapter
        work.mkdir(parents=True, exist_ok=True)
        IngestArtifact(
            series=SERIES,
            chapter=chapter,
            strip_width=100,
            strip_height=30000,
            files=[
                SourceFile(
                    index=i,
                    name=f"{i + 1:04d}.jpg",
                    sha256=f"{i:02x}" * 32,
                    width=100,
                    height=150,
                    y0=150 * i,
                    y1=150 * (i + 1),
                )
                for i in range(6)
            ],
        ).save(work / "ingest.json")
        SlicesArtifact(
            strip_width=100,
            strip_height=30000,
            bands=[Band(y0=70, y1=90, color=(255, 255, 255))],
            slices=[Slice(index=7, y0=20100, y1=23000), Slice(index=9, y0=100, y1=200)],
        ).save(work / "slices.json")
    FilterArtifact(
        decisions=[
            FilterDecision(
                target="file", index=3, decision="filtered", score=0.97, matched_example="global/end_card.jpg"
            ),
            FilterDecision(target="file", index=5, decision="keep", score=0.10),
            FilterDecision(
                target="slice", index=7, decision="filtered", score=0.93, matched_example="global/ad.jpg"
            ),
            FilterDecision(
                target="slice", index=9, decision="filtered", score=0.88, matched_example="global/ad.jpg"
            ),
            FilterDecision(target="slice", index=9, decision="restored", score=1.0, method="manual"),
        ]
    ).save(work_root / CHAPTER / "filter.json")
    FilterArtifact(decisions=[FilterDecision(target="file", index=0, decision="keep", score=0.0)]).save(
        work_root / "Chapter 2" / "filter.json"
    )
    return work_root


def filtered_url(series: str = SERIES) -> str:
    return f"/api/series/{quote(series)}/filtered"


def restore_url(series: str = SERIES, chapter: str = CHAPTER) -> str:
    return f"/api/series/{quote(series)}/chapters/{quote(chapter)}/filter/restore"


def test_filtered_lists_items_in_documented_shape_and_order(tmp_path: Path) -> None:
    write_filter_fixtures(tmp_path)
    client = make_client(tmp_path)
    response = client.get(filtered_url())
    assert response.status_code == 200
    # keep-only pairs are excluded; items are file-before-slice then by index; a restored pair keeps
    # the score of its last "filtered" decision.
    assert response.json() == [
        {
            "chapter": CHAPTER,
            "items": [
                {
                    "target": "file",
                    "index": 3,
                    "state": "filtered",
                    "score": 0.97,
                    "matched_example": "global/end_card.jpg",
                    "method": "phash",
                    "name": "0004.jpg",
                    "y0": None,
                    "y1": None,
                },
                {
                    "target": "slice",
                    "index": 7,
                    "state": "filtered",
                    "score": 0.93,
                    "matched_example": "global/ad.jpg",
                    "method": "phash",
                    "name": None,
                    "y0": 20100,
                    "y1": 23000,
                },
                {
                    "target": "slice",
                    "index": 9,
                    "state": "restored",
                    "score": 0.88,
                    "matched_example": "global/ad.jpg",
                    "method": "phash",
                    "name": None,
                    "y0": 100,
                    "y1": 200,
                },
            ],
        }
    ]

    # ingest.json missing -> name: null, everything else unchanged.
    (tmp_path / "work" / SERIES / CHAPTER / "ingest.json").unlink()
    response = client.get(filtered_url())
    assert response.json()[0]["items"][0]["name"] is None

    # Unknown series -> []; a `..` series -> clean 404.
    assert client.get(filtered_url("No Such Series")).json() == []
    assert client.get("/api/series/..%2F..%2Fsecret/filtered").status_code == 404


def test_filtered_invalid_filter_json_skips_chapter(tmp_path: Path) -> None:
    write_filter_fixtures(tmp_path)
    chapter_2 = tmp_path / "work" / SERIES / "Chapter 2"
    FilterArtifact(
        decisions=[
            FilterDecision(target="file", index=0, decision="filtered", score=0.99, matched_example="x.jpg")
        ]
    ).save(chapter_2 / "filter.json")
    (chapter_2 / "filter.json").write_bytes(b"{not json")
    client = make_client(tmp_path)
    response = client.get(filtered_url())
    assert response.status_code == 200
    assert [c["chapter"] for c in response.json()] == [CHAPTER]


def test_restore_appends_one_manual_decision(tmp_path: Path) -> None:
    work = write_filter_fixtures(tmp_path)
    filtered_dir = tmp_path / "out" / SERIES / "_filtered" / CHAPTER
    filtered_dir.mkdir(parents=True)
    (filtered_dir / "0004.jpg").write_bytes(b"promo bytes")
    filter_path = work / CHAPTER / "filter.json"
    before_filter = filter_path.read_bytes()
    before_image = (filtered_dir / "0004.jpg").read_bytes()
    client = make_client(tmp_path)
    response = client.post(restore_url(), json={"target": "file", "index": 3})
    assert response.status_code == 200
    assert response.json() == {
        "target": "file",
        "index": 3,
        "decision": "restored",
        "score": 1.0,
        "matched_example": None,
        "method": "manual",
    }
    artifact = FilterArtifact.load(filter_path)
    assert len(artifact.decisions) == 5 + 1
    assert artifact.decisions[-1].decision == "restored"
    assert artifact.decisions[-1].method == "manual"
    # restore is a metadata override only: _filtered/ files are untouched.
    assert (filtered_dir / "0004.jpg").read_bytes() == before_image
    # A following filtered listing shows the item as restored.
    items = client.get(filtered_url()).json()[0]["items"]
    assert next(i for i in items if i["target"] == "file")["state"] == "restored"
    assert before_filter != filter_path.read_bytes()


def test_restore_rejects_non_filtered_items_missing_artifact_and_bad_bodies(tmp_path: Path) -> None:
    work = write_filter_fixtures(tmp_path)
    client = make_client(tmp_path)
    url = restore_url()
    filter_path = work / CHAPTER / "filter.json"
    before = filter_path.read_bytes()

    # never filtered (keep-only pair) -> 409, nothing written
    response = client.post(url, json={"target": "file", "index": 5})
    assert response.status_code == 409
    assert response.json() == {"detail": "item is not currently filtered"}
    # already restored -> 409
    response = client.post(url, json={"target": "slice", "index": 9})
    assert response.status_code == 409
    # never mentioned at all -> 409
    response = client.post(url, json={"target": "file", "index": 42})
    assert response.status_code == 409
    assert filter_path.read_bytes() == before

    # filter.json missing -> 404
    response = client.post(restore_url(chapter="Chapter 3"), json={"target": "file", "index": 0})
    assert response.status_code == 404
    assert response.json() == {"detail": "filter.json not found"}

    # bad bodies -> 422 (invalid target, negative index, extra field)
    for body in (
        {"target": "page", "index": 0},
        {"target": "file", "index": -1},
        {"target": "file", "index": 0, "x": 1},
    ):
        response = client.post(url, json=body)
        assert response.status_code == 422, body
        assert filter_path.read_bytes() == before


def test_restore_rejects_non_json_content_types(tmp_path: Path) -> None:
    """The CSRF guard: a cross-site page can send text/plain POSTs without a preflight; only JSON passes."""
    work = write_filter_fixtures(tmp_path)
    filter_path = work / CHAPTER / "filter.json"
    before = filter_path.read_bytes()
    client = make_client(tmp_path)
    body = '{"target": "file", "index": 3}'
    for headers in (
        {"Content-Type": "text/plain"},
        {"Content-Type": "application/x-www-form-urlencoded"},
        {},
    ):
        response = client.post(restore_url(), content=body, headers=headers)
        assert response.status_code == 415, headers
        assert filter_path.read_bytes() == before, headers
    response = client.post(
        restore_url(), content=body, headers={"Content-Type": "application/json; charset=utf-8"}
    )
    assert response.status_code == 200


def test_restore_cors_preflight_allows_post_only_from_allowed_origins(tmp_path: Path) -> None:
    write_filter_fixtures(tmp_path)
    client = make_client(tmp_path)
    preflight = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options(restore_url(), headers=preflight)
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response.headers["access-control-allow-methods"]
    response = client.options(restore_url(), headers={**preflight, "Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_restore_traversal_cannot_escape_roots(tmp_path: Path) -> None:
    """`..` / absolute segments in series/chapter on the restore route must yield a clean 404."""
    write_filter_fixtures(tmp_path)
    client = make_client(tmp_path)
    escapes = (
        "/api/series/..%2F..%2Fsecret/chapters/x/filter/restore",
        f"/api/series/..%2F..%2Fsecret/chapters/{quote(CHAPTER)}/filter/restore",
        f"/api/series/{quote(SERIES)}/chapters/..%2F..%2Fsecret/filter/restore",
        f"/api/series/{quote(SERIES)}/chapters/%2Fetc%2Fpasswd/filter/restore",
        f"/api/series/%2Fetc%2Fpasswd/chapters/{quote(CHAPTER)}/filter/restore",
    )
    for url in escapes:
        response = client.post(url, json={"target": "file", "index": 0})
        assert response.status_code == 404, url
