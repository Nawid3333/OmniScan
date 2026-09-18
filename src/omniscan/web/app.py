"""FastAPI app serving existing pipeline artifacts and raw images (read-only, local)."""

from __future__ import annotations

import mimetypes
from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths, natural_key
from omniscan.core.schemas import FinalArtifact, GlossaryEntry, IngestArtifact, RegionsArtifact
from omniscan.glossary.match import find_terms, term_present
from omniscan.glossary.store import GlossaryStore


def _under(root: Path, candidate: Path) -> bool:
    """True if `candidate` resolves to a location inside `root` (blocks `..` / absolute escapes)."""
    return candidate.resolve().is_relative_to(root.resolve())


def create_app(cfg: Config, *, cors_origins: Sequence[str] = ("http://localhost:5173",)) -> FastAPI:
    """Build the debug API app: series/chapter browsing + ingest/slices artifacts + raw pages."""
    app = FastAPI(title="OmniScan debug API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def series_paths(series: str) -> SeriesPaths:
        """SeriesPaths for a validated series name (404 on any escape attempt)."""
        paths = SeriesPaths.from_config(cfg, series)
        if not _under(cfg.paths.library_root, paths.library_dir) or not _under(
            cfg.paths.work_root, paths.work_dir
        ):
            raise HTTPException(status_code=404, detail=f"unknown series {series!r}")
        return paths

    def chapter_paths(series: str, chapter: str) -> ChapterPaths:
        """ChapterPaths for a validated series/chapter pair (404 on any escape attempt)."""
        paths = series_paths(series).chapter(chapter)
        if not _under(cfg.paths.library_root, paths.raw_dir) or not _under(
            cfg.paths.work_root, paths.work_dir
        ):
            raise HTTPException(status_code=404, detail=f"unknown chapter {chapter!r} of series {series!r}")
        return paths

    def artifact_bytes(chapter: ChapterPaths, name: str) -> Response:
        """Serve a JSON artifact's exact file bytes (no model round-trip)."""
        path = chapter.artifact(name)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"{name} not found")
        return Response(content=path.read_bytes(), media_type="application/json")

    def glossary_entries(series: SeriesPaths) -> list[GlossaryEntry]:
        """All store entries in store order; [] when series.db does not exist (never creates the db)."""
        if not series.db.is_file():
            return []
        with GlossaryStore(series.db) as store:
            return store.list()

    def final_lines(chapter: ChapterPaths) -> dict[str, str]:
        """region_id -> final text; {} when final.json is missing or invalid (never a 500)."""
        path = chapter.artifact("final.json")
        if not path.is_file():
            return {}
        try:
            final = FinalArtifact.load(path)
        except Exception:
            return {}
        lines: dict[str, str] = {}
        for line in final.lines:
            lines.setdefault(line.region_id, line.text)
        return lines

    @app.get("/api/series")
    def list_series() -> list[str]:
        """Series names in the library (missing library root -> empty list)."""
        root = cfg.paths.library_root
        if not root.is_dir():
            return []
        names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]
        return sorted(names, key=natural_key)

    @app.get("/api/series/{series}/chapters")
    def list_chapters(series: str) -> list[str]:
        """Chapter folder names of a series in reading order (unknown series -> empty list)."""
        return series_paths(series).chapters()

    @app.get("/api/series/{series}/chapters/{chapter}/ingest")
    def get_ingest(series: str, chapter: str) -> Response:
        """The chapter's ingest.json, byte-for-byte as written by the ingest stage."""
        return artifact_bytes(chapter_paths(series, chapter), "ingest.json")

    @app.get("/api/series/{series}/chapters/{chapter}/slices")
    def get_slices(series: str, chapter: str) -> Response:
        """The chapter's slices.json, byte-for-byte as written by the slicer."""
        return artifact_bytes(chapter_paths(series, chapter), "slices.json")

    @app.get("/api/series/{series}/chapters/{chapter}/ocr")
    def get_ocr(series: str, chapter: str) -> Response:
        """The chapter's ocr.json, byte-for-byte as written by the OCR stage."""
        return artifact_bytes(chapter_paths(series, chapter), "ocr.json")

    @app.get("/api/series/{series}/chapters/{chapter}/translations")
    def list_translations(series: str, chapter: str) -> list[str]:
        """Run ids (file stems) of the chapter's translation runs in natural order."""
        directory = chapter_paths(series, chapter).work_dir / "translations"
        if not directory.is_dir():
            return []
        stems = [p.stem for p in directory.glob("*.json") if p.is_file() and not p.stem.startswith(".")]
        return sorted(stems, key=natural_key)

    @app.get("/api/series/{series}/chapters/{chapter}/translations/{run_id}")
    def get_translation(series: str, chapter: str, run_id: str) -> Response:
        """One translation run's JSON, byte-for-byte as written by the translation stage."""
        if "/" in run_id or "\\" in run_id or ".." in run_id:
            raise HTTPException(status_code=404, detail="translation run not found")
        directory = chapter_paths(series, chapter).work_dir / "translations"
        path = directory / f"{run_id}.json"
        if not _under(directory, path) or not path.is_file():
            raise HTTPException(status_code=404, detail="translation run not found")
        return Response(content=path.read_bytes(), media_type="application/json")

    @app.get("/api/series/{series}/chapters/{chapter}/final")
    def get_final(series: str, chapter: str) -> Response:
        """The chapter's final.json, byte-for-byte as written by the judge stage."""
        return artifact_bytes(chapter_paths(series, chapter), "final.json")

    @app.get("/api/series/{series}/glossary")
    def get_glossary(series: str) -> list[dict[str, object]]:
        """Every glossary entry of the series in store order ([] when series.db is missing)."""
        return [entry.model_dump(mode="json") for entry in glossary_entries(series_paths(series))]

    @app.get("/api/series/{series}/chapters/{chapter}/glossary-hits")
    def get_glossary_hits(series: str, chapter: str) -> dict[str, dict[str, list[dict[str, object]]]]:
        """Glossary hits per OCR region, with locked-term target checks against the final text."""
        paths = chapter_paths(series, chapter)
        ocr_path = paths.artifact("ocr.json")
        if not ocr_path.is_file():
            raise HTTPException(status_code=404, detail="ocr.json not found")
        ocr = RegionsArtifact.load(ocr_path)
        entries = [
            entry
            for entry in glossary_entries(series_paths(series))
            if entry.status in ("proposed", "locked")
        ]
        finals = final_lines(paths)
        hits: dict[str, list[dict[str, object]]] = {}
        for region in ocr.regions:
            region_hits: list[dict[str, object]] = []
            for match in find_terms(region.text, entries):
                entry = next(e for e in entries if e.id == match.entry_id)
                final_text = finals.get(region.id)
                region_hits.append(
                    {
                        "entry_id": match.entry_id,
                        "source": match.source,
                        "target": entry.target,
                        "status": entry.status,
                        "start": match.start,
                        "end": match.end,
                        "particle": match.particle,
                        "target_in_final": (
                            term_present(final_text, entry) if final_text is not None else None
                        ),
                    }
                )
            hits[region.id] = region_hits
        return {"regions": hits}

    @app.get("/api/series/{series}/chapters/{chapter}/pages/{index}")
    def get_page(series: str, chapter: str, index: int) -> FileResponse:
        """Raw image bytes of the SourceFile at `index` in the chapter's ingest.json."""
        paths = chapter_paths(series, chapter)
        ingest_path = paths.artifact("ingest.json")
        if not ingest_path.is_file():
            raise HTTPException(status_code=404, detail="ingest.json not found")
        ingest = IngestArtifact.load(ingest_path)
        if not 0 <= index < len(ingest.files):
            raise HTTPException(
                status_code=404,
                detail=f"page index {index} out of range (0..{len(ingest.files) - 1})",
            )
        source_file = ingest.files[index]
        image_path = paths.raw_dir / source_file.name
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"raw file {source_file.name} not found")
        media_type = mimetypes.guess_type(source_file.name)[0] or "application/octet-stream"
        return FileResponse(image_path, media_type=media_type)

    return app
