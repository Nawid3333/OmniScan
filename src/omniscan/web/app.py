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
from omniscan.core.schemas import IngestArtifact


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
