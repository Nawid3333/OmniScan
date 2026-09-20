"""FastAPI app serving existing pipeline artifacts and raw images (read-only except filter restore)."""

from __future__ import annotations

import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import Field, ValidationError

from omniscan.core.config import Config
from omniscan.core.paths import IMAGE_SUFFIXES, ChapterPaths, SeriesPaths, list_images, natural_key
from omniscan.core.schemas import (
    FilterArtifact,
    FilterDecision,
    FinalArtifact,
    GlossaryEntry,
    IngestArtifact,
    Model,
    RegionsArtifact,
    SlicesArtifact,
)
from omniscan.filter.decide import effective_decision, restore
from omniscan.glossary.match import find_terms, term_present
from omniscan.glossary.store import GlossaryStore


class RestoreBody(Model):
    """Body of the filter-restore POST request."""

    target: Literal["file", "slice"]
    index: int = Field(ge=0)


def _under(root: Path, candidate: Path) -> bool:
    """True if `candidate` resolves to a location inside `root` (blocks `..` / absolute escapes)."""
    return candidate.resolve().is_relative_to(root.resolve())


def create_app(cfg: Config, *, cors_origins: Sequence[str] = ("http://localhost:5173",)) -> FastAPI:
    """Build the debug API app: series/chapter browsing + ingest/slices artifacts + raw pages."""
    app = FastAPI(title="OmniScan debug API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_methods=["GET", "POST"],
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

    def output_paths(series: str, chapter: str) -> Path:
        """The chapter's output dir; 404 unless it sits inside cfg.paths.output_root."""
        paths = chapter_paths(series, chapter)
        # chapter_paths never checks the output root, so a chapter of ".." would resolve to the root
        # itself; that (and any escape out of the root) must 404.
        if ".." in chapter or not _under(cfg.paths.output_root, paths.output_dir):
            raise HTTPException(status_code=404, detail=f"unknown chapter {chapter!r} of series {series!r}")
        return paths.output_dir

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

    def source_names(chapter: ChapterPaths) -> dict[int, str]:
        """SourceFile index -> name from ingest.json; {} when missing or invalid (never a 500)."""
        path = chapter.artifact("ingest.json")
        if not path.is_file():
            return {}
        try:
            return {file.index: file.name for file in IngestArtifact.load(path).files}
        except Exception:
            return {}

    def slice_ranges(chapter: ChapterPaths) -> dict[int, tuple[int, int]]:
        """Slice index -> (y0, y1) from slices.json; {} when missing or invalid (never a 500)."""
        path = chapter.artifact("slices.json")
        if not path.is_file():
            return {}
        try:
            return {s.index: (s.y0, s.y1) for s in SlicesArtifact.load(path).slices}
        except Exception:
            return {}

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
        """Raw image bytes of the SourceFile whose index is `index` (kept files may have gaps)."""
        paths = chapter_paths(series, chapter)
        ingest_path = paths.artifact("ingest.json")
        if not ingest_path.is_file():
            raise HTTPException(status_code=404, detail="ingest.json not found")
        ingest = IngestArtifact.load(ingest_path)
        source_file = next((f for f in ingest.files if f.index == index), None)
        if source_file is None:
            raise HTTPException(status_code=404, detail=f"no page with index {index} in ingest.json")
        image_path = paths.raw_dir / source_file.name
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"raw file {source_file.name} not found")
        media_type = mimetypes.guess_type(source_file.name)[0] or "application/octet-stream"
        return FileResponse(image_path, media_type=media_type)

    @app.get("/api/series/{series}/chapters/{chapter}/output")
    def list_output(series: str, chapter: str) -> list[str]:
        """File names of the chapter's finished output images in natural order ([] when absent)."""
        return [p.name for p in list_images(output_paths(series, chapter))]

    @app.get("/api/series/{series}/chapters/{chapter}/output/{name}")
    def get_output_image(series: str, chapter: str, name: str) -> FileResponse:
        """One finished output image's bytes; never a path outside the chapter's output dir."""
        directory = output_paths(series, chapter)
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=404, detail="output image not found")
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(status_code=404, detail="output image not found")
        path = directory / name
        if not _under(directory, path) or not path.is_file():
            raise HTTPException(status_code=404, detail="output image not found")
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.get("/api/series/{series}/filtered")
    def list_filtered(series: str) -> list[dict[str, object]]:
        """Per chapter, every (target, index) the filter ever marked filtered, with restore state."""
        result: list[dict[str, object]] = []
        spaths = series_paths(series)
        for chapter in spaths.chapters():
            paths = spaths.chapter(chapter)
            filter_path = paths.artifact("filter.json")
            if not filter_path.is_file():
                continue
            try:
                artifact = FilterArtifact.load(filter_path)
            except Exception:
                continue  # invalid filter.json: skip the chapter, never a 500
            # (target, index) -> the LAST "filtered" decision of that pair (score/example source).
            last_filtered: dict[tuple[Literal["file", "slice"], int], FilterDecision] = {}
            for decision in artifact.decisions:
                if decision.decision == "filtered":
                    last_filtered[(decision.target, decision.index)] = decision
            if not last_filtered:
                continue
            names = source_names(paths)
            ranges = slice_ranges(paths)
            items: list[dict[str, object]] = []
            for (target, index), decision in sorted(
                last_filtered.items(), key=lambda kv: (0 if kv[0][0] == "file" else 1, kv[0][1])
            ):
                y0: int | None = None
                y1: int | None = None
                if target == "slice" and index in ranges:
                    y0, y1 = ranges[index]
                items.append(
                    {
                        "target": target,
                        "index": index,
                        "state": effective_decision(artifact, target, index),
                        "score": decision.score,
                        "matched_example": decision.matched_example,
                        "method": decision.method,
                        "name": names.get(index) if target == "file" else None,
                        "y0": y0,
                        "y1": y1,
                    }
                )
            result.append({"chapter": chapter, "items": items})
        return result

    @app.post("/api/series/{series}/chapters/{chapter}/filter/restore")
    async def restore_filtered(series: str, chapter: str, request: Request) -> dict[str, object]:
        """Append a manual `restored` decision for the requested (target, index); metadata only."""
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("application/json"):
            # A cross-site page can send text/plain POSTs without a CORS preflight; close that hole.
            # Checked before anything else — FastAPI would 422 the body first if it parsed it itself.
            raise HTTPException(status_code=415, detail="Content-Type must be application/json")
        paths = chapter_paths(series, chapter)
        filter_path = paths.artifact("filter.json")
        if not filter_path.is_file():
            raise HTTPException(status_code=404, detail="filter.json not found")
        try:
            body = RestoreBody.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors())) from exc
        try:
            artifact = FilterArtifact.load(filter_path)
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"filter.json is not a valid filter artifact: {exc}"
            ) from exc
        if effective_decision(artifact, body.target, body.index) != "filtered":
            raise HTTPException(status_code=409, detail="item is not currently filtered")
        decision = restore(paths, body.target, body.index)
        return decision.model_dump(mode="json")

    return app
