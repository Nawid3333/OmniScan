"""FastAPI app serving existing pipeline artifacts and raw images (mostly read-only; see `run_worker`)."""

from __future__ import annotations

import io
import mimetypes
import threading
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import Field, ValidationError

from omniscan.core.config import Config, SeriesConfigError, series_config
from omniscan.core.paths import IMAGE_SUFFIXES, ChapterPaths, SeriesPaths, list_images, natural_key
from omniscan.core.schemas import (
    BBox,
    FilterArtifact,
    FilterDecision,
    FinalArtifact,
    GlossaryEntry,
    IngestArtifact,
    InpaintArtifact,
    Lang,
    Model,
    RegionKind,
    RegionsArtifact,
    SlicesArtifact,
)
from omniscan.edits import store as edit_store
from omniscan.filter.decide import effective_decision, restore
from omniscan.glossary.match import find_terms, term_present
from omniscan.glossary.store import GlossaryStore
from omniscan.inpaint.patches import load_patches
from omniscan.pipeline.stages import STAGE_ORDER
from omniscan.queue.store import QueueStore, queue_db_path
from omniscan.queue.worker import run_queue


class RestoreBody(Model):
    """Body of the filter-restore POST request."""

    target: Literal["file", "slice"]
    index: int = Field(ge=0)


class RunBody(Model):
    """Body of the on-demand run POST request."""

    through: str  # a STAGE_ORDER name; every stage up to and including it is queued
    start: str | None = None  # a STAGE_ORDER name to start from instead of the first stage (e.g. "inpaint")


class FinalEditBody(Model):
    """Body of the manual final-line edit PUT request."""

    text: str


class EmptyBody(Model):
    """Body of a POST request that carries no data (sent as `{}`, so it must come as application/json)."""


class RegionPatchBody(Model):
    """Body of the region edit PATCH request: every field given replaces the region's value."""

    kind: RegionKind | None = None
    bbox: BBox | None = None
    bubble_bbox: BBox | None = None
    text: str | None = None


class NewRegionBody(Model):
    """Body of the hand-drawn region POST request."""

    bbox: BBox
    kind: RegionKind = "bubble_text"
    text: str = ""
    bubble_bbox: BBox | None = None


def _under(root: Path, candidate: Path) -> bool:
    """True if `candidate` resolves to a location inside `root` (blocks `..` / absolute escapes)."""
    return candidate.resolve().is_relative_to(root.resolve())


def _drain_loop(cfg: Config, stop: threading.Event) -> None:
    """Run queued jobs one at a time for as long as `stop` is not set; idles 1s between empty checks.

    Started as a daemon thread by `create_app(..., run_worker=True)` so the debug server can produce
    missing artifacts on request instead of only ever reading what already exists. Exactly one such
    loop should run against a given `queue.db` at a time (the store's own single-worker design) — do
    not also run `omniscan queue run` against the same library while `omniscan serve` is up.

    Opens its own `QueueStore` rather than reusing one built elsewhere: sqlite3 connections are only
    usable from the thread that created them, and this loop is its own dedicated thread.
    """
    from omniscan.queue.executor import (
        stage_executor,
    )  # deferred: it imports torch; a read-only app never does

    store = QueueStore(queue_db_path(cfg))
    executor = stage_executor(cfg)
    while not stop.is_set():
        summary = run_queue(store, executor, max_jobs=1)
        if summary.done == 0 and summary.failed == 0 and summary.retried == 0:
            stop.wait(1.0)


def create_app(
    cfg: Config, *, cors_origins: Sequence[str] = ("http://localhost:5173",), run_worker: bool = False
) -> FastAPI:
    """Build the debug API app: series/chapter browsing + ingest/slices artifacts + raw pages.

    `run_worker=True` (the real `omniscan serve` command's default, not this function's) also starts
    a background thread that drains `queue.db`, so `POST .../run` requests actually execute instead of
    only ever sitting queued; tests and other embedders that just want to read existing artifacts
    should leave it `False` (the default here) to avoid touching the GPU/queue at all.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = threading.Event()
        thread: threading.Thread | None = None
        if run_worker:
            thread = threading.Thread(target=_drain_loop, args=(cfg, stop), daemon=True)
            thread.start()
        yield
        stop.set()
        if thread is not None:
            thread.join(timeout=5.0)

    app = FastAPI(title="OmniScan debug API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
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

    def _merged_inpaint(paths: ChapterPaths) -> InpaintArtifact:
        """inpaint.json with inpaint_lama.json's items overriding the matching flat ones by
        region_id, when a LaMa pass exists. The flat inpaint stage writes one item per cleaned
        region, flagging any it could not clean as `needs_lama`; LaMa re-cleans only that subset
        into its own inpaint_lama.json/patches_lama.npz rather than rewriting the flat pass's files
        (so re-running the cheap flat pass alone never invalidates LaMa's expensive output). Assumes
        inpaint.json exists (callers check first)."""
        artifact = InpaintArtifact.load(paths.artifact("inpaint.json"))
        lama_path = paths.artifact("inpaint_lama.json")
        if not lama_path.is_file():
            return artifact
        lama_by_id = {item.region_id: item for item in InpaintArtifact.load(lama_path).items}
        items = [lama_by_id.get(item.region_id, item) for item in artifact.items]
        return artifact.model_copy(update={"items": items})

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

    @app.get("/api/series/{series}/chapters/{chapter}/inpaint")
    def get_inpaint(series: str, chapter: str) -> Response:
        """The chapter's inpaint.json; when inpaint_lama.json also exists, its items override the
        matching flat ones by region_id (the same override export/stage.py applies at export time),
        so this reflects the chapter as actually cleaned instead of the flat pass's placeholder.
        Byte-for-byte the file on disk when there is no LaMa pass yet."""
        paths = chapter_paths(series, chapter)
        if not paths.artifact("inpaint.json").is_file():
            raise HTTPException(status_code=404, detail="inpaint.json not found")
        if not paths.artifact("inpaint_lama.json").is_file():
            return artifact_bytes(paths, "inpaint.json")
        return Response(content=_merged_inpaint(paths).model_dump_json(), media_type="application/json")

    @app.get("/api/series/{series}/chapters/{chapter}/layout")
    def get_layout(series: str, chapter: str) -> Response:
        """The chapter's layout.json, byte-for-byte as written by the typeset stage."""
        return artifact_bytes(chapter_paths(series, chapter), "layout.json")

    @app.get("/api/series/{series}/chapters/{chapter}/inpaint/patches/{region_id}.png")
    def get_inpaint_patch(series: str, chapter: str, region_id: str) -> Response:
        """One region's inpaint patch as an RGBA PNG (alpha = the patch's mask, scaled 0/255);
        prefers patches_lama.npz over patches.npz for a region LaMa has re-cleaned (mirrors the
        override export/stage.py applies at export time)."""
        paths = chapter_paths(series, chapter)
        inpaint_path = paths.artifact("inpaint.json")
        if not inpaint_path.is_file():
            raise HTTPException(status_code=404, detail="inpaint.json not found")
        artifact = _merged_inpaint(paths)
        # region_id is never used to build a filesystem path, only as a lookup key; validating it
        # against the known list is this route's traversal guard in place of the usual _under/`..` check.
        if all(item.region_id != region_id for item in artifact.items):
            raise HTTPException(status_code=404, detail="region not found")
        lama_npz = paths.artifact("patches_lama.npz")
        lama_patches = load_patches(lama_npz) if lama_npz.is_file() else {}
        if region_id in lama_patches:
            pixels, mask = lama_patches[region_id]
        else:
            patches_path = paths.artifact("patches.npz")
            if not patches_path.is_file():
                raise HTTPException(status_code=404, detail="patches.npz not found")
            patches = load_patches(patches_path)
            if region_id not in patches:
                raise HTTPException(status_code=404, detail="no patch stored for this region")
            pixels, mask = patches[region_id]
        img = Image.fromarray(pixels, mode="RGB").convert("RGBA")
        alpha = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        img.putalpha(alpha)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

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

    async def json_body[B: Model](request: Request, body_type: type[B]) -> B:
        """The request's JSON body as `body_type`: 415 unless it is sent as application/json (a cross-site
        page can send text/plain without a CORS preflight), 422 when it does not validate."""
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("application/json"):
            raise HTTPException(status_code=415, detail="Content-Type must be application/json")
        try:
            return body_type.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors())) from exc

    def edit_settings(series: str) -> tuple[edit_store.Direction, Lang]:
        """The series' reading direction and OCR language (series.toml applied) for a hand edit."""
        try:
            scfg = series_config(cfg, series_paths(series).library_dir)
        except SeriesConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return scfg.detect.reading_direction, scfg.ocr.lang

    def run_edit[T](operation: Callable[[], T]) -> T:
        """Run one edit operation, mapping its errors to 404 (missing artifact or region) and 422 (bad box)."""
        try:
            return operation()
        except edit_store.EditNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/series/{series}/chapters/{chapter}/final/{region_id}")
    async def edit_final_line(
        series: str, chapter: str, region_id: str, request: Request
    ) -> dict[str, object]:
        """Write one region's English line by hand; its decision becomes "manual".

        Recorded in edits.json and applied to final.json at once (creating final.json when no judge has run
        yet), and re-applied by every later judge run, so the pipeline never overwrites it."""
        body = await json_body(request, FinalEditBody)
        paths = chapter_paths(series, chapter)
        direction, _lang = edit_settings(series)
        line = run_edit(lambda: edit_store.set_translation(paths, region_id, body.text, direction=direction))
        return line.model_dump(mode="json")

    @app.post("/api/series/{series}/chapters/{chapter}/final/{region_id}/revert")
    async def revert_final_line(
        series: str, chapter: str, region_id: str, request: Request
    ) -> dict[str, object]:
        """Drop a region's hand-written line; returns the judge's line again (null when it has none)."""
        await json_body(request, EmptyBody)
        paths = chapter_paths(series, chapter)
        direction, _lang = edit_settings(series)
        line = run_edit(lambda: edit_store.revert_translation(paths, region_id, direction=direction))
        return {"line": None if line is None else line.model_dump(mode="json")}

    @app.get("/api/series/{series}/chapters/{chapter}/edits")
    def get_edits(series: str, chapter: str) -> dict[str, object]:
        """The chapter's hand edits (edits.json; empty lists when none), the regions they deleted, and which
        current regions carry a region edit or a hand-written line."""
        paths = chapter_paths(series, chapter)
        edits = edit_store.load_edits(paths)
        edited, translated = edit_store.edited_ids(paths)
        return {
            **edits.model_dump(mode="json"),
            "deleted_regions": [r.model_dump(mode="json") for r in edit_store.deleted_regions(paths)],
            "edited_region_ids": edited,
            "manual_translation_ids": translated,
        }

    @app.post("/api/series/{series}/chapters/{chapter}/regions", status_code=201)
    async def add_region(series: str, chapter: str, request: Request) -> dict[str, object]:
        """Add a hand-drawn region (clamped into the strip); returns it with its new m-prefixed id."""
        body = await json_body(request, NewRegionBody)
        paths = chapter_paths(series, chapter)
        direction, lang = edit_settings(series)
        region = run_edit(
            lambda: edit_store.add_region(
                paths,
                body.bbox,
                direction=direction,
                kind=body.kind,
                text=body.text,
                bubble_bbox=body.bubble_bbox,
                lang=lang,
            )
        )
        return region.model_dump(mode="json")

    @app.patch("/api/series/{series}/chapters/{chapter}/regions/{region_id}")
    async def patch_region(series: str, chapter: str, region_id: str, request: Request) -> dict[str, object]:
        """Change a region's kind, text box, bubble box and/or source text; returns the region as rebuilt."""
        body = await json_body(request, RegionPatchBody)
        paths = chapter_paths(series, chapter)
        direction, _lang = edit_settings(series)
        region = run_edit(
            lambda: edit_store.update_region(
                paths,
                region_id,
                direction=direction,
                kind=body.kind,
                bbox=body.bbox,
                bubble_bbox=body.bubble_bbox,
                text=body.text,
            )
        )
        return region.model_dump(mode="json")

    @app.delete("/api/series/{series}/chapters/{chapter}/regions/{region_id}")
    def delete_region(series: str, chapter: str, region_id: str) -> dict[str, object]:
        """Remove a region (a false detection stays on the page untranslated; revert brings it back)."""
        paths = chapter_paths(series, chapter)
        direction, _lang = edit_settings(series)
        run_edit(lambda: edit_store.delete_region(paths, region_id, direction=direction))
        return {"deleted": region_id}

    @app.post("/api/series/{series}/chapters/{chapter}/regions/{region_id}/revert")
    async def revert_region(series: str, chapter: str, region_id: str, request: Request) -> dict[str, object]:
        """Drop every hand edit of a region; returns it as the pipeline read it (null for a hand-added one)."""
        await json_body(request, EmptyBody)
        paths = chapter_paths(series, chapter)
        direction, _lang = edit_settings(series)
        region = run_edit(lambda: edit_store.revert_region(paths, region_id, direction=direction))
        return {"region": None if region is None else region.model_dump(mode="json")}

    @app.post("/api/series/{series}/chapters/{chapter}/run", status_code=202)
    async def run_chapter(series: str, chapter: str, request: Request) -> dict[str, object]:
        """Queue every stage through `through` (inclusive) for one chapter — from `start` when given (the
        editor re-renders a chapter with start "inpaint" without re-translating it).

        Actually executes only when the app was built with `run_worker=True` (`omniscan serve`'s
        default) — its background thread drains `queue.db`; poll `GET /api/jobs/{job_id}` with the
        returned id for progress. `through` must be one of `omniscan.pipeline.stages.STAGE_ORDER`."""
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("application/json"):
            raise HTTPException(status_code=415, detail="Content-Type must be application/json")
        chapter_paths(series, chapter)  # 404s on any path-traversal attempt (existence checked next)
        if chapter not in series_paths(series).chapters():
            raise HTTPException(status_code=404, detail=f"unknown chapter {chapter!r} of series {series!r}")
        try:
            body = RunBody.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors())) from exc
        if body.through not in STAGE_ORDER:
            raise HTTPException(
                status_code=422,
                detail=f"unknown stage {body.through!r} (known: {', '.join(STAGE_ORDER)})",
            )
        first = 0
        if body.start is not None:
            if body.start not in STAGE_ORDER:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown stage {body.start!r} (known: {', '.join(STAGE_ORDER)})",
                )
            first = STAGE_ORDER.index(body.start)
        stages = STAGE_ORDER[first : STAGE_ORDER.index(body.through) + 1]
        if not stages:
            raise HTTPException(status_code=422, detail=f"{body.start!r} comes after {body.through!r}")
        store = QueueStore(queue_db_path(cfg))
        job = store.add(series, list(stages), chapters=[chapter])
        return {"job_id": job.id, "stages": list(stages)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int) -> dict[str, object]:
        """One queued job's current status ({} fields never populated -> a plain 404, never a 500)."""
        store = QueueStore(queue_db_path(cfg))
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "id": job.id,
            "series": job.series,
            "chapters": list(job.chapters) if job.chapters is not None else None,
            "stages": list(job.stages),
            "status": job.status,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "error": job.error,
        }

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
            for match in find_terms(region.text, entries, region.lang):
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
