"""OCR stage wrapper: ingest.json + slices.json + regions.json + chapter JPEGs -> ocr.json (text filled).

The strip is cut into overlapping tiles (only tiles that touch a region), the line detector runs in
batches, lines are merged across tiles, assigned to regions, cropped and read by the recognizer, and
the results are assembled into the regions of ocr.json via ocr/pipeline.py. With `ocr.engine` set to
a crop-reading engine (manga_ocr or paddleocr_vl) every region is read as one whole crop instead and
no line detection runs. Stored fixed-position watermarks pass through unread; with `sfx.sweep` CRAFT
then looks over every active tile for sound effects the detector missed (ocr/sweep.py), read by the
same engine.
The series' learned lessons (learn/: word fixes the editor kept making, texts they keep deleting or
labelling) are applied to the reading. The reading is kept as ocr_auto.json; ocr.json is that reading with
the chapter's hand edits (edits.json) applied, so a re-run never loses them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, RegionsArtifact, SlicesArtifact
from omniscan.core.stage import ChapterContext
from omniscan.detect.tiles import keep_tiles, plan_tiles
from omniscan.edits.apply import apply_region_edits
from omniscan.edits.store import OCR_AUTO_FILE, load_edits
from omniscan.gpu.groups import VISION_GROUP
from omniscan.ingest.strip import load_strip
from omniscan.learn.apply import apply_to_regions
from omniscan.learn.memory import current_memory
from omniscan.ocr.engines import engine_rec_model
from omniscan.ocr.pipeline import read_region_crops, read_regions
from omniscan.ocr.sfx import (
    default_sfx_text_paths,
    load_sfx_lexicon,
    measure_lettering_style,
    reclassify_sfx_regions,
)
from omniscan.ocr.sweep import SweepRules, sweep
from omniscan.ocr.watermark_text import (
    default_watermark_text_paths,
    load_watermark_patterns,
    reclassify_watermark_regions,
)


class OcrStage:
    """Read the text of detected regions (ppocr or a crop-reading engine; satisfies core.stage.Stage)."""

    name: ClassVar[str] = "ocr"
    version: ClassVar[int] = (
        5  # 3: watermark text reclassification; 4: sound effects found, lettering measured;
        # 5: stored watermark regions passed through unread, whole-page sweep for missed effects
    )
    gpu_group: ClassVar[str | None] = VISION_GROUP

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        return [
            ctx.paths.artifact("ingest.json"),
            ctx.paths.artifact("slices.json"),
            ctx.paths.artifact("regions.json"),
            *list_images(ctx.paths.raw_dir),
        ]

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["ocr.json", OCR_AUTO_FILE]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        patterns = load_watermark_patterns(default_watermark_text_paths())
        lexicon = load_sfx_lexicon(default_sfx_text_paths())
        words = "\n".join(f"{lang}:{word}" for lang in sorted(lexicon) for word in sorted(lexicon[lang]))
        return {
            **cfg.ocr.model_dump(),
            "reading_direction": cfg.detect.reading_direction,
            # fingerprint of the watermark text patterns (card F2b): editing either TOML re-runs the stage
            "watermark_patterns": hashlib.sha256("\n".join(patterns).encode("utf-8")).hexdigest(),
            "sfx": {
                **cfg.sfx.model_dump(exclude={"mode"}),  # the mode only matters to inpaint and typeset
                "lexicon": hashlib.sha256(words.encode("utf-8")).hexdigest(),
            },
        }

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        regions_path = ctx.paths.artifact("regions.json")
        if not regions_path.is_file():
            raise FileNotFoundError("regions.json missing — run the detect stage first")
        ingest_path = ctx.paths.artifact("ingest.json")
        if not ingest_path.is_file():
            raise FileNotFoundError("ingest.json missing — run the slice stage first")
        ingest = IngestArtifact.load(ingest_path)
        regions = RegionsArtifact.load(regions_path)
        strip = load_strip(ctx, ingest)
        cfg = ctx.cfg
        # a stored fixed-position watermark (F2c) is never translated: not read, erased whole by inpaint
        to_read = [r for r in regions.regions if r.kind != "watermark"]
        if cfg.ocr.engine == "ppocr":
            engine = cfg.ocr.rec_model or cfg.ocr.rec_repo.split("/")[-1]
            reader = models["recognizer"]
            ocr_regions, metrics = read_regions(
                strip,
                to_read,
                models["line_detector"],
                reader,
                cfg.ocr,
                direction=cfg.detect.reading_direction,
                engine=engine,
            )
        else:
            rec_model = engine_rec_model(cfg.ocr)
            if rec_model is None:
                raise ValueError(f"OCR engine '{cfg.ocr.engine}' needs ocr.rec_model")
            engine, reader = rec_model, models["reader"]
            ocr_regions, metrics = read_region_crops(strip, to_read, reader, cfg.ocr, engine=engine)
        # false-positive detections read as junk with a low score (or nothing at all): leave those pixels alone
        kept = [r for r in ocr_regions if r.lines and r.confidence >= ctx.cfg.ocr.drop_conf]
        metrics["regions_dropped"] = float(len(ocr_regions) - len(kept))
        # OCR'd ad/spam text is a source-injected watermark (card F2b): never translated, erased by inpaint
        patterns = load_watermark_patterns(default_watermark_text_paths())
        kept = reclassify_watermark_regions(kept, patterns)
        order = {r.id: i for i, r in enumerate(regions.regions)}
        kept = sorted(
            [*kept, *(r for r in regions.regions if r.kind == "watermark")], key=lambda r: order[r.id]
        )
        # the detector has no working sfx class (M10): onomatopoeia in free text becomes kind "sfx", and
        # the sweep finds the effects (and stamped watermarks) it missed; the lettering of effects and
        # free text (fill vs outline colour, an effect's tilt and weight) is measured for the typesetter
        if cfg.sfx.detect:
            lexicon = load_sfx_lexicon(default_sfx_text_paths())
            kept = reclassify_sfx_regions(kept, lexicon, cfg.sfx)
            if cfg.sfx.sweep:
                slices = SlicesArtifact.load(ctx.paths.artifact("slices.json")).slices
                active = [(s.y0, s.y1) for s in slices if not s.blank and not s.filtered]
                tiles = keep_tiles(
                    plan_tiles(ingest.strip_width, ingest.strip_height, cfg.ocr.tile_px, cfg.ocr.overlap),
                    active,
                )
                rules = SweepRules(
                    words=lexicon.get(cfg.ocr.lang, frozenset()),
                    watermark_patterns=tuple(patterns),
                    min_score=cfg.ocr.drop_conf,
                    max_chars=cfg.sfx.max_chars,
                    engine=f"{engine}+craft",
                    lang=cfg.ocr.lang,
                )
                kept, swept = sweep(
                    strip,
                    kept,
                    slices,
                    tiles,
                    models["sfx_sweeper"],
                    reader,
                    rules,
                    min_px=cfg.sfx.sweep_min_px,
                )
                metrics.update(swept)
        if cfg.learn.enabled:  # what the series' hand corrections taught: word fixes, drops, labels (learn/)
            kept, learned = apply_to_regions(kept, current_memory(ctx.series), cfg.learn)
            metrics.update(learned)
        metrics["watermarked"] = float(sum(1 for r in kept if r.kind == "watermark"))
        kept = [measure_lettering_style(strip, r) if r.kind in ("sfx", "free_text") else r for r in kept]
        metrics["sfx"] = float(sum(1 for r in kept if r.kind == "sfx"))
        RegionsArtifact(regions=kept).save(ctx.paths.artifact(OCR_AUTO_FILE))
        edits = load_edits(ctx.paths)
        if edits.regions:
            slices = SlicesArtifact.load(ctx.paths.artifact("slices.json")).slices
            kept, orphans = apply_region_edits(kept, edits, slices, direction=cfg.detect.reading_direction)
            metrics["edits_orphaned"] = float(orphans)
        RegionsArtifact(regions=kept).save(ctx.paths.artifact("ocr.json"))
        return metrics
