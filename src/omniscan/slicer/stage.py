"""Slice stage wrapper: ingest.json + chapter JPEGs -> slices.json (via the shared strip tensor)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from omniscan.core.config import Config
from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact
from omniscan.core.stage import ChapterContext
from omniscan.filter.apply import apply_slice_filter, example_files, examples_fingerprint, load_overrides
from omniscan.filter.decide import load_examples
from omniscan.gpu.timeline import mark
from omniscan.ingest.strip import load_strip
from omniscan.slicer import slice_with_strategy


class SliceStage:
    """Cut the chapter strip into slices (satisfies core.stage.Stage); no GPU model group needed."""

    name: ClassVar[str] = "slice"
    version: ClassVar[int] = 4  # 4: slice-level promo filter
    gpu_group: ClassVar[str | None] = None

    def __init__(self) -> None:
        self._fingerprint = ""  # of the examples loaded in inputs() (config_subset has no ctx)

    def inputs(self, ctx: ChapterContext) -> list[Path]:
        """Files whose content determines this stage's output (raw images and/or upstream artifacts)."""
        inputs = [ctx.paths.artifact("ingest.json"), *list_images(ctx.paths.raw_dir)]
        filter_json = ctx.paths.artifact("filter.json")
        if filter_json.is_file():
            inputs.append(filter_json)
        if ctx.cfg.filter.enabled:  # disabled: no automatic match counts, examples are irrelevant
            self._fingerprint = examples_fingerprint(
                load_examples(ctx.cfg.paths.promo_examples, ctx.paths.series)
            )
            inputs.extend(example_files(ctx.cfg.paths.promo_examples, ctx.paths.series))
        else:
            self._fingerprint = examples_fingerprint([])
        return inputs

    def outputs(self, ctx: ChapterContext) -> list[str]:
        """Artifact names (relative to the chapter work dir) this stage writes."""
        return ["slices.json"]

    def config_subset(self, cfg: Config) -> Mapping[str, Any]:
        """Only the config values that affect this stage's output (hashed for invalidation)."""
        return {
            **cfg.slicer.model_dump(),
            "filter": cfg.filter.model_dump(),
            "examples": self._fingerprint,
        }

    def run(self, ctx: ChapterContext, models: Mapping[str, Any]) -> Mapping[str, float]:
        """Do the work, write outputs, return metrics (seconds are added by the runner)."""
        ingest_path = ctx.paths.artifact("ingest.json")
        if not ingest_path.is_file():
            raise FileNotFoundError("ingest.json missing — run the ingest stage first")
        ingest = IngestArtifact.load(ingest_path)

        strip = load_strip(ctx, ingest)
        mark("slice: strip decoded")
        slices = slice_with_strategy(strip, ctx.cfg.slicer, ingest.files)

        overrides = load_overrides(ctx.paths.artifact("filter.json"))
        examples = (
            load_examples(ctx.cfg.paths.promo_examples, ctx.paths.series) if ctx.cfg.filter.enabled else []
        )
        if ctx.cfg.filter.enabled and (examples or overrides.slices_forced):
            slices = apply_slice_filter(
                strip,
                slices,
                examples,
                ctx.cfg.filter.threshold,
                overrides,
                ctx.paths.chapter,
                filtered_dir=ctx.paths.filtered_dir,
            )

        slices.save(ctx.paths.artifact("slices.json"))
        return {
            "slices": float(len(slices.slices)),
            "blank": float(sum(1 for s in slices.slices if s.blank)),
            "forced": float(sum(1 for s in slices.slices if s.forced_cut)),
            "filtered": float(sum(1 for s in slices.slices if s.filtered)),
            "bands": float(len(slices.bands)),
            "strip_height": float(slices.strip_height),
        }
