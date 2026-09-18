"""Promo-filter decisions: match raws and strip slices against example hashes, record and restore them."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import FilterArtifact, FilterDecision, IngestArtifact, SlicesArtifact
from omniscan.filter.hashing import dhash, similarity

EXAMPLE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
FILTER_ARTIFACT = "filter.json"


@dataclass(frozen=True, slots=True)
class ExampleHash:
    name: str  # relative path, e.g. "global/end_card_1.jpg" or "<series>/promo_2.jpg"
    hash: int


def load_examples(promo_examples_dir: Path, series: str) -> list[ExampleHash]:
    """dhash every *.jpg/*.jpeg/*.png (case-insensitive suffix) directly under
    promo_examples_dir/'global' and promo_examples_dir/series, non-recursively. Skips missing directories
    (returns [] if neither exists). `name` = "<dirname>/<filename>" where dirname is 'global' or the
    series name, using forward slashes always (even on Windows)."""
    examples: list[ExampleHash] = []
    for dirname in ("global", series):
        directory = promo_examples_dir / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if not (path.is_file() and path.suffix.lower() in EXAMPLE_SUFFIXES):
                continue
            with Image.open(path) as img:
                value = dhash(img)
            examples.append(ExampleHash(name=f"{dirname}/{path.name}", hash=value))
    return examples


def best_match(img_hash: int, examples: Sequence[ExampleHash]) -> tuple[ExampleHash, float] | None:
    """Highest-similarity example and its score; None if examples is empty."""
    if not examples:
        return None
    example = max(examples, key=lambda e: similarity(img_hash, e.hash))
    return example, similarity(img_hash, example.hash)


def decide_files(
    paths: ChapterPaths, ingest: IngestArtifact, examples: Sequence[ExampleHash], threshold: float = 0.90
) -> list[FilterDecision]:
    """One FilterDecision(target="file", index=source_file.index, method="phash") per entry in ingest.files,
    in ingest.files order. dhash the raw file at paths.raw_dir / source_file.name. decision="filtered" when
    best_match score >= threshold (matched_example=that example's `name`), else decision="keep"
    (matched_example=None, score=0.0 if examples is empty else the best score found). For every "filtered"
    decision, copy the raw file byte-for-byte (shutil.copy2) into paths.filtered_dir / source_file.name,
    creating paths.filtered_dir if needed. Never deletes or modifies the source file."""
    decisions: list[FilterDecision] = []
    for source_file in ingest.files:
        raw_path = paths.raw_dir / source_file.name
        with Image.open(raw_path) as img:
            value = dhash(img)
        match = best_match(value, examples)
        if match is not None and match[1] >= threshold:
            decisions.append(
                FilterDecision(
                    target="file",
                    index=source_file.index,
                    decision="filtered",
                    score=match[1],
                    matched_example=match[0].name,
                )
            )
            paths.filtered_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, paths.filtered_dir / source_file.name)
        else:
            decisions.append(
                FilterDecision(
                    target="file",
                    index=source_file.index,
                    decision="keep",
                    score=0.0 if match is None else match[1],
                )
            )
    return decisions


def decide_slices(
    paths: ChapterPaths,
    chapter: str,
    strip: Image.Image,
    slices: SlicesArtifact,
    examples: Sequence[ExampleHash],
    threshold: float = 0.90,
) -> list[FilterDecision]:
    """One FilterDecision(target="slice", index=slice_.index, method="phash") per entry in slices.slices, in
    slices.slices order. Crop strip to the box (0, slice_.y0, strip.width, slice_.y1) and dhash that crop.
    Same threshold rule as decide_files. For every "filtered" decision, save the crop as
    paths.filtered_dir / f"{chapter}_slice_{slice_.index:04d}.jpg" at quality=95, creating paths.filtered_dir
    if needed."""
    decisions: list[FilterDecision] = []
    for slice_ in slices.slices:
        crop = strip.crop((0, slice_.y0, strip.width, slice_.y1))
        match = best_match(dhash(crop), examples)
        if match is not None and match[1] >= threshold:
            decisions.append(
                FilterDecision(
                    target="slice",
                    index=slice_.index,
                    decision="filtered",
                    score=match[1],
                    matched_example=match[0].name,
                )
            )
            paths.filtered_dir.mkdir(parents=True, exist_ok=True)
            crop.save(paths.filtered_dir / f"{chapter}_slice_{slice_.index:04d}.jpg", quality=95)
        else:
            decisions.append(
                FilterDecision(
                    target="slice",
                    index=slice_.index,
                    decision="keep",
                    score=0.0 if match is None else match[1],
                )
            )
    return decisions


def effective_decision(artifact: FilterArtifact, target: Literal["file", "slice"], index: int) -> str | None:
    """The decision of the LAST entry in artifact.decisions matching (target, index) in list order
    ("last write wins" — this is how restore overrides an earlier "filtered"); None if no entry matches."""
    result: str | None = None
    for decision in artifact.decisions:
        if decision.target == target and decision.index == index:
            result = decision.decision
    return result


def restore(paths: ChapterPaths, target: Literal["file", "slice"], index: int) -> FilterDecision:
    """Load filter.json from paths.artifact("filter.json") (raise FileNotFoundError with a clear message if
    absent), append a new FilterDecision(target=target, index=index, decision="restored", score=1.0,
    matched_example=None, method="manual"), save the artifact back (atomic, via FilterArtifact.save), return
    the new decision. Does not touch any file already copied into paths.filtered_dir — restore is a metadata
    override only, per the "never delete" policy."""
    artifact_path = paths.artifact(FILTER_ARTIFACT)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"filter: no {FILTER_ARTIFACT} for {paths.series}/{paths.chapter} — run 'omniscan filter run' first"
        )
    artifact = FilterArtifact.load(artifact_path)
    decision = FilterDecision(
        target=target, index=index, decision="restored", score=1.0, matched_example=None, method="manual"
    )
    artifact.decisions.append(decision)
    artifact.save(artifact_path)
    return decision
