"""Applying the promo filter in the pipeline: overrides from filter.json, verdicts, slice marking.

`filter.json` is the user's override file: the stages read it as an *input* (only `method="manual"`
entries count, last one per (target, index) wins), so `filter restore` / `filter force` re-run what
depends on it. Automatic matching (dHash against the example images) happens in `ingest` and `slice`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import FilterArtifact, FilterDecision, Slice, SlicesArtifact
from omniscan.filter.decide import EXAMPLE_SUFFIXES, FILTER_ARTIFACT, ExampleHash, best_match
from omniscan.filter.hashing import dhash_tensor


@dataclass(frozen=True, slots=True)
class Overrides:
    """Manual filter.json decisions, per raw-file index (position in `list_images(raw_dir)`) and slice index."""

    files_restored: frozenset[int]
    files_forced: frozenset[int]
    slices_restored: frozenset[int]
    slices_forced: frozenset[int]


EMPTY_OVERRIDES = Overrides(frozenset(), frozenset(), frozenset(), frozenset())


def load_overrides(filter_json: Path) -> Overrides:
    """The manual entries of filter.json (last entry per (target, index) wins); a missing or invalid
    file gives empty Overrides."""
    if not filter_json.is_file():
        return EMPTY_OVERRIDES
    try:
        artifact = FilterArtifact.load(filter_json)
    except Exception:  # corrupt/unreadable override file must never break the pipeline
        return EMPTY_OVERRIDES
    latest: dict[tuple[Literal["file", "slice"], int], str] = {}
    for decision in artifact.decisions:
        if decision.method == "manual":
            latest[(decision.target, decision.index)] = decision.decision
    return Overrides(
        files_restored=frozenset(i for (t, i), d in latest.items() if t == "file" and d == "restored"),
        files_forced=frozenset(i for (t, i), d in latest.items() if t == "file" and d == "filtered"),
        slices_restored=frozenset(i for (t, i), d in latest.items() if t == "slice" and d == "restored"),
        slices_forced=frozenset(i for (t, i), d in latest.items() if t == "slice" and d == "filtered"),
    )


def file_verdict(
    index: int, similarity_score: float, matched: str | None, threshold: float, overrides: Overrides
) -> Literal["filtered", "keep"]:
    """The file-level verdict: a forced override wins, then a restored one, else the automatic match."""
    return _verdict(
        index, similarity_score, matched, threshold, overrides.files_forced, overrides.files_restored
    )


def slice_verdict(
    index: int, similarity_score: float, matched: str | None, threshold: float, overrides: Overrides
) -> Literal["filtered", "keep"]:
    """The slice-level verdict: a forced override wins, then a restored one, else the automatic match."""
    return _verdict(
        index, similarity_score, matched, threshold, overrides.slices_forced, overrides.slices_restored
    )


def _verdict(
    index: int,
    similarity_score: float,
    matched: str | None,
    threshold: float,
    forced: frozenset[int],
    restored: frozenset[int],
) -> Literal["filtered", "keep"]:
    if index in forced:
        return "filtered"
    if index in restored:
        return "keep"
    return "filtered" if similarity_score >= threshold and matched is not None else "keep"


def examples_fingerprint(examples: Sequence[ExampleHash]) -> str:
    """sha256 hex of the examples' "name:hash" lines in order (changes with any name or hash)."""
    payload = "\n".join(f"{example.name}:{example.hash}" for example in examples)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def example_files(promo_examples_dir: Path, series: str) -> list[Path]:
    """Every *.jpg/*.jpeg/*.png directly under promo_examples_dir/'global' and .../<series> (name order)."""
    files: list[Path] = []
    for dirname in ("global", series):
        directory = promo_examples_dir / dirname
        if directory.is_dir():
            files.extend(
                path
                for path in sorted(directory.iterdir(), key=lambda p: p.name)
                if path.is_file() and path.suffix.lower() in EXAMPLE_SUFFIXES
            )
    return files


def record_override(
    paths: ChapterPaths,
    target: Literal["file", "slice"],
    index: int,
    decision: Literal["filtered", "restored"],
) -> FilterDecision:
    """Append a manual `decision` for (target, index) to filter.json — creating the file when absent —
    and return it. Metadata only: nothing under filtered_dir is ever touched."""
    artifact_path = paths.artifact(FILTER_ARTIFACT)
    artifact = FilterArtifact.load(artifact_path) if artifact_path.is_file() else FilterArtifact(decisions=[])
    entry = FilterDecision(
        target=target, index=index, decision=decision, score=1.0, matched_example=None, method="manual"
    )
    artifact.decisions.append(entry)
    artifact.save(artifact_path)
    return entry


def apply_slice_filter(
    strip: torch.Tensor,
    slices: SlicesArtifact,
    examples: Sequence[ExampleHash],
    threshold: float,
    overrides: Overrides,
    chapter: str,
    filtered_dir: Path | None = None,
) -> SlicesArtifact:
    """Flag every non-blank slice matching an example (or a manual override) as `filtered` and save each
    newly filtered slice's pixels to filtered_dir/f"{chapter}_slice_{index:04d}.jpg" (quality 95)."""
    updated: list[Slice] = []
    for slice_ in slices.slices:
        if slice_.blank:
            updated.append(slice_)
            continue
        crop = strip[:, slice_.y0 : slice_.y1, :]
        match = best_match(dhash_tensor(crop), examples)
        score, matched = (match[1], match[0].name) if match is not None else (0.0, None)
        if slice_verdict(slice_.index, score, matched, threshold, overrides) == "filtered":
            if filtered_dir is not None:
                _save_filtered_slice(crop, filtered_dir / f"{chapter}_slice_{slice_.index:04d}.jpg")
            updated.append(slice_.model_copy(update={"filtered": True}))
        else:
            updated.append(slice_)
    return slices.model_copy(update={"slices": updated})


def _save_filtered_slice(crop: torch.Tensor, dest: Path) -> None:
    """Write a [3, h, w] uint8 crop as a quality-95 JPEG (one host transfer per filtered slice)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop.permute(1, 2, 0).contiguous().cpu().numpy()).save(dest, format="JPEG", quality=95)
