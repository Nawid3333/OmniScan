"""Ingest stage: raw chapter files -> normalised JPEGs on disk + strip layout artifact."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, SourceFile
from omniscan.filter.apply import EMPTY_OVERRIDES, Overrides, file_verdict
from omniscan.filter.decide import ExampleHash, best_match
from omniscan.filter.hashing import dhash

from .convert import ConvertedImage, convert_to_jpeg
from .layout import dominant_width, stack_layout

__all__ = [
    "ConvertedImage",
    "IngestResult",
    "convert_to_jpeg",
    "dominant_width",
    "ingest_chapter",
    "stack_layout",
]


@dataclass(frozen=True, slots=True)
class IngestResult:
    """One chapter's ingest: the strip-layout artifact plus the JPEG paths to decode downstream."""

    artifact: IngestArtifact
    converted_paths: list[
        Path
    ]  # the ConvertedImage.jpeg_path for every file kept in the strip, in order (for the caller to decode)


def ingest_chapter(
    raw_dir: Path,
    series: str,
    chapter: str,
    cache_dir: Path,
    quality: int = 95,
    *,
    examples: Sequence[ExampleHash] = (),
    threshold: float = 0.90,
    overrides: Overrides | None = None,
    filtered_dir: Path | None = None,
) -> IngestResult:
    """Normalise every image of a chapter to JPEG, compute its strip position, and build the artifact.

    With promo `examples`, a raw file whose dHash matches one — or that `overrides` forces/restores —
    is left out of the strip (kept files keep their original raw index; gaps are allowed). Filtered
    files are listed in `filtered_files` and copied byte-for-byte into `filtered_dir`; never deleted."""
    paths = list_images(raw_dir)
    if not paths:
        raise ValueError(f"no images found in {raw_dir}")
    overrides = EMPTY_OVERRIDES if overrides is None else overrides

    kept: list[tuple[int, Path, ConvertedImage]] = []
    filtered_names: list[str] = []
    for index, path in enumerate(paths):
        if examples:
            with Image.open(path) as img:
                match = best_match(dhash(img), examples)
            score, matched = (match[1], match[0].name) if match is not None else (0.0, None)
        else:  # without examples no automatic match can count (forced overrides still apply)
            score, matched = 0.0, None
        if file_verdict(index, score, matched, threshold, overrides) == "filtered":
            filtered_names.append(path.name)
            if filtered_dir is not None:
                _copy_filtered(path, filtered_dir / path.name)
            continue
        kept.append((index, path, convert_to_jpeg(path, cache_dir, index, quality)))

    if not kept:
        raise ValueError(f"every image of {raw_dir} matched a promo example")
    strip_width = dominant_width([img.width for _, _, img in kept])
    layout = stack_layout([(img.width, img.height) for _, _, img in kept], strip_width)
    files = [
        SourceFile(
            index=index,
            name=path.name,
            sha256=img.sha256,
            width=img.width,
            height=img.height,
            y0=y0,
            y1=y1,
            scale=scale,
            converted_from=path.suffix.lower() if img.converted else None,
        )
        for (index, path, img), (scale, y0, y1) in zip(kept, layout, strict=True)
    ]
    artifact = IngestArtifact(
        series=series,
        chapter=chapter,
        strip_width=strip_width,
        strip_height=layout[-1][2],
        files=files,
        filtered_files=filtered_names,
    )
    return IngestResult(artifact=artifact, converted_paths=[img.jpeg_path for _, _, img in kept])


def _copy_filtered(source: Path, dest: Path) -> None:
    """Byte-copy a filtered raw into _filtered (an existing identical copy is left alone; the source never moves)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.read_bytes() == source.read_bytes():
        return
    shutil.copy2(source, dest)
