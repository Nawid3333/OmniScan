"""Ingest stage: raw chapter files -> normalised JPEGs on disk + strip layout artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omniscan.core.paths import list_images
from omniscan.core.schemas import IngestArtifact, SourceFile

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
    ]  # the ConvertedImage.jpeg_path for every file, in order (for the caller to decode)


def ingest_chapter(
    raw_dir: Path, series: str, chapter: str, cache_dir: Path, quality: int = 95
) -> IngestResult:
    """Normalise every image of a chapter to JPEG, compute its strip position, and build the artifact."""
    paths = list_images(raw_dir)
    if not paths:
        raise ValueError(f"no images found in {raw_dir}")

    converted = [convert_to_jpeg(path, cache_dir, index, quality) for index, path in enumerate(paths)]
    strip_width = dominant_width([img.width for img in converted])
    layout = stack_layout([(img.width, img.height) for img in converted], strip_width)

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
        for index, (path, img, (scale, y0, y1)) in enumerate(zip(paths, converted, layout, strict=True))
    ]
    artifact = IngestArtifact(
        series=series,
        chapter=chapter,
        strip_width=strip_width,
        strip_height=layout[-1][2],
        files=files,
    )
    return IngestResult(artifact=artifact, converted_paths=[img.jpeg_path for img in converted])
