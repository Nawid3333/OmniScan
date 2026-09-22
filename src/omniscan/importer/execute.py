"""Import execution: apply a planned import into the library layout (idempotent copy/move, JPEG conversion)."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from omniscan.core.manifest import hash_file
from omniscan.importer.plan import JPEG_SUFFIXES, ImportPlan, ImportPlanError

Progress = Callable[[int, int | None], None]

# Same convention as the ingest stage (src/omniscan/ingest/convert.py): an imported file must be a
# plain baseline RGB JPEG, so ingest never re-converts it. Alpha is composited over white and EXIF
# rotation is applied, matching ingest's _flatten.
JPEG_QUALITY = 95
_EXIF_ORIENTATION = 0x0112
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Counts for one executed import; `chapters_written` is in plan order."""

    chapters_written: list[str]  # plan.items chapters, in plan order
    files_copied: int
    files_skipped_duplicate: int
    files_converted: int = 0  # sources re-encoded to JPEG instead of being copied byte-for-byte
    converted: list[str] = field(default_factory=list)  # their source file names, in plan order


def execute_import(
    plan: ImportPlan,
    library_root: Path,
    *,
    move: bool = False,
    on_progress: Progress | None = None,
) -> ImportResult:
    """Apply `plan` under `library_root / plan.series / <chapter>`.

    Fail-fast: stops at the first destination file that exists with different content and raises
    `ImportPlanError`; files already written before that point are left in place. Sources that are
    not already .jpg/.jpeg are decoded and re-encoded as JPEG (quality 95) instead of being copied;
    a re-run finds the identical bytes already in place and skips them.
    """
    total = sum(len(item.files) for item in plan.items)
    done = 0
    chapters_written: list[str] = []
    files_copied = files_converted = files_skipped = 0
    converted: list[str] = []
    for item in plan.items:
        dest_dir = library_root / plan.series / item.chapter
        dest_dir.mkdir(parents=True, exist_ok=True)
        for source_file in item.files:
            convert = source_file.suffix.lower() not in JPEG_SUFFIXES
            encoded = converted_jpeg_bytes(source_file) if convert else None
            dest = dest_dir / (source_file.stem + ".jpg" if convert else source_file.name)
            if not dest.exists():
                if encoded is not None:
                    dest.write_bytes(encoded)
                    files_converted += 1
                    converted.append(source_file.name)
                    if move:  # a converted source is consumed just like a copied one
                        source_file.unlink()
                elif move:
                    shutil.move(source_file, dest)
                    files_copied += 1
                else:
                    shutil.copy2(source_file, dest)
                    files_copied += 1
            elif hash_file(dest) == (_sha256(encoded) if encoded is not None else hash_file(source_file)):
                if move:  # identical content already at the destination; drop the redundant source
                    source_file.unlink()
                files_skipped += 1
            else:
                raise ImportPlanError(
                    f"destination exists with different content: {source_file} -> {dest} "
                    f"({files_copied} file(s) already written in this call)"
                )
            done += 1
            if on_progress is not None:
                on_progress(done, total)
        chapters_written.append(item.chapter)
    return ImportResult(
        chapters_written=chapters_written,
        files_copied=files_copied,
        files_skipped_duplicate=files_skipped,
        files_converted=files_converted,
        converted=converted,
    )


def converted_jpeg_bytes(source: Path) -> bytes:
    """The exact JPEG bytes `source` is committed as (decoded, flattened, re-encoded at quality 95)."""
    try:
        with Image.open(source) as img:
            out = _flatten(img)
            buffer = BytesIO()
            out.save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True)
    except Exception as exc:  # unreadable/corrupt images stop the import with a clear message
        raise ImportPlanError(f"can't convert {source} to JPEG: {exc}") from exc
    return buffer.getvalue()


def _flatten(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation and flatten to RGB (alpha composited over opaque white), as ingest does."""
    transposed = ImageOps.exif_transpose(img)
    if transposed is not None:
        img = transposed
    if img.mode in _ALPHA_MODES or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _sha256(data: bytes) -> str:
    """sha256 hex digest of `data` (the same check `hash_file` applies to files)."""
    return hashlib.sha256(data).hexdigest()
