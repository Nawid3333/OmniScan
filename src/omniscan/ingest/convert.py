"""Convert raw chapter images to plain baseline RGB JPEGs (EXIF-oriented, alpha flattened)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from omniscan.core.manifest import hash_file

_EXIF_ORIENTATION = 0x0112
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})


@dataclass(frozen=True, slots=True)
class ConvertedImage:
    """Result of normalising one raw image: where the final JPEG bytes live and their geometry."""

    jpeg_path: Path  # where the final JPEG bytes live (original file, or a written cache copy)
    sha256: str  # sha256 of jpeg_path's bytes
    width: int  # final pixel width, after EXIF orientation is applied
    height: int  # final pixel height, after EXIF orientation is applied
    converted: bool  # True iff a new file was written (source was not already a plain baseline RGB JPEG)


def _needs_rotation(img: Image.Image) -> bool:
    """True if the image's EXIF Orientation tag requires a transpose."""
    return img.getexif().get(_EXIF_ORIENTATION, 1) != 1


def needs_conversion(path: Path) -> bool:
    """True if `path` is not already usable as-is: not a plain RGB JPEG, has alpha/CMYK mode, or needs rotation."""
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return True
    with Image.open(path) as img:
        if img.format != "JPEG" or img.mode != "RGB":
            return True
        return _needs_rotation(img)


def _flatten(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation and flatten to RGB (alpha composited over opaque white)."""
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


def convert_to_jpeg(src: Path, cache_dir: Path, index: int, quality: int = 95) -> ConvertedImage:
    """Return `src` itself if already a plain baseline RGB JPEG, else write a normalised JPEG into `cache_dir`."""
    if not needs_conversion(src):
        with Image.open(src) as img:
            width, height = img.size
        return ConvertedImage(
            jpeg_path=src, sha256=hash_file(src), width=width, height=height, converted=False
        )

    with Image.open(src) as img:
        out = _flatten(img)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{index:04d}_{src.stem}.jpg"
    out.save(dest, format="JPEG", quality=quality, subsampling=0, optimize=True)
    return ConvertedImage(
        jpeg_path=dest, sha256=hash_file(dest), width=out.width, height=out.height, converted=True
    )
