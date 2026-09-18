"""PDF packaging: one PDF page per chapter image via Pillow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image


def pack_pdf(images: Sequence[Path], dest: Path) -> None:
    """One PDF page per image, in the order given. Open every image with PIL and .convert("RGB") (so RGBA PNGs and
    grayscale JPEGs work), then first.save(tmp, format="PDF", save_all=True, append_images=rest, resolution=100.0).
    Same rules as pack_cbz for: ValueError("no images to pack") on empty input (dest not created), creating
    dest.parent, atomic tmp + replace, and removing the tmp file on any exception. Close every opened image."""
    if not images:
        raise ValueError("no images to pack")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    opened: list[Image.Image] = []
    try:
        pages = []
        for image in images:
            page = Image.open(image).convert("RGB")
            opened.append(page)
            pages.append(page)
        pages[0].save(tmp, format="PDF", save_all=True, append_images=pages[1:], resolution=100.0)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        for page in opened:
            page.close()
