"""CBZ packaging: a ZIP_STORED zip of the chapter's images plus a ComicInfo.xml."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from pathlib import Path


def _comic_info_xml(page_count: int, *, title: str | None, series: str | None, number: str | None) -> bytes:
    """Serialise a ComicInfo.xml with the given metadata (PageCount and LanguageISO are always present)."""
    root = ET.Element("ComicInfo")
    for tag, value in (("Title", title), ("Series", series), ("Number", number)):
        if value is not None:
            ET.SubElement(root, tag).text = value
    ET.SubElement(root, "PageCount").text = str(page_count)
    ET.SubElement(root, "LanguageISO").text = "en"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def pack_cbz(
    images: Sequence[Path],
    dest: Path,
    *,
    title: str | None = None,
    series: str | None = None,
    number: str | None = None,
) -> None:
    """Write a CBZ (a zip file, stored WITHOUT compression: zipfile.ZIP_STORED — JPEGs are already compressed).
    Entries, in this order: every image as f"{i:04d}{path.suffix.lower()}" for i = 1..N in the order given (never
    sorted, never renamed otherwise), then "ComicInfo.xml" last. ComicInfo.xml is built with
    xml.etree.ElementTree and serialised with ET.tostring(root, encoding="utf-8", xml_declaration=True); root tag
    "ComicInfo"; children in exactly this order, each only if applicable: "Title" (if title is not None),
    "Series" (if series is not None), "Number" (if number is not None), "PageCount" (always, text = str(N)),
    "LanguageISO" (always, text = "en").
    Raises ValueError("no images to pack") if images is empty (dest is not created). Creates dest.parent if
    missing. Atomic: write to dest.with_suffix(dest.suffix + ".tmp"), then Path.replace(dest); on ANY exception
    remove the .tmp file (if it exists) and re-raise, leaving no partial dest."""
    if not images:
        raise ValueError("no images to pack")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for i, image in enumerate(images, start=1):
                zf.write(image, f"{i:04d}{image.suffix.lower()}")
            zf.writestr(
                "ComicInfo.xml",
                _comic_info_xml(len(images), title=title, series=series, number=number),
            )
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
