import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

C = "src/omniscan/share/contribution.py"
MUTANTS = [
    (
        C,
        "y0=min(max(box.y0 - page.y0, 0), height),",
        "y0=min(max(box.y0, 0), height),",
        "boxes not moved onto the page",
    ),
    (
        C,
        "font=PureWindowsPath(edit.font).name if edit.font else None,",
        "font=edit.font,",
        "the font's folder path is shared",
    ),
    (
        C,
        'return region.edited or region.english_from in ("suggestion", "typed") or region.lettering is not None',
        "return region.edited or region.lettering is not None",
        "a page whose only correction is an English line is left out",
    ),
    (
        C,
        'return region.edited or region.english_from in ("suggestion", "typed") or region.lettering is not None',
        'return region.edited or region.english_from in ("suggestion", "typed")',
        "a page whose only correction is a lettering is left out",
    ),
    (
        C,
        'return region.edited or region.english_from in ("suggestion", "typed") or region.lettering is not None',
        "return True",
        "every page is contributed",
    ),
    (
        C,
        "                deleted=True,\n                edited=True,",
        "                deleted=True,",
        "a page whose only correction is a deletion is left out",
    ),
    (
        C,
        "    if edit is not None and edit.auto_text is not None:\n        return edit.auto_text\n",
        "",
        "the reading after a re-run replaces the reading that was corrected",
    ),
    (
        C,
        'return "typed" if edit.suggested_by is None else "suggestion"',
        'return "typed"',
        "kept suggestions count as typed",
    ),
    (
        C,
        '            if entry.status == "locked"',
        '            if entry.status != "rejected"',
        "proposed glossary terms shared",
    ),
    (
        C,
        "    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)\n    info.compress_type = compression\n"
        "    info.create_system = 0\n    archive.writestr(info, data)",
        "    archive.writestr(name, data, compress_type=compression)",
        "entries carry the export time",
    ),
    (
        C,
        '    Image.fromarray(pixels).save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=0)',
        '    Image.fromarray(pixels).save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=0, exif=b"Exif\\x00\\x00MM\\x00*\\x00\\x00\\x00\\x08\\x00\\x00")',
        "pages carry an EXIF block",
    ),
    (
        C,
        "    if not series_config(cfg, series.library_dir).share.enabled:",
        "    if not cfg.share.enabled:",
        "series.toml opt-out ignored",
    ),
    (C, "series_id=digest(series.series),", "series_id=series.series,", "the series name is shared"),
    (
        C,
        "        if not any(_corrected(region) for region in regions):\n            continue\n",
        "",
        "uncorrected pages contributed",
    ),
]
