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
        '{"deleted": True, "edited": True, "replaced_by": replaced_by}',
        '{"deleted": True, "replaced_by": replaced_by}',
        "a page whose only correction is a deletion is left out",
    ),
    (
        C,
        '{"deleted": True, "edited": True, "replaced_by": replaced_by}',
        '{"deleted": True, "edited": True}',
        "a redrawn box is not paired with the box drawn over it",
    ),
    (
        C,
        "            if replaced_by is not None:\n                removed.append((region, replaced_by))\n",
        "            pass\n",
        "a detected box replaced by a drawn one vanishes",
    ),
    (
        C,
        "            if region.id in claimed:\n                continue\n",
        "",
        "a detection an edit claims counted as replaced",
    ),
    (
        C,
        '    if edit is not None:\n        return "typed" if edit.suggested_by is None else "suggestion"\n'
        '    return "machine" if line else None',
        "    if not line:\n        return None\n    if edit is not None:\n"
        '        return "typed" if edit.suggested_by is None else "suggestion"\n    return "machine"',
        "a line cleared by hand is not a correction",
    ),
    (
        C,
        "    for name in dict.fromkeys(chapters) if chapters is not None else order:",
        "    for name in chapters if chapters is not None else order:",
        "a chapter asked twice is built twice",
    ),
    (
        C,
        "        deleted=sum(1 for r in regions if r.deleted) - redrawn,",
        "        deleted=sum(1 for r in regions if r.deleted),",
        "redrawn boxes counted as deleted",
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
        "    if False:",
        "series.toml opt-out ignored",
    ),
    (
        C,
        "    if not cfg.share.enabled:\n        raise",
        "    if False:\n        raise",
        "a series.toml opts a series back in although the machine opted out",
    ),
    (C, "series_id=digest(series.series, salt),", "series_id=series.series,", "the series name is shared"),
    (
        C,
        "    return hmac.new(salt, text.encode(), hashlib.sha256).hexdigest()[:16]",
        "    return hashlib.sha256(text.encode()).hexdigest()[:16]",
        "ids reversible by hashing known titles",
    ),
    (
        C,
        '                with path.open("x", encoding="ascii") as out:',
        '                with path.open("w", encoding="ascii") as out:',
        "an existing salt could be replaced (ids change)",
    ),
    (
        C,
        "        if len(salt) >= 16:\n            return salt\n",
        "        return salt\n",
        "a damaged salt used",
    ),
    (
        C,
        "        if not any(_corrected(region) or region.checked for region in regions):\n            continue\n",
        "",
        "uncorrected pages contributed",
    ),
    (
        C,
        "        if not any(_corrected(region) or region.checked for region in regions):",
        "        if not any(_corrected(region) for region in regions):",
        "a page whose only news is a checked line is left out",
    ),
    (
        C,
        "            or chapter_edits.checked\n",
        "",
        "a chapter with checks but no edits is skipped",
    ),
    (
        C,
        '    checked = {rid for rid, status in line_statuses(paths, touched=()).items() if status == "checked"}',
        "    checked = {check.region_id for check in chapter_edits.checked}",
        "a stale check counts",
    ),
    (
        C,
        "                checked=region.id in checked,",
        "                checked=False,",
        "checks not shared",
    ),
    (
        C,
        "        checked=sum(1 for r in regions if r.checked),\n",
        "",
        "checked lines missing from the summary",
    ),
]
