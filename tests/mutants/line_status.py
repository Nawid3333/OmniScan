import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

ST = "src/omniscan/edits/store.py"
SE = "src/omniscan/edits/session.py"
MUTANTS = [
    (
        ST,
        "    return (check.source, check.english) == (now.source, now.english)",
        "    return check.source == now.source",
        "a check survives a changed English line",
    ),
    (
        ST,
        "    return (check.source, check.english) == (now.source, now.english)",
        "    return check.english == now.english",
        "a check survives a changed source text",
    ),
    (
        ST,
        "            if not checked:\n                if index is not None:\n                    checks[index] = None\n",
        "            if not checked:\n                if index is not None:\n                    pass\n",
        "unchecking does nothing",
    ),
    (
        ST,
        "            elif index is None:\n                checks.append(_line_check(region, english))",
        "            elif True:\n                checks.append(_line_check(region, english))",
        "checking again piles up checks",
    ),
    (
        ST,
        "            elif not _holds(edits.checked[index], region, english):\n",
        "            else:\n",
        "re-checking an unchanged line records an undo step",
    ),
    (
        ST,
        "                checks[index] = _line_check(region, english)  # re-checked where it was: the order stays\n",
        "                checks[index] = None\n                checks.append(_line_check(region, english))\n",
        "re-checking reorders the checks",
    ),
    (
        ST,
        "        _ensure_auto(paths)\n        regions = current_regions(paths)\n        wanted =",
        "        regions = current_regions(paths)\n        wanted =",
        "undoing a first check empties ocr.json",
    ),
    (
        ST,
        "    if check_index is not None:\n        edits.checked[check_index]",
        "    if False:\n        edits.checked[check_index]",
        "a check stays behind when its region moves",
    ),
    (
        ST,
        "        if check is not None:  # else it could claim an overlapping neighbour before that one's own check\n"
        "            del edits.checked[check]\n",
        "",
        "a deleted region's check lands on its neighbour",
    ),
    (
        ST,
        "        elif followers[2] is not None:  # a hand-added region goes, and its check with it\n"
        "            del edits.checked[followers[2]]\n",
        "",
        "a reverted hand-added region leaves its check",
    ),
    (
        ST,
        'region.id: "checked" if region.id in held else "edited" if region.id in touched else "todo"',
        'region.id: "checked" if region.id in held else "todo"',
        "edited lines shown as todo",
    ),
    (
        "src/omniscan/edits/apply.py",
        "    return _match_all(edits.checked, regions)",
        "    return {i: c.region_id for i, c in enumerate(edits.checked) if any(r.id == c.region_id for r in regions)}",
        "a check matched by id only (lost on a re-detection)",
    ),
    (
        SE,
        "                if ids:\n                    store.set_checked(self.paths, ids, checked=checked)\n",
        "                pass\n",
        "the desktop session never saves its checks",
    ),
    (
        SE,
        '        if check:\n            return "checked"\n',
        "",
        "a pending check does not show before save",
    ),
    (
        SE,
        '        if check is None and self._status.get(region_id) == "checked" and not self._rewrites(region_id):',
        '        if check is None and self._status.get(region_id) == "checked":',
        "an unsaved edit keeps showing the saved check",
    ),
    (
        SE,
        '        if check is None and self._status.get(region_id) == "checked" and not self._rewrites(region_id):',
        '        if check is None and self._status.get(region_id) == "checked" and not self._unsaved(region_id):',
        "a new speaker shows a checked line as edited",
    ),
    (
        SE,
        '            if (self._status.get(region_id) == "checked") != checked\n'
        "            or (checked and self._rewrites(region_id))\n",
        "            if True\n",
        "no-op checks keep the session dirty",
    ),
]
