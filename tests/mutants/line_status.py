import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

ST = "src/omniscan/edits/store.py"
SE = "src/omniscan/edits/session.py"
MUTANTS = [
    (
        ST,
        '        and edits.checked[i].english == _norm(english.get(region_id, ""))\n',
        "",
        "a check survives a changed English line",
    ),
    (
        ST,
        "        if edits.checked[i].source == _norm(by_id[region_id].text)\n        and ",
        "        if ",
        "a check survives a changed source text",
    ),
    (
        ST,
        "        kept = [check for i, check in enumerate(edits.checked) if claims.get(i) not in wanted]",
        "        kept = list(edits.checked)",
        "checking again piles up checks and unchecking does nothing",
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
        "            for checked in (True, False):  # after the text edits: a check approves the saved lines\n"
        "                ids = [region_id for region_id, value in self._checks.items() if value is checked]\n"
        "                if ids:\n"
        "                    store.set_checked(self.paths, ids, checked=checked)\n",
        "",
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
        "        if check is False or unsaved:\n",
        "        if check is False:\n",
        "an unsaved edit keeps showing the saved check",
    ),
]
