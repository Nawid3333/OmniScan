import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

P = "src/omniscan/interchange/project.py"
MUTANTS = [
    (P, "        or pure.as_posix() != path\n", "", "unnormalised member names accepted"),
    (
        P,
        '        or any(part != safe_filename(part) or part == ".." for part in parts)\n',
        '        or any(part == ".." for part in parts)\n',
        "names no filesystem accepts",
    ),
    (
        P,
        '        or (parts[0] == "series" and (len(parts) != 2 or parts[1] not in SERIES_FILES))\n',
        "",
        "any file under series/",
    ),
    (
        P,
        "    if len({path.casefold() for path in listed}) != len(listed):",
        "    if len(set(listed)) != len(listed):",
        "files differing in case only",
    ),
    (
        P,
        "    if set(archive.namelist()) - {PROJECT_FILE} != set(listed):",
        "    if not set(listed) <= set(archive.namelist()):",
        "unlisted members accepted",
    ),
    (P, "    if sum(file.bytes for file in project.files) > MAX_BYTES:", "    if False:", "no size cap"),
    (P, "            if size > file.bytes:\n", "            if False:\n", "no stop at the declared size"),
    (
        P,
        "    if size != file.bytes or digest.hexdigest() != file.sha256:",
        "    if size != file.bytes:",
        "content not checked against its sha256",
    ),
    (
        P,
        "        if exists and not force:",
        "        if False:",
        "an existing chapter overwritten without force",
    ),
    (P, "    if target.exists():\n        target.rename(old)\n", "", "force keeps the old chapter's files"),
    (
        P,
        "        if target.exists():\n            kept.append(name)\n        else:\n",
        "        if False:\n            kept.append(name)\n        else:\n",
        "the receiver's series files overwritten",
    ),
    (P, '        and not path.name.endswith(".tmp")\n', "", "half-written files packed"),
    (
        P,
        'stages = {part: root.with_name(f".{root.name}.unpacking") for part, root in targets.items()}',
        "stages = dict(targets)",
        "unpacked straight into place (a bad archive leaves half a chapter)",
    ),
]
