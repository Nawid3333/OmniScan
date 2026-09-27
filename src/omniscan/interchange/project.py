"""Chapter project archives: one chapter's raw pages and all its work in one file, passed between OmniScan users.

A group moves a chapter from translator to proofreader to typesetter; each of them needs the same raw pages and
the chapter's whole work folder — stage artifacts, edits.json with its undo history, hand cleanup — so every
stage already done stays done (the stage manifest hashes file contents, which travel unchanged) and every hand
edit stays editable. `pack` writes them, plus the series' series.toml and voices.toml and optionally the
finished pages, to one zip file whose project.json lists every file with its sha256.

`unpack` is written for archives from someone else. It refuses members project.json does not list, paths that
could leave the chapter's folders, series or chapter names that are not plain folder names, more bytes than
declared (a zip bomb) and files whose content does not match their sha256. It unpacks into hidden folders next
to the targets and moves them into place only once every file checked out. An existing chapter is replaced only
with `force`; the series files are added where missing, never overwritten.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from pydantic import ValidationError

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import ChapterProject, ProjectFile
from omniscan.packaging.names import safe_filename
from omniscan.update.version import current_version

PROJECT_FILE = "project.json"
SUFFIX = ".omniscan"
SERIES_FILES = ("series.toml", "voices.toml")
MAX_BYTES = 16 * 1024**3  # everything an archive may unpack to (a chapter with its work is far below)
MAX_SERIES_FILE_BYTES = 1_000_000  # a series.toml / voices.toml, read into memory
_CHUNK = 1 << 20
_PARTS = ("raw", "work", "series", "output")
_TEXT = frozenset(
    {".json", ".toml", ".yaml", ".yml", ".txt"}
)  # compressed; images and npz are stored as they are
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class ProjectError(ValueError):
    """The file is not a chapter project OmniScan can unpack safely."""


@dataclass(frozen=True, slots=True)
class Unpacked:
    """Where a chapter project went."""

    paths: ChapterPaths
    series_files: list[str]  # series files added (the receiver had none)
    kept_series_files: list[str]  # series files the receiver already had, left as they were
    replaced: bool  # an existing chapter was replaced (`force`)


def _files(root: Path) -> list[Path]:
    """Every file under `root`, recursively and sorted, leaving out temporary (*.tmp) and hidden files."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.endswith(".tmp")
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def _add(archive: zipfile.ZipFile, member: str, source: Path) -> ProjectFile:
    """Stream one file into the archive, hashing it on the way."""
    info = zipfile.ZipInfo(member, date_time=_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED if source.suffix.lower() in _TEXT else zipfile.ZIP_STORED
    digest, size = hashlib.sha256(), 0
    with source.open("rb") as src, archive.open(info, "w", force_zip64=True) as out:
        while chunk := src.read(_CHUNK):
            digest.update(chunk)
            out.write(chunk)
            size += len(chunk)
    return ProjectFile(path=member, sha256=digest.hexdigest(), bytes=size)


def pack(series: SeriesPaths, chapter: str, dest: Path, *, with_output: bool = False) -> ChapterProject:
    """Write chapter `chapter` of `series` to the project archive `dest` (atomically); returns its project.json.
    FileNotFoundError when the series has no such chapter folder."""
    paths = series.chapter(chapter)
    if not paths.raw_dir.is_dir():
        raise FileNotFoundError(f"no chapter {chapter!r} in {series.series}")
    roots = [("raw", paths.raw_dir), ("work", paths.work_dir)] + (
        [("output", paths.output_dir)] if with_output else []
    )
    sources = [
        (f"{part}/{file.relative_to(root).as_posix()}", file) for part, root in roots for file in _files(root)
    ]
    sources += [
        (f"series/{name}", series.library_dir / name)
        for name in SERIES_FILES
        if (series.library_dir / name).is_file()
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w") as archive:
            files = [_add(archive, member, source) for member, source in sources]
            project = ChapterProject(
                app_version=str(current_version()), series=series.series, chapter=chapter, files=files
            )
            info = zipfile.ZipInfo(PROJECT_FILE, date_time=_ZIP_TIME)
            archive.writestr(info, project.model_dump_json(indent=2), compress_type=zipfile.ZIP_DEFLATED)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return project


def _check_name(name: str, what: str) -> str:
    """`name` when it can be one plain folder name (a series or a chapter), else ProjectError."""
    if name != safe_filename(name) or name.startswith(("_", ".")):
        raise ProjectError(f"the {what} name {name!r} cannot be a folder name")
    return name


def _check_path(path: str) -> None:
    """ProjectError unless `path` is a member OmniScan writes: raw/…, work/…, output/… or series/<series file>."""
    pure = PurePosixPath(path)
    parts = pure.parts
    if (
        "\\" in path
        or pure.as_posix() != path
        or len(parts) < 2
        or parts[0] not in _PARTS
        or any(part != safe_filename(part) or part == ".." for part in parts)
        or (parts[0] == "series" and (len(parts) != 2 or parts[1] not in SERIES_FILES))
    ):
        raise ProjectError(f"the archive holds a file OmniScan does not write: {path!r}")


def read_project(archive: zipfile.ZipFile) -> ChapterProject:
    """The archive's project.json, after checking that it describes exactly the archive's files, safely."""
    try:
        project = ChapterProject.model_validate_json(archive.read(PROJECT_FILE))
    except KeyError as exc:
        raise ProjectError("not an OmniScan chapter project (no project.json)") from exc
    except ValidationError as exc:
        raise ProjectError(f"project.json is not valid: {exc.errors()[0]['msg']}") from exc
    listed = [file.path for file in project.files]
    for path in listed:
        _check_path(path)
    if len({path.casefold() for path in listed}) != len(listed):
        raise ProjectError("project.json lists a file twice")  # also twice up to case (Windows, macOS)
    if set(archive.namelist()) - {PROJECT_FILE} != set(listed):
        raise ProjectError("the archive's files do not match its project.json")
    if not any(path.startswith("raw/") for path in listed):
        raise ProjectError("the project has no raw pages")
    if sum(file.bytes for file in project.files) > MAX_BYTES:
        raise ProjectError(f"the project would unpack to more than {MAX_BYTES // 1024**3} GB")
    _check_name(project.series, "series")
    _check_name(project.chapter, "chapter")
    return project


@contextmanager
def _open(path: Path) -> Iterator[zipfile.ZipFile]:
    """The archive at `path`; ProjectError when it is not a zip file."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ProjectError(f"{path.name} is not an OmniScan chapter project (not a zip file)") from exc
    with archive:
        yield archive


def show(path: Path) -> ChapterProject:
    """The project.json of the archive at `path` (checked like `unpack` checks it)."""
    with _open(path) as archive:
        return read_project(archive)


def _copy(archive: zipfile.ZipFile, file: ProjectFile, out: BinaryIO) -> None:
    """Copy one member to `out`, stopping at more bytes than declared; ProjectError when it does not match."""
    digest, size = hashlib.sha256(), 0
    with archive.open(file.path) as src:
        while chunk := src.read(_CHUNK):
            size += len(chunk)
            if size > file.bytes:
                raise ProjectError(f"{file.path} is larger than its project.json says")
            digest.update(chunk)
            out.write(chunk)
    if size != file.bytes or digest.hexdigest() != file.sha256:
        raise ProjectError(f"{file.path} is damaged (its content does not match project.json)")


def _move_into_place(stage: Path, target: Path) -> None:
    """Replace `target` by `stage`; the old folder is removed only after the new one is in place."""
    old = target.with_name(f".{target.name}.replaced")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    shutil.rmtree(old, ignore_errors=True)


def unpack(
    path: Path, cfg: Config, *, series: str | None = None, chapter: str | None = None, force: bool = False
) -> Unpacked:
    """Unpack the chapter project at `path` into the library and work folders of `cfg` — as the series and chapter
    it was packed as, or as `series` / `chapter`. ProjectError for an archive that is not a safe chapter project,
    FileExistsError when the chapter exists already and `force` is not set (then it is replaced)."""
    with _open(path) as archive:
        project = read_project(archive)
        series_paths = SeriesPaths.from_config(cfg, _check_name(series or project.series, "series"))
        paths = series_paths.chapter(_check_name(chapter or project.chapter, "chapter"))
        exists = bool(_files(paths.raw_dir) or _files(paths.work_dir))
        if exists and not force:
            raise FileExistsError(f"{paths.series}/{paths.chapter} exists already (force replaces it)")
        targets = {"raw": paths.raw_dir, "work": paths.work_dir, "output": paths.output_dir}
        stages = {part: root.with_name(f".{root.name}.unpacking") for part, root in targets.items()}
        series_data: dict[str, bytes] = {}
        try:
            for stage in stages.values():
                shutil.rmtree(stage, ignore_errors=True)
            for file in project.files:
                part, _, rest = file.path.partition("/")
                if part == "series":
                    if file.bytes > MAX_SERIES_FILE_BYTES:
                        raise ProjectError(f"{file.path} is too large for a settings file")
                    buffer = io.BytesIO()
                    _copy(archive, file, buffer)
                    series_data[rest] = buffer.getvalue()
                    continue
                dest = stages[part] / rest
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as out:
                    _copy(archive, file, out)
            for part in ("raw", "work"):
                stages[part].mkdir(
                    parents=True, exist_ok=True
                )  # an unprocessed chapter has no work files yet
            for part, stage in stages.items():
                if stage.is_dir():
                    _move_into_place(stage, targets[part])
        finally:
            for stage in stages.values():
                shutil.rmtree(stage, ignore_errors=True)
    added, kept = [], []
    for name, data in series_data.items():
        target = series_paths.library_dir / name
        if target.exists():
            kept.append(name)
        else:
            target.write_bytes(data)
            added.append(name)
    return Unpacked(paths=paths, series_files=added, kept_series_files=kept, replaced=exists)
