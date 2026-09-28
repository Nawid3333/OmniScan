"""Chapter project archives: one chapter's raw pages and all its work in one file, passed between OmniScan users.

A group moves a chapter from translator to proofreader to typesetter; each of them needs the same raw pages and
the chapter's whole work folder — stage artifacts, edits.json with its undo history, hand cleanup — so every
stage already done stays done (the stage manifest hashes file contents, which travel unchanged) and every hand
edit stays editable. `pack` writes them, plus the series' series.toml and voices.toml and optionally the
finished pages, to one zip file whose project.json lists every file with its sha256.

`unpack` is written for archives from someone else. It refuses members project.json does not list, paths that
could leave the chapter's folders, names Windows cannot create (NUL, COM1.txt, …), series or chapter names that
are not plain folder names, more bytes or files than it allows (a zip bomb), damaged or encrypted members, files
whose content does not match their sha256, and work artifacts naming files outside the chapter. It unpacks into
hidden folders next to the targets and moves them into place only once every file checked out, putting the old
folders back when a move fails. An existing chapter is replaced only with `force`. The series files are added
where missing, never overwritten; from someone else's series.toml only numbers, switches and fixed choices are
taken — never a URL, a file name, a model or a font, which would make OmniScan fetch, load or write whatever the
sender chose.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import tomllib
import zipfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal, get_origin

from pydantic import BaseModel, ValidationError

from omniscan.core.config import SERIES_SECTIONS, Config, dumps_toml
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import ChapterProject, ProjectFile
from omniscan.packaging.names import safe_filename
from omniscan.translate.voices import parse_voices
from omniscan.update.version import current_version

log = logging.getLogger(__name__)

PROJECT_FILE = "project.json"
SUFFIX = ".omniscan"
SERIES_FILES = ("series.toml", "voices.toml")
MAX_BYTES = 16 * 1024**3  # everything an archive may unpack to (a chapter with its work is far below)
MAX_SERIES_FILE_BYTES = 1_000_000  # a series.toml / voices.toml, read into memory
MAX_PROJECT_JSON_BYTES = 64 * 1024**2  # project.json itself, read into memory (a line per file)
MAX_FILES = 100_000  # files one archive may hold (a chapter with its work has a few hundred)
_CHUNK = 1 << 20
_PARTS = ("raw", "work", "series", "output")
_TEXT = frozenset(
    {".json", ".toml", ".yaml", ".yml", ".txt"}
)  # compressed; images and npz are stored as they are
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{device}{n}" for device in ("COM", "LPT") for n in (*"123456789", "\u00b9", "\u00b2", "\u00b3")}
)  # Windows device names: "NUL", "nul.txt" or "COM1 .jpg" open a device, not a file
_DAMAGED = (
    zipfile.BadZipFile,  # a bad CRC or local header
    zlib.error,  # damaged compressed data
    EOFError,  # cut off
    NotImplementedError,  # a compression method zipfile does not read
    RuntimeError,  # an encrypted member
)
_NAMED_FILES = (
    ("work/ingest.json", "files", "name"),
    ("work/ingest.json", "filtered_files", None),
    ("work/export.json", "files", "name"),
    ("output/omniscan-chapter.json", "pages", "file"),
)  # (artifact, list, field) whose file names OmniScan joins onto a chapter folder


class ProjectError(ValueError):
    """The file is not a chapter project OmniScan can unpack safely."""


@dataclass(frozen=True, slots=True)
class Unpacked:
    """Where a chapter project went."""

    paths: ChapterPaths
    series_files: list[str]  # series files added (the receiver had none)
    kept_series_files: list[str]  # series files the receiver already had, left as they were
    replaced: bool  # an existing chapter was replaced (`force`)
    dropped_settings: list[str]  # settings of the added series.toml left out ("inpaint.lama_url", …)


def _files(root: Path) -> list[Path]:
    """Every file under `root`, recursively and sorted, leaving out temporary (*.tmp) and hidden files."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()  # a link's content is not the chapter's
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
    FileNotFoundError when the series has no such chapter folder, ProjectError when a name in it is one `unpack`
    would refuse."""
    paths = series.chapter(chapter)
    if not paths.raw_dir.is_dir():
        raise FileNotFoundError(f"no chapter {chapter!r} in {series.series}")
    _check_name(series.series, "series")
    _check_name(chapter, "chapter")
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
    for member, source in sources:
        try:
            _check_path(member)
        except ProjectError as exc:
            raise ProjectError(f"{source} cannot be packed: its name is not valid on every system") from exc
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


def _plain(name: str) -> bool:
    """`name` is one file or folder name every filesystem takes as it is: no separator, no '..', nothing
    `safe_filename` would change and no Windows device name."""
    return name == safe_filename(name) and name.split(".")[0].rstrip(" ").upper() not in _RESERVED


def _check_name(name: str, what: str) -> str:
    """`name` when it can be one plain folder name (a series or a chapter), else ProjectError."""
    if not _plain(name) or name.startswith(("_", ".")):
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
        or not all(_plain(part) for part in parts)
        or (parts[0] == "series" and (len(parts) != 2 or parts[1] not in SERIES_FILES))
    ):
        raise ProjectError(f"the archive holds a file OmniScan does not write: {path!r}")


def _read_member(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    """Member `name` read into memory, ProjectError past `limit` bytes (checked before inflating any of it)."""
    info = archive.getinfo(name)
    if info.file_size > limit:
        raise ProjectError(f"{name} is too large ({info.file_size} bytes)")
    try:
        with archive.open(info) as src:
            return src.read(limit)  # zipfile stops at the declared size, and so does this
    except _DAMAGED as exc:
        raise ProjectError(f"{name} is damaged or encrypted ({exc})") from exc


def read_project(archive: zipfile.ZipFile) -> ChapterProject:
    """The archive's project.json, after checking that it describes exactly the archive's files, safely."""
    if len(archive.infolist()) > MAX_FILES + 1:
        raise ProjectError(f"the archive holds more than {MAX_FILES} files")
    try:
        project = ChapterProject.model_validate_json(
            _read_member(archive, PROJECT_FILE, MAX_PROJECT_JSON_BYTES)
        )
    except KeyError as exc:
        raise ProjectError("not an OmniScan chapter project (no project.json)") from exc
    except ValidationError as exc:
        raise ProjectError(f"project.json is not valid: {exc.errors()[0]['msg']}") from exc
    listed = [file.path for file in project.files]
    for path in listed:
        _check_path(path)
    folded = {path.casefold() for path in listed}
    if len(folded) != len(listed):
        raise ProjectError("project.json lists a file twice")  # also twice up to case (Windows, macOS)
    if folded & {str(folder).casefold() for path in listed for folder in PurePosixPath(path).parents}:
        raise ProjectError("project.json lists a file where it also lists a folder")
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
    except (zipfile.BadZipFile, EOFError) as exc:
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
    try:
        with archive.open(file.path) as src:
            while chunk := src.read(_CHUNK):
                size += len(chunk)
                if size > file.bytes:
                    raise ProjectError(f"{file.path} is larger than its project.json says")
                digest.update(chunk)
                out.write(chunk)
    except _DAMAGED as exc:
        raise ProjectError(f"{file.path} is damaged or encrypted ({exc})") from exc
    if size != file.bytes or digest.hexdigest() != file.sha256:
        raise ProjectError(f"{file.path} is damaged (its content does not match project.json)")


def _enumerable(annotation: object) -> bool:
    """A setting type every value of which is harmless to take from someone else: a number, a switch or one of
    fixed choices — never free text such as a URL, a file name, a model id or a font (nor an optional one)."""
    return annotation in (bool, int, float) or get_origin(annotation) is Literal


def _setting(section: str, key: str, value: Any) -> tuple[bool, Any]:
    """Whether `[section] key = value` from someone else's series.toml is taken, and its validated value."""
    model = Config.model_fields[section].annotation if section in SERIES_SECTIONS else None
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return False, None
    field = model.model_fields.get(key)
    if field is None or not _enumerable(field.annotation) or key.endswith("batch_size"):
        return False, None  # batch sizes fit the sender's GPU memory, not the receiver's
    try:
        return True, getattr(model.model_validate({key: value}), key)
    except ValidationError:
        return False, None


def safe_series_settings(text: str) -> tuple[str, list[str]]:
    """Someone else's series.toml reduced to the settings safe to take (numbers, switches, fixed choices), as
    TOML ("" when none is left), and the settings left out as "section.key". ProjectError when it is not TOML."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"series/series.toml is not valid TOML: {exc}") from exc
    kept: dict[str, dict[str, Any]] = {}
    dropped: list[str] = []
    for section, table in data.items():
        if not isinstance(table, dict):
            dropped.append(section)
            continue
        for key, value in table.items():
            ok, valid = _setting(section, key, value)
            if ok:
                kept.setdefault(section, {})[key] = valid
            else:
                dropped.append(f"{section}.{key}")
    if not dropped:
        return text, []  # nothing left out: the sender's own file, comments and all
    return (dumps_toml(kept) if kept else ""), dropped


def _series_file(name: str, data: bytes) -> tuple[str, list[str]]:
    """A received series file as the text to install ("" = none) and the settings left out; ProjectError when
    it is not a valid series file."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectError(f"series/{name} is not UTF-8 text") from exc
    if name == "series.toml":
        return safe_series_settings(text)
    try:
        parse_voices(text)
    except ValueError as exc:
        raise ProjectError(f"series/{exc}") from exc
    return text, []


def _check_references(stages: Mapping[str, Path]) -> None:
    """ProjectError when a received artifact names a file by anything but a plain file name: the stages join
    these names onto the chapter's folders, so '../…' would make them read or list files outside the chapter."""
    for member, key, field in _NAMED_FILES:
        part, _, rest = member.partition("/")
        root = stages.get(part)
        path = root / rest if root is not None else None
        if path is None or not path.is_file():
            continue
        try:
            data = json.loads(path.read_bytes())
        except ValueError as exc:
            raise ProjectError(f"{member} is not valid JSON") from exc
        items = data.get(key, []) if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ProjectError(f"{member} is not valid")
        for item in items:
            name = item.get(field) if field is not None and isinstance(item, dict) else item
            if not isinstance(name, str) or not _plain(name):
                raise ProjectError(f"{member} names a file outside the chapter: {name!r}")


def _has_files(root: Path) -> bool:
    """`root` holds at least one file (at any depth)."""
    return root.is_dir() and any(path.is_file() for path in root.rglob("*"))


def _swap_in(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Move every (staged folder, target) into place together. The old targets are set aside first and put back
    when a move fails, so a chapter is never left half old and half new; they are deleted once all moves worked.
    ProjectError when an earlier unpack left a set-aside folder behind (it may be the only copy, so it is kept)."""
    olds = [target.with_name(f".{target.name}.replaced") for _, target in pairs]
    for old in olds:
        if old.exists():
            raise ProjectError(f"{old} is left from an earlier unpack: move it back or delete it first")
    set_aside: list[tuple[Path, Path]] = []
    placed: list[tuple[Path, Path]] = []
    try:
        for (stage, target), old in zip(pairs, olds, strict=True):
            if target.exists():
                target.rename(old)
                set_aside.append((target, old))
            target.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(target)
            placed.append((stage, target))
    except OSError:
        for stage, target in reversed(placed):
            target.rename(stage)
        for target, old in reversed(set_aside):
            old.rename(target)
        raise
    for _, old in set_aside:
        try:
            shutil.rmtree(old)
        except OSError as exc:
            log.warning("could not delete the replaced chapter folder %s (%s); delete it by hand", old, exc)


def unpack(
    path: Path, cfg: Config, *, series: str | None = None, chapter: str | None = None, force: bool = False
) -> Unpacked:
    """Unpack the chapter project at `path` into the library, work and output folders of `cfg` — as the series
    and chapter it was packed as, or as `series` / `chapter`. ProjectError for an archive that is not a safe
    chapter project, FileExistsError when the chapter exists already and `force` is not set (then its raw
    pages, work and finished pages are replaced)."""
    with _open(path) as archive:
        project = read_project(archive)
        series_paths = SeriesPaths.from_config(cfg, _check_name(series or project.series, "series"))
        paths = series_paths.chapter(_check_name(chapter or project.chapter, "chapter"))
        targets = {"raw": paths.raw_dir, "work": paths.work_dir, "output": paths.output_dir}
        exists = any(_has_files(root) for root in targets.values())
        if exists and not force:
            raise FileExistsError(f"{paths.series}/{paths.chapter} exists already (force replaces it)")
        with_output = any(file.path.startswith("output/") for file in project.files)
        parts = ["raw", "work"] + (["output"] if with_output or targets["output"].exists() else [])
        stages = {part: targets[part].with_name(f".{targets[part].name}.unpacking") for part in parts}
        series_text: dict[str, str] = {}
        dropped: list[str] = []
        try:
            for stage in stages.values():
                shutil.rmtree(stage, ignore_errors=True)
                stage.mkdir(parents=True)  # an unprocessed chapter has no work files yet
            for file in project.files:
                part, _, rest = file.path.partition("/")
                if part == "series":
                    if file.bytes > MAX_SERIES_FILE_BYTES:
                        raise ProjectError(f"{file.path} is too large for a settings file")
                    buffer = io.BytesIO()
                    _copy(archive, file, buffer)
                    series_text[rest], left_out = _series_file(rest, buffer.getvalue())
                    dropped += left_out
                    continue
                dest = stages[part] / rest
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as out:
                    _copy(archive, file, out)
            _check_references(stages)
            _swap_in([(stages[part], targets[part]) for part in parts])
        finally:
            for stage in stages.values():
                shutil.rmtree(stage, ignore_errors=True)
    added, kept = [], []
    for name, text in series_text.items():
        target = series_paths.library_dir / name
        if target.exists():
            kept.append(name)
        elif text:
            target.write_text(text, encoding="utf-8", newline="\n")
            added.append(name)
    took_settings = "series.toml" not in kept
    return Unpacked(
        paths=paths,
        series_files=added,
        kept_series_files=kept,
        replaced=exists,
        dropped_settings=dropped if took_settings else [],
    )
