"""Hand edits of a series glossary: add a term, correct it, lock or reject it, remove it (#53).

The CLI (`omniscan glossary add|set|lock|reject|remove`), the web Studio's Glossary tab and the desktop app share
these functions. Words typed by hand are recorded as `origin="user"`, which the proposal and reference merges never
overwrite (glossary.reference.merge_into_store); a status change alone keeps the entry's origin. After every change
the series' glossary.yaml is written again, as `propose` and `reference` do, so a later `glossary import` of an older
copy cannot quietly undo a hand edit. The next translate/judge run redoes just the lines holding a changed term:
their incremental keys include the entries that match them (translate/incremental.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import GlossaryEntry, TermType
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml

TermStatus = Literal["proposed", "locked", "rejected"]


class TermError(ValueError):
    """A term that cannot be stored as asked: empty words, or a source text already in the glossary."""


class DuplicateTermError(TermError):
    """A source text that another entry of the glossary already has."""


def _open(series: SeriesPaths) -> GlossaryStore:
    if not series.library_dir.is_dir():
        raise FileNotFoundError(f"no series {series.series!r} in the library")
    series.work_dir.mkdir(parents=True, exist_ok=True)
    return GlossaryStore(series.db)


def _words(value: str, what: str) -> str:
    words = " ".join(value.split())  # a pasted line break or double space would never match the page's text
    if not words:
        raise TermError(f"the {what} is empty")
    return words


def _aliases(values: Sequence[str], source: str) -> list[str]:
    aliases: list[str] = []
    for value in values:
        alias = " ".join(value.split())
        if alias and alias != source and alias not in aliases:
            aliases.append(alias)
    return aliases


def _unique(store: GlossaryStore, source: str, entry_id: int | None) -> None:
    clash = store.find_by_source(source)
    if clash is not None and clash.id != entry_id:
        raise DuplicateTermError(f"{source!r} is already in the glossary (id {clash.id}, {clash.status})")


def _existing(store: GlossaryStore, entry_id: int) -> GlossaryEntry:
    entry = store.get(entry_id)
    if entry is None:
        raise KeyError(f"the glossary has no entry {entry_id}")
    return entry


def find_term(series: SeriesPaths, key: str) -> GlossaryEntry:
    """The entry `key` names: its id (digits) or its exact source text; KeyError when there is none."""
    entry = None
    if series.db.is_file():
        with GlossaryStore(series.db) as store:
            words = " ".join(key.split())
            entry = store.get(int(words)) if words.isdecimal() else store.find_by_source(words)
    if entry is None:
        raise KeyError(f"the glossary has no entry {key!r}")
    return entry


def add_term(
    series: SeriesPaths,
    source: str,
    target: str,
    *,
    type: TermType = "other",
    status: TermStatus = "locked",
    notes: str | None = None,
    aliases: Sequence[str] = (),
) -> GlossaryEntry:
    """Add a term typed by hand, locked unless `status` says otherwise, and return it with its id. TermError for
    empty words or a source already in the glossary; FileNotFoundError for a series not in the library."""
    words = _words(source, "source")
    entry = GlossaryEntry(
        source=words,
        target=_words(target, "target"),
        type=type,
        status=status,
        notes=(notes or "").strip() or None,
        aliases=_aliases(aliases, words),
        origin="user",
    )
    with _open(series) as store:
        _unique(store, entry.source, None)
        added = store.add(entry)
        export_yaml(store, series.glossary_yaml)
    return added


def update_term(
    series: SeriesPaths,
    entry_id: int,
    *,
    source: str | None = None,
    target: str | None = None,
    type: TermType | None = None,
    status: TermStatus | None = None,
    notes: str | None = None,
    aliases: Sequence[str] | None = None,
) -> GlossaryEntry:
    """Change an entry and return it; a field left None stays, `notes=""` clears the notes. New words make the
    entry the user's (never overwritten by a later proposal). KeyError for no such entry, TermError as add_term."""
    with _open(series) as store:
        entry = _existing(store, entry_id)
        values = entry.model_dump()
        if source is not None:
            values["source"] = _words(source, "source")
            _unique(store, values["source"], entry_id)
        if target is not None:
            values["target"] = _words(target, "target")
        if (values["source"], values["target"]) != (entry.source, entry.target):
            values["origin"] = "user"
        if type is not None:
            values["type"] = type
        if status is not None:
            values["status"] = status
        if notes is not None:
            values["notes"] = notes.strip() or None
        values["aliases"] = _aliases(entry.aliases if aliases is None else aliases, values["source"])
        changed = GlossaryEntry.model_validate(values)
        store.update(changed)
        export_yaml(store, series.glossary_yaml)
    return changed


def set_status(series: SeriesPaths, entry_ids: Sequence[int], status: TermStatus) -> list[GlossaryEntry]:
    """Lock, reject or re-propose several entries at once and return them; KeyError, before anything is changed,
    when one of them does not exist."""
    with _open(series) as store:
        entries = [_existing(store, entry_id) for entry_id in entry_ids]
        changed = [entry.model_copy(update={"status": status}) for entry in entries]
        for entry in changed:
            store.update(entry)
        export_yaml(store, series.glossary_yaml)
    return changed


def remove_term(series: SeriesPaths, entry_id: int) -> GlossaryEntry:
    """Delete an entry and return what it was; KeyError for no such entry. A later `glossary propose` may suggest
    the term again; rejecting it instead keeps it out."""
    with _open(series) as store:
        entry = _existing(store, entry_id)
        store.delete(entry_id)
        export_yaml(store, series.glossary_yaml)
    return entry
