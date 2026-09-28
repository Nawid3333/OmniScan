"""Tests for hand edits of a series glossary (#53): omniscan.glossary.edit on a real sqlite db, no GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import GlossaryEntry
from omniscan.glossary.edit import (
    DuplicateTermError,
    TermError,
    add_term,
    find_term,
    remove_term,
    set_status,
    update_term,
)
from omniscan.glossary.store import GlossaryStore


def series(tmp_path: Path, name: str = "S") -> SeriesPaths:
    """A series with a library folder and nothing else yet (no work dir, no db)."""
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    paths = SeriesPaths.from_config(cfg, name)
    paths.library_dir.mkdir(parents=True)
    return paths


def stored(paths: SeriesPaths) -> list[GlossaryEntry]:
    with GlossaryStore(paths.db) as store:
        return store.list()


def exported(paths: SeriesPaths) -> list[dict[str, object]]:
    return yaml.safe_load(paths.glossary_yaml.read_text(encoding="utf-8"))


def machine_entry(paths: SeriesPaths, **fields: object) -> GlossaryEntry:
    """An entry the proposal pass wrote (origin llm, proposed), straight into the store."""
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {"source": "헌터", "target": "Hunter", "origin": "llm", "count": 4, **fields}
    with GlossaryStore(paths.db) as store:
        return store.add(GlossaryEntry.model_validate(values))


def test_add_term_locks_a_user_entry_and_exports_the_yaml(tmp_path: Path) -> None:
    paths = series(tmp_path)
    entry = add_term(
        paths,
        "  성진우 ",
        "Sung Jinwoo",
        type="person",
        notes=" main character ",
        aliases=["진우", " 진우 ", ""],
    )
    assert entry.id is not None
    assert (entry.source, entry.target, entry.type) == ("성진우", "Sung Jinwoo", "person")
    assert (entry.status, entry.origin, entry.notes, entry.aliases) == (
        "locked",
        "user",
        "main character",
        ["진우"],
    )
    assert stored(paths) == [entry]
    assert exported(paths) == [entry.model_dump(mode="json")]


def test_add_term_as_a_proposal_without_notes(tmp_path: Path) -> None:
    paths = series(tmp_path)
    entry = add_term(paths, "게이트", "Gate", status="proposed", notes="   ")
    assert (entry.status, entry.notes, entry.type, entry.aliases) == ("proposed", None, "other", [])


def test_words_are_collapsed_to_single_spaces(tmp_path: Path) -> None:
    paths = series(tmp_path)
    entry = add_term(paths, "헌터\n협회", "Hunter  Association")
    assert (entry.source, entry.target) == ("헌터 협회", "Hunter Association")


@pytest.mark.parametrize(("source", "target", "what"), [(" ", "Gate", "source"), ("게이트", "\n", "target")])
def test_empty_words_are_refused(tmp_path: Path, source: str, target: str, what: str) -> None:
    paths = series(tmp_path)
    with pytest.raises(TermError, match=f"the {what} is empty"):
        add_term(paths, source, target)
    assert not paths.db.exists() or stored(paths) == []


def test_a_source_already_in_the_glossary_is_refused(tmp_path: Path) -> None:
    paths = series(tmp_path)
    first = add_term(paths, "게이트", "Gate")
    with pytest.raises(
        DuplicateTermError, match=rf"'게이트' is already in the glossary \(id {first.id}, locked\)"
    ):
        add_term(paths, " 게이트", "Portal")
    assert stored(paths) == [first]


def test_a_series_not_in_the_library_gets_no_folder(tmp_path: Path) -> None:
    paths = series(tmp_path)
    ghost = SeriesPaths.from_config(
        Config(paths=PathsConfig(library_root=tmp_path / "lib", work_root=tmp_path / "work")), "Ghost"
    )
    with pytest.raises(FileNotFoundError, match="no series 'Ghost' in the library"):
        add_term(ghost, "게이트", "Gate")
    assert not ghost.work_dir.exists()
    assert paths.library_dir.is_dir()


def test_new_words_make_a_machine_entry_the_users(tmp_path: Path) -> None:
    paths = series(tmp_path)
    proposal = machine_entry(paths, notes="seen in 4 chapters", aliases=["헌터들"])
    assert proposal.id is not None
    changed = update_term(paths, proposal.id, target=" Hunters ")
    assert (changed.target, changed.origin, changed.status) == ("Hunters", "user", "proposed")
    assert (changed.source, changed.count, changed.notes, changed.aliases) == (
        "헌터",
        4,
        "seen in 4 chapters",
        ["헌터들"],
    )
    assert stored(paths) == [changed]
    assert exported(paths) == [changed.model_dump(mode="json")]


def test_same_words_or_a_status_change_keep_the_origin(tmp_path: Path) -> None:
    paths = series(tmp_path)
    proposal = machine_entry(paths)
    assert proposal.id is not None
    same = update_term(paths, proposal.id, source="헌터", target="Hunter", type="rank", status="locked")
    assert (same.origin, same.type, same.status) == ("llm", "rank", "locked")
    renamed = update_term(paths, proposal.id, source="헌터들")
    assert (renamed.source, renamed.target, renamed.origin) == ("헌터들", "Hunter", "user")


def test_notes_are_kept_changed_or_cleared(tmp_path: Path) -> None:
    paths = series(tmp_path)
    entry = add_term(paths, "게이트", "Gate", notes="dungeon entrance")
    assert entry.id is not None
    assert update_term(paths, entry.id, type="place").notes == "dungeon entrance"
    assert update_term(paths, entry.id, notes=" portal ").notes == "portal"
    assert update_term(paths, entry.id, notes="  ").notes is None


def test_aliases_are_replaced_or_kept_and_never_repeat_the_source(tmp_path: Path) -> None:
    paths = series(tmp_path)
    entry = add_term(paths, "성진우", "Sung Jinwoo", aliases=["진우", "성진우"])
    assert entry.id is not None
    assert entry.aliases == ["진우"]
    assert update_term(paths, entry.id, target="Jinwoo Sung").aliases == ["진우"]
    assert update_term(paths, entry.id, source="진우").aliases == []
    assert update_term(paths, entry.id, aliases=["성진우", "진우 씨"]).aliases == ["성진우", "진우 씨"]
    assert update_term(paths, entry.id, aliases=[""]).aliases == []


def test_renaming_onto_another_entrys_source_is_refused(tmp_path: Path) -> None:
    paths = series(tmp_path)
    gate = add_term(paths, "게이트", "Gate")
    hunter = add_term(paths, "헌터", "Hunter")
    assert hunter.id is not None
    with pytest.raises(DuplicateTermError, match="'게이트' is already in the glossary"):
        update_term(paths, hunter.id, source="게이트")
    assert update_term(paths, hunter.id, source="헌터", target="Hunters").target == "Hunters"
    assert [e.source for e in stored(paths)] == [gate.source, "헌터"]


def test_invalid_changes_leave_the_entry_as_it_was(tmp_path: Path) -> None:
    paths = series(tmp_path)
    entry = add_term(paths, "게이트", "Gate")
    assert entry.id is not None
    with pytest.raises(TermError, match="the target is empty"):
        update_term(paths, entry.id, target=" ", status="rejected")
    with pytest.raises(ValueError, match="type"):
        update_term(paths, entry.id, type="planet")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="no entry 99"):
        update_term(paths, 99, target="Portal")
    assert stored(paths) == [entry]


def test_set_status_changes_every_entry_or_none(tmp_path: Path) -> None:
    paths = series(tmp_path)
    first = machine_entry(paths)
    second = machine_entry(paths, source="마석", target="mana stone")
    assert first.id is not None and second.id is not None
    with pytest.raises(KeyError, match="no entry 7"):
        set_status(paths, [first.id, 7], "locked")
    assert [e.status for e in stored(paths)] == ["proposed", "proposed"]
    locked = set_status(paths, [first.id, second.id], "locked")
    assert [(e.id, e.status, e.origin, e.target) for e in locked] == [
        (first.id, "locked", "llm", "Hunter"),
        (second.id, "locked", "llm", "mana stone"),
    ]
    assert stored(paths) == locked
    assert [e["status"] for e in exported(paths)] == ["locked", "locked"]
    assert [e.status for e in set_status(paths, [second.id], "rejected")] == ["rejected"]
    assert [e.status for e in stored(paths)] == ["locked", "rejected"]


def test_remove_term_returns_what_it_deleted(tmp_path: Path) -> None:
    paths = series(tmp_path)
    gate = add_term(paths, "게이트", "Gate")
    hunter = add_term(paths, "헌터", "Hunter")
    assert gate.id is not None
    assert remove_term(paths, gate.id) == gate
    assert stored(paths) == [hunter]
    assert exported(paths) == [hunter.model_dump(mode="json")]
    with pytest.raises(KeyError, match=f"no entry {gate.id}"):
        remove_term(paths, gate.id)


def test_find_term_by_id_or_source(tmp_path: Path) -> None:
    paths = series(tmp_path)
    with pytest.raises(KeyError, match="no entry '게이트'"):
        find_term(paths, "게이트")
    assert not paths.work_dir.exists()  # a lookup makes no db
    gate = add_term(paths, "게이트", "Gate")
    number = add_term(paths, "一", "One")
    assert find_term(paths, f" {gate.id} ") == gate
    assert find_term(paths, " 게이트\n") == gate
    assert find_term(paths, "一") == number  # a numeral that is not a decimal digit is a source text
    with pytest.raises(KeyError, match="no entry '42'"):
        find_term(paths, "42")
