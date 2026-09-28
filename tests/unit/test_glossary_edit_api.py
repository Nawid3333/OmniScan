"""Tests for the glossary hand-edit surfaces (#53): `omniscan glossary add|set|lock|reject|remove|list` and the web
API's POST/PATCH/DELETE on /api/series/{series}/glossary."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import omniscan.cli
from omniscan.cli import app
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import GlossaryEntry
from omniscan.glossary.edit import update_term
from omniscan.glossary.reference import AggregatedTerm, merge_into_store
from omniscan.glossary.store import GlossaryStore
from omniscan.web.app import create_app

SERIES = "Solo Leveling"
runner = CliRunner()


def make_cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SeriesPaths:
    """The series (library folder only) of a config the CLI uses too."""
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(omniscan.cli, "get_config", lambda: cfg)
    series = SeriesPaths.from_config(cfg, SERIES)
    series.library_dir.mkdir(parents=True)
    return series


def stored(paths: SeriesPaths) -> list[GlossaryEntry]:
    with GlossaryStore(paths.db) as store:
        return store.list()


def proposal(paths: SeriesPaths, source: str, target: str) -> GlossaryEntry:
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    with GlossaryStore(paths.db) as store:
        return store.add(GlossaryEntry(source=source, target=target, origin="llm"))


def cli(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["glossary", *args])
    return result.exit_code, result.output


def test_cli_add_locks_by_default_and_list_shows_ids(paths: SeriesPaths) -> None:
    code, out = cli(
        "add", SERIES, "성진우", "Sung Jinwoo", "--type", "person", "--alias", "진우", "--notes", "hero"
    )
    assert (code, out) == (0, "glossary: added 1: 성진우 -> Sung Jinwoo (person, locked)\n")
    code, out = cli("add", SERIES, "게이트", "Gate", "--proposed")
    assert (code, out) == (0, "glossary: added 2: 게이트 -> Gate (other, proposed)\n")
    entries = stored(paths)
    assert [(e.aliases, e.notes, e.origin) for e in entries] == [
        (["진우"], "hero", "user"),
        ([], None, "user"),
    ]
    code, out = cli("list", SERIES)
    assert code == 0
    rows = [line for line in out.splitlines() if "Source" in line or "Sung Jinwoo" in line or "Gate" in line]
    assert [re.sub(r"[│┃|]", " ", row).split()[0] for row in rows] == [
        "ID",
        "1",
        "2",
    ]  # the ID column


def test_cli_add_refuses_a_duplicate_empty_words_and_an_unknown_series(paths: SeriesPaths) -> None:
    assert cli("add", SERIES, "게이트", "Gate")[0] == 0
    code, out = cli("add", SERIES, "게이트", "Portal")
    assert (code, out) == (2, "glossary: '게이트' is already in the glossary (id 1, locked)\n")
    assert cli("add", SERIES, "헌터", " ") == (2, "glossary: the target is empty\n")
    assert cli("add", "Ghost", "헌터", "Hunter") == (2, "glossary: no series 'Ghost' in the library\n")
    assert [e.target for e in stored(paths)] == ["Gate"]


def test_cli_set_by_source_or_id(paths: SeriesPaths) -> None:
    entry = proposal(paths, "헌터", "Hunter")
    code, out = cli("set", SERIES, "헌터", "--target", "Hunters", "--type", "rank", "--notes", "plural")
    assert (code, out) == (0, f"glossary: {entry.id}: 헌터 -> Hunters (rank, proposed)\n")
    code, out = cli("set", SERIES, str(entry.id), "--source", "헌터들", "--alias", "헌터", "--notes", "")
    assert code == 0, out
    [changed] = stored(paths)
    assert (changed.source, changed.target, changed.origin, changed.notes, changed.aliases) == (
        "헌터들",
        "Hunters",
        "user",
        None,
        ["헌터"],
    )
    assert cli("set", SERIES, "헌터들", "--alias", "")[0] == 0
    assert stored(paths)[0].aliases == []


def test_cli_set_errors(paths: SeriesPaths) -> None:
    proposal(paths, "헌터", "Hunter")
    proposal(paths, "게이트", "Gate")
    assert cli("set", SERIES, "마석", "--target", "x") == (2, "glossary: the glossary has no entry '마석'\n")
    code, out = cli("set", SERIES, "게이트", "--source", "헌터")
    assert (code, out) == (2, "glossary: '헌터' is already in the glossary (id 1, proposed)\n")
    assert [e.source for e in stored(paths)] == ["헌터", "게이트"]


def test_cli_lock_and_reject_several_at_once(paths: SeriesPaths) -> None:
    hunter = proposal(paths, "헌터", "Hunter")
    gate = proposal(paths, "게이트", "Gate")
    code, out = cli("lock", SERIES, "헌터", str(gate.id))
    assert (code, out) == (
        0,
        f"glossary: {hunter.id}: 헌터 -> Hunter (other, locked)\nglossary: {gate.id}: 게이트 -> Gate (other, locked)\n",
    )
    assert cli("reject", SERIES, "게이트")[0] == 0
    assert [(e.status, e.origin) for e in stored(paths)] == [("locked", "llm"), ("rejected", "llm")]
    assert cli("lock", SERIES, "헌터", "마석") == (2, "glossary: the glossary has no entry '마석'\n")
    assert cli("reject", SERIES, "헌터", "99") == (2, "glossary: the glossary has no entry '99'\n")
    assert [e.status for e in stored(paths)] == ["locked", "rejected"]


def test_cli_remove(paths: SeriesPaths) -> None:
    hunter = proposal(paths, "헌터", "Hunter")
    assert cli("remove", SERIES, "헌터") == (
        0,
        f"glossary: removed {hunter.id}: 헌터 -> Hunter (other, proposed)\n",
    )
    assert stored(paths) == []
    assert cli("remove", SERIES, "헌터") == (2, "glossary: the glossary has no entry '헌터'\n")


def test_a_later_proposal_pass_keeps_the_users_target(paths: SeriesPaths) -> None:
    entry = proposal(paths, "헌터", "Hunter")
    assert entry.id is not None
    update_term(paths, entry.id, target="Awakened")
    term = AggregatedTerm(
        source="헌터",
        target="Hunter",
        type="other",
        chapters=3,
        occurrences=9,
        first_seen_chapter=1.0,
        alternatives=(),
    )
    with GlossaryStore(paths.db) as store:
        report = merge_into_store(store, [term], min_locks=99, origin="llm")
    assert [(c.source, c.existing_target) for c in report.conflicts] == [("헌터", "Awakened")]
    assert [(e.target, e.origin, e.count) for e in stored(paths)] == [("Awakened", "user", 0)]


def glossary_url(entry_id: int | None = None) -> str:
    base = f"/api/series/{quote(SERIES)}/glossary"
    return base if entry_id is None else f"{base}/{entry_id}"


def test_post_adds_a_locked_user_term(tmp_path: Path, paths: SeriesPaths) -> None:
    client = TestClient(create_app(make_cfg(tmp_path)))
    response = client.post(glossary_url(), json={"source": " 게이트 ", "target": "Gate", "type": "place"})
    assert response.status_code == 201
    body = response.json()
    assert (body["id"], body["source"], body["status"], body["origin"], body["type"]) == (
        1,
        "게이트",
        "locked",
        "user",
        "place",
    )
    assert client.get(glossary_url()).json() == [body]
    proposed = client.post(glossary_url(), json={"source": "헌터", "target": "Hunter", "status": "proposed"})
    assert proposed.json()["status"] == "proposed"


def test_post_errors(tmp_path: Path, paths: SeriesPaths) -> None:
    client = TestClient(create_app(make_cfg(tmp_path)))
    assert client.post(glossary_url(), json={"source": "게이트", "target": "Gate"}).status_code == 201
    duplicate = client.post(glossary_url(), json={"source": "게이트", "target": "Portal"})
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "'게이트' is already in the glossary (id 1, locked)"
    empty = client.post(glossary_url(), json={"source": "헌터", "target": "  "})
    assert (empty.status_code, empty.json()["detail"]) == (422, "the target is empty")
    assert (
        client.post(glossary_url(), json={"source": "헌터", "target": "x", "type": "planet"}).status_code
        == 422
    )
    assert client.post(glossary_url(), content=b'{"source": "a", "target": "b"}').status_code == 415
    ghost = client.post("/api/series/Ghost/glossary", json={"source": "헌터", "target": "Hunter"})
    assert ghost.status_code == 404
    assert not (tmp_path / "work" / "Ghost").exists()
    assert (
        client.post("/api/series/..%2F..%2Fx/glossary", json={"source": "a", "target": "b"}).status_code
        == 404
    )
    assert [e.source for e in stored(paths)] == ["게이트"]


def test_patch_corrects_locks_and_clears(tmp_path: Path, paths: SeriesPaths) -> None:
    entry = proposal(paths, "헌터", "Hunter")
    client = TestClient(create_app(make_cfg(tmp_path)))
    status = client.patch(glossary_url(entry.id), json={"status": "locked"})
    assert status.status_code == 200
    assert (status.json()["status"], status.json()["origin"], status.json()["target"]) == (
        "locked",
        "llm",
        "Hunter",
    )
    words = client.patch(
        glossary_url(entry.id), json={"target": "Hunters", "notes": "n", "aliases": ["헌터들"]}
    )
    assert (words.json()["target"], words.json()["origin"], words.json()["aliases"]) == (
        "Hunters",
        "user",
        ["헌터들"],
    )
    cleared = client.patch(glossary_url(entry.id), json={"notes": "", "aliases": []})
    assert (cleared.json()["notes"], cleared.json()["aliases"], cleared.json()["status"]) == (
        None,
        [],
        "locked",
    )
    assert stored(paths)[0].model_dump(mode="json") == cleared.json()


def test_patch_errors(tmp_path: Path, paths: SeriesPaths) -> None:
    hunter = proposal(paths, "헌터", "Hunter")
    gate = proposal(paths, "게이트", "Gate")
    client = TestClient(create_app(make_cfg(tmp_path)))
    missing = client.patch(glossary_url(99), json={"target": "x"})
    assert (missing.status_code, missing.json()["detail"]) == (404, "no glossary entry 99")
    assert client.patch(glossary_url(gate.id), json={"source": "헌터"}).status_code == 409
    assert client.patch(glossary_url(gate.id), json={"source": " "}).status_code == 422
    assert client.patch(glossary_url(gate.id), json={"status": "maybe"}).status_code == 422
    assert client.patch(glossary_url(gate.id), json={"colour": "red"}).status_code == 422
    assert client.patch(f"/api/series/Ghost/glossary/{hunter.id}", json={"target": "x"}).status_code == 404
    assert stored(paths) == [hunter, gate]


def test_delete_returns_the_removed_entry(tmp_path: Path, paths: SeriesPaths) -> None:
    hunter = proposal(paths, "헌터", "Hunter")
    client = TestClient(create_app(make_cfg(tmp_path)))
    response = client.delete(glossary_url(hunter.id))
    assert (response.status_code, response.json()) == (200, hunter.model_dump(mode="json"))
    assert stored(paths) == []
    again = client.delete(glossary_url(hunter.id))
    assert (again.status_code, again.json()["detail"]) == (404, f"no glossary entry {hunter.id}")
