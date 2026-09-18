"""Tests for omniscan.glossary.yaml_io (tmp_path yaml files)."""

from __future__ import annotations

from pathlib import Path

import yaml

from omniscan.core.schemas import GlossaryEntry
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml, import_yaml


def entry(**overrides: object) -> GlossaryEntry:
    """A GlossaryEntry with the given field overrides."""
    fields: dict[str, object] = {"source": "지훈", "target": "Jihoon"}
    fields.update(overrides)
    return GlossaryEntry(**fields)  # type: ignore[arg-type]


def sample_entries() -> list[GlossaryEntry]:
    return [
        entry(
            source="지훈",
            target="Jihoon",
            type="person",
            gender="male",
            pronouns="he/him",
            aliases=["훈이형"],
            notes="protagonist",
            status="locked",
            origin="reference",
            first_seen_chapter=1.5,
            count=12,
        ),
        entry(source="학교", target="school", type="place", first_seen_chapter=2.0, count=4),
        entry(source="묵룡문", target="Muryongmun", type="org", status="rejected", origin="llm"),
    ]


def test_replace_round_trips_every_field(tmp_path: Path) -> None:
    yaml_path = tmp_path / "glossary.yaml"
    with GlossaryStore(tmp_path / "src.db") as store:
        for e in sample_entries():
            store.add(e)
        export_yaml(store, yaml_path)
    with GlossaryStore(tmp_path / "dst.db") as store:
        assert import_yaml(store, yaml_path, mode="replace") == 3
        imported = {e.source: e for e in store.list()}
        for original in sample_entries():
            got = imported[original.source]
            expected = original.model_copy(update={"id": got.id})
            assert got == expected


def test_merge_updates_existing_by_source_in_place(tmp_path: Path) -> None:
    yaml_path = tmp_path / "glossary.yaml"
    with GlossaryStore(tmp_path / "series.db") as store:
        added = store.add(entry(source="지훈", target="Jihoon"))
        export_yaml(store, yaml_path)
        text = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        text[0]["target"] = "Ji-hoon"
        yaml_path.write_text(yaml.safe_dump(text, allow_unicode=True), encoding="utf-8")

        assert import_yaml(store, yaml_path, mode="merge") == 1
        fetched = store.find_by_source("지훈")
        assert fetched is not None
        assert fetched.id == added.id
        assert fetched.target == "Ji-hoon"
        assert len(store.list()) == 1


def test_merge_adds_new_sources_and_counts_writes(tmp_path: Path) -> None:
    yaml_path = tmp_path / "glossary.yaml"
    yaml_path.write_text(
        yaml.safe_dump([e.model_dump(mode="json") for e in sample_entries()], allow_unicode=True),
        encoding="utf-8",
    )
    with GlossaryStore(tmp_path / "series.db") as store:
        assert store.list() == []
        assert import_yaml(store, yaml_path, mode="merge") == 3
        assert len(store.list()) == 3


def test_hand_edited_yaml_tweak_takes_effect(tmp_path: Path) -> None:
    yaml_path = tmp_path / "glossary.yaml"
    with GlossaryStore(tmp_path / "series.db") as store:
        store.add(entry(source="지훈", target="Jihoon"))
        export_yaml(store, yaml_path)

        text = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        text[0]["target"] = "Ji-hoon"
        text[0]["notes"] = "hand-edited"
        yaml_path.write_text(yaml.safe_dump(text, sort_keys=False, allow_unicode=True), encoding="utf-8")

        assert import_yaml(store, yaml_path, mode="merge") == 1
        fetched = store.find_by_source("지훈")
        assert fetched is not None
        assert fetched.target == "Ji-hoon" and fetched.notes == "hand-edited"


def test_replace_ignores_file_ids(tmp_path: Path) -> None:
    yaml_path = tmp_path / "glossary.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            [entry(id=77, source="지훈", target="Jihoon").model_dump(mode="json")], allow_unicode=True
        ),
        encoding="utf-8",
    )
    with GlossaryStore(tmp_path / "series.db") as store:
        import_yaml(store, yaml_path, mode="replace")
        fetched = store.find_by_source("지훈")
        assert fetched is not None
        assert fetched.id != 77
