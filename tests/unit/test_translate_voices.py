"""Tests for speakers and character voices (translate/voices.py) through the prompt, the keys and the edits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import omniscan.edits.cli as edit_cli
from omniscan.core.config import Config, PathsConfig
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import (
    BBox,
    ChapterEdits,
    Region,
    RegionEdit,
    RegionsArtifact,
    Slice,
    SlicesArtifact,
)
from omniscan.edits import store
from omniscan.edits.apply import apply_region_edits
from omniscan.edits.session import StudioSession
from omniscan.llm.ollama import ChatResponse
from omniscan.translate.incremental import translation_key
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import chat_json_messages
from omniscan.translate.run import run_profile
from omniscan.translate.voices import (
    Character,
    character_named,
    load_voices,
    parse_voices,
    relevant_characters,
    voice_key,
)

VOICES = """
[[character]]
name = "Jinwoo"
aliases = ["진우", "성진우"]
voice = "calm and terse; plain speech"

[[character]]
name = "Jinah"
aliases = ["진아"]
voice = "cheerful, teases her brother"

[[character]]
name = "Extra"
"""
JINWOO = Character("Jinwoo", ("진우", "성진우"), "calm and terse; plain speech")
JINAH = Character("Jinah", ("진아",), "cheerful, teases her brother")
CHARACTERS = [JINWOO, JINAH, Character("Extra")]
PROFILE = TranslationProfile(name="p", endpoint="local", model="m", style="chat_json")


def region(rid: str, text: str, *, speaker: str | None = None, y0: int = 0) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=0, y0=y0, x1=50, y1=y0 + 40),
        text=text,
        speaker=speaker,
    )


# ---------------------------------------------------------------- voices.toml


def test_voices_toml_is_read_and_checked(tmp_path: Path) -> None:
    assert parse_voices(VOICES) == CHARACTERS
    series = SeriesPaths("S", tmp_path / "lib" / "S", tmp_path / "work" / "S", tmp_path / "out" / "S")
    assert load_voices(series) == []
    series.library_dir.mkdir(parents=True)
    (series.library_dir / "voices.toml").write_text(VOICES, encoding="utf-8")
    assert load_voices(series) == CHARACTERS
    for bad, message in (
        ("[[character]\n", "invalid TOML"),
        ("character = 3\n", "list of"),
        ("[[character]]\nvoice = 'x'\n", "character 1 has no name"),
        ("[[character]]\nname = 'A'\naliases = 'B'\n", "'aliases' must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_voices(bad)


def test_speaker_names_find_their_character() -> None:
    assert character_named("jinwoo", CHARACTERS) is JINWOO
    assert character_named(" 성진우 ", CHARACTERS) is JINWOO
    assert character_named("Hae-in", CHARACTERS) is None and character_named(None, CHARACTERS) is None


def test_relevant_characters_are_the_speakers_and_the_named_ones_with_a_voice() -> None:
    chunk = [region("r1", "밥 먹었어?", speaker="Jinah"), region("r2", "진우 형!")]
    assert relevant_characters(chunk, CHARACTERS) == [JINWOO, JINAH]
    assert relevant_characters([region("r1", "hi")], CHARACTERS, ["진아야"]) == [
        JINAH
    ]  # named in the context
    assert relevant_characters([region("r1", "Extra", speaker="Extra")], CHARACTERS) == []  # no voice to show


# ---------------------------------------------------------------- prompt and keys


def test_the_prompt_names_speakers_and_voices_only_when_there_are_some() -> None:
    plain = [region("r1", "가자")]
    assert chat_json_messages(plain, []) == chat_json_messages(plain, [], characters=())
    assert "Characters" not in chat_json_messages(plain, [])[1]["content"]
    spoken = [region("r1", "가자", speaker="Jinwoo"), region("r2", "응")]
    user = chat_json_messages(spoken, [], characters=[JINWOO])[1]["content"]
    assert "- Jinwoo (진우, 성진우): calm and terse; plain speech" in user
    assert user.index("Characters") < user.index("Regions (reading order):")
    listed = json.loads(user.split("Regions (reading order):\n", 1)[1])
    assert listed == [
        {"id": "r1", "kind": "bubble_text", "text": "가자", "speaker": "Jinwoo"},
        {"id": "r2", "kind": "bubble_text", "text": "응"},
    ]


def test_a_speaker_or_its_voice_changes_only_that_lines_key() -> None:
    plain = region("r1", "가자")
    assert translation_key(plain, [], PROFILE) == translation_key(plain, [], PROFILE, CHARACTERS)
    assert voice_key(plain, CHARACTERS) == {}
    spoken = plain.model_copy(update={"speaker": "Jinwoo"})
    keys = {
        translation_key(plain, [], PROFILE),
        translation_key(spoken, [], PROFILE, CHARACTERS),
        translation_key(spoken, [], PROFILE, [Character("Jinwoo", (), "loud")]),
        translation_key(spoken.model_copy(update={"speaker": "Jinah"}), [], PROFILE, CHARACTERS),
    }
    assert len(keys) == 4
    assert voice_key(spoken.model_copy(update={"speaker": "Nobody"}), CHARACTERS) == {
        "speaker": "Nobody",
        "voice": "",
    }


class RecordingClient:
    """Answers chat_json with a line per region; keeps every request's user message."""

    def __init__(self) -> None:
        self.users: list[str] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **_: Any) -> ChatResponse:
        user = messages[-1]["content"]
        self.users.append(user)
        ids = [r["id"] for r in json.loads(user.split("Regions (reading order):\n", 1)[1])]
        content = json.dumps({"translations": [{"id": i, "text": f"T {i}"} for i in ids]})
        return ChatResponse(content, model, True, None, 1, 1, {})


def test_each_request_shows_the_voices_of_its_own_characters() -> None:
    regions = [
        region("r1", "진우야", speaker="Jinah"),
        region("r2", "응", y0=100),
        region("r3", "배고파", y0=200),
    ]
    client = RecordingClient()
    run_profile(client, PROFILE.model_copy(update={"chunk_regions": 2}), regions, [], characters=CHARACTERS)
    first, second = client.users
    assert "Jinah (진아)" in first and "Jinwoo (진우, 성진우)" in first  # the speaker, and the one she calls
    assert "Characters" not in second  # nobody speaks or is named in r3


# ---------------------------------------------------------------- edits, the Studio session, the CLI


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    paths = ChapterPaths(
        "S",
        "Chapter 1",
        tmp_path / "lib" / "S" / "Chapter 1",
        tmp_path / "work" / "S" / "Chapter 1",
        tmp_path / "out" / "S" / "Chapter 1",
        tmp_path / "out" / "S" / "_filtered" / "Chapter 1",
    )
    paths.raw_dir.mkdir(parents=True)
    SlicesArtifact(strip_width=200, strip_height=600, bands=[], slices=[Slice(index=0, y0=0, y1=600)]).save(
        paths.artifact("slices.json")
    )
    RegionsArtifact(regions=[region("r0001", "가자"), region("r0002", "응", y0=200)]).save(
        paths.artifact("ocr.json")
    )
    return paths


def test_a_hand_set_speaker_survives_a_rerun_and_can_be_cleared(paths: ChapterPaths) -> None:
    assert store.update_region(paths, "r0001", direction="ltr", speaker="Jinwoo").speaker == "Jinwoo"
    edits = store.load_edits(paths)
    rerun, _ = apply_region_edits(
        store.auto_regions(paths), edits, [], direction="ltr"
    )  # the ocr stage again
    assert [r.speaker for r in rerun] == ["Jinwoo", None]
    assert store.update_region(paths, "r0001", direction="ltr", speaker="").speaker is None
    cleared = ChapterEdits(
        regions=[RegionEdit(region_id="r0001", anchor=BBox(x0=0, y0=0, x1=50, y1=40), speaker="")]
    )
    assert (
        apply_region_edits([region("r0001", "가자", speaker="X")], cleared, [], direction="ltr")[0][0].speaker
        is None
    )


def test_the_studio_session_sets_speakers(paths: ChapterPaths) -> None:
    session = StudioSession(paths, direction="ltr")
    session.set_speaker("r0002", "Jinah")
    assert session.dirty and [row.speaker for row in session.rows()] == ["", "Jinah"]
    session.set_speaker("r0002", "")  # back to nobody before saving: nothing to save
    assert not session.dirty
    session.set_speaker("r0002", "Jinah")
    session.set_source("r0002", "응!")
    assert session.save() == 2
    saved = {r.id: r for r in store.current_regions(paths)}["r0002"]
    assert (saved.speaker, saved.text) == ("Jinah", "응!")
    reopened = StudioSession(paths, direction="ltr")
    assert reopened.rows()[1].speaker == "Jinah" and reopened.rows()[1].edited
    with pytest.raises(KeyError):
        reopened.set_speaker("r0009", "x")


def test_edit_speaker_on_the_command_line(
    paths: ChapterPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib", work_root=tmp_path / "work", output_root=tmp_path / "out"
        )
    )
    monkeypatch.setattr(edit_cli, "get_config", lambda: cfg)
    (tmp_path / "lib" / "S" / "voices.toml").write_text(VOICES, encoding="utf-8")
    runner = CliRunner()
    args = ["speaker", "S", "Chapter 1", "r0001"]
    assert runner.invoke(edit_cli.edit_app, [*args, "진우"]).output == "edit: r0001 is said by 진우\n"
    assert "not in voices.toml" in runner.invoke(edit_cli.edit_app, [*args, "Hae-in"]).output
    shown = json.loads(runner.invoke(edit_cli.edit_app, ["show", "S", "Chapter 1", "--json"]).output)
    assert shown["regions"][0]["speaker"] == "Hae-in"
    assert runner.invoke(edit_cli.edit_app, [*args, ""]).output == "edit: r0001 has no speaker\n"
