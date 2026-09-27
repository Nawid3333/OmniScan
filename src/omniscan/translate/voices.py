"""Character voices: how each character of a series talks, and who says a line (`Region.speaker`).

A series' `voices.toml` sits next to its `series.toml` in the library, written by hand:

    [[character]]
    name = "Jinwoo"
    aliases = ["진우", "성진우"]
    voice = "calm and terse; plain speech"

The chat_json translation prompt shows the voices of the characters who speak in a request or are named in it,
and every region's speaker, so each character keeps one voice across a chapter and across the series.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import Region

VOICES_FILE = "voices.toml"


@dataclass(frozen=True, slots=True)
class Character:
    """One character of a series: the name used as a region's speaker, other names, and how they talk."""

    name: str
    aliases: tuple[str, ...] = ()
    voice: str = ""

    def names(self) -> tuple[str, ...]:
        """The character's name and aliases."""
        return (self.name, *self.aliases)


def parse_voices(text: str) -> list[Character]:
    """The characters of a voices.toml text; ValueError naming what is wrong."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{VOICES_FILE}: invalid TOML: {exc}") from exc
    tables = data.get("character", [])
    if not isinstance(tables, list):
        raise ValueError(f"{VOICES_FILE}: 'character' must be a list of [[character]] tables")
    characters: list[Character] = []
    for index, table in enumerate(tables, start=1):
        name = table.get("name") if isinstance(table, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{VOICES_FILE}: character {index} has no name")
        aliases = table.get("aliases", [])
        voice = table.get("voice", "")
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError(f"{VOICES_FILE}: {name}: 'aliases' must be a list of strings")
        if not isinstance(voice, str):
            raise ValueError(f"{VOICES_FILE}: {name}: 'voice' must be a string")
        characters.append(
            Character(name.strip(), tuple(a.strip() for a in aliases if a.strip()), voice.strip())
        )
    return characters


def load_voices(series: SeriesPaths) -> list[Character]:
    """The series' characters (voices.toml in its library folder), or none when it has no such file."""
    path = series.library_dir / VOICES_FILE
    return parse_voices(path.read_text(encoding="utf-8")) if path.is_file() else []


def character_named(name: str | None, characters: Sequence[Character]) -> Character | None:
    """The character a speaker name refers to (its name or an alias, case-insensitive), or None."""
    if not name:
        return None
    wanted = name.strip().casefold()
    return next((c for c in characters if any(n.casefold() == wanted for n in c.names())), None)


def voice_key(region: Region, characters: Sequence[Character]) -> dict[str, str]:
    """What a region's translation depends on through its speaker: the speaker and their voice ({} when the
    region has no speaker, so the key of an unassigned line never changes)."""
    if not region.speaker:
        return {}
    character = character_named(region.speaker, characters)
    return {"speaker": region.speaker, "voice": character.voice if character is not None else ""}


def relevant_characters(
    regions: Sequence[Region], characters: Sequence[Character], texts: Iterable[str] = ()
) -> list[Character]:
    """The characters (with a voice) who speak one of `regions` or are named in their text or in `texts`,
    in the file's order."""
    speakers = {id(c) for r in regions if (c := character_named(r.speaker, characters)) is not None}
    corpus = " ".join([*(r.text for r in regions), *texts])
    return [
        c for c in characters if c.voice and (id(c) in speakers or any(n and n in corpus for n in c.names()))
    ]
