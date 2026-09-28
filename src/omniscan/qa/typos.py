"""Typos in the translated lines: words the release language's dictionary does not know (pyspellchecker, offline;
English, German or Spanish after the series' `[translate] target_lang`).

The words of the series' glossary targets and character names (voices.toml) count as known, and so do the words
the editor marked "not a typo" (`typo_words.txt` in the series' library folder, one word per line). Sound effects
and watermarks are never checked; a capitalised word inside a sentence is taken for a name and left alone.
Nothing is ever changed: each typo carries the dictionary's closest words for the editor to take or dismiss.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache

from spellchecker import SpellChecker

from omniscan.core.config import SeriesConfigError, get_config, series_config
from omniscan.core.paths import SeriesPaths
from omniscan.qa.consistency import Line, glossary
from omniscan.translate.voices import load_voices

WORDS_FILE = "typo_words.txt"
MAX_SUGGESTIONS = 3

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_LETTERS_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")  # other releases: any letters (ä, ß, ñ, é ...)
_STRETCHED = re.compile(r"([a-z])\1\1")  # "Nooo", "Ahhh": drawn out on purpose
_SENTENCE_END = frozenset(".!?…")
_OPENERS = "\"'“”‘’([{-–—*~ \t\n"  # what may stand before a sentence's first word
_SUFFIXES = ("s", "d", "m", "re", "ve", "ll")  # Jinwoo's, I'd, I'm, we're, I've, you'll
_IRREGULAR = frozenset({"won't", "can't", "shan't", "ain't", "y'know", "o'clock", "ma'am"})
# interjections and romanised honorifics common in manhwa/manga English that a dictionary lacks
DEFAULT_WORDS = frozenset(
    {
        *("hmph", "heh", "hehe", "ahaha", "ugh", "argh", "urgh", "gah", "tch", "tsk", "eek", "kyaa", "hngh"),
        *("ngh", "pfft", "psst", "shh", "grr", "oof", "hm", "mhm", "mm", "uh", "um", "hah", "wha", "geez"),
        *(
            "jeez",
            "sheesh",
            "whoa",
            "yup",
            "nope",
            "okay",
            "ok",
            "oi",
            "huh",
            "ssi",
            "nim",
            "hyung",
            "hyungnim",
        ),
        *("noona", "nuna", "oppa", "unnie", "eonni", "sunbae", "sunbaenim", "hoobae", "ahjussi", "ajussi"),
        *("ahjumma", "ajumma", "halmeoni", "harabeoji", "eomma", "omma", "appa", "abeoji", "eomeoni", "san"),
        *("sama", "kun", "chan", "senpai", "sensei", "dono", "shishou", "gege", "jiejie", "shixiong"),
        *("shijie", "shifu", "daoist", "qi"),
    }
)


@dataclass(frozen=True, slots=True)
class Typo:
    """A word of an English line the dictionary does not know, and its closest known words (best first)."""

    word: str
    suggestions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SeriesTypo:
    """A typo found in one region of a series' chapter."""

    chapter: str
    region_id: str
    word: str
    suggestions: tuple[str, ...]
    english: str


@cache
def _dictionary(language: str = "en") -> SpellChecker:
    """The release language's dictionary, loaded once per language (a few MB of word frequencies)."""
    return SpellChecker(language=language, distance=1)


def _word_pattern(language: str) -> re.Pattern[str]:
    """What a word is in lines of `language`: ASCII letters in English, any letters otherwise."""
    return _WORD if language == "en" else _LETTERS_WORD


def _words_of(text: str, language: str = "en") -> set[str]:
    """The lower-cased words of a phrase ("Kim Dokja" -> kim, dokja)."""
    return {word.lower() for word in _word_pattern(language).findall(text.replace("’", "'"))}


def _starts_sentence(text: str, start: int) -> bool:
    """Whether the word at `start` opens the line or a sentence (a capital there is not a name's)."""
    before = text[:start].rstrip(_OPENERS)
    return not before or before[-1] in _SENTENCE_END


class TypoChecker:
    """Finds the words of English lines that neither the dictionary nor the allowed words know."""

    def __init__(self, allowed: Iterable[str] = (), language: str = "en") -> None:
        """`allowed`: extra known words (any case; phrases are split into words); `language`: the release's."""
        self._language = language
        self._word = _word_pattern(language)
        self._allowed = set(DEFAULT_WORDS)
        for phrase in allowed:
            self._allowed |= _words_of(phrase, language)
        self._verdicts: dict[str, bool] = {}

    def _known(self, word: str) -> bool:
        """Whether a lower-cased word (possibly with an apostrophe) is a word the checker accepts."""
        verdict = self._verdicts.get(word)
        if verdict is None:
            verdict = self._judge(word)
            self._verdicts[word] = verdict
        return verdict

    def _judge(self, word: str) -> bool:
        if word in self._allowed or word in _IRREGULAR or _STRETCHED.search(word):
            return True
        dictionary = _dictionary(self._language)
        if dictionary.known([word]):
            return True
        if "'" not in word:
            return False
        stem, _, suffix = word.rpartition("'")
        if suffix == "t" and stem.endswith("n"):  # don't, couldn't, isn't
            stem = stem[:-1]
        elif suffix not in _SUFFIXES:
            return False
        return "'" not in stem and (stem in self._allowed or bool(dictionary.known([stem])))

    def check(self, text: str) -> list[Typo]:
        """The unknown words of one English line, in order, each once."""
        text = text.replace("’", "'")
        shouting = text.upper() == text  # an all-caps line: its capitals say nothing about names
        seen: set[str] = set()
        typos: list[Typo] = []
        for match in self._word.finditer(text):
            word = match.group()
            key = word.lower()
            if len(word) < 2 or key in seen:
                continue
            capital = not shouting and word[0].isupper()
            if capital and (word.isupper() or not _starts_sentence(text, match.start())):
                continue  # an acronym, or a name inside a sentence
            seen.add(key)
            if not self._known(key):
                typos.append(Typo(word, self.suggestions(key)))
        return typos

    def suggestions(self, word: str) -> tuple[str, ...]:
        """The dictionary's closest words to `word`, most common first (at most MAX_SUGGESTIONS)."""
        dictionary = _dictionary(self._language)
        candidates = (dictionary.candidates(word) or set()) - {word}
        ranked = sorted(candidates, key=lambda c: (-dictionary.word_usage_frequency(c), c))
        return tuple(ranked[:MAX_SUGGESTIONS])


def series_words(series: SeriesPaths) -> list[str]:
    """The words the series marked "not a typo" (sorted); none when it has no list yet."""
    path = series.library_dir / WORDS_FILE
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return sorted({line.lower() for line in lines if line and not line.startswith("#")})


def allow_word(series: SeriesPaths, word: str) -> list[str]:
    """Mark a word "not a typo" for the whole series and return the series' list; ValueError for no single word.
    The series' library folder must exist (FileNotFoundError otherwise: never creates a series)."""
    words = _words_of(word, "any")  # a word of any release language (Straße, año)
    if len(words) != 1 or _LETTERS_WORD.fullmatch(word.strip().replace("’", "'")) is None:
        raise ValueError(f"not a single word: {word!r}")
    wanted = sorted(set(series_words(series)) | words)
    path = series.library_dir / WORDS_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(f"{w}\n" for w in wanted), encoding="utf-8")
    tmp.replace(path)
    return wanted


def allowed_words(series: SeriesPaths) -> set[str]:
    """What counts as known in the series besides the dictionary: its "not a typo" words, the English of its
    glossary terms (rejected ones aside) and its characters' names and aliases."""
    allowed = set(series_words(series))
    allowed |= {entry.target for entry in glossary(series) if entry.status != "rejected"}
    try:
        characters = load_voices(series)
    except (
        ValueError
    ):  # a broken voices.toml is reported by the translate stage; the check goes on without it
        characters = []
    allowed |= {name for character in characters for name in character.names()}
    return allowed


def release_language(series: SeriesPaths) -> str:
    """The series' release language (`[translate] target_lang`); English when its series.toml is broken (the
    stages report that)."""
    try:
        return series_config(get_config(), series.library_dir).translate.target_lang
    except SeriesConfigError:
        return "en"


def checker_for(series: SeriesPaths) -> TypoChecker:
    """A checker in the series' release language that knows the series' own words (allowed_words)."""
    return TypoChecker(allowed_words(series), release_language(series))


def series_typos(lines: Sequence[Line], checker: TypoChecker) -> list[SeriesTypo]:
    """The typos of a series' translated lines (qa.consistency.series_lines), sound effects and watermarks aside."""
    typos: list[SeriesTypo] = []
    for line in lines:
        if line.kind in ("sfx", "watermark") or not line.english.strip():
            continue
        typos += [
            SeriesTypo(line.chapter, line.region_id, typo.word, typo.suggestions, line.english)
            for typo in checker.check(line.english)
        ]
    return typos
