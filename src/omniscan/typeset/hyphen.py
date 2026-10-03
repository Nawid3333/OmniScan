"""Line-breaking rules per release language (`[translate] target_lang`): the little words a line should not end
on, and where a long word may be hyphenated.

English keeps the rules the fitter always had: its own list of articles and prepositions, and a long word cut
wherever the width allows (`hyphen_points` returns None). Spanish is hyphenated only between syllables, which
follow regular rules (Real Academia Española, "Ortografía", 4.1): one consonant between vowels starts the next
syllable (ca-sa), two are split (can-to) unless they are a cluster that starts a syllable (ha-blar, mu-cho,
pe-rro), three or more keep the last cluster together (ins-truc-ción); two strong vowels, or an accented i/u next
to another vowel, are separate syllables (le-o, dí-a), any other vowel pair is one (cie-lo). German keeps the
width-based cut (its compound words need a dictionary) but gets its own list of little words.
"""

from __future__ import annotations

from itertools import pairwise

_DANGLING: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "a",
            "an",
            "the",
            "to",
            "of",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "from",
            "as",
            "is",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "this",
            "that",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "los",
            "las",
            "lo",
            "un",
            "una",
            "unos",
            "unas",
            "y",
            "e",
            "o",
            "u",
            "ni",
            "de",
            "del",
            "a",
            "al",
            "en",
            "con",
            "por",
            "para",
            "sin",
            "que",
            "mi",
            "mis",
            "tu",
            "tus",
            "su",
            "sus",
            "se",
            "me",
            "te",
            "nos",
            "le",
            "les",
            "este",
            "esta",
            "ese",
            "esa",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einen",
            "einem",
            "einer",
            "und",
            "oder",
            "aber",
            "zu",
            "zum",
            "zur",
            "von",
            "vom",
            "mit",
            "im",
            "in",
            "am",
            "an",
            "auf",
            "für",
            "bei",
            "mein",
            "dein",
            "sein",
            "ihr",
            "unser",
            "ist",
            "als",
        ]
    ),
}

_VOWELS = frozenset("aeiouáéíóúü")
_STRONG = frozenset("aeoáéíóú")  # an accented i or u counts as strong: it never joins a diphthong (dí-a)
_ONSETS = frozenset(
    ["bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kl", "kr", "pl", "pr", "tr", "ch", "ll", "rr"]
)


def dangling_words(lang: str) -> frozenset[str]:
    """The articles, prepositions and conjunctions a line of `lang` should not end on (English for unknown codes)."""
    return _DANGLING.get(lang, _DANGLING["en"])


def hyphen_points(word: str, lang: str) -> list[int] | None:
    """Where `word` may be hyphenated (indices between its characters), or None when any place will do."""
    if lang != "es":
        return None
    return _spanish_points(word)


def _spanish_points(word: str) -> list[int]:
    """The syllable boundaries inside the letters of a Spanish word (punctuation around it is kept whole)."""
    start = next((i for i, ch in enumerate(word) if ch.isalpha()), len(word))
    end = len(word)
    while end > start and not word[end - 1].isalpha():
        end -= 1
    core = word[start:end].lower()
    if not core.isalpha():
        return []  # a hyphenated or apostrophised word: leave it alone
    vowel = [_is_vowel(core, i) for i in range(len(core))]
    nuclei: list[tuple[int, int]] = []  # [start, end) of each syllable's vowels
    i = 0
    while i < len(core):
        if not vowel[i]:
            i += 1
            continue
        j = i + 1
        while j < len(core) and vowel[j] and not (core[j - 1] in _STRONG and core[j] in _STRONG):
            j += 1
        nuclei.append((i, j))
        i = j
    points: list[int] = []
    for (_, gap_start), (gap_end, _) in pairwise(nuclei):
        cluster = core[gap_start:gap_end]
        if len(cluster) <= 1:
            cut = gap_start
        elif len(cluster) == 2:
            cut = gap_start if cluster in _ONSETS else gap_start + 1
        elif len(cluster) == 3:
            cut = gap_start + 1 if cluster[1:] in _ONSETS else gap_start + 2
        else:
            cut = gap_start + 2
        points.append(start + cut)
    return points


def _is_vowel(word: str, index: int) -> bool:
    """Whether the letter at `index` is a vowel; y is one only where no vowel follows it (hoy, muy, y)."""
    letter = word[index]
    if letter == "y":
        return index + 1 == len(word) or word[index + 1] not in _VOWELS
    return letter in _VOWELS
