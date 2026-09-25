"""Typeset layout engine: lettering that follows the balloon, as a professional letterer sets it.

Greedy wrapping into the rectangle inscribed in a balloon wastes most of an oval and leaves ragged,
top-heavy blocks. Letterers instead break dialogue into lines whose lengths follow the balloon — short
lines at the top and bottom, the longest across the middle (the "lozenge") — and keep neighbouring
lines similar in length. This module does that with pure metric math over font advances:

- a `Shape` gives the usable width of every line of a block of `n` lines (an ellipse's chord at the
  line's outer edge, or a rectangle's width);
- `fit_shape` tries every line count, keeps those that let the text grow (nearly) largest — the
  layouts that match the balloon's proportions — and re-breaks the words of each so every line's
  length is proportional to its width and breaks fall after sentences and clauses, never after an
  article (dynamic programming instead of greedy); the best size-plus-phrasing score wins;
- a word too long for any line at the smallest size is hyphenated instead of overflowing.

Nothing is drawn; the result feeds `LayoutItem`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from omniscan.core.schemas import BBox
from omniscan.typeset.fonts import load_font


class Measurable(Protocol):
    """Anything that can measure a single line's advance width in pixels (PIL fonts, test fakes)."""

    def getlength(self, text: str) -> float: ...


FontFactory = Callable[[Path, int], Measurable]

# Process-wide caches for the bundled-font loader (`font_factory is load_font`): fonts per (resolved
# path, size) and measurements per (resolved path, size, text), cleared when full so they stay
# bounded. Any other factory (test fakes) never touches them — it only gets the per-call memo.
_FONT_CACHE_MAX = 512
_MEASURE_CACHE_MAX = 200_000
_font_cache: dict[tuple[str, int], Measurable] = {}
_measure_cache: dict[tuple[str, int, str], float] = {}


class _MeasuredFont:
    """A Measurable that memoises getlength for one font within a single layout call, so a line measured
    while searching for a size is not measured again while balancing or for the block width."""

    __slots__ = ("_cache", "_font")

    def __init__(self, font: Measurable) -> None:
        self._font = font
        self._cache: dict[str, float] = {}

    def getlength(self, text: str) -> float:
        value = self._cache.get(text)
        if value is None:
            value = self._font.getlength(text)
            self._cache[text] = value
        return value


class _CachedFont:
    """A Measurable backed by the process-wide measurement cache for one (resolved path, size)."""

    __slots__ = ("_font", "_path", "_size")

    def __init__(self, font: Measurable, path: str, size: int) -> None:
        self._font = font
        self._path = path
        self._size = size

    def getlength(self, text: str) -> float:
        key = (self._path, self._size, text)
        value = _measure_cache.get(key)
        if value is None:
            value = self._font.getlength(text)
            if len(_measure_cache) >= _MEASURE_CACHE_MAX:
                _measure_cache.clear()
            _measure_cache[key] = value
        return value


def _default_font(path: Path, size: int) -> Measurable:
    """load_font with the process-wide font cache; its measurements go through _CachedFont."""
    resolved = str(path.resolve())
    key = (resolved, size)
    font = _font_cache.get(key)
    if font is None:
        font = load_font(path, size)
        if len(_font_cache) >= _FONT_CACHE_MAX:
            _font_cache.clear()
        _font_cache[key] = font
    return _CachedFont(font, resolved, size)


def measurer(font_factory: FontFactory) -> FontFactory:
    """The factory to measure with: the process-wide cached loader for the bundled `load_font`, the
    given factory itself otherwise (test fakes never touch the shared caches)."""
    return _default_font if font_factory is load_font else font_factory


def line_height(size_px: int, line_spacing: float) -> int:
    """Line pitch in pixels: size times spacing, rounded, never below 1."""
    return max(1, round(size_px * line_spacing))


_GLYPH_EXTENT = 0.8  # share of the font size the letters really occupy vertically (cap height + a little)
_MIN_HYPHEN_HEAD = 3  # a hyphenated word keeps at least this many letters before the hyphen ...
_MIN_HYPHEN_TAIL = 2  # ... and after it
# Break quality, in units of (a tenth of the mean line width)^2: letterers break after a sentence or a
# clause and never leave an article or preposition dangling at the end of a line.
_SENTENCE_END = frozenset(".!?…")
_CLAUSE_END = frozenset(",;:—–")
_DANGLING = frozenset(
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
)
_SENTENCE_BONUS = -6.0
_CLAUSE_BONUS = -3.0
_DANGLING_PENALTY = 8.0
_SIZE_TOLERANCE = 0.9  # line counts whose largest size is within 10 % of the best compete on phrasing
_PHRASING_PX = 0.5  # font pixels one unit of break cost is worth when choosing the line count
_SIZE_STEPS = (
    3  # each line count is also tried 1 and 2 px smaller: a letterer shrinks a little for a better break
)


@dataclass(frozen=True, slots=True)
class Shape:
    """Where lettering may go: an ellipse inscribed in `box`, or the rectangle `box` itself."""

    kind: Literal["ellipse", "rect"]
    box: BBox

    def line_widths(self, n: int, size_px: int, pitch: int) -> list[float]:
        """Usable width of each of `n` lines of a vertically centred block (0.0 where a line does not fit)."""
        width, height = float(self.box.width), float(self.box.height)
        if self.kind == "rect":
            return [width] * n if n * pitch <= height else [0.0] * n
        a, b = width / 2, height / 2
        extent = _GLYPH_EXTENT * size_px
        widths: list[float] = []
        for i in range(n):
            centre = (i + 0.5) * pitch - n * pitch / 2  # line centre relative to the block centre
            dy = abs(centre) + extent / 2  # the edge of the letters farther from the middle
            widths.append(2 * a * math.sqrt(1 - (dy / b) ** 2) if dy < b else 0.0)
        return widths


@dataclass(frozen=True, slots=True)
class Fit:
    """One lettering decision: size, lines, their block size and whether the text really fits."""

    size_px: int
    lines: list[str]
    width: int  # ceil of the widest line's advance
    height: int  # len(lines) * line pitch
    overflow: bool
    best_px: int  # largest size this text could have in the shape (before any size cap); 0 on overflow


@dataclass(frozen=True, slots=True)
class _Token:
    """One breakable unit: a word, or the head of a hyphenated word (`joins` its successor on one line)."""

    text: str
    joins: bool = False


def _line_text(tokens: list[_Token]) -> str:
    """The rendered text of tokens set on one line: joined pieces merge, a trailing head keeps its hyphen."""
    parts: list[str] = []
    glue = False
    for i, token in enumerate(tokens):
        last = i == len(tokens) - 1
        piece = f"{token.text}-" if token.joins and last else token.text
        if glue:
            parts[-1] += piece
        else:
            parts.append(piece)
        glue = token.joins
    return " ".join(parts)


class _Setter:
    """Measures lines of one text at one size (cached) and breaks it into lines of given widths."""

    def __init__(self, tokens: list[_Token], font: Measurable) -> None:
        self.tokens = tokens
        self.font = font
        self._widths: dict[tuple[int, int], float] = {}

    def width(self, start: int, end: int) -> float:
        """Advance width of tokens[start:end] set on one line."""
        key = (start, end)
        value = self._widths.get(key)
        if value is None:
            value = self.font.getlength(_line_text(self.tokens[start:end]))
            self._widths[key] = value
        return value

    def greedy(self, widths: list[float]) -> list[int] | None:
        """Line end indices when every line, top down, takes as many tokens as fit its width (the text may
        end before the last line); None when it does not fit. Fitting as much as possible early is optimal
        for "fits in at most these lines", and that question only gets easier as the text shrinks."""
        ends: list[int] = []
        start = 0
        for limit in widths:
            if start == len(self.tokens):
                break
            end = start
            while end < len(self.tokens) and self.width(start, end + 1) <= limit:
                end += 1
            if end == start:
                return None
            ends.append(end)
            start = end
        return ends if start == len(self.tokens) else None

    def balanced(self, widths: list[float]) -> list[int] | None:
        """Line end indices minimising the squared difference between every line's length and its share
        of the text (proportional to the line's width) plus the phrasing cost of every break, every line
        within its width; None when impossible."""
        n, count = len(widths), len(self.tokens)
        total = self.width(0, count)
        capacity = sum(widths)
        if capacity <= 0:
            return None
        ratio = total / capacity
        unit = (0.1 * capacity / n) ** 2  # scale of the phrasing terms
        inf = math.inf
        cost = [[inf] * (count + 1) for _ in range(n + 1)]
        back = [[0] * (count + 1) for _ in range(n + 1)]
        cost[0][0] = 0.0
        for line in range(1, n + 1):
            limit = widths[line - 1]
            ideal = ratio * limit
            remaining = n - line
            for end in range(line, count - remaining + 1):
                for start in range(end - 1, line - 2, -1):  # longer and longer lines ending at `end`
                    length = self.width(start, end)
                    if length > limit:
                        break
                    if cost[line - 1][start] == inf:
                        continue
                    value = cost[line - 1][start] + (length - ideal) ** 2
                    if end < count:  # the last line's end is not a break
                        value += unit * self.break_cost(end)
                    if value < cost[line][end]:
                        cost[line][end] = value
                        back[line][end] = start
        if cost[n][count] == inf:
            return None
        ends = [count]
        for line in range(n, 1, -1):
            ends.append(back[line][ends[-1]])
        return ends[::-1]

    def break_cost(self, end: int) -> float:
        """Phrasing cost of ending a line after token `end - 1` (negative: a good place to break)."""
        token = self.tokens[end - 1]
        if token.joins:
            return 0.0  # a hyphenated word must break here
        word = token.text.rstrip("\"')”’»")
        if word[-1:] in _SENTENCE_END:
            return _SENTENCE_BONUS
        if word[-1:] in _CLAUSE_END or word.endswith("..."):
            return _CLAUSE_BONUS
        if word.casefold() in _DANGLING:
            return _DANGLING_PENALTY
        return 0.0

    def lines(self, ends: list[int]) -> list[str]:
        """The rendered line strings for line end indices."""
        starts = [0, *ends[:-1]]
        return [_line_text(self.tokens[s:e]) for s, e in zip(starts, ends, strict=True)]


def _max_lines(shape: Shape, min_px: int, line_spacing: float, count: int) -> int:
    """Most lines worth trying: as many as fit the shape's height at the smallest size, at most one per token."""
    return max(1, min(count, int(shape.box.height // line_height(min_px, line_spacing)) + 1))


def _best_size(
    tokens: list[_Token],
    shape: Shape,
    n: int,
    font_path: Path,
    min_px: int,
    max_px: int,
    line_spacing: float,
    factory: FontFactory,
) -> int:
    """Largest size in [min_px, max_px] at which the tokens fit the shape in exactly `n` lines; 0 if none
    (binary search: shrinking the text only ever frees room)."""

    def fits(size: int) -> bool:
        setter = _Setter(tokens, _MeasuredFont(factory(font_path, size)))
        return setter.greedy(shape.line_widths(n, size, line_height(size, line_spacing))) is not None

    if not fits(min_px):
        return 0
    lo, hi = min_px, max_px
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _hyphenated(tokens: list[_Token], font: Measurable, limit: float) -> list[_Token]:
    """Tokens with every word wider than `limit` split into hyphen-joined pieces that each fit."""
    out: list[_Token] = []
    for token in tokens:
        text = token.text
        while font.getlength(text) > limit and len(text) >= _MIN_HYPHEN_HEAD + _MIN_HYPHEN_TAIL:
            cut = _MIN_HYPHEN_HEAD
            while cut + 1 <= len(text) - _MIN_HYPHEN_TAIL and font.getlength(text[: cut + 1] + "-") <= limit:
                cut += 1
            out.append(_Token(text[:cut], joins=True))
            text = text[cut:]
        out.append(_Token(text, joins=token.joins))
    return out


def fit_shape(
    text: str,
    shape: Shape,
    font_path: Path,
    *,
    min_px: int = 14,
    max_px: int = 48,
    size_cap: int | None = None,
    line_spacing: float = 1.15,
    hyphenate: bool = True,
    font_factory: FontFactory = load_font,
) -> Fit:
    """Set `text` in `shape` at the largest size its words allow (at most `size_cap`), lines balanced.

    Every line count is tried; those whose largest size is within `_SIZE_TOLERANCE` of the best are set
    at that size and up to two pixels smaller, and the layout scoring best on size plus phrasing (breaks
    after sentences and clauses, none after an article) wins, fewer lines and larger sizes on ties. With nothing fitting at `min_px` the long words are hyphenated
    (when allowed); failing that, a greedy wrap at `min_px` is returned with `overflow=True`.
    """
    if min_px < 1 or max_px < min_px:
        raise ValueError(f"need 1 <= min_px <= max_px, got {min_px}..{max_px}")
    words = text.split()
    if not words:
        return Fit(size_px=max_px, lines=[], width=0, height=0, overflow=False, best_px=max_px)
    factory = measurer(font_factory)
    tokens = [_Token(word) for word in words]
    sizes = _line_count_sizes(tokens, shape, font_path, min_px, max_px, line_spacing, factory)
    if not sizes and hyphenate:
        widest = max(shape.line_widths(1, min_px, line_height(min_px, line_spacing)))
        if widest > 0:
            tokens = _hyphenated(tokens, _MeasuredFont(factory(font_path, min_px)), widest)
            sizes = _line_count_sizes(tokens, shape, font_path, min_px, max_px, line_spacing, factory)
    if not sizes:
        return _overflow(tokens, shape, font_path, min_px, line_spacing, factory)
    best_px = max(sizes.values())
    chosen: tuple[float, int, list[str], int] | None = None  # (score, size, lines, pitch)
    floor = max(min_px, math.ceil(_SIZE_TOLERANCE * best_px))
    for n, largest in sorted(sizes.items()):
        top = min(largest, size_cap) if size_cap is not None else largest
        for size in sorted({max(min_px, top - step) for step in range(_SIZE_STEPS)}, reverse=True):
            if size < floor and size != top:
                continue
            setter = _Setter(tokens, _MeasuredFont(factory(font_path, size)))
            pitch = line_height(size, line_spacing)
            ends = _balanced_ends(setter, shape, n, size, pitch)
            score = size - _PHRASING_PX * sum(setter.break_cost(end) for end in ends[:-1])
            if chosen is None or score > chosen[0]:
                chosen = (score, size, setter.lines(ends), pitch)
    assert chosen is not None  # the best line count itself always qualifies
    _, size, lines, pitch = chosen
    font = _MeasuredFont(factory(font_path, size))
    width = math.ceil(max(font.getlength(line) for line in lines))
    return Fit(
        size_px=size, lines=lines, width=width, height=len(lines) * pitch, overflow=False, best_px=best_px
    )


def _balanced_ends(setter: _Setter, shape: Shape, n: int, size: int, pitch: int) -> list[int]:
    """Balanced line ends in exactly `n` lines, else in the most lines below `n` that can be balanced,
    else the greedy ends in `n`-line widths (which fit: `n` lines were feasible at a size >= `size`)."""
    for m in range(min(n, len(setter.tokens)), 0, -1):
        ends = setter.balanced(shape.line_widths(m, size, pitch))
        if ends is not None:
            return ends
    ends = setter.greedy(shape.line_widths(n, size, pitch))
    assert ends is not None
    return ends


def _line_count_sizes(
    tokens: list[_Token],
    shape: Shape,
    font_path: Path,
    min_px: int,
    max_px: int,
    line_spacing: float,
    factory: FontFactory,
) -> dict[int, int]:
    """Largest fitting size per line count, for every count that fits at `min_px` and could still come
    within `_SIZE_TOLERANCE` of the best (taller blocks are skipped once they cannot)."""
    sizes: dict[int, int] = {}
    best = 0
    for n in range(1, _max_lines(shape, min_px, line_spacing, len(tokens)) + 1):
        if min(max_px, int(shape.box.height / (n * line_spacing))) < _SIZE_TOLERANCE * best:
            break  # n lines of any size that could still qualify are taller than the shape
        size = _best_size(tokens, shape, n, font_path, min_px, max_px, line_spacing, factory)
        if size:
            sizes[n] = size
            best = max(best, size)
    return sizes


def _overflow(
    tokens: list[_Token],
    shape: Shape,
    font_path: Path,
    min_px: int,
    line_spacing: float,
    factory: FontFactory,
) -> Fit:
    """Greedy lines at `min_px` against the shape's widest line (the text is too long for the shape)."""
    setter = _Setter(tokens, _MeasuredFont(factory(font_path, min_px)))
    pitch = line_height(min_px, line_spacing)
    limit = max(1.0, max(shape.line_widths(1, min_px, pitch)))
    ends: list[int] = []
    start = 0
    while start < len(tokens):
        end = start + 1
        while end < len(tokens) and setter.width(start, end + 1) <= limit:
            end += 1
        ends.append(end)
        start = end
    lines = setter.lines(ends)
    width = math.ceil(max(setter.font.getlength(line) for line in lines))
    return Fit(size_px=min_px, lines=lines, width=width, height=len(lines) * pitch, overflow=True, best_px=0)
