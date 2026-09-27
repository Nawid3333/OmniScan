"""Which regions still show their original text on the finished pages (pure functions, no GPU, no I/O but qa.json).

The `qa` stage re-reads every cleaned region of the lettered page with the chapter's OCR engine. A region is
flagged only when what is read there is recognisably the region's original text — source-script characters
similar to its OCR text — so an OCR model misreading the English lettering as stray kana never raises an issue.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence

from omniscan.core.config import Config
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import ExportArtifact, QaArtifact, QaIssue, Region
from omniscan.export.segments import Segment
from omniscan.learn.apply import similarity

QA_FILE = "qa.json"
MIN_SIMILARITY = 0.3  # what is read shares at least this much (character bigrams) with the original text
MIN_CONFIDENCE = 0.5  # the OCR's own confidence in what it read
# Hangul (jamo, compatibility jamo, syllables), kana (incl. half-width), CJK ideographs
_SOURCE_SCRIPT = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-鿿가-힯ｦ-ﾟ]")


def source_chars(text: str) -> str:
    """The source-script characters of `text` (Korean, Japanese, Chinese), in order."""
    return "".join(_SOURCE_SCRIPT.findall(text))


def cleaned_kinds(cfg: Config) -> frozenset[str]:
    """The region kinds the pipeline erases and letters anew: text always, sound effects in `replace` mode,
    watermarks when `inpaint.remove_watermarks` is on."""
    kinds = {"bubble_text", "free_text"}
    if cfg.sfx.mode == "replace":
        kinds.add("sfx")
    if cfg.inpaint.remove_watermarks:
        kinds.add("watermark")
    return frozenset(kinds)


def regions_to_check(
    regions: Sequence[Region], kinds: Collection[str], segments: Sequence[Segment]
) -> list[Region]:
    """The regions to re-read: of a cleaned kind, with source text, and on an exported page."""
    return [
        r
        for r in regions
        if r.kind in kinds
        and source_chars(r.text)
        and any(s.y0 <= (r.bbox.y0 + r.bbox.y1) / 2 < s.y1 for s in segments)
    ]


def leftover_issues(regions: Sequence[Region], readings: Mapping[str, tuple[str, float]]) -> list[QaIssue]:
    """An issue for every region whose re-read (text, confidence) is recognisably its original text."""
    issues: list[QaIssue] = []
    for region in regions:
        reading = readings.get(region.id)
        if reading is None:
            continue
        text, confidence = reading
        found = source_chars(text)
        if not found or confidence < MIN_CONFIDENCE:
            continue
        if similarity(found, source_chars(region.text)) < MIN_SIMILARITY:
            continue
        watermark = region.kind == "watermark"
        what = "the watermark is still visible" if watermark else "the original text is still readable"
        issues.append(
            QaIssue(
                region_id=region.id,
                kind="watermark_left" if watermark else "source_left",
                message=f"{what} on the finished page: {text.strip()!r}",
                read=text,
            )
        )
    return issues


def segment_rows(export: ExportArtifact, segments: Sequence[Segment]) -> list[tuple[str, int, int]]:
    """(output file, first strip row, end row) of every exported image; ValueError when the output folder was
    written for other slices or cuts than the chapter has now."""
    if len(export.files) != len(segments) or any(
        f.height != s.y1 - s.y0 for f, s in zip(export.files, segments, strict=False)
    ):
        raise ValueError("the exported images do not match the chapter's slices and cuts — run export again")
    return [(f.name, s.y0, s.y1) for f, s in zip(export.files, segments, strict=True)]


def load_issues(paths: ChapterPaths) -> list[QaIssue]:
    """The issues of the chapter's last `qa` run (qa.json), or none when it never ran."""
    path = paths.artifact(QA_FILE)
    return QaArtifact.load(path).issues if path.is_file() else []
