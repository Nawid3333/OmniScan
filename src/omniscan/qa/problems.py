"""A chapter's problems in one list, for whoever proofreads it: the web Studio's Problems view and
`omniscan edit problems`.

Two sources, both read-only: the desktop Studio's checks (studio/qa.py through edits.session.StudioSession, so the
two Studios agree) over the chapter as it is saved — lines with no English, source script left in the English,
lettering that overflows its balloon, lines the judge was unsure of or that break a locked glossary term, much
too long lines, typos — and what the last `omniscan qa` re-read still found on the finished pages (qa.json).
Each problem carries its line's review state, so a proofreader can leave out the lines already checked.
"""

from __future__ import annotations

from dataclasses import dataclass

from omniscan.core.paths import ChapterPaths
from omniscan.edits import store
from omniscan.edits.session import StudioSession
from omniscan.qa.leftover import load_issues


@dataclass(frozen=True, slots=True)
class Problem:
    """One problem on one region."""

    region_id: str
    kind: str  # studio/qa.py's IssueKind, or the qa stage's source_left / watermark_left on a finished page
    message: str
    status: str  # the line's review state: todo, edited or checked
    finished_page: bool = False  # found by re-reading the exported page (qa.json)
    word: str = ""  # the unknown word of a typo
    read: str = ""  # what the OCR read on the finished page


def chapter_problems(paths: ChapterPaths) -> list[Problem]:
    """The chapter's problems in reading order, each region's checks before what its finished page showed; a
    damaged qa.json counts as no re-read, like one that was never written."""
    session = StudioSession(paths)
    status = store.line_statuses(paths)
    order = {region.id: i for i, region in enumerate(session.regions())}
    problems = [
        Problem(
            issue.region_id, issue.kind, issue.message, status.get(issue.region_id, "todo"), word=issue.word
        )
        for issue in session.issues()
    ]
    try:
        leftover = load_issues(paths)
    except OSError, ValueError:
        leftover = []
    problems += [
        Problem(
            issue.region_id,
            issue.kind,
            issue.message,
            status.get(issue.region_id, "todo"),
            finished_page=True,
            read=issue.read,
        )
        for issue in leftover
    ]
    return sorted(problems, key=lambda problem: order.get(problem.region_id, len(order)))
