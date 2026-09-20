"""Tests for omniscan.acquire.plan."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from omniscan.acquire.completeness import Finding
from omniscan.acquire.plan import (
    ACCEPTED_NAME,
    AcquirePlan,
    PlanItem,
    build_plan,
    chapter_state,
    read_accepted,
    to_json,
    write_accepted,
)
from omniscan.acquire.sources import ChapterSource


def three_sources() -> list[ChapterSource]:
    return [ChapterSource(f"Chapter {n}", f"https://x.test/{n}") for n in (1, 2, 3)]


# --- build_plan / chapter_state ---


def test_build_plan_states_and_credits(tmp_path: Path) -> None:
    sources = three_sources()
    (tmp_path / "Chapter 1").mkdir(parents=True)
    (tmp_path / "Chapter 1" / ACCEPTED_NAME).write_text(
        json.dumps({"status": "review", "pages": 4}), encoding="utf-8"
    )
    (tmp_path / "Chapter 2").mkdir()
    (tmp_path / "Chapter 2" / ACCEPTED_NAME).write_text("{not json", encoding="utf-8")
    (tmp_path / "Chapter 3").mkdir()
    (tmp_path / "Chapter 3" / ACCEPTED_NAME).write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    plan = build_plan(sources, tmp_path)

    assert [item.state for item in plan.items] == ["done", "todo", "todo"]
    assert [item.credits for item in plan.items] == [0, 1, 1]
    assert (plan.credits_total, plan.todo, plan.done) == (2, 2, 1)
    assert plan.mode == "basic"


def test_build_plan_advanced_costs_two_credits(tmp_path: Path) -> None:
    sources = three_sources()
    plan = build_plan(sources, tmp_path, mode="advanced")
    assert [item.credits for item in plan.items] == [2, 2, 2]
    assert plan.credits_total == 6


def test_build_plan_force_makes_everything_todo(tmp_path: Path) -> None:
    sources = three_sources()
    done_dir = tmp_path / "Chapter 1"
    done_dir.mkdir(parents=True)
    (done_dir / ACCEPTED_NAME).write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    plan = build_plan(sources, tmp_path, force=True)

    assert [item.state for item in plan.items] == ["todo", "todo", "todo"]
    assert (plan.credits_total, plan.todo, plan.done) == (3, 3, 0)


def test_chapter_state_ok_status_is_done(tmp_path: Path) -> None:
    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    write_accepted(ok_dir, "ok", [], pages=10, source_url="https://x.test/1")
    assert chapter_state(ok_dir) == "done"
    assert chapter_state(tmp_path / "missing") == "todo"


# --- write_accepted / read_accepted ---


def test_write_then_read_accepted_round_trip(tmp_path: Path) -> None:
    chapter = tmp_path / "Chapter 4"
    chapter.mkdir()
    findings = [Finding("warn", "few_pages", "4 pages; the series median is 10")]

    write_accepted(chapter, "review", findings, pages=4, source_url="https://x.test/4")

    record = read_accepted(chapter)
    assert record is not None
    assert record["status"] == "review"
    assert record["pages"] == 4
    assert record["source_url"] == "https://x.test/4"
    assert record["findings"] == [
        {"level": "warn", "code": "few_pages", "message": "4 pages; the series median is 10"}
    ]
    accepted_at = datetime.fromisoformat(record["accepted_at"])
    assert accepted_at.utcoffset() == timedelta(0)
    text = (chapter / ACCEPTED_NAME).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert not list(chapter.glob("*.tmp"))


def test_read_accepted_missing_or_corrupt(tmp_path: Path) -> None:
    assert read_accepted(tmp_path / "nope") is None
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / ACCEPTED_NAME).write_text("{oops", encoding="utf-8")
    assert read_accepted(bad) is None


# --- to_json ---


def test_to_json_exact_dict() -> None:
    plan = AcquirePlan(
        items=(
            PlanItem(ChapterSource("Chapter 1", "https://x.test/1"), "done", 0),
            PlanItem(ChapterSource("Chapter 2", "https://x.test/2"), "todo", 1),
        ),
        mode="basic",
        credits_total=1,
        todo=1,
        done=1,
    )
    assert to_json(plan) == {
        "mode": "basic",
        "credits_total": 1,
        "todo": 1,
        "done": 1,
        "items": [
            {"name": "Chapter 1", "url": "https://x.test/1", "state": "done", "credits": 0},
            {"name": "Chapter 2", "url": "https://x.test/2", "state": "todo", "credits": 1},
        ],
    }
