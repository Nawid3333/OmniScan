"""Tests for omniscan.menu (card MENU1)."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
import typer

import omniscan.menu as menu
from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.pipeline.stages import STAGE_ORDER
from omniscan.queue.store import STATUSES

Read = Callable[[str], str]


def make_read(*answers: str) -> tuple[Read, list[str]]:
    """A read that pops the canned answers (EOFError once they run out); also returns the remainder."""
    remaining = list(answers)

    def read(_prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return read, remaining


def counting_read() -> tuple[Read, list[str]]:
    """A read returning "" and recording every prompt it was asked (proves it was never called)."""
    calls: list[str] = []

    def read(prompt: str) -> str:
        calls.append(prompt)
        return ""

    return read, calls


def menu_cfg(tmp_path: Path) -> Config:
    """Config with every root under tmp_path (no GPU, nothing outside tmp_path)."""
    return Config(
        gpu=GpuConfig(device="cpu"),
        paths=PathsConfig(
            library_root=tmp_path / "library",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        ),
    )


# ---------------------------------------------------------------- pick_from


def test_pick_from_valid_choice() -> None:
    read, remaining = make_read("2")
    assert menu.pick_from(["a", "b", "c"], "Pick one", read=read) == 1
    assert remaining == []


def test_pick_from_zero_returns_none() -> None:
    read, remaining = make_read("0")
    assert menu.pick_from(["a", "b"], "Pick one", read=read) is None
    assert remaining == []


def test_pick_from_out_of_range_reprompts(capsys: pytest.CaptureFixture[str]) -> None:
    read, remaining = make_read("9", "2")
    assert menu.pick_from(["a", "b", "c"], "Pick one", read=read) == 1
    assert remaining == []  # both answers consumed: exactly one reprompt
    assert "not a valid choice, try again" in capsys.readouterr().out


def test_pick_from_non_numeric_reprompts() -> None:
    read, remaining = make_read("not a number", "1")
    assert menu.pick_from(["a", "b"], "Pick one", read=read) == 0
    assert remaining == []


def test_pick_from_eof_returns_none() -> None:
    read, _remaining = make_read()
    assert menu.pick_from(["a", "b"], "Pick one", read=read) is None


def test_pick_from_zero_is_invalid_without_allow_back() -> None:
    read, remaining = make_read("0", "1")
    assert menu.pick_from(["a", "b"], "Pick one", read=read, allow_back=False) == 0
    assert remaining == []


# ---------------------------------------------------------------- pick_series


def test_pick_series_lists_library_dirs_sorted_naturally(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = menu_cfg(tmp_path)
    root = cfg.paths.library_root
    for name in ("Z", "A", "_meta", ".hidden", "Chapter 10", "Chapter 2"):
        (root / name).mkdir(parents=True)
    (root / "stray.txt").write_text("not a series", encoding="utf-8")
    read, remaining = make_read("1")

    assert menu.pick_series(cfg, read=read) == "A"

    listed = [
        line.split(". ", 1)[1]
        for line in capsys.readouterr().out.splitlines()
        if re.fullmatch(r"[1-9]\d*\. .+", line)
    ]
    assert listed == ["A", "Chapter 2", "Chapter 10", "Z"]
    output = "\n".join(listed)
    assert "_meta" not in output and ".hidden" not in output and "stray.txt" not in output
    assert remaining == []


def test_pick_series_empty_library_returns_none_without_prompting(tmp_path: Path) -> None:
    cfg = menu_cfg(tmp_path)  # library_root never created
    read, calls = counting_read()

    assert menu.pick_series(cfg, read=read) is None
    assert calls == []


def test_pick_series_library_without_dirs_returns_none_without_prompting(tmp_path: Path) -> None:
    cfg = menu_cfg(tmp_path)
    cfg.paths.library_root.mkdir()  # exists but empty
    read, calls = counting_read()

    assert menu.pick_series(cfg, read=read) is None
    assert calls == []


# ---------------------------------------------------------------- pick_chapters


def chapter_cfg(tmp_path: Path) -> tuple[Config, SeriesPaths]:
    """Config plus a SeriesPaths over three real chapter folders."""
    cfg = menu_cfg(tmp_path)
    for chapter in ("Chapter 1", "Chapter 2", "Chapter 3"):
        (cfg.paths.library_root / "S" / chapter).mkdir(parents=True)
    return cfg, SeriesPaths.from_config(cfg, "S")


def test_pick_chapters_all_via_blank_and_via_a(tmp_path: Path) -> None:
    _cfg, sp = chapter_cfg(tmp_path)

    assert menu.pick_chapters(sp, read=make_read("")[0]) is None
    assert menu.pick_chapters(sp, read=make_read("a")[0]) is None
    assert menu.pick_chapters(sp, read=make_read("ALL")[0]) is None
    assert menu.pick_chapters(sp, read=make_read("1,3")[0]) == ["Chapter 1", "Chapter 3"]
    assert menu.pick_chapters(sp, read=make_read("3,1")[0]) == ["Chapter 3", "Chapter 1"]

    read, remaining = make_read("5", "2")  # "5" is out of range: one reprompt
    assert menu.pick_chapters(sp, read=read) == ["Chapter 2"]
    assert remaining == []


def test_pick_chapters_empty_series_returns_empty_list_without_prompting(tmp_path: Path) -> None:
    cfg = menu_cfg(tmp_path)
    (cfg.paths.library_root / "S").mkdir(parents=True)  # a series folder with no chapters
    sp = SeriesPaths.from_config(cfg, "S")
    read, calls = counting_read()

    assert menu.pick_chapters(sp, read=read) == []
    assert calls == []


# ---------------------------------------------------------------- pick_bool / pick_text / pick_int


def test_pick_bool_default_on_blank() -> None:
    assert menu.pick_bool("Skip the LaMa inpaint stage?", True, read=make_read("")[0]) is True
    assert menu.pick_bool("Skip the LaMa inpaint stage?", False, read=make_read("")[0]) is False
    assert menu.pick_bool("Skip the LaMa inpaint stage?", False, read=make_read("Y")[0]) is True
    assert menu.pick_bool("Skip the LaMa inpaint stage?", True, read=make_read("no")[0]) is False

    read, remaining = make_read("maybe", "y")  # anything else reprints and asks again
    assert menu.pick_bool("Skip the LaMa inpaint stage?", False, read=read) is True
    assert remaining == []


def test_pick_text_default_on_blank() -> None:
    assert menu.pick_text("Path", None, read=make_read("")[0]) is None
    assert menu.pick_text("Path", "default", read=make_read("")[0]) == "default"
    assert menu.pick_text("Path", "default", read=make_read("  typed  ")[0]) == "typed"


def test_pick_int_reprompts_on_non_integer() -> None:
    read, remaining = make_read("x", "7")
    assert menu.pick_int("Count", 3, read=read) == 7
    assert remaining == []
    assert menu.pick_int("Count", 3, read=make_read("")[0]) == 3


# ---------------------------------------------------------------- run_menu


def test_run_menu_exit_immediately(tmp_path: Path) -> None:
    read, remaining = make_read("0")
    assert menu.run_menu(menu_cfg(tmp_path), read=read) == 0
    assert remaining == []  # the top-level prompt was read exactly once


def test_run_menu_eof_returns_zero(tmp_path: Path) -> None:
    read, _remaining = make_read()
    assert menu.run_menu(menu_cfg(tmp_path), read=read) == 0


def test_run_menu_q_exits_cleanly(tmp_path: Path) -> None:
    read, remaining = make_read("q")
    assert menu.run_menu(menu_cfg(tmp_path), read=read) == 0
    assert remaining == []


# ---------------------------------------------------------------- menu_pipeline


def recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace omniscan.menu.cmd_run with a fake recording its kwargs."""
    calls: list[dict[str, object]] = []

    def fake_cmd_run(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(menu, "cmd_run", fake_cmd_run)
    return calls


def test_menu_pipeline_run_everything_calls_cmd_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, sp = chapter_cfg(tmp_path)
    (cfg.paths.library_root / sp.series).mkdir(parents=True, exist_ok=True)
    calls = recorder(monkeypatch)
    read, remaining = make_read("1", "1", "2", "n", "n", "0", "0")

    menu.menu_pipeline(cfg, read=read)

    assert calls == [
        {
            "series": "S",
            "chapter": ["Chapter 2"],
            "stage": None,
            "no_lama": False,
            "force": False,
            "step": False,
            "preview_chapter": None,
        }
    ]
    assert remaining == []


def test_menu_pipeline_run_one_stage_calls_cmd_run_with_single_stage_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = recorder(monkeypatch)
    read, remaining = make_read("2", "1", "a", "3", "n", "0", "0")

    menu.menu_pipeline(cfg, read=read)

    assert calls == [
        {
            "series": "S",
            "chapter": None,
            "stage": [STAGE_ORDER[2]],
            "no_lama": False,
            "force": False,
            "step": False,
            "preview_chapter": None,
        }
    ]
    assert remaining == []


def test_menu_pipeline_nothing_to_do_when_series_has_no_chapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = menu_cfg(tmp_path)
    (cfg.paths.library_root / "Empty").mkdir(parents=True)
    calls = recorder(monkeypatch)
    read, remaining = make_read("1", "1", "0")

    menu.menu_pipeline(cfg, read=read)

    assert calls == []
    assert "nothing to do" in capsys.readouterr().out
    assert remaining == []


def test_menu_pipeline_eof_during_chapter_pick_aborts_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOF after the series pick: the flow aborts without calling cmd_run (card Definitions)."""
    cfg, _sp = chapter_cfg(tmp_path)
    calls = recorder(monkeypatch)
    read, _remaining = make_read("1", "1")  # EOF hits at the chapter pick

    menu.menu_pipeline(cfg, read=read)  # must not raise

    assert calls == []


# ---------------------------------------------------------------- other submenus


def test_menu_dispatch_catches_typer_exit_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[bool] = []

    def fake_doctor(as_json: bool) -> None:
        calls.append(as_json)
        raise typer.Exit(1)

    monkeypatch.setattr(menu, "doctor", fake_doctor)
    read, remaining = make_read("1", "0", "1", "0", "0")

    menu.menu_diagnostics(menu_cfg(tmp_path), read=read)  # must not raise

    assert calls == [False, False]  # the follow-up call ran: the loop survived the exit
    assert remaining == []
    assert "(doctor: exit 1)" in capsys.readouterr().out


def test_menu_launch_serve_keyboard_interrupt_returns_to_submenu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_serve(**kwargs: object) -> None:
        calls.append(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(menu, "cmd_serve", fake_serve)
    read, remaining = make_read("1", "0", "0")

    menu.menu_launch(menu_cfg(tmp_path), read=read)  # must not raise

    assert calls == [{"host": "127.0.0.1", "port": 8000, "reload": False}]
    assert remaining == []


# ---------------------------------------------------------------- MENU2: menu_translate


def spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[dict[str, object]]:
    """Replace the menu module's `name` attribute with a fake recording its kwargs."""
    calls: list[dict[str, object]] = []

    def fake(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(menu, name, fake)
    return calls


def test_menu_translate_run_calls_cmd_translate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "cmd_translate")
    read, remaining = make_read("1", "1", "a", "", "n", "", "0")

    menu.menu_translate(cfg, read=read)

    assert calls == [{"series": "S", "chapter": None, "profile": None, "force": False}]
    assert remaining == []


def test_menu_translate_run_splits_profile_list_and_forces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "cmd_translate")
    read, remaining = make_read("1", "1", "1", "a, b ,c", "y", "", "0")

    menu.menu_translate(cfg, read=read)

    assert calls == [{"series": "S", "chapter": ["Chapter 1"], "profile": ["a", "b", "c"], "force": True}]
    assert remaining == []


def test_menu_translate_nothing_to_do_when_series_has_no_chapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = menu_cfg(tmp_path)
    (cfg.paths.library_root / "Empty").mkdir(parents=True)
    calls = spy(monkeypatch, "cmd_translate")
    read, remaining = make_read("1", "1", "0")

    menu.menu_translate(cfg, read=read)

    assert calls == []
    assert "nothing to do" in capsys.readouterr().out
    assert remaining == []


def test_menu_translate_judge_calls_cmd_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "cmd_judge")
    read, remaining = make_read("2", "1", "a", "", "n", "", "0")

    menu.menu_translate(cfg, read=read)

    assert calls == [{"series": "S", "chapter": None, "run": None, "force": False}]
    assert remaining == []


def test_menu_translate_judge_splits_run_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "cmd_judge")
    read, remaining = make_read("2", "1", "1", "r1, r2", "y", "", "0")

    menu.menu_translate(cfg, read=read)

    assert calls == [{"series": "S", "chapter": ["Chapter 1"], "run": ["r1", "r2"], "force": True}]
    assert remaining == []


def test_menu_translate_story_calls_story_summarize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "story_summarize")
    read, remaining = make_read("3", "1", "a", "", "n", "", "0")

    menu.menu_translate(cfg, read=read)

    assert calls == [
        {"series": "S", "chapter": None, "model": "gemma4:31b-cloud", "force": False, "as_json": False}
    ]
    assert remaining == []


def test_menu_translate_eof_during_chapter_pick_aborts_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOF after the series pick: the flow aborts without calling cmd_translate (card Definitions)."""
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "cmd_translate")
    read, _remaining = make_read("1", "1")  # EOF hits at the chapter pick

    menu.menu_translate(cfg, read=read)  # must not raise

    assert calls == []


# ---------------------------------------------------------------- MENU2: menu_glossary


def test_menu_glossary_list_passes_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "glossary_list")

    menu.menu_glossary(cfg, read=make_read("1", "1", "2", "", "0")[0])
    assert calls == [{"series": "S", "status": "locked"}]

    menu.menu_glossary(cfg, read=make_read("1", "1", "4", "", "0")[0])  # "all" -> status=None
    assert calls == [{"series": "S", "status": "locked"}, {"series": "S", "status": None}]


def test_menu_glossary_export_and_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    exports = spy(monkeypatch, "glossary_export")
    imports = spy(monkeypatch, "glossary_import")

    menu.menu_glossary(cfg, read=make_read("2", "1", "", "0")[0])
    assert exports == [{"series": "S"}]

    menu.menu_glossary(cfg, read=make_read("3", "1", "1", "", "0")[0])
    menu.menu_glossary(cfg, read=make_read("3", "1", "2", "", "0")[0])
    assert imports == [{"series": "S", "mode": "merge"}, {"series": "S", "mode": "replace"}]


def test_menu_glossary_propose_calls_glossary_propose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "glossary_propose")
    read, remaining = make_read("4", "1", "a", "", "5", "y", "", "0")

    menu.menu_glossary(cfg, read=read)

    assert calls == [
        {
            "series": "S",
            "chapter": None,
            "model": "gemma4:31b-cloud",
            "min_chapters": 5,
            "dry_run": True,
            "as_json": False,
        }
    ]
    assert remaining == []


def test_menu_glossary_bootstrap_calls_cmd_reference_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pre-check runs first — the leaf goes straight to cmd_reference and nothing else (card)."""
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "cmd_reference")
    read, remaining = make_read("5", "1", "", "", "", "", "", "0")

    menu.menu_glossary(cfg, read=read)

    assert calls == [
        {
            "series": "S",
            "min_locks": 3,
            "model": "gemma4:31b-cloud",
            "dry_run": False,
            "force": False,
        }
    ]
    assert remaining == []


# ---------------------------------------------------------------- MENU2: menu_watermark


def test_pick_float_parses_reprompts_and_defaults() -> None:
    read, remaining = make_read("x", "0.25")
    assert menu.pick_float("Left edge", 0.0, read=read) == 0.25
    assert remaining == []
    assert menu.pick_float("Left edge", 0.5, read=make_read("")[0]) == 0.5
    assert menu.pick_float("Right edge", 1.0, read=make_read("1")[0]) == 1.0


def test_menu_watermark_add_calls_watermark_add(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "watermark_add")
    read, remaining = make_read("1", "1", "0.1", "", "0.9", "", "promo banner", "", "0")

    menu.menu_watermark(cfg, read=read)

    assert calls == [{"series": "S", "x0": 0.1, "y0": 0.0, "x1": 0.9, "y1": 1.0, "note": "promo banner"}]
    assert remaining == []


def test_menu_watermark_list_and_remove(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    lists = spy(monkeypatch, "watermark_list")
    removes = spy(monkeypatch, "watermark_remove")

    menu.menu_watermark(cfg, read=make_read("2", "1", "", "0")[0])
    assert lists == [{"series": "S"}]

    menu.menu_watermark(cfg, read=make_read("3", "1", "2", "", "0")[0])
    assert removes == [{"series": "S", "index": 2}]


# ---------------------------------------------------------------- MENU2: menu_filter


def test_menu_filter_run_calls_filter_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "filter_run")
    read, remaining = make_read("1", "1", "a", "", "0")

    menu.menu_filter(cfg, read=read)

    assert calls == [{"series": "S", "chapter": None, "as_json": False}]
    assert remaining == []


def test_menu_filter_add_checks_path_and_adds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "filter_add")
    example = tmp_path / "promo.jpg"
    example.write_bytes(b"fake")  # only .exists() is checked here; filter_add validates the image

    menu.menu_filter(cfg, read=make_read("2", "1", str(tmp_path / "nope.jpg"), "0")[0])
    assert calls == []
    assert "does not exist" in capsys.readouterr().out

    menu.menu_filter(cfg, read=make_read("2", "1", str(example), "y", "", "", "0")[0])
    assert calls == [{"series": "S", "path": example, "global_": True, "name": None}]


def test_menu_filter_restore_picks_one_chapter_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly one chapter name (not a list) reaches the wrapper; the two picks don't cross-wire."""
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "filter_restore")
    read, remaining = make_read("3", "1", "2", "2", "3", "", "0")

    menu.menu_filter(cfg, read=read)

    assert calls == [{"series": "S", "chapter": "Chapter 2", "target": "slice", "index": 3}]
    assert remaining == []


def test_menu_filter_force_picks_one_chapter_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "filter_force")
    read, remaining = make_read("4", "1", "1", "1", "0", "", "0")

    menu.menu_filter(cfg, read=read)

    assert calls == [{"series": "S", "chapter": "Chapter 1", "target": "file", "index": 0}]
    assert remaining == []


# ---------------------------------------------------------------- MENU2: menu_queue


def test_menu_queue_add_defaults_and_stage_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _sp = chapter_cfg(tmp_path)
    calls = spy(monkeypatch, "queue_add")

    menu.menu_queue(cfg, read=make_read("1", "1", "", "a", "", "", "y", "", "0")[0])
    assert calls == [
        {"series": "S", "stage": None, "chapter": None, "priority": 0, "max_attempts": 2, "force": True}
    ]

    menu.menu_queue(cfg, read=make_read("1", "1", "ingest, slice", "1", "2", "3", "n", "", "0")[0])
    assert calls[-1] == {
        "series": "S",
        "stage": ["ingest", "slice"],
        "chapter": ["Chapter 1"],
        "priority": 2,
        "max_attempts": 3,
        "force": False,
    }


def test_menu_queue_list_passes_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = menu_cfg(tmp_path)
    calls = spy(monkeypatch, "queue_list")

    menu.menu_queue(cfg, read=make_read("2", "1", "", "0")[0])
    assert calls == [{"status": "queued"}]

    menu.menu_queue(cfg, read=make_read("2", str(len(STATUSES) + 1), "", "0")[0])  # last option = "all"
    assert calls == [{"status": "queued"}, {"status": None}]


def test_menu_queue_run_zero_means_no_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = menu_cfg(tmp_path)
    calls = spy(monkeypatch, "queue_run")

    menu.menu_queue(cfg, read=make_read("3", "", "0", "", "0")[0])
    assert calls == [{"webhook": None, "max_jobs": None}]  # "0" is no limit, not stop-after-zero

    menu.menu_queue(cfg, read=make_read("3", "http://hook", "5", "", "0")[0])
    assert calls == [{"webhook": None, "max_jobs": None}, {"webhook": "http://hook", "max_jobs": 5}]


def test_menu_queue_job_action_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = menu_cfg(tmp_path)
    cancels = spy(monkeypatch, "queue_cancel")
    pauses = spy(monkeypatch, "queue_pause")

    menu.menu_queue(cfg, read=make_read("4", "3", "7", "", "0")[0])
    assert cancels == [{"job_id": 7}]

    menu.menu_queue(cfg, read=make_read("4", "1", "1", "", "0")[0])
    assert pauses == [{"job_id": 1}]


def test_menu_queue_clear_calls_queue_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = menu_cfg(tmp_path)
    calls = spy(monkeypatch, "queue_clear")

    menu.menu_queue(cfg, read=make_read("5", "", "0")[0])

    assert calls == [{}]


# ---------------------------------------------------------------- MENU2: top-level structure


def test_top_level_options_grow_to_ten_with_new_categories() -> None:
    assert menu.TOP_LEVEL_OPTIONS == (
        "Run pipeline (a series' chapters through the full pipeline, or one stage at a time)",
        "Library (import chapters, package finished chapters, covers/metadata, match two chapter sets)",
        "Translate (translate chapters, judge candidates, summarize story)",
        "Glossary (list, export, import, propose terms, bootstrap from reference chapters)",
        "Watermark (fixed-position regions excluded from translation)",
        "Filter (promo-filter overrides: restore or force-filter a file/slice)",
        "Queue (queue pipeline jobs, drain, pause/resume/cancel/retry)",
        "Models (list, download, remove, verify)",
        "Diagnostics (doctor, hardware report, check for updates)",
        "Launch (web debug viewer, desktop app)",
    )


def test_run_menu_smoke_walk_reaches_each_new_submenu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pick each new top-level entry then back out; run_menu's own dispatch reaches all five."""
    cfg = menu_cfg(tmp_path)
    reached: list[str] = []

    def fake_submenu(name: str) -> Callable[..., None]:
        def submenu(_cfg: Config, *, read: Callable[[str], str] = input) -> None:
            read("")  # consume the "0" that backs out of the real submenu
            reached.append(name)

        return submenu

    for name in ("menu_translate", "menu_glossary", "menu_watermark", "menu_filter", "menu_queue"):
        monkeypatch.setattr(menu, name, fake_submenu(name))
    read, remaining = make_read("3", "0", "4", "0", "5", "0", "6", "0", "7", "0", "0")

    assert menu.run_menu(cfg, read=read) == 0

    assert reached == ["menu_translate", "menu_glossary", "menu_watermark", "menu_filter", "menu_queue"]
    assert remaining == []
