"""Tests for scripts/omni_builder.py `ask` (card G1): fake claude CLI, no Ollama, no network."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import IO, Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "omni_builder.py"
_spec = importlib.util.spec_from_file_location("omni_builder", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

ANSWER = "pages land under data/slices/<chapter> (scripts/slicer.py:120)."

# The fake `claude` records argv/stdin/cwd to calls.jsonl next to itself, then plays back transcript.txt.
PAYLOAD = """\
import json, os, sys
from pathlib import Path

here = Path(__file__).resolve().parent
record = {
    "argv": sys.argv[1:],
    "stdin": sys.stdin.buffer.read().decode("utf-8"),
    "cwd": os.getcwd(),
}
with (here / "calls.jsonl").open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\\n")
sys.stdout.write((here / "transcript.txt").read_text(encoding="utf-8"))
"""


def write_fake_claude(tmp_path: Path, transcript_text: str) -> Path:
    """Write the fake claude executable (payload + platform shim) and its canned transcript; return its path."""
    (tmp_path / "fake_claude_payload.py").write_text(PAYLOAD, encoding="utf-8")
    if sys.platform == "win32":
        fake = tmp_path / "claude.cmd"
        fake.write_text(f'@"{sys.executable}" "%~dp0fake_claude_payload.py" %*\n', encoding="utf-8")
    else:
        fake = tmp_path / "claude"
        fake.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/fake_claude_payload.py" "$@"\n',
            encoding="utf-8",
        )
        fake.chmod(0o755)
    (tmp_path / "transcript.txt").write_text(transcript_text, encoding="utf-8")
    return fake


def transcript(answer: str, *, is_error: bool = False) -> str:
    """A canned stream-json transcript ending in the given result text."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "fake-session"}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "working..."}]}}),
        json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}),
        json.dumps({"type": "result", "is_error": is_error, "result": answer}),
    ]
    return "\n".join(lines) + "\n"


def calls(tmp_path: Path) -> list[dict[str, Any]]:
    """What the fake claude recorded, one dict per invocation."""
    path = tmp_path / "calls.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def patch_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate the script: no Ollama check, 2 slots, tmp lock/log dirs."""
    monkeypatch.setattr(script, "check_ollama", lambda: None)
    monkeypatch.setattr(script, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(script, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(script, "SLOTS", 2)


def run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    """Run script.main() with the given command line; returns its exit code."""
    monkeypatch.setattr(sys, "argv", ["omni_builder.py", *argv])
    return script.main()


def arg_after(argv: list[Any], flag: str) -> str:
    """The argv element following `flag`."""
    return argv[argv.index(flag) + 1]


# --- 1. extract_answer ------------------------------------------------------------


def test_extract_answer_strips_final_result() -> None:
    stream = (
        b'{"type":"assistant","message":{}}\n{"type":"result","is_error":false,"result":"  the answer\\n"}\n'
    )
    assert script.extract_answer(stream) == "the answer"


def test_extract_answer_last_result_wins() -> None:
    stream = b'{"type":"result","is_error":false,"result":"first"}\n{"type":"result","result":"second"}\n'
    assert script.extract_answer(stream) == "second"


def test_extract_answer_is_error() -> None:
    stream = b'{"type":"result","result":"good"}\n{"type":"result","is_error":true,"result":"boom"}\n'
    assert script.extract_answer(stream) is None


def test_extract_answer_empty_result() -> None:
    assert script.extract_answer(b'{"type":"result","result":"   "}\n') is None
    assert script.extract_answer(b'{"type":"result"}\n') is None


def test_extract_answer_skips_garbage() -> None:
    stream = b"not json at all\n[1, 2]\n{broken\n" + b'{"type":"result","result":"ok"}\n'
    assert script.extract_answer(stream) == "ok"


def test_extract_answer_empty_stream() -> None:
    assert script.extract_answer(b"") is None
    assert script.extract_answer(b'{"type":"assistant","message":{}}\n') is None


# --- 2. build_ask_prompt -----------------------------------------------------------


def test_build_ask_prompt_with_files() -> None:
    prompt = script.build_ask_prompt("Where is X?", ["a.md", "dir/b.py"])
    assert prompt == (
        script.ASK_SYSTEM
        + "\n\nQuestion: Where is X?"
        + "\n\nStart with these files (relative to the repo root): a.md, dir/b.py"
    )


def test_build_ask_prompt_without_files() -> None:
    assert script.build_ask_prompt("Q?", []) == script.ASK_SYSTEM + "\n\nQuestion: Q?"


# --- 3. try_acquire_slot / acquire_slot --------------------------------------------


def test_try_acquire_slot_cycles_and_acquire_slot_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(script, "SLOTS", 2)
    first, second = script.try_acquire_slot(), script.try_acquire_slot()
    assert first is not None and second is not None
    assert script.try_acquire_slot() is None  # all 2 busy
    second.close()
    third = script.try_acquire_slot()  # freed slot re-acquired
    assert third is not None
    third.close()

    def no_sleep(_seconds: float) -> None:
        raise AssertionError("acquire_slot slept although a slot was free")

    monkeypatch.setattr(script.time, "sleep", no_sleep)
    assert script.acquire_slot() is not None


# --- 4. ask end to end --------------------------------------------------------------


def test_ask_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.setenv("CLAUDE_BIN", str(write_fake_claude(tmp_path, transcript(ANSWER))))
    patch_script(monkeypatch, tmp_path)
    code = run_main(
        monkeypatch,
        [
            "ask",
            "Where do slicer pages go?",
            "--file",
            "docs/ARCHITECTURE.md",
            "--file",
            "src/omniscan/core/slicer.py",
            "--cwd",
            str(workdir),
        ],
    )
    out, err = capsys.readouterr()
    assert code == 0
    assert out == ANSWER + "\n"
    assert "slicer" not in err  # stdout carries only the answer

    recorded = calls(tmp_path)
    assert len(recorded) == 1
    argv = recorded[0]["argv"]
    assert arg_after(argv, "--settings").endswith("ask-settings.json")
    assert arg_after(argv, "--model") == "glm-5.3-flash:cloud"
    assert arg_after(argv, "--max-turns") == "25"
    assert arg_after(argv, "--permission-mode") == "default"
    assert "acceptEdits" not in argv
    stdin_text = recorded[0]["stdin"]
    assert stdin_text.startswith(script.ASK_SYSTEM)
    assert "\n\nQuestion: Where do slicer pages go?" in stdin_text
    assert stdin_text.endswith(
        "Start with these files (relative to the repo root): docs/ARCHITECTURE.md, src/omniscan/core/slicer.py"
    )
    assert os.path.normcase(str(recorded[0]["cwd"])) == os.path.normcase(str(workdir))
    assert len(list((tmp_path / "logs").glob("ask-*.jsonl"))) == 1


# --- 5. slot handling ----------------------------------------------------------------


def test_ask_busy_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CLAUDE_BIN", str(write_fake_claude(tmp_path, transcript("x"))))
    patch_script(monkeypatch, tmp_path)
    held = [script.try_acquire_slot(), script.try_acquire_slot()]  # keep the handles open for the whole test
    assert all(h is not None for h in held)
    code = run_main(monkeypatch, ["ask", "q"])
    out, err = capsys.readouterr()
    assert code == 3
    assert out == ""
    assert "no free slot (all 2 builders busy); use --wait or read the file yourself" in err
    assert calls(tmp_path) == []  # claude never ran


def test_ask_wait_proceeds_when_slot_frees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CLAUDE_BIN", str(write_fake_claude(tmp_path, transcript("late answer"))))
    patch_script(monkeypatch, tmp_path)
    held: list[IO[bytes]] = [
        h for h in (script.try_acquire_slot(), script.try_acquire_slot()) if h is not None
    ]
    assert len(held) == 2

    def release(_seconds: float) -> None:
        held[0].close()

    monkeypatch.setattr(script.time, "sleep", release)
    code = run_main(monkeypatch, ["ask", "q", "--wait"])
    out, _err = capsys.readouterr()
    assert code == 0
    assert out == "late answer\n"


# --- 6. failure paths -----------------------------------------------------------------


def test_ask_error_result_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CLAUDE_BIN", str(write_fake_claude(tmp_path, transcript("boom", is_error=True))))
    patch_script(monkeypatch, tmp_path)
    code = run_main(monkeypatch, ["ask", "q"])
    out, err = capsys.readouterr()
    assert code == 1
    assert out == ""
    assert "no answer" in err


def test_ask_no_result_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    no_result = '{"type":"assistant","message":{}}\n{"type":"system","subtype":"init"}\n'
    monkeypatch.setenv("CLAUDE_BIN", str(write_fake_claude(tmp_path, no_result)))
    patch_script(monkeypatch, tmp_path)
    code = run_main(monkeypatch, ["ask", "q"])
    out, err = capsys.readouterr()
    assert code == 1
    assert out == ""
    assert "no answer" in err


# --- 7. run/smoke regression ------------------------------------------------------------


def test_smoke_argv_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_BIN", str(write_fake_claude(tmp_path, transcript("SMOKE OK"))))
    patch_script(monkeypatch, tmp_path)
    assert run_main(monkeypatch, ["smoke"]) == 0
    recorded = calls(tmp_path)
    assert len(recorded) == 1
    argv = recorded[0]["argv"]
    assert arg_after(argv, "--settings").endswith("settings.json")
    assert arg_after(argv, "--permission-mode") == "acceptEdits"
    assert arg_after(argv, "--max-turns") == "3"


# --- 8. ask settings file ---------------------------------------------------------------


def test_ask_settings_json() -> None:
    data = json.loads((script.REPO / ".builder" / "ask-settings.json").read_text(encoding="utf-8"))
    permissions = data["permissions"]
    assert sorted(permissions["allow"]) == ["Glob", "Grep", "Read"]
    for tool in ("Bash", "Edit", "Write"):
        assert tool in permissions["deny"]
    for secret in ("Read(~/.config/omniscan/**)", "Read(**/.env*)", "Read(**/secrets.env)"):
        assert secret in permissions["deny"]


# --- 9. print_answer console-encoding fallback ------------------------------------------


class _NarrowConsoleStdout:
    """Fakes a console bound to a narrow codepage (e.g. Windows cp1252): `write` raises
    UnicodeEncodeError for any character that codepage cannot represent, exactly like the real
    stream — this is what made `ask`'s answer crash the whole command instead of printing it."""

    encoding = "cp1252"

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> int:
        text.encode(self.encoding)  # raises UnicodeEncodeError, same failure mode as the real console
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        pass


def test_print_answer_falls_back_on_a_narrow_console_codepage(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _NarrowConsoleStdout()
    monkeypatch.setattr(sys, "stdout", fake)
    script.print_answer("before → after")  # the real arrow character that crashed this command
    assert "".join(fake.written) == "before ? after\n"


def test_print_answer_prints_unchanged_when_the_console_can_encode_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    script.print_answer("plain ascii answer")
    out, _err = capsys.readouterr()
    assert out == "plain ascii answer\n"
