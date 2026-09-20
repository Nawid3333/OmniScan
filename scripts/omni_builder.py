"""omni-builder — run a task card headless with Claude Code on Ollama (the GLM builder). Windows, Linux and macOS.

Model policy (2026-09-18): glm-5.3-flash:cloud is the default and should be used for essentially every card — flash
is meant to be used MORE than glm-5.3:cloud because the full model is noticeably more token-costly. Use `--model glm`
only for a card that is genuinely hard/large or that flash already failed on. Up to 3 builders may run at once (the owner allowed 3 on 2026-09-19; Ollama
Pro allows 3 concurrent requests, so with 3 builders running nothing is left for a live Ollama check: use OMNI_SLOTS=2 then). Avoid deepseek/kimi as builders: one deepseek run
burned ~5M input tokens / $26.77 over 71 turns and hit the account's session limit before finishing a card.

Usage (from the repo root):
  uv run python scripts/omni_builder.py run <CARD_ID> [--model MODEL] [--max-turns N]     # new run in its own worktree
  uv run python scripts/omni_builder.py resume <CARD_ID> <FEEDBACK_FILE> [--model MODEL]  # continue with review feedback
  uv run python scripts/omni_builder.py smoke [--model MODEL]                             # 3-turn connectivity test
  uv run python scripts/omni_builder.py ask "QUESTION" [--file PATH ...] [--max-turns N] [--wait] [--cwd DIR]
                                                                                          # read-only lookup, prints only the answer
Model aliases: flash (default) | glm | deepseek | kimi — or any full Ollama model name.

Env overrides: OMNI_REPO (default: this repo), OMNI_WT (default: "<repo>-wt" next to it), OMNI_SLOTS (default 3),
               OLLAMA_URL (default http://localhost:11434), CLAUDE_BIN (path to the claude executable).
The card text is passed to `claude -p` on stdin (a multi-KB prompt through a Windows .cmd shim would be mangled).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import IO, NoReturn

REPO = Path(os.environ.get("OMNI_REPO", Path(__file__).resolve().parents[1])).resolve()
WT_ROOT = Path(os.environ.get("OMNI_WT", REPO.parent / f"{REPO.name}-wt"))
SLOTS = int(os.environ.get("OMNI_SLOTS", "3"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LOG_DIR = REPO / ".builder" / "logs"
LOCK_DIR = Path(tempfile.gettempdir()) / "omni-builder"
DEFAULT_MODEL = "glm-5.3-flash:cloud"
MODEL_ALIASES = {
    "flash": "glm-5.3-flash:cloud",  # default: token-efficient
    "glm": "glm-5.3:cloud",  # escalation / harder Python
    "deepseek": "deepseek-v4-pro:cloud",  # algorithmic / numeric GPU code (expensive, see policy note)
    "kimi": "kimi-k2.7-code:cloud",  # TypeScript / web UI (has vision)
}
SESSION_RE = re.compile(rb'"session_id":"([^"]*)"')
ASK_SYSTEM = (
    "You are a read-only research assistant for the OmniScan repository. Answer the question using only the Read, "
    "Grep and Glob tools. Be concise: at most 40 lines, plain text or short bullets, quote `path:line` for every "
    "claim about code, give exact numbers when asked for numbers. If the answer is not in the files, say so instead "
    "of guessing. Never print the content of secrets, API keys or `.env` files. Do not suggest changes unless asked."
)


def build_ask_prompt(question: str, files: Sequence[str]) -> str:
    """The stdin prompt for an `ask` session: system rules, the question, optional starting files."""
    prompt = ASK_SYSTEM + "\n\nQuestion: " + question
    if files:
        prompt += "\n\nStart with these files (relative to the repo root): " + ", ".join(files)
    return prompt


def extract_answer(stream: bytes) -> str | None:
    """The stripped `result` text of the last stream-json result line; None on error results or when absent."""
    result: dict[str, object] | None = None
    for line in stream.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            result = obj
    if result is None or result.get("is_error"):
        return None
    text = result.get("result")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def die(message: str) -> NoReturn:
    print(f"omni-builder: {message}", file=sys.stderr)
    raise SystemExit(1)


def say(message: str) -> None:
    print(f"omni-builder: {message}", file=sys.stderr, flush=True)


def claude_env(workdir: Path | None = None) -> dict[str, str]:
    """Environment for the builder; `workdir` pins `import omniscan` to that worktree's src.

    Worktrees share one .venv whose editable install is re-pointed by every `uv run`, so concurrent builders could otherwise
    import each other's code. PYTHONPATH entries precede the .pth entry.
    """
    env = dict(os.environ)
    env.update(
        {
            "ANTHROPIC_BASE_URL": OLLAMA_URL,
            "ANTHROPIC_AUTH_TOKEN": "ollama",
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": DEFAULT_MODEL,
            "CLAUDE_CODE_SUBAGENT_MODEL": DEFAULT_MODEL,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "API_TIMEOUT_MS": "600000",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000",  # GLM has 1M; cap at 200k to keep token use low
        }
    )
    if workdir is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(workdir / "src"), env.get("PYTHONPATH", "")]))
    if sys.platform == "win32" and "CLAUDE_CODE_GIT_BASH_PATH" not in env:
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if Path(candidate).is_file():
                env["CLAUDE_CODE_GIT_BASH_PATH"] = candidate
                break
    return env


def claude_bin() -> str:
    override = os.environ.get("CLAUDE_BIN")
    if override:
        return override
    found = shutil.which("claude")
    if found is None:
        die("claude CLI not found (install it: npm install -g @anthropic-ai/claude-code)")
    if sys.platform == "win32" and Path(found).suffix.lower() in {".cmd", ".ps1", ".bat"}:
        native = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.is_file():
            return str(native)
    return found


def check_ollama() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=5):
            return
    except OSError as exc:
        die(f"Ollama not reachable at {OLLAMA_URL} ({exc}); is Ollama running?")


def try_lock(handle: IO[bytes]) -> bool:
    """Non-blocking exclusive lock on the first byte of `handle`."""
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def try_acquire_slot() -> IO[bytes] | None:
    """One non-blocking pass over the SLOTS lock files; the handle holds the slot until closed, None when all are busy."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(1, SLOTS + 1):
        handle = (LOCK_DIR / f"slot{index}.lock").open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        if try_lock(handle):
            say(f"acquired slot {index}/{SLOTS}")
            return handle
        handle.close()
    return None


def acquire_slot() -> IO[bytes]:
    """Block until one of the SLOTS builder slots is free; the returned handle holds it until the process exits."""
    while True:
        handle = try_acquire_slot()
        if handle is not None:
            return handle
        say(f"all {SLOTS} slots busy, waiting...")
        time.sleep(30)


def link_venv(worktree: Path) -> None:
    """Share the main repo's .venv with a worktree (avoids re-downloading multi-GB ROCm wheels per worktree)."""
    target, link = REPO / ".venv", worktree / ".venv"
    if not target.is_dir() or link.exists():
        return
    if sys.platform == "win32":  # a junction needs no symlink privilege
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def run_claude(
    workdir: Path,
    log: Path,
    model: str,
    max_turns: int,
    prompt: str,
    *extra: str,
    settings: Path | None = None,
    permission_mode: str = "acceptEdits",
) -> int:
    """Run `claude -p` in `workdir`, appending stream-json to `log`; returns its exit code."""
    log.parent.mkdir(parents=True, exist_ok=True)
    start = log.stat().st_size if log.exists() else 0
    cmd = [
        claude_bin(),
        "-p",
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        permission_mode,
        "--settings",
        str(settings or REPO / ".builder" / "settings.json"),
        "--output-format",
        "stream-json",
        "--verbose",
        *extra,
    ]
    # Write straight to the log files (no pipe to another process while running).
    with log.open("ab") as out, log.with_suffix(".stderr").open("ab") as err:
        code = subprocess.run(
            cmd, input=prompt.encode("utf-8"), cwd=workdir, stdout=out, stderr=err, env=claude_env(workdir)
        ).returncode
    session = None
    with log.open("rb") as fh:
        fh.seek(start)
        match = SESSION_RE.search(fh.read())
        if match:
            session = match.group(1).decode()
            log.with_suffix(".session").write_text(session, encoding="utf-8")
    say(f"claude exited with code {code} (session {session or 'unknown'})")
    return code


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def main() -> int:
    parser = argparse.ArgumentParser(prog="omni_builder.py", description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume", "smoke"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("card")
        if name == "resume":
            p.add_argument("card")
            p.add_argument("feedback", type=Path)
        p.add_argument("--model", default=DEFAULT_MODEL, type=resolve_model)
        p.add_argument("--max-turns", type=int, default=150)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--file", dest="files", action="append", default=[], metavar="PATH")
    p.add_argument("--model", default=DEFAULT_MODEL, type=resolve_model)
    p.add_argument("--max-turns", type=int, default=25)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--cwd", type=Path, default=REPO)
    args = parser.parse_args()

    check_ollama()
    if args.command == "smoke":
        prompt = "Run: uv run python --version. Then reply with exactly one line: SMOKE OK <python version>."
        return run_claude(REPO, LOG_DIR / "smoke.jsonl", args.model, 3, prompt)
    if args.command == "ask":
        if args.wait:
            slot = acquire_slot()  # held until the call ends
        else:
            slot = try_acquire_slot()  # held until the call ends
            if slot is None:
                say(f"no free slot (all {SLOTS} builders busy); use --wait or read the file yourself")
                return 3
        log = LOG_DIR / f"ask-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.jsonl"
        say(f"asking {args.model} (log: {log})")
        code = run_claude(
            args.cwd,
            log,
            args.model,
            args.max_turns,
            build_ask_prompt(args.question, args.files),
            settings=REPO / ".builder" / "ask-settings.json",
            permission_mode="default",
        )
        answer = extract_answer(log.read_bytes()) if log.is_file() else None
        if answer is None:
            say(f"ask produced no answer (claude exit code {code}; log: {log})")
            return 1
        print(answer)
        return 0

    slot = acquire_slot()  # held until the builder process exits
    log = LOG_DIR / f"{args.card}.jsonl"
    if args.command == "run":
        spec = REPO / "docs" / "tasks" / f"{args.card}.md"
        if not spec.is_file():
            die(f"missing task card {spec}")
        worktree = WT_ROOT / args.card
        if not worktree.is_dir():
            subprocess.run(
                ["git", "-C", str(REPO), "worktree", "add", "-q", str(worktree), "-b", args.card, "main"],
                check=True,
            )
            link_venv(worktree)
        say(f"running {args.card} with {args.model} in {worktree} (log: {log})")
        code = run_claude(worktree, log, args.model, args.max_turns, spec.read_text(encoding="utf-8"))
        say(f"{args.card} finished; report: {worktree / 'docs' / 'reports' / (args.card + '.md')}")
        return code
    session_file = log.with_suffix(".session")
    if not session_file.is_file() or not session_file.read_text(encoding="utf-8").strip():
        die(f"no session recorded for {args.card}")
    session = session_file.read_text(encoding="utf-8").strip()
    feedback = args.feedback.read_text(encoding="utf-8")
    return run_claude(WT_ROOT / args.card, log, args.model, args.max_turns, feedback, "--resume", session)


if __name__ == "__main__":
    raise SystemExit(main())
