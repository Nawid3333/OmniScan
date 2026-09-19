"""mutate — run a list of hand-written mutants against a set of test files and report which ones the tests miss.

A mutants file is a Python file that defines `MUTANTS`, a list of `(file, old, new)` tuples: in `file` (path relative to the repo
root) the text `old` — which must occur exactly ONCE — is replaced by `new`, the tests are run, and the file is restored.
Add a short label as a 4th element if you like: `(file, old, new, "flip < to <=")`.

Usage (from the repo root or a worktree root):
  python scripts/mutate.py check MUTANTS_FILE                          # every `old` is found exactly once and the result compiles
  python scripts/mutate.py run MUTANTS_FILE -t TEST [-t TEST ...]      # run all mutants; prints KILLED / SURVIVED per mutant
      [--only 3,5-9] [--timeout 120]

`-t` accepts a test file or `file::test_name`. Mutants run one after the other (they edit the real files) and each file is
restored in a `finally` block, and a run that is killed hard is undone by the next `check`/`run` (backup in `.mutate_backup/`); `git diff --stat -- src` must still be empty afterwards.
Exit code 0 when no mutant survived, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
PENDING = (
    ROOT / ".mutate_backup" / "pending.json"
)  # the file being mutated right now, so a killed run can be undone

Mutant = tuple[str, str, str, str]


def load_mutants(path: Path) -> list[Mutant]:
    """Read `MUTANTS` from a Python file and normalise every entry to (file, old, new, label)."""
    raw = runpy.run_path(str(path)).get("MUTANTS")
    if not isinstance(raw, list) or not raw:
        sys.exit(f"{path}: define a non-empty list MUTANTS")
    out: list[Mutant] = []
    for i, entry in enumerate(raw):
        if len(entry) not in (3, 4) or not all(isinstance(x, str) for x in entry):
            sys.exit(f"{path}: entry {i} must be (file, old, new) or (file, old, new, label) with strings")
        file, old, new = entry[0], entry[1], entry[2]
        label = entry[3] if len(entry) == 4 else ""
        out.append((file, old, new, label))
    return out


def parse_only(spec: str | None, total: int) -> list[int]:
    """Turn '3,5-9' into sorted zero-based indices; a bad range aborts."""
    if not spec:
        return list(range(total))
    chosen: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        first, last = int(lo), int(hi or lo)
        chosen.update(range(first, last + 1))
    bad = [i for i in chosen if not 0 <= i < total]
    if bad:
        sys.exit(f"--only: indices out of range: {sorted(bad)} (0..{total - 1})")
    return sorted(chosen)


def apply(mutant: Mutant) -> tuple[Path, str, str] | str:
    """Return (path, original text, mutated text), or a string describing why the mutant is invalid."""
    file, old, new, _ = mutant
    path = ROOT / file
    if not path.is_file():
        return f"file not found: {file}"
    original = path.read_bytes().decode("utf-8")
    count = original.count(old)
    if count != 1:
        return f"`old` occurs {count} times in {file} (must be exactly 1 — add surrounding context)"
    mutated = original.replace(old, new)
    if mutated == original:
        return "`new` equals `old`"
    if file.endswith(".py"):
        try:
            compile(mutated, file, "exec")
        except SyntaxError as exc:
            return f"mutant does not compile: {exc.msg} (line {exc.lineno})"
    return path, original, mutated


def recover() -> None:
    """Restore the file of a previous run that was killed while a mutant was applied."""
    if not PENDING.is_file():
        return
    pending = json.loads(PENDING.read_text(encoding="utf-8"))
    (ROOT / pending["file"]).write_bytes(pending["original"].encode("utf-8"))
    PENDING.unlink()
    print(f"recovered {pending['file']} from an interrupted run (a mutant was still applied)")


def line_of(mutant: Mutant) -> int:
    """1-based line of `old` in its file (0 when unknown)."""
    path = ROOT / mutant[0]
    if not path.is_file():
        return 0
    text = path.read_bytes().decode("utf-8")
    at = text.find(mutant[1])
    return text.count("\n", 0, at) + 1 if at >= 0 else 0


def run_tests(tests: list[str], timeout: int) -> tuple[bool, str]:
    """Run pytest on `tests` from the repo root; returns (passed, last output lines)."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", "pytest", *tests, "-x", "-q", "-p", "no:warnings", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, "timeout (counts as killed: an infinite loop is a visible failure)"
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-3:])
    return proc.returncode == 0, tail


def cmd_check(args: argparse.Namespace) -> int:
    """Validate a mutants file without running any test."""
    recover()
    mutants = load_mutants(Path(args.mutants))
    bad = 0
    for i, m in enumerate(mutants):
        res = apply(m)
        if isinstance(res, str):
            bad += 1
            print(f"[{i}] INVALID {m[0]}: {res}")
    print(f"{len(mutants)} mutants, {bad} invalid")
    return 1 if bad else 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the selected mutants against the tests and print one line per mutant plus a summary."""
    recover()
    mutants = load_mutants(Path(args.mutants))
    if not args.tests:
        sys.exit("give at least one -t TEST")
    ok, tail = run_tests(args.tests, args.timeout)
    if not ok:
        print("baseline: the tests fail WITHOUT any mutation — fix that first:\n" + tail)
        return 2
    killed = survived = invalid = 0
    for i in parse_only(args.only, len(mutants)):
        m = mutants[i]
        res = apply(m)
        if isinstance(res, str):
            invalid += 1
            print(f"[{i}] INVALID  {m[0]}: {res}")
            continue
        path, original, mutated = res
        started = time.monotonic()
        PENDING.parent.mkdir(exist_ok=True)
        PENDING.write_text(json.dumps({"file": m[0], "original": original}), encoding="utf-8")
        try:
            path.write_bytes(mutated.encode("utf-8"))
            passed, _ = run_tests(args.tests, args.timeout)
        finally:
            path.write_bytes(original.encode("utf-8"))
            PENDING.unlink(missing_ok=True)
        state = "SURVIVED" if passed else "KILLED"
        killed += not passed
        survived += passed
        what = m[3] or f"{m[1].strip()[:60]!r} -> {m[2].strip()[:60]!r}"
        print(f"[{i}] {state:8} {m[0]}:{line_of(m)}  {what}  ({time.monotonic() - started:.1f}s)", flush=True)
    print(f"summary: {killed} killed, {survived} survived, {invalid} invalid")
    return 1 if survived or invalid else 0


def main() -> None:
    """Parse arguments and dispatch."""
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # labels may hold non-ASCII (Korean) snippets
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="validate a mutants file")
    check.add_argument("mutants")
    check.set_defaults(func=cmd_check)
    run = sub.add_parser("run", help="run mutants against tests")
    run.add_argument("mutants")
    run.add_argument(
        "-t", "--tests", action="append", default=[], help="test file or file::test (repeatable)"
    )
    run.add_argument("--only", help="comma list / ranges of mutant indices, e.g. 0-9,14")
    run.add_argument("--timeout", type=int, default=120, help="seconds per pytest run (default 120)")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
