---
name: windows-shell-tips
description: "Hard-won rules for working on this Windows machine with the PowerShell/Bash tools, and for tracking background builder runs reliably"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 18b96bfa-62d0-4f52-b6c0-2ca8b649a9c3
  modified: 2026-09-18T22:53:28.778Z
---

**Background task tracking — launch-and-track, don't detach-then-poll.**
**Why:** early in the builder loop I detached builder runs and polled with `pgrep`-style wait loops; the pattern matched by model name, so back-to-back
runs merged, no completion fired, and the user had to ask "is it done?". The real complaint was slow noticing.
**How to apply:** launch the blocking command itself as a `run_in_background: true` tool call —
`uv run python scripts/omni_builder.py run <ID>` (run from `V:\OmniScan`) — so the completion notification is tied 1:1 to that card. When a notification
arrives, act on it in that same turn (review/merge/dispatch next). Never `sleep`-poll.

**PowerShell tool:** the working directory resets to `C:\Users\limex` on every call → `Set-Location V:\OmniScan` in each command. Multi-line here-strings with
quotes/apostrophes get tangled (terminator errors) → write files with the Write/Edit tools, or put a small script file in the scratchpad and run it with
`uv run python <file>`. Use `uv run --frozen` to avoid re-locking. Output of German-locale tools (robocopy, wsl) is localised; robocopy exit code 1 = success.

**Windows quirks that bit us:** creating symlinks needs Developer Mode/admin (WinError 1314) → tests guard for it; `git checkout -- path` restores from the
index, not HEAD; npm blocks package postinstall scripts unless `--allow-scripts=<pkg>` is passed; pass long prompts to CLIs on stdin, not as arguments
(.cmd shims mangle them).

**Line endings:** the repo has `.gitattributes` (`* text=auto eol=lf`); scripts written from PowerShell should use `newline="\n"` in Python `write_text`.
