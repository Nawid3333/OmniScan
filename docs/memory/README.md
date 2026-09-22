# Memory snapshot (2026-09-19)

A copy of the Claude Code auto-memory notes about the owner, the environment and the project, kept in the repo so tools that
do not load Claude's memory (other AIs, a fresh machine) still get them. The live copy for the director's
interactive Claude Code sessions is `C:\Users\limex\.claude\projects\V--OmniScan\memory\` (seeded from these files). `MEMORY.md` is the index.
These notes summarise; `docs/HANDOFF.md`, `docs/DECISIONS.md` and `docs/CHECKPOINT.md` are the maintained sources. Not secrets.
**GLM builder sessions (`scripts/omni_builder.py`) must never read or write the live copy** — it is the director's
cross-session memory, not a builder's; `.builder/settings.json` denies `~/.claude/**` for this reason (2026-09-22,
after a builder session edited it directly). A builder that needs project background reads these repo-tracked files instead.
