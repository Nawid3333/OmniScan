# Next session — work plan and builder queue

Rewritten 2026-09-23, after a session that fixed three real performance/correctness bugs in OCR/translation and
changed the translation default to cloud-first. Read `docs/HANDOFF.md` and `docs/CHECKPOINT.md` first; this file says
*what to do next and in which order*, with as much of the volume work as possible handed to `glm-5.3-flash:cloud` builders.

## Where we are
The whole pipeline runs end to end: `omniscan run SERIES` takes raw chapters through ingest → slice → detect → OCR →
translate → judge → inpaint → LaMa → typeset → export in three passes, now with a cloud-first translation default
(`gemma4:31b-cloud`, `translategemma:12b` as its automatic rate-limit fallback), an OCR qualification suite with real
per-language recommendations, story memory, glossary term proposals (with or without a reference translation), and
cover/metadata fetch. ~3 880 tests, protected by an end-to-end golden GPU test.
What is **not** done: sound effects, tuning on real commercial manhwa (data is now imported — SoloLeveling — but
nobody has tuned against it yet), a visual review of finished pages this session, web views for the render stages,
portability (the torch pin is ROCm-only), the KO/ZH OCR default promotion the qualification suite's own numbers
already support. **The rule stays: the director writes cards and looks at real output; builders write the code and tests.**

## Start of session (director, ~10 min)
```powershell
Set-Location V:\OmniScan
git status; git log -5 --oneline; git worktree list      # expect: clean main, no worktrees left
uv run --frozen pytest -q                                 # ~3 880 passed, few minutes including the GPU tests
uv run --frozen omniscan doctor                           # GPU = RX 9070 XT (cuda:1), Ollama up
uv run python scripts/omni_builder.py smoke               # connectivity test of glm-5.3-flash:cloud
```
Then: ask the owner the questions at the bottom (1 and 2 first), write the next card while a builder runs the previous
one. Launch builders as **tracked background calls**; never sleep-poll.

## Ordered queue

| # | ID | What | Tier | State |
|---|---|---|---|---|
| 1 | **(decision)** | **Promote PaddleOCR-VL as the KO/ZH OCR default.** The qualification suite's own numbers (post use_cache fix, `docs/benchmarks/ocr-qualification-results.md`) say PaddleOCR-VL wins both: KO 0.567 vs 0.475 chrF, ZH 0.741 vs 0.642. `ja` was already promoted to `ppocr-v6-medium` this session (the model that wins there). Needs owner go-ahead, then a small card: `config/models.toml` `recommended_for`/notes (copy the `ja` promotion's shape from this session's `8e75d05`) + `OcrConfig` defaults if the catalog alone doesn't drive runtime dispatch (check first — `recommended_for` was purely informational as of last session, confirm before assuming it's now wired) | T1 (director writes, flash can do the mechanical part) | **numbers ready, not applied** |
| 2 | **A1 (real tuning)** | **Tune on real commercial manhwa.** Data is in place: `data/raws/SoloLeveling/Chapter 1`/`Chapter 2` (Korean) + `_reference_en/Chapter 1`/`Chapter 2` (English), imported this session, deliberately bounded to 2 of 200+/204 chapters. Run `omniscan match chapters` (CM1) and `omniscan reference` (GL1) against it, then tune `detect.threshold`, `ocr.drop_conf`, typeset sizes against genuine commercial line art/lettering (Pepper&Carrot is a different, simpler style). Also worth a plain look: run the chapter through `omniscan run` and *view* the finished pages — nobody has done a visual quality check this session | T3 (director) | not started |
| 3 | **B11b** | Web views for render stages: raw↔clean inpaint slider + patch outlines, layout overlay (card already written, `docs/tasks/B11b.md`, close structural clone of the existing OCR/Translation views) | T2 | **card written, never launched — confirm still wanted before spending a slot** |
| 4 | **(small)** | Wire `ImportView` into the main window's sidebar (`src/omniscan/gui/main_window.py` + the existing `import_view.py`) — both pieces exist and are unblocked, just not connected | T1 | not started, small |
| 5 | **M10** | Sound effects: detection of SFX (`sfx` kind exists in schemas/fixtures), a policy (translate as text, keep, or replace with lettered English) — design first, the director decides | T3 | design needed; blocked on real raws for detection (SoloLeveling now gives some — revisit after item 2) |
| 6 | **(new, from TL1)** | Glossary term matcher (`glossary/match.py`) only strips Korean particles; harmless-but-inert for zh/ja today, will matter once Japanese/Chinese term matching is exercised for real. Small follow-up once Japanese has raws | T2 | flagged by TL1's builder, not scheduled |
| 7 | **F2b → F2c** | Promo filter tier 1 (post-OCR text patterns: Discord/Patreon/"translated by"/URLs) then tier 3 (position/shape heuristics + batch over all chapters). F2a (file/slice-level) is merged | T2 | F2b card next |
| 8 | **B32** | `doctor` checks: every model named in `translation_profiles.toml`/`judge.toml` exists in Ollama (now more important — the fallback chain must be checked too); Hugging Face models pinned to a `revision` with files verified | T1 | filler, card to write |
| 9 | **B31** | Duplicate / near-duplicate chapter detection (reuse the promo filter's dHash, or `CM1`'s page hashing) | T1 | filler, card to write |
| 10 | **C4b/c** | zh/ja OCR model packs beyond what O1c/O1d already qualified; second-opinion OCR use of PaddleOCR-VL now that it's fast enough to actually run alongside a primary engine (was previously ruled out on speed alone — re-open now that use_cache is fixed) | T3 | after item 2 |
| 11 | **C7d** | Polygon fitting to the bubble outline, font roles (dialogue/shout/thought/narration), colour matching | T3 | after item 2 |
| 12 | **P1/P2** | Portability: torch backend selectable in `pyproject.toml` (ROCm / CUDA / CPU / MPS via `uv` extras + `conflicts`), then a GitHub Actions matrix. Today nothing except this PC can `uv sync`. Needs care: use `--model glm` or do it as director | T3 | not started, decide the shape soon |
| 13 | **C2** | Hybrid GPU JPEG codec decision (question F1); CPU `turbo` is ~145 Mp/s and not the bottleneck so far — translation was the bottleneck this session (now fixed), so this drops in relative priority | director | open, low urgency |
| 14 | **W1 follow-through** | `model-watch` GitHub Action is merged and runs, but no live run has yet fed a real catalog update back through the qualification suite; do that once item 1 needs fresh candidates | T1 | merged, not yet exercised live |
| later | **M13** | Desktop shell polish, LAN mode | — | `docs/PLAN.md` |

Merged and reviewed since the last rewrite of this file (all with mutation checks where applicable; details in
`docs/reports/<ID>.md`): **O1c** (OCR qualification suite), **O1d** (PaddleOCR-VL engine), **O1e** (fixed a
`series.toml`-clobbers-candidate-config bug, core-side fix by the director), **O1f** (batching investigation for
PaddleOCR-VL — reversed course after real-hardware measurement showed it was 9-32x *slower*, not faster), **C5c**
(story memory + glossary term proposals from raw OCR text alone), **L1** (cover/metadata: AniList/MangaDex/Jikan +
user upload), **W1** (`model-watch` Action), **Q3/Q4/Q6** (mutation reviews), **U2a/U2b** (model catalog + manager,
loaders read downloaded folders), **U6** (updater core), **TL1** (language-aware LLM prompts — every prompt now
names the chapter's real source language instead of hard-coding Korean; Korean prompts byte-identical to before).
Plus three director-only fixes this session (no card, too small/hardware-bound for a builder): PaddleOCR-VL
`use_cache=True` + no-batching + truncation-scoring fix; Ollama `num_ctx` + one-local-model-at-a-time; cloud-first
translation default with rate-limit fallback.

## Card checklist (what made cards succeed)
- Follow `docs/tasks/_TEMPLATE.md` and the O1e/TL1 level of detail: exact signatures, **golden values computed from
  real code first**, numbered acceptance tests, file allowlist, stop-and-ask rule.
- Tell the builder which existing files to imitate — it copies patterns well and invents badly.
- GPU code: pin the exact preprocessing and add an equivalence test.
- After the builder: `git rebase main` (README/USER_GUIDE conflicts are routine: keep both sides), read
  `docs/reports/<ID>.md` in full (the "Questions" section catches real spec bugs every time), ruff/pyright/pytest,
  fix problems yourself, mutation-check test-only cards, look at real output, `git merge --ff-only`, push, remove
  worktree and branch.
- Builders that hit the turn limit have usually finished the work but not committed: check `git status` in the
  worktree before deciding anything. **`git worktree remove --force` can leave an empty directory behind** — always
  `rm -rf` it before retrying, or `omni_builder.py` silently runs in a git-less folder.
- **Never run multiple heavy tasks at once** (pytest suites, GPU pipelines, benchmarks) — it starves everything
  (looks like a hang) and invalidates any timing data collected during the overlap.
- Measure before changing anything performance-related; be ready to reverse a hypothesis when real hardware
  disagrees (O1f's batching assumption was wrong three separate ways before the actual fix — `use_cache` — was found).

## Working rule (owner): use the GLM builders more, the director does less by hand
The director writes cards (exact interfaces, golden values), launches builders, merges and looks at real output;
**implementation, tests, docs and routine mutation reviews go to the builders** (keep slots busy; write the next
card while one runs). One-off work that needs this machine (real-hardware benchmarking, GPU-timed measurement,
anything requiring the live Ollama server) is the director's own job — that was most of this session's work.
