# Next session — work plan and builder queue

Rewritten 2026-09-23 (later the same day), after a session that fixed three performance/correctness bugs in
OCR/translation, changed the translation default to cloud-first, then did a real visual-review tuning pass on
Solo Leveling and fixed four more bugs found that way. Read `docs/HANDOFF.md` and `docs/CHECKPOINT.md` first;
this file says *what to do next and in which order*, with as much of the volume work as possible handed to
`glm-5.3-flash:cloud` builders.

## Where we are
The whole pipeline runs end to end: `omniscan run SERIES` takes raw chapters through ingest → slice → detect →
OCR → translate → judge → inpaint → LaMa → typeset → export in three passes, with a cloud-first translation
default (`gemma4:31b-cloud`, `translategemma:12b` as its automatic rate-limit fallback), an OCR qualification
suite with real per-language recommendations, story memory, glossary term proposals, cover/metadata fetch, a
web viewer with six chapter views, and a desktop app with six pages including Import. ~3 900 tests, protected
by an end-to-end golden GPU test. **The real Solo Leveling chapters have now actually been rendered and looked
at**, which is what found four of this session's bugs — none of them showed up in the test suite.
What is **not** done: sound effects (confirmed this session: the detector never emits an `sfx` region on real
content — a design-level gap, not a quick fix), F2c (fixed-position watermark tool, unwired), portability (the
torch pin is ROCm-only), a full switch to PaddleOCR-VL as the sole KO/ZH default (it's recorded as *a*
recommendation now, not a replacement — a real tradeoff). **The rule stays: the director writes cards and
looks at real output; builders write the code and tests.**

## Start of session (director, ~10 min)
```powershell
Set-Location V:\OmniScan
git status; git log -5 --oneline; git worktree list      # expect: clean main, no worktrees left
uv run --frozen pytest -q                                 # ~3 900 passed, few minutes including the GPU tests
uv run --frozen omniscan doctor                           # GPU = RX 9070 XT (cuda:1), Ollama up
uv run python scripts/omni_builder.py smoke               # connectivity test of glm-5.3-flash:cloud
```
Then: ask the owner the questions at the bottom, write the next card while a builder runs the previous one.
Launch builders as **tracked background calls**; never sleep-poll.

## Ordered queue

| # | ID | What | Tier | State |
|---|---|---|---|---|
| 1 | **M10** | Sound effects: design first (translate as text, keep original, or reletter — the director decides). **Now confirmed blocked on more than "no real raws"** — zero `sfx`-kind regions on either real Solo Leveling chapter; the detector itself never fires that class on real content. Concrete example pulled 2026-09-23 and shown to the owner: Chapter 2 region `r0077` (page 9, a small yellow sign in the background art) is `kind="free_text"`, OCR'd as `"KMA-"`, translated as `"KMA-"` (passed through unchanged) — the pipeline has no notion of "incidental art text that isn't meant to be read/translated at all", only dialogue vs. free text. Needs a decision on whether to tune/retrain detection for SFX/incidental text or take a different approach before any card can be written | T3 (director decides) | design needed, genuinely blocked |
| 2 | ~~(decision)~~ | **Full KO/ZH PaddleOCR-VL switch — resolved 2026-09-23, no switch.** Owner's call: keep the fast engine (`ppocr`) as the global default; PaddleOCR-VL (`ocr-vl-1.6`) stays an opt-in per series for when the extra quality is worth the extra time. **No code work needed** — this was already fully wired before this session: `cfg.ocr.engine: Literal["ppocr","manga_ocr","paddleocr_vl"]` (`core/config.py`), settable per series in `<library_root>/<series>/series.toml`'s `[ocr]` section (`engine = "paddleocr_vl"`, `rec_model = "ocr-vl-1.6"`) or via the desktop app's Settings → Per-series override editor — documented in `docs/USER_GUIDE.md` under "OCR engines and models". | — | **done, nothing to build** |
| 3 | **F2c** | Promo filter tier 3: wire the already-built fixed-position `omniscan watermark` tool (`watermark/store.py`, `watermark/resolve.py` — persistence and box-resolution exist, nothing calls `resolve_watermark_regions` anywhere in the real pipeline) into `detect`/`ocr` the way F2b (tier 1, text patterns, merged) wired its own reclassification | T2 | not started |
| 4 | **(new, from TL1)** | Glossary term matcher (`glossary/match.py`) only strips Korean particles; harmless-but-inert for zh/ja today, will matter once Japanese/Chinese term matching is exercised for real | T2 | flagged by TL1's builder, not scheduled |
| 5 | **(new, from D2)** | `_MAX_BUBBLE_TO_TEXT_AREA_RATIO`'s cap scales with the text box's own area, so a very small text box (e.g. a one-character "!") shrinks the acceptable bubble ceiling with it. No known incident (225-pairing evidence had none), but worth a look if a tiny-text-box mispairing ever shows up in a real chapter | T2 | flagged by D2's builder, watch for it |
| 6 | **A1 continuation** | `omniscan match chapters` (CM1) / `omniscan reference` (GL1) against the Solo Leveling `_reference_en` side — not run yet (the visual-review pass took priority this session and found more real bugs). Worth doing for glossary-quality signal on genuine commercial content | T3 (director) | not started |
| 7 | **B31** | Duplicate / near-duplicate chapter detection (reuse the promo filter's dHash, or `CM1`'s page hashing) | T1 | filler, card to write |
| 8 | **C4b/c** | zh/ja OCR model packs beyond what O1c/O1d already qualified; second-opinion OCR use of PaddleOCR-VL now that it's fast enough to run alongside a primary engine | T3 | after item 6 |
| 9 | **C7d** | Polygon fitting to the bubble outline, font roles (dialogue/shout/thought/narration), colour matching | T3 | after item 6 |
| 10 | **P1/P2** | Portability: torch backend selectable in `pyproject.toml` (ROCm / CUDA / CPU / MPS via `uv` extras + `conflicts`), then a GitHub Actions matrix. Today nothing except this PC can `uv sync` | T3 | not started, decide the shape soon |
| 11 | **C2** | Hybrid GPU JPEG codec decision (question F1); CPU `turbo` is ~145 Mp/s and not the bottleneck | director | open, low urgency |
| 12 | **W1 follow-through** | `model-watch` GitHub Action is merged and runs, but no live run has yet fed a real catalog update back through the qualification suite | T1 | merged, not yet exercised live |
| 13 | **F14** (from B11b) | `web/app.py` now transitively imports torch (via `inpaint/patches.py::load_patches`) for the new patch-PNG route; no functional issue, but if `omniscan serve` running on a no-torch machine ever matters, needs a numpy-only npz reader outside `inpaint/patches.py` | T2 | low priority, noted in `docs/OPEN_QUESTIONS.md` |
| — | **MENU1** | Interactive menu, part 1: framework (numbered pickers, `run_menu`) + core flows (run pipeline, import/pack, models, doctor/hardware/update, launch serve/gui). Owner's instruction 2026-09-23: no more argument-based CLI as the primary interface — bare `omniscan` opens a menu. Additive: every existing subcommand keeps working for scripts/tests/automation | T2 | **in flight** (launched 2026-09-23) |
| — | **MENU2** | Interactive menu, part 2, on top of MENU1's framework: translate/judge/story/glossary/reference workflow, watermark regions, promo filter, job queue | T2 | queued (write after MENU1 merges — same framework, avoid a merge race) |
| later | **M13** | Desktop shell polish, LAN mode | — | `docs/PLAN.md` |

Merged and reviewed since the last rewrite of this file (all with mutation checks where applicable; details in
`docs/reports/<ID>.md`): **D1** (cross-class duplicate detection — a `text_bubble`/`text_free` pair where one
box is contained in the other, IoU alone missed it), **D2** (reject a bubble pairing more than 8x the text's
own area — a real detector artifact, found on Solo Leveling), **F2b** (promo filter tier 1: OCR'd ad/spam text
reclassified as `kind="watermark"`, excluded from translation but not removed — found the source aggregator
injects a gambling ad on most pages, sometimes rendered in English as if official content), **B11b** (inpaint
raw↔clean slider + patch outlines, layout overlay web views — verified against real Solo Leveling data, not
just synthetic tests). Plus two director-only changes with no card (hardware-verified GPU tensor code /
small Qt wiring, not builder-suited): LaMa now tiles an over-size region instead of skipping it (found the
same way as D1/D2 — a real rendered page with untranslated Korean bleeding through); the desktop app's Import
page wired into `MainWindow`'s sidebar as a sixth page (verified with real offscreen screenshots).
**B32** (doctor's required-Ollama-models list derived from the real profile/judge config instead of a stale
hardcoded one) was in flight when this file was last rewritten — check `git log`/`docs/reports/B32.md` for
whether it landed.

Earlier in the same session (first half — full detail in the previous revision of this file, kept in git
history): **O1c** (OCR qualification suite), **O1d** (PaddleOCR-VL engine), **O1e** (series.toml-clobbers-fix,
core-side), **O1f** (batching investigation, reversed after measurement), **C5c** (story memory + glossary
proposals), **L1** (covers/metadata), **W1** (model-watch), **TL1** (language-aware prompts), plus the
PaddleOCR-VL `use_cache`/truncation fix, the Ollama `num_ctx`/one-model-at-a-time fix, and the cloud-first
translation default with rate-limit fallback.

## Card checklist (what made cards succeed)
- Follow `docs/tasks/_TEMPLATE.md` and the O1e/D1/D2/F2b level of detail: exact signatures, **real numbers
  computed from the actual bug first** (not invented), numbered acceptance tests, file allowlist, stop-and-ask
  rule. Every card this session that pinned the *exact* real boxes/scores/ratios from a genuine bug produced a
  clean, minimal, first-try-correct diff.
- Tell the builder which existing files to imitate — it copies patterns well and invents badly.
- GPU code / real-hardware-timed code: the director does it directly rather than writing a card (this
  session: LaMa tiling, all the OCR/translation performance fixes) — a card can't hand a builder a GPU to
  measure against.
- After the builder: `git rebase main` (README/USER_GUIDE conflicts are routine: keep both sides), read
  `docs/reports/<ID>.md` in full (the "Questions"/"Deviations" sections catch real spec bugs and spec gaps
  every time — this session's B11b report caught a genuine contradiction in its own card: "use load_patches"
  vs "don't import torch"), ruff/pyright/pytest, **for a web/GUI card, verify against real data or a real
  screenshot, not just the test suite** (this session: hit the new inpaint/layout/patch routes against the
  real Solo Leveling chapter and got back a real image; rendered real offscreen screenshots for the Import
  page), `git merge --ff-only`, push, remove worktree and branch.
- Builders that hit the turn limit have usually finished the work but not committed: check `git status` in the
  worktree before deciding anything. **`git worktree remove --force` can leave an empty directory behind** —
  always `rm -rf` it before retrying, or `omni_builder.py` silently runs in a git-less folder.
- **Never run multiple heavy tasks at once** (pytest suites, GPU pipelines, benchmarks) — it starves everything
  and invalidates timing data. A background web/API server started for manual verification counts as a "heavy
  task" that must be cleanly killed afterward — `Start-Process`'s own PID is a wrapper, not necessarily the
  process holding the port; check `Get-NetTCPConnection` if a route mysteriously 404s after a restart.
- Measure before changing anything performance-related; be ready to reverse a hypothesis when real hardware
  disagrees.
- **The highest-value work this session came from actually running the pipeline on real, non-trivial content
  and reading the output — not from more tests.** All four second-half bugs (LaMa skip, cross-class duplicate,
  oversized bubble, ad watermark) were invisible in ~3 900 passing tests and only showed up on genuine
  commercial manhwa. When real raw data exists, budget time to actually look at rendered pages before moving
  on to the next feature.

## Working rule (owner): use the GLM builders more, the director does less by hand
The director writes cards (exact interfaces, real numbers from the actual bug), launches builders, merges and
verifies against real output; **implementation, tests, docs and routine mutation reviews go to the builders**
(keep slots busy; write the next card while one runs). One-off work that needs this machine (real-hardware
benchmarking, GPU-timed measurement, anything requiring the live Ollama server or a real rendered page) is the
director's own job.

## Questions for the owner
1. **Sound effects / incidental art text (M10)** — shown a concrete example (`r0077`, "KMA-" sign,
   see item 1 above) 2026-09-23; owner reviewed it and is deciding next steps. Worth pursuing
   (retrain/fine-tune the detector for a real SFX/incidental-text class, or a different approach), or
   leave such regions as they are for now?
2. ~~Full PaddleOCR-VL switch for KO/ZH~~ — **resolved 2026-09-23**: keep `ppocr` as the default, keep
   PaddleOCR-VL opt-in per series (already fully wired, see item 2 above).
3. Is `omniscan match chapters`/`omniscan reference` against the Solo Leveling `_reference_en` side (item 6)
   worth doing now, or should real tuning continue elsewhere first?
