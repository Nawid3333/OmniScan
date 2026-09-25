# Next session — work plan and builder queue

> **Development paused on 2026-09-25** (owner's decision; back in some years or when the program is needed).
> Start with the "Development paused" section at the top of `docs/HANDOFF.md` — it has the state and the resume steps.

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

## 2026-09-25: lettering quality pass (director, cloud session — do this first)
The owner asked for output "on the level of an official translation": no cleaning artifacts, good font
matching, sound effects redrawn in a matching style. Built and CPU-verified on synthetic pages over drawn art
with the real LaMa model (`docs/benchmarks/lettering-quality.md`, decisions in `docs/DECISIONS.md` →
Pipeline design): glyph-precise cleaning, LaMa without lettering as context, balloon-shaped phrase-aware
lettering with chapter-wide sizes, `webtoon`/`manga` presets with the owner's own fonts pluggable, SFX
detection by lexicon + a whole-page CRAFT sweep for the effects the detector misses + style-matched
redraw, watermarks erased. **Not yet run on real chapters** (the cloud session had no GPU
and no Hugging Face access). First thing on this machine:

1. `uv run --frozen pytest -q` — including the GPU golden test `test_e2e_synthetic.py`, which the cloud
   session could not run (its "English drawn where the Korean was" check depends on the new layout).
2. `uv run python scripts/lettering_demo.py` and look at `data/lettering_demo/*.png`.
3. `uv run python scripts/sfx_sweep_check.py --out data/sweep_check` — the sound-effect sweep with this
   machine's models (PaddleOCR-VL reads the swept effects here; the cloud session only had the PP-OCRv5
   Korean recogniser: 26/30). The first run downloads CRAFT (78 MB) from EasyOCR's GitHub release; to serve
   it from our own release, upload `craft_mlt_25k.zip` (sha256 in `config/models.toml`) to `models-v1`.
4. `uv run omniscan run SoloLeveling --force -s detect -s ocr -s inpaint -s inpaint_lama -s typeset -s export`
   (detect, ocr, inpaint and typeset stage versions were bumped) and **look at the pages**: SFX found by
   the sweep (`sweep_*` metrics of the ocr stage; extend `config/sfx_text.toml` with misses), watermarks
   gone, the chapter's typical size, LaMa on real screentone, any lettering overflow.

## Ordered queue

| # | ID | What | Tier | State |
|---|---|---|---|---|
| 1 | **M10** | Sound effects. 2026-09-23 "leave as-is" was **superseded 2026-09-25** by the owner's request for style-matched SFX: lexicon reclassification after OCR + measured style + redraw (`ocr/sfx.py`, `typeset/sfx.py`); size-only detection exists but is off (signs such as `r0077` would pass it). Still open: the detector itself has no working SFX class — unknown effects outside the lexicon stay free text | director | **built, real-chapter check pending** |
| 2 | ~~(decision)~~ | **Full KO/ZH PaddleOCR-VL switch — resolved 2026-09-23, no switch.** Owner's call: keep the fast engine (`ppocr`) as the global default; PaddleOCR-VL (`ocr-vl-1.6`) stays an opt-in per series for when the extra quality is worth the extra time. **No code work needed** — this was already fully wired before this session: `cfg.ocr.engine: Literal["ppocr","manga_ocr","paddleocr_vl"]` (`core/config.py`), settable per series in `<library_root>/<series>/series.toml`'s `[ocr]` section (`engine = "paddleocr_vl"`, `rec_model = "ocr-vl-1.6"`) or via the desktop app's Settings → Per-series override editor — documented in `docs/USER_GUIDE.md` under "OCR engines and models". | — | **done, nothing to build** |
| 3 | ~~**F2c**~~ | Promo filter tier 3: wire the already-built fixed-position `omniscan watermark` tool into `detect`, the way F2b (tier 1) wired its own reclassification. Position-based, no OCR text needed — hooks in at `detect` time, before OCR. **Merged 2026-09-23** (`detect/watermark_position.py`: `_MIN_OVERLAP_IOA = 0.5` IoA threshold, `DetectStage` version 2→3) | T2 | **done, merged** |
| 4 | ~~**GL2**~~ | Glossary term matcher (`glossary/match.py`) only stripped Korean particles; harmless-but-inert for zh, a real gap for ja. **Merged 2026-09-23**: `find_terms(text, entries, lang)` now takes the source language, `JAPANESE_PARTICLES`/`PARTICLES_BY_LANG` added, zh/en get `()` on purpose (documented correct, not a gap). The builder caught and fixed 5 more `check_locked_terms` call sites in `judge.py` the card's own file list had missed | T2 | **done, merged** |
| 5 | **(new, from D2)** | `_MAX_BUBBLE_TO_TEXT_AREA_RATIO`'s cap scales with the text box's own area, so a very small text box (e.g. a one-character "!") shrinks the acceptable bubble ceiling with it. No known incident (225-pairing evidence had none), but worth a look if a tiny-text-box mispairing ever shows up in a real chapter | T2 | flagged by D2's builder, watch for it |
| 6 | **A1 continuation** | `omniscan match chapters` (CM1) run 2026-09-23 against the real Solo Leveling raw vs `_reference_en` sides: **0 matched, both chapters unmatched on both sides.** Not a matcher bug — probed the raw quality matrix directly: Chapter 1 raw has 12 pages, Chapter 1 reference has 22 (nearly 2x); best per-page dhash similarity across the whole chapter tops out at ~0.72 (CM1's own "clean match" fixture expects >0.8). The two sides do not appear to be the same underlying chapter content/pagination — likely a real mismatch between the wfwf504 (raw) and mgread (reference) sources for these specific 2 chapters, not a code defect. `omniscan reference` (GL1) was **not** run against this pair — it would have nothing usable to extract. Needs either different (genuinely-corresponding) reference chapters, or accepting this test pair can't validate GL1 | T3 (director) | **run, real finding — needs owner input on next step, see below** |
| 7 | ~~**B31**~~ | Duplicate / near-duplicate chapter detection, reusing CM1's own page-hash quality metric self-compared (not the 1:1 `align`). **Merged 2026-09-23**: `match/duplicates.py`, `omniscan match duplicates ROOT`, `min_quality=0.9` default (deliberately conservative, no real incident to tune against yet) | T1 | **done, merged** |
| 8 | **C4b/c** | zh/ja OCR model packs beyond what O1c/O1d already qualified; second-opinion OCR use of PaddleOCR-VL now that it's fast enough to run alongside a primary engine | T3 | after item 6 (which found a data problem, not a green light) |
| 9 | **C7d** | Polygon fitting to the bubble outline, font roles (dialogue/shout/thought/narration). Colour matching is **partially done** already: the director's own text-colour sampling fix (`ocr/assemble.py::_sample_text_color`, Otsu-split ink colour from the source art, merged 2026-09-23, no card) covers the "match the original ink colour" half of this item; polygon fitting and font roles remain | T3 | after item 6 |
| 10 | ~~**P1/P2**~~ | Portability: torch backend selectable in `pyproject.toml` (ROCm / CUDA / CPU / MPS via `uv` extras + `conflicts`), then a GitHub Actions matrix. **Merged and green 2026-09-23** on ubuntu-latest/windows-latest/macos-latest after 5 CI-driven fix rounds (see `docs/PLAN.md` item 8 for the full root-cause chain — mostly one recurring pattern: AMD's rocm10 index mirroring torch's whole pure-Python dependency closure with incomplete platform coverage) | T3 | **done, merged** |
| 11 | **C2** | Hybrid GPU JPEG codec decision (question F1); CPU `turbo` is ~145 Mp/s and not the bottleneck | director | open, low urgency |
| 12 | **W1 follow-through** | `model-watch` GitHub Action is merged and runs, but no live run has yet fed a real catalog update back through the qualification suite. First live run 2026-09-25 (issue 6): 65 "new" repos, none usable — 64 Paddle-inference/ONNX builds of PP-OCR (the PaddlePaddle patterns matched every build, not only `*_safetensors`) and `ogkalu/comic-text-segmenter-yolov8m` (ultralytics YOLO). Patterns narrowed to `*PP-OCRv*_safetensors`, the YOLO repo ignored | T1 | exercised live, no candidate yet |
| 13 | ~~**F14**~~ (from B11b) | `web/app.py` transitively imported torch via `inpaint/patches.py::load_patches` even though `load_patches` itself never touches torch. **Merged 2026-09-23**: the module-level `import torch` moved into a `TYPE_CHECKING` guard; `omniscan serve`'s import path is now torch-free (regression-tested via a subprocess `sys.modules` check) | T2 | **done, merged** |
| — | ~~**MENU1**~~ | Interactive menu, part 1: framework (numbered pickers, `run_menu`) + core flows. **Merged 2026-09-23** | T2 | **done, merged** |
| — | ~~**MENU2**~~ | Interactive menu, part 2: translate/judge/story, glossary, watermark regions, promo filter, job queue — 5 new submenus, ~19 leaves, on top of MENU1's exact framework. **Merged 2026-09-23** | T2 | **done, merged** |
| — | **B33** | Web viewer: on-demand run buttons on every chapter view + an in-place, click-to-edit translation overlay (reuses InpaintView's clean-compositing + LayoutView's box positioning). **Merged 2026-09-23** | T2 | **done, merged** |
| later | **M13** | Desktop shell polish, LAN mode | — | `docs/PLAN.md` |

Also merged 2026-09-23, director-only (no card, GPU/pixel work or tooling): text-colour sampling for
inpainted regions (`ocr/assemble.py::_sample_text_color`, Otsu-split ink colour from the source art instead of
guessing black/white — see C7d above); the web debug viewer's Inpaint/Edit views now merge in a chapter's
`inpaint_lama.json`/`patches_lama.npz` pass instead of only ever showing the flat pass's uncleaned placeholder;
pixel-exact raw|final scroll sync + an independent-scroll toggle in the Reader view; on-demand pipeline
run + editable final-line endpoints in `web/app.py`; `scripts/omni_builder.py ask` no longer crashes printing
an answer with a character the console's codepage can't represent; `uv sync --all-extras` (not bare `uv sync`)
is now the documented setup command in `CLAUDE.md`/`README.md` — a plain `uv sync` silently drops the `gui`
extra (PySide6), which was producing the same ~44 stale pyright errors and skipped `tests/gui/**` on every
fresh worktree all session before being traced to the docs rather than the environment.

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
1. ~~Sound effects (M10)~~ — **resolved 2026-09-25**: style-matched replacement, built (item 1 above). New
   question: which official release's look should the presets be tuned to (E3 — 2–3 reference pages)?
2. ~~Full PaddleOCR-VL switch for KO/ZH~~ — **resolved 2026-09-23**: keep `ppocr` as the default, keep
   PaddleOCR-VL opt-in per series (already fully wired, see item 2 above).
3. Is `omniscan match chapters`/`omniscan reference` against the Solo Leveling `_reference_en` side (item 6)
   worth doing now, or should real tuning continue elsewhere first?
