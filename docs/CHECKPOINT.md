# Checkpoint — 2026-09-19

Where the project stands and exactly how to continue. `docs/PLAN.md` is the master plan; this file is the
current-state snapshot. Repo: `~/projects/omniscan` (WSL), `main` is pushed to `Nawid3333/OmniScan`.

## State in one paragraph
Everything up to a working **ingest → slice** pipeline, the **web debug tool** (5 views), the **translation candidate
runner**, glossary, promo filter, job queue, import, packaging and docs is built, reviewed and merged (≈1 900 tests, ruff
and pyright clean). The first real-data checks have been done: slicing on real comic pages, live translation through
three Ollama models, and the text/bubble detector on real pages. **Detection (C3) is half built on branch `C3`**;
OCR, judge, inpaint, typeset and export do not exist yet, so there is no end-to-end run yet.

## What works today (CLI)
`doctor`, `import`, `ingest`, `slice`, `filter run|restore`, `glossary list|export|import`, `watermark add|list|remove`,
`pack` (CBZ/PDF), `serve` (+ `npm run dev` in `webui/`), `queue add|list|run|pause|resume|cancel|retry|clear`,
`translate` (candidate runs; needs an `ocr.json`). Stubs (exit 2): `acquire detect ocr judge inpaint typeset export run
reference`. Web views: Slicer, OCR (reads `ocr.json`), Translation, Reader, Filtered (Restore = the only write action).

## Evidence gathered (all reproducible)
- **Codec:** rocJPEG cannot run in WSL; CPU `turbo` ≈ 141–148 Mp/s (`docs/benchmarks/codec.md`). Hybrid GPU codec (C2) is
  still an open decision — my recommendation: build detect/OCR first, profile a real chapter, then decide.
- **Translation probe** (`docs/benchmarks/translation-probe.md`, `scripts/probe_translation.py`): gemma4 needs
  `think:false`; cloud models ignore Ollama's JSON `format` (parser is tolerant); translategemma ignores the glossary
  unless locked terms are pre-substituted. Live `omniscan translate` run: 3 profiles, 10/10 regions each, 0 missing.
- **Detector** (`scripts/probe_detector.py` on branch `C3`): `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2, 42.7 M
  params, classes bubble / text_bubble / text_free) loads and runs on the RX 9070 XT via `transformers`; on real pages it
  finds the speech bubbles, the text inside and free-standing SFX (missed two small "PLOP" SFX at threshold 0.3).
  Not benchmarked for speed; fp16 untested.
- **Real-data slice run:** two Pepper&Carrot episodes (CC BY 4.0, David Revoy) in `~/omniscan/library/PepperCarrot/`
  (with `ATTRIBUTION.txt`; never in the repo). 2481 px pages stitched to 11 222 / 19 078 px strips, cut into 3 / 6 slices,
  no forced cuts, 1.8 s / 0.6 s. Artifacts in `~/omniscan/work/PepperCarrot/`.

## In flight: C3 detection (branch `C3`, worktree `~/projects/omniscan-wt/C3`, pushed, NOT merged)
Done: `[detect]` config section, shared `ingest.strip.load_strip` (SliceStage now uses it and also hashes the raw
images, fixing stale slices when pixels change but sizes do not), `detect/tiles.py`, `detect/postprocess.py`
(no tests yet), `scripts/probe_detector.py`.
Still to do, in order:
1. `detect/model.py` — RT-DETR-v2 wrapper: tile → `F.interpolate` to 640×640 (bilinear, antialias, no mean/std
   normalisation, /255 only), batch forward, `processor.post_process_object_detection`, boxes back in tile pixels.
2. `gpu/groups.py` — `build_vram_manager(cfg)` registering the `"vision"` group (`{"detector": Detector}`).
3. `detect/stage.py` — `DetectStage` (`gpu_group="vision"`, inputs = raw images + `slices.json`, outputs
   `regions.json` as `RegionsArtifact` with empty text): plan tiles, skip tiles that only touch blank/filtered slices,
   detect, `merge_detections`, `build_regions`.
4. CLI: replace the `detect` stub; `_run_stages` must build the VRAM manager when a stage has a `gpu_group`.
5. Tests: tiles, postprocess (merge/association/reading order/slice assignment), stage with a fake detector and a fake
   scheduler, GPU smoke test (`@pytest.mark.gpu`), SliceStage-invalidates-on-pixel-change test; docs (README status row,
   USER_GUIDE, ARCHITECTURE).
6. Look at real results: copy `regions.json` to `ocr.json` for a chapter and open the OCR view, check threshold/tile size.
Design decisions already taken: tile side = min(1280, strip width), overlap 0.5; tile-edge boxes are penalised and
truncated duplicates dropped; text↔bubble by containment (≥ 0.6); several text boxes in one bubble become one region;
regions belong to the slice holding their centre and are dropped in blank/filtered slices; reading order = rows top→bottom,
left→right (or right→left for manga). **Deliberately not in detect:** text masks (C6 derives them from OCR line boxes) and
the PLAN's "cut validator" (all later stages work in strip space; a region crossing a cut only matters when export
re-cuts slices, so it belongs to export).

## Next, in priority order
1. Finish C3 (above), then **C4 OCR** (`PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors` recognition is verified via
   `scripts/paddle_models_check.py`; detection / line grouping is not).
2. C5 judge + story memory + glossary post-check (text-only work; can start before OCR exists), then C6 inpaint,
   C7 typeset, export.
3. Unblocked builder cards (flash, up to 2 at a time): **acquire plumbing** (sources.toml, resumable downloader, non-chapter
   image filter, DRM-domain warning — everything except the extract.pics request/response shapes), duplicate-chapter
   detection, `doctor` check that translation-profile models exist, model pinning/integrity.

## Waiting on you (the user)
The full list (about 50 questions with my defaults) is **[docs/OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)** — ask 2–3 of them at natural pauses
and record answers there. The most pressing ones:
- **1–2 real Korean raw chapters** (`omniscan import "<folder>" --series Sample --chapter "Chapter 1"`): needed to tune
  detection/OCR on real Korean text. Everything so far was validated on English/synthetic material.
- Optional: `gh auth login` in WSL (enables PRs); Cloudflare token + account id, extract.pics key, and the hook URL
  (plan M0 step 8); paste the extract.pics API docs so `acquire` can be finished (the site is a JS app I cannot fetch);
  decide whether the hybrid GPU codec (C2) should come sooner.

## How to work the loop (for the next session)
- **Builders:** `cd ~/projects/omniscan && scripts/omni-builder run <ID>` (default `glm-5.3-flash:cloud`; `--model glm`
  for glm-5.3 only when a card is genuinely hard). Launch it as a background tool call so the completion notification
  arrives; at most 2 at once. Do not use deepseek/kimi as builders (a single run burned ~$27 and hit the account limit).
- **Card → review → merge:** write `docs/tasks/<ID>.md` (exact interfaces, numbered acceptance tests, file allowlist,
  stop-and-ask rule), `ruff format` it and commit; after the builder finishes: `git fetch ~/projects/omniscan main:main-sync`,
  `GIT_EDITOR=true git rebase main-sync` in the worktree (every card edits the README status table, so expect that
  conflict; keep both rows), read `docs/reports/<ID>.md`, run pytest + ruff + pyright (+ `npm run check/test/build` in
  `webui/` for web cards), fix issues yourself, `git merge --ff-only`, push, remove worktree and branch. Builders' "Questions"
  sections have caught real spec bugs every time — read them. Mutation-check test-only cards (break the code, see the tests fail).
- **Gotchas:** `chmod +x scripts/omni-builder` before committing it; delete builder scratch files in a worktree before
  rebasing; `uv run` from another worktree just re-points the shared venv's editable install (harmless); files with
  apostrophes are easier to write through the `\\wsl.localhost\Ubuntu-26.04\…` path than through `wsl -e bash -lc '…'`.
- **Data:** never commit raws or samples (`samples/`, `library/` are gitignored); tests use synthetic fixtures.
