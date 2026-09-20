# Product spec — owner requirements of 2026-09-19/20 and what we decided

This file captures the requirements the owner stated in chat, the evidence gathered, the decisions, and the card queue. `docs/PLAN.md` stays the master plan; this is its addendum for the desktop product. Status of every card: `docs/NEXT.md`.

## 0. Principles
1. **Nothing heavy ships inside the program.** The app is small; a settings screen lists every model (purpose, size, languages, hardware needs, status) with a download button; the user decides what to install (decision U2a/U2b, `docs/DECISIONS.md`).
2. **Hardware-aware.** The program detects the GPUs/CPU/RAM/VRAM/backends and tells the user, per model, whether it is fine, slow, or incompatible and why ("needs 8 GB VRAM, you have 4 GB"). Every model must be usable on some hardware: GPU (CUDA, ROCm, MPS) and CPU.
3. **Defaults come from evidence, not from names.** A qualification suite (scoring against Pepper&Carrot ground truth in ko/cn/ja) decides which model is the default per language; users can override.
4. **Every step is inspectable.** Each stage has a debugger view and a manual mode (for translation groups); a run can be stepwise (pause after each step with a one-chapter preview) or fully automatic.
5. **Legal boundary.** No crawlers or per-site downloaders for unlicensed sites (see §9).

## 1. OCR engines and models
**Owner wish:** PaddleOCR v6 or newer, all models and sizes, CPU and GPU builds; user can choose between engines (PP-OCR, manga-ocr, PaddleOCR-VL); default = the biggest PP-OCR v6.
**Evidence (2026-09-20, `omniscan eval` on Pepper&Carrot Korean episodes 6 and 9, same detector and pipeline, only the OCR repos changed):**

| Combination | Ep 6 char-recall / CER micro / page chrF | Ep 9 char-recall / CER micro / page chrF |
|---|---|---|
| v5 server det + **v5 Korean mobile rec** (current) | 0.852 / 0.113 / **0.475** | 0.949 / 0.032 / 0.193 |
| v6 medium det + v6 medium rec | 0.743 / 0.113 / 0.105 | 0.944 / 0.039 / 0.189 |
| v6 medium det + v5 Korean rec | 0.778 / 0.100 / 0.348 | 0.954 / 0.029 / 0.218 |

The published PP-OCRv6 models on Hugging Face (`PaddlePaddle/PP-OCRv6_{tiny,small,medium}_{det,rec}_safetensors`, Apache-2.0, 2–88 MB) are tagged **en + zh only**; there is no Korean-specific v6 recogniser (v5 has one). On Korean the v6 recogniser collapses page-level accuracy (0.475 → 0.105), and the v6 detector finds fewer characters. **Consequence: "v6 biggest as default" is adopted per language where it wins** — expected zh/en (to be measured on the Chinese Pepper&Carrot pages), while Korean stays on the v5 Korean model until a v6 Korean recogniser exists or the suite shows a better one. The catalog carries all sizes so the user can pick.
**Catalog scope (card O1a):** PP-OCRv6 tiny/small/medium det+rec; PP-OCRv5 det (server/mobile) and per-language mobile rec; PaddleOCR-VL 1.6 / 1.5 / base (+ GGUF builds for CPU/llama.cpp); manga-ocr (`kha-white/manga-ocr-base`, Japanese, 444 MB; a 2025 safetensors variant, 121 MB); the comic text/bubble detector already used (`ogkalu/comic-text-and-bubble-detector`). ONNX variants exist for the PP-OCR models and are a later CPU-speed option (needs an onnxruntime engine).
**Engines (card O1b):** `ppocr` (line detector + recogniser), `manga_ocr` (recogniser on region/bubble crops), `paddleocr_vl` (VLM on crops). All sit behind the existing recogniser protocol (`read(crops) -> [(text, score)]`) so the OCR stage, translation and typesetting do not change. Settings: `ocr.engine`, model ids per role, language. The owner's "heavy stack" table is a starting point; sizes there are approximate and unverified (the real ones are in the catalog).

## 2. Model manager, hardware detection, updates, qualification
- **Done:** catalog `config/models.toml` + `omniscan models list|download|remove|verify` (`--json` for the GUI), the vision/OCR/LaMa mirror (release `models-v1`, private repo → upstream fallback), loaders use installed folders, `doctor` reports missing required models.
- **H1 (hardware detection + compatibility):** `omniscan hardware [--json]`; per-model requirements in the catalog; `models list` shows ok / slow / warn / incompatible with the reason; the GUI reads the same JSON.
- **O1a (catalog for all OCR models and sizes)**, **O1b (engines)**, **O1c (qualification suite):** `scripts/qualify_ocr.py` runs candidates over the Pepper&Carrot sets per language and writes a table (recall, CER, page chrF, time, VRAM); a candidate becomes a default only if it beats the current default and the pipeline still passes the golden test.
- **model-watch (GitHub Action):** scheduled workflow that checks the upstream repos of catalog entries for new revisions/versions, opens an issue or PR with the diff and the CPU-runnable part of the qualification; the GPU part is run locally (`scripts/qualify_ocr.py`). Updating a model = new catalog revision + new mirror asset; older models stay listed as "older version" so users can keep or remove them.
- **App updates:** GitHub Releases built by Actions (U6 = client side, U4/U5 = packaging and swap).

## 3. Slicer
**Analysis of the three options in the owner's notes:**
- *`webtoon-slicer` (PyPI):* unverified third-party package; we would add a dependency for something we already have.
- *Gutter script (row variance < 5, search backwards for a gutter):* our slicer is a superset — band detection with colour-drift tolerance, dynamic-programming cut planning between min/target/max heights, forced cuts at the least-detailed row when no gutter exists, a page-mode variant (B14), blank/promo flags; 129 mutants checked (Q1).
- *Overlapping sliding window:* we already use it where it matters — tiled detection/OCR with overlap and NMS/containment merging in strip space.
**Decision:** one engine is not enough for every series, so **strategies** behind one interface, chosen per series in settings: `smart` (current, default), `page` (page-mode), `fixed_overlap` (fixed height + overlap, for continuous art), `simple_gutter` (the owner's script, as a baseline). **Compare mode:** run all strategies on one chapter and show the cut lines side by side in the debugger; the user picks and the choice is saved per series (card S2).

## 4. Promo / credit / banner filtering
Owner's 3-tier design, mapped on what exists:
- **Tier 2 (pHash of repeated banners): exists** (promo filter, dHash + examples, `filter run|restore`, Filtered view).
- **Tier 1 (post-OCR text patterns: Discord, Patreon, "translated by", URLs, ...): missing** → card F2: configurable regex list (global + per series), applied after OCR to drop boxes and mark slices, results visible in the debugger and restorable.
- **Tier 3 (position and shape heuristics: first 5 % / last 10 % of the chapter, no bubbles + URL-like lines, very wide short slices): missing** → in F2; only the first/last slices are examined (cheap).
- **Batch:** the filter runs across all chapters of a series in one pass before OCR; only first/last slices per chapter are hashed/OCR-probed.

## 5. Pipeline control and preview
Run modes: full run; any subset of stages (`omniscan run -s ...` exists); **step-by-step**: after each stage a one-chapter preview is shown and the user continues or aborts; **automatic**: no previews, for series whose settings are known to work. Needs a hook in the pipeline runner (`after_stage(chapter, stage, outcome)` that may pause/abort) and the preview renderers per stage (slice cuts, detection boxes, OCR text, translation, inpaint before/after, typeset, final) — cards P1 (runner gates) and B11' (views). The same views are the debugger and the manual-tool panes (edit boxes/text/masks, re-run one step).

## 6. Library, reader, side-by-side, manual tools
- Library folders configurable with defaults (`data/raws`, `data/output`, `data/translated-check` today); per series: raws, output, reference translations.
- **Reader** inside the app for output (and raws); **side-by-side** raw | output with synced or independent scrolling — also the core of the debugger and of manual review by translation groups.
- **Manual tools** for groups: edit regions/OCR text/translations, brush and clone/inpaint tools, re-typeset; export edited output.
- **Fonts:** automatic font matching to the raw lettering (font recognition from the crops, matched against the installed OFL font set) as default, with a manual override in advanced settings.
- **Advanced settings to expose** (research and ideas, to be prioritised): slicer strategy and heights; detector threshold; OCR engine/model; per-series glossary/style; number of translation candidates and judge model; inpaint mode (flat/LaMa) and mask growth; font family/size limits/stroke/alignment; bubble fitting margins; SFX policy (keep/translate/replace); output format/quality; run mode (step/auto); VRAM budget; cache location; per-stage re-run.

## 7. Covers and metadata
**Question:** local LLM/Ollama with web search, or Tavily, or something better? **Decision:** none of them for covers. Use structured metadata APIs (free, no LLM): AniList (GraphQL, `Media(type: MANGA)` with `coverImage`), MangaDex API (covers and metadata), Jikan (MyAnimeList) as fallback; the user can always upload their own cover from a folder. Card L1 verifies the APIs' terms and rate limits first, caches covers locally, and stores the source URL. An LLM is only useful for fuzzy title matching (translated/romanised titles) and can be an optional fallback.

## 8. Acquisition
- **`omniscan acquire` with the extract.pics client (B33b).** API shapes as given by the owner: `POST https://api.extract.pics/v0/extractions` with the key in `Authorization` and body `{"url": <chapter page>, "mode": "basic"|"advanced"}` (advanced = executes JavaScript and scrolls, for lazy-loaded pages); the response carries an id with status `pending`; poll `GET /v0/extractions/{id}` until `done`; the result lists `data.images[].url`. Cost: 1 credit (basic) / 2 credits (advanced) per URL; free plan 100 credits/month; failed extractions cost nothing; a page that offers an "all chapters" reader costs one extraction for many chapters. The client must show the credit cost before running, respect the monthly limit, send the chapter page as `Referer` when downloading, and keep page order.
- **User-supplied links, made less tedious:** paste a list, a text file (one URL per line), or a URL template with a number range; select which chapters to run (checkbox list, ranges), see the estimated credits, resume a half-finished series.
- **Completeness checks per chapter** (so no broken chapter reaches the pipeline): all images decoded, page count compared with the series median and with the previous chapter, width consistency, gaps in numbering, last-image sanity (not a banner), optional re-extraction in advanced mode when a chapter looks short; a preview/debug panel with thumbnails before the chapter is accepted into the library.
- **Boundary (unchanged):** no HTML crawler that lists chapters of arbitrary sites, no per-site downloaders for unlicensed aggregator sites, no gallery-dl/Playwright scraping integration aimed at them, and the assistant does not run downloads against such sites. Sites the owner is licensed to use can be added later as explicit, individually reviewed adapters. Paid DRM platforms stay refused.

## 9. Automation on GitHub
Actions: CPU tests on push (P2), app release builds (U4), `model-watch` (§2), `relay-deploy` (exists). Repository visibility (private today) decides whether end users can pull the model mirror and app updates anonymously (owner decision pending).

## 10. Working method
Builders (`glm-5.3-flash:cloud`) implement, test and mutation-review; the director writes cards and checks real output. `.claude/settings.json` deny rules (the owner removed `.claudeignore`, which Claude Code does not read) keep lockfiles, `data/`, `models/`, caches and binaries out of context. When a big or generated file must be inspected, a builder does it (card G1: `omni_builder.py ask`).

## 11. Card queue for this spec (see `docs/NEXT.md` for states)
H1 hardware + compatibility · O1a catalog of all OCR models/sizes · O1b OCR engines (ppocr sizes, manga-ocr, PaddleOCR-VL) · O1c qualification suite + model-watch Action · F2 filter tiers + batch · S2 slicer strategies + compare · P1 runner gates and previews · B11' debugger/side-by-side views · R1' reader · L1 library covers/metadata · B33b extract.pics client + chapter selection + completeness checks · G1 `ask` helper · then the PySide6 shell (U3) on top of the JSON interfaces.
