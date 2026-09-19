# Decision log

Design and process decisions with the reason and the evidence, so nobody has to rediscover why. Newest first inside each
group. "Status": **firm** (measured or explicitly decided), **default** (my choice until the owner answers, see
`docs/OPEN_QUESTIONS.md`), **planned** (not built yet). Master plan: `docs/PLAN.md`.

## Platform and environment
| Decision | Why / evidence | Status |
|---|---|---|
| **Develop and run on Windows 11 natively; no WSL, no Linux container** | AMD publishes `win_amd64` ROCm wheels for our exact pins; GPU torch, `uv sync`, all tests, the detector and PaddleOCR models gave the same results as WSL, the suite runs ~4× faster. `docs/benchmarks/windows-native.md` | firm |
| **The shipped app is one executable for all systems; native packaging, runtime chosen per hardware, no embedded container** | Containers need WSL2/Hyper-V on Windows and have no Apple GPU access; PyTorch runs natively everywhere (M13, PySide6) | firm goal, packaging planned |
| **`gpu.device = "auto"`** = strongest discrete GPU, else MPS, else CPU; never hard-code `cuda:0` | On Windows the integrated GPU is `cuda:0` and crashes on the first kernel; torch exposes `is_integrated` | firm |
| Python 3.14 + `uv`; PyTorch only from AMD's ROCm index; `triton` required on Linux only | ROCm wheels cover cp310–cp314; AMD lists triton unconditionally but ships no Windows wheel | firm |
| LF line endings everywhere (`.gitattributes`) | Windows checkouts otherwise converted files to CRLF | firm |
| **Never use the PaddlePaddle framework**; run Paddle *models* through PyTorch/`transformers` | Paddle has no ROCm build; the Korean PP-OCRv5 and PP-OCRv6 zh/ja recognition models load from safetensors and read correctly on GPU on Windows and WSL (`scripts/paddle_models_check.py`) | firm |
| Raws are never committed; tests use synthetic images; CC-licensed comics allowed as extra material | Copyright; Pepper&Carrot (CC BY 4.0) is used for real-page checks, kept in `data/` | firm |

## Pipeline design
| Decision | Why / evidence | Status |
|---|---|---|
| **Stages with resumable manifests**: each stage declares inputs (content-hashed), a config subset and outputs; a re-run is a no-op | Hundreds of chapters, cloud rate limits, cheap re-runs after editing one thing (`core/stage.py`) | firm |
| **Strip space**: a chapter is one vertically stitched, width-normalised strip; slices are virtual y-ranges | Webtoon reading order, no physical slice files until export | firm |
| Slicer: cut only inside uniform bands ≥ 50 px, DP toward a target height, forced cuts flagged | Never cut through art/bubbles; verified on synthetic property tests (150 seeds) and real pages | firm |
| **CPU `turbo` codec is the baseline**; the hybrid GPU codec (C2) waits for a real-data profile | rocJPEG hardware decode unavailable (WSL had no VCN; not on Windows); turbo ≈ 141–148 Mp/s (`docs/benchmarks/codec.md`) | firm baseline, C2 open (Q F1) |
| Detection: `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2, Apache-2.0) on overlapping square tiles (tile = min(1280, strip width), overlap 0.5, resized to 640²) | The model was trained on 640² resized images with tall webtoons split vertically; finds bubbles, their text and free SFX on real pages | firm (C3 in review) |
| Detect writes `regions.json` only — **no text masks, no "cut validator"** | Masks come from OCR line boxes in the inpaint stage; all later stages work in strip space, so a region crossing a cut only matters when export re-cuts slices | firm |
| OCR: PP-OCRv5 **server text-line detector** + PP-OCRv5 Korean / PP-OCRv6 zh+ja recognition via `transformers`; PaddleOCR-VL as second opinion | Probe: 7/7 lines, recognition CER 0.048 on boxes from the detector, no false positives on real art (`docs/benchmarks/ocr-probe.md`) | firm (C4a), second opinion planned |
| **All torch vision models run in fp32** (comic detector, PP-OCR detector and recognizer): no `.half()` / autocast | On this ROCm-Windows stack MIOpen's FP16 kernels are unreliable: the PP-OCR detector fails in fp16/bf16 at every shape tried, and RT-DETR fp16 works only at batch 8 (142.7 tiles/s) but fails at batch 1, 2 and 3 (`miopenStatusUnknownError`, `invalid device function`), so a short chapter or a partial last batch would crash. fp32 works at every batch size (RT-DETR 31–65 tiles/s, PP-OCR det 25 ms per 960×704) — a 200-tile chapter costs ~3 s. `docs/benchmarks/ocr-probe.md`, `windows-native.md` addendum | firm (re-test after driver/torch updates) |
| **DB post-processing of the text-line heatmap runs on the CPU (OpenCV)** — an exception to "GPU end-to-end" | Measured 3.4 ms per page vs ~41 ms for a GPU connected-components version; ~0.1 s per chapter; needs `opencv-python-headless` | firm on evidence (question F9 covers the other CPU exception) |
| Inpaint: flat fill for uniform bubble interiors (C6a), LaMa for text over art (C6b); export composites only the masked pixels onto the original | Minimal quality loss and IO | flat fill firm; LaMa firm on evidence |
| **LaMa = TorchScript `big-lama.pt`, fp32, one fixed 512×512 window per region, warmed once per process** | 30 ms per crop steady state, output identical outside the mask, text on painted art removed convincingly; every new input shape costs a 10–25 s warm-up, fp16 fails (`docs/benchmarks/lama-probe.md`) | firm (C6b planned) |
| Model groups are never co-resident with the local LLM (VramManager evicts Ollama models) | 16 GB VRAM; WDDM silently pages instead of failing | firm |

## Translation
| Decision | Why / evidence | Status |
|---|---|---|
| **Several candidate runs + a judge**, profiles in TOML; judge = `gemma4:31b-cloud` | Different models resolve ambiguity differently (probe); the judge picks/merges using glossary and context | firm design, judge planned (C5) |
| **Glossary**: auto-proposed, owner locks entries, locked terms enforced in the prompt and verified afterwards; particle-aware matcher; honorifics kept romanised | Consistency over hundreds of chapters; Korean particles break naive matching | firm (honorifics = default, Q D1) |
| Send `think:false` for thinking models; parse answers tolerantly; pre-substitute locked terms for translategemma | gemma4 spent 353 s and returned nothing with thinking on; cloud models ignore Ollama's `format` and wrap JSON in prose/fences; translategemma ignored the glossary until substituted | firm (`docs/benchmarks/translation-probe.md`) |
| Regions are sent with whitespace collapsed to one line | Source line breaks leaked into the English; the typesetter re-wraps | firm |
| `endpoint = "local"` for all shipped profiles, including `*-cloud` models | The local Ollama daemon proxies cloud models; no API key needed | firm |
| Reference mode (already-translated chapters as baseline, auto-lock terms seen in ≥ 3 chapters) | Human translations are the best glossary/style source | planned (C12, Q D7) |

## Acquisition
| Decision | Why / evidence | Status |
|---|---|---|
| extract.pics → **Cloudflare Worker relay** (kept in the repo, deployed by GitHub Actions) → WebSocket to the app; GitHub itself cannot be the webhook | The extract.pics form has only a URL (no headers/secret); GitHub's inbound triggers need an `Authorization` header — a live test got 401/405 | firm; relay built, not deployed; client unfinished (docs not fetchable, Q A4) |
| **No acquisition from paid DRM platforms** | Obfuscated canvases; a different, riskier problem; scope boundary in the plan | firm (Q C2 asks to confirm) |
| Local import (`omniscan import`) is a first-class second path | Groups already have raws on disk | firm, built |

## Tooling and process
| Decision | Why / evidence | Status |
|---|---|---|
| **Director + builders**: the director writes exact cards and reviews; Claude Code CLI on Ollama (`glm-5.3-flash:cloud`) builds; ≤ 3 concurrent (owner, 2026-09-19); no deepseek/kimi | Token cost and the $26.77 incident; flash landed every well-specified card cleanly | firm |
| Builder cards: exact interfaces, numbered acceptance tests, file allowlist, "stop and ask" | Vague specs made builders burn tokens; their questions caught real spec bugs | firm |
| Web debug tool: FastAPI + Svelte/TypeScript, read-only except the Filtered view's Restore (JSON content-type required as a CSRF guard) | Reviewable stage output for scanlation groups; a cross-site page can send `text/plain` POSTs without a preflight | firm |
| Job queue: SQLite, single worker per queue, notifications never fail a job | Simple, resumable; multi-worker not needed yet | firm |
| Questions live in `docs/OPEN_QUESTIONS.md`; state lives in `docs/CHECKPOINT.md`; conversation learnings live here and in `docs/HANDOFF.md` | The owner asked that nothing depend on chat history | firm |
| Test and tuning data must be legally usable: own chapters via `omniscan import`, open-licensed comics (Pepper&Carrot, CC BY 4.0: Korean/English/text-free versions of the same art), official free chapters. Claude does not scrape unlicensed aggregator sites (toongod, wfwf, …) or fetch fan translations, and does not pick commercial titles to download | Those copies are unlicensed; Pepper&Carrot gives real Korean lettering plus exact ground truth without that problem (owner asked for scraping on 2026-09-19; declined, alternative delivered) | firm |
| `acquire` is built against the extract.pics API only after its docs text is available (the docs site is a JS app: `WebFetch`, `llms.txt` and `openapi.json` all fail); the API key lives only in `~/.config/omniscan/secrets.env` (`EXTRACTPICS_API_KEY`), never in the repo, cards or builder prompts | Builders test against fakes; a key pasted in chat should be rotated if the transcript is shared | firm |
