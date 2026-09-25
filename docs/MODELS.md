# Models and third-party notices

`config/models.toml` is the model catalog: every model OmniScan can download, what it is for, its
size and licence. `omniscan models list` shows the entries with their install status, `omniscan
models download <id>` installs one. The shipped app stays small — the user decides what to download
(owner decision 2026-09-19).

Every entry can carry hardware requirements (card H1): `min_vram_gb` (GPU memory needed to run
comfortably), `min_ram_gb`, `backends` (which of `cuda`/`rocm`/`mps`/`xpu`/`cpu` the model supports;
empty = all), `cpu_ok` (usable without a GPU) and `cpu_speed` (`fast`/`ok`/`slow`/`unusable` — how
it feels on the CPU), plus free-text `notes`. `omniscan hardware` detects the machine and
`omniscan models list` shows the resulting `fit` per model — `ok`, `slow`, `warn` or `incompatible`
— with the reason (e.g. "needs 21 GB of GPU memory, your AMD Radeon RX 9070 XT has 16 GB") in its
messages; `models download` warns for a non-`ok` fit and refuses `incompatible` models unless
`--force` is given.

Three download styles (catalog `format`):

- `zip` — mirrored **unchanged** as assets of the GitHub release
  [`models-v1`](https://github.com/Nawid3333/OmniScan/releases/tag/models-v1) of `Nawid3333/OmniScan`
  (the mirror repo is private today, so a failed download falls back to the upstream source
  automatically); sha256 of the whole zip.
- `hf` — downloaded straight from the Hugging Face repo at a pinned commit, with a sha256 per file
  (`[model.files]`); no mirror. Installed to `<models_dir>/<id>/` with a `.installed.json` marker
  recording the revision and verified files; `omniscan models verify --deep` re-hashes every file.
- `ollama` — pulled through the local Ollama daemon (local or Ollama Cloud; needs an Ollama account).

Hardware numbers below are informational (docs only) — the per-model `min_vram_gb`/`min_ram_gb`/
`cpu_speed` fields are pending card H1. Language codes are from the upstream model cards; entries
whose card lists no language have `langs = []` and a catalog note explaining why.

## OCR stack (PP-OCR v5/v6, PaddleOCR-VL, manga-ocr)

| Model | Role | Languages | Size | Licence | Upstream | Hardware |
|---|---|---|---|---|---|---|
| `detector-comic-text-bubble` | detector | — (script-agnostic) | 159 MB | Apache-2.0 | [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)@`16e8a622` | — (pending H1) |
| `ocr-det-ppocrv5-server` | text-line detector | en, zh | 80 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv5_server_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors)@`cbea9f3c` | ~0.8 GB VRAM; CPU ok |
| `ocr-rec-korean-ppocrv5-mobile` | recognizer | ko | 23 MB | Apache-2.0 | [PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors)@`6ef525a3` | ~0.5 GB VRAM; CPU fast |
| `ocr-det-ppocrv6-tiny` | text-line detector | en, zh | 2 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv6_tiny_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_safetensors)@`07595f98` | ~0.5 GB VRAM; CPU fast |
| `ocr-det-ppocrv6-small` | text-line detector | en, zh | 10 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv6_small_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_safetensors)@`eae2ee92` | ~0.5 GB VRAM; CPU fast |
| `ocr-det-ppocrv6-medium` | text-line detector | en, zh | 89 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv6_medium_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_safetensors)@`4236c2b6` | ~0.8 GB VRAM; CPU ok |
| `ocr-rec-ppocrv6-tiny` | recognizer | en, zh | 5 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv6_tiny_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_safetensors)@`6f2d2d51` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-ppocrv6-small` | recognizer | en, zh | 22 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv6_small_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_safetensors)@`fe049fb1` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-ppocrv6-medium` | recognizer | en, zh | 77 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv6_medium_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_safetensors)@`024cad6a` | ~0.8 GB VRAM; CPU ok |
| `ocr-det-ppocrv5-mobile` | text-line detector | en, zh | 15 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv5_mobile_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors)@`c5041d22` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-ppocrv5-server` | recognizer | en, zh | 85 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv5_server_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors)@`542979d7` | ~0.8 GB VRAM; CPU ok |
| `ocr-rec-ppocrv5-mobile` | recognizer | en, zh | 32 MB | Apache-2.0 | [PaddlePaddle/PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_safetensors)@`485a9a97` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-en-ppocrv5-mobile` | recognizer | en | 23 MB | Apache-2.0 | [PaddlePaddle/en_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_safetensors)@`e4abce2b` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-latin-ppocrv5-mobile` | recognizer | — (script model) | 23 MB | Apache-2.0 | [PaddlePaddle/latin_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec_safetensors)@`47901138` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-eslav-ppocrv5-mobile` | recognizer | — (script model) | 23 MB | Apache-2.0 | [PaddlePaddle/eslav_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec_safetensors)@`feb3757f` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-th-ppocrv5-mobile` | recognizer | th | 23 MB | Apache-2.0 | [PaddlePaddle/th_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/th_PP-OCRv5_mobile_rec_safetensors)@`4cc4c996` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-el-ppocrv5-mobile` | recognizer | el | 23 MB | Apache-2.0 | [PaddlePaddle/el_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/el_PP-OCRv5_mobile_rec_safetensors)@`0d7a6ed5` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-arabic-ppocrv5-mobile` | recognizer | ar | 23 MB | Apache-2.0 | [PaddlePaddle/arabic_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec_safetensors)@`4941f944` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-cyrillic-ppocrv5-mobile` | recognizer | — (script model) | 23 MB | Apache-2.0 | [PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_safetensors)@`365e51a4` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-devanagari-ppocrv5-mobile` | recognizer | — (script model) | 23 MB | Apache-2.0 | [PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_safetensors)@`4b1f1989` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-te-ppocrv5-mobile` | recognizer | te | 23 MB | Apache-2.0 | [PaddlePaddle/te_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/te_PP-OCRv5_mobile_rec_safetensors)@`1949a6f8` | ~0.5 GB VRAM; CPU fast |
| `ocr-rec-ta-ppocrv5-mobile` | recognizer | ta | 23 MB | Apache-2.0 | [PaddlePaddle/ta_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/ta_PP-OCRv5_mobile_rec_safetensors)@`d1a8d42f` | ~0.5 GB VRAM; CPU fast |
| `ocr-vl-1.6` | vlm_ocr | en, zh, multilingual | 1931 MB | Apache-2.0 | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)@`c5630aba` | ~2 GB weights; 6–8 GB VRAM, 12 GB RAM; CPU unusable |
| `ocr-vl-1.5` | vlm_ocr | en, zh, multilingual | 1931 MB | Apache-2.0 | [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)@`2a4195fa` | ~2 GB weights; 6–8 GB VRAM, 12 GB RAM; CPU unusable |
| `ocr-vl` | vlm_ocr | en, zh, multilingual | 1931 MB | Apache-2.0 | [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)@`7fa00a8c` | ~2 GB weights; 6–8 GB VRAM, 12 GB RAM; CPU unusable |
| `ocr-vl-1.6-gguf` | vlm_ocr | — | 1818 MB | Apache-2.0 | [PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF)@`511b0964` | ~2 GB; CPU-friendly via llama.cpp |
| `ocr-vl-1.5-gguf` | vlm_ocr | — | 1818 MB | Apache-2.0 | [PaddlePaddle/PaddleOCR-VL-1.5-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-GGUF)@`cc977c16` | ~2 GB; CPU-friendly via llama.cpp |
| `ocr-rec-manga-ocr-base` | recognizer | ja | 445 MB | Apache-2.0 | [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base)@`aa6573bd` | ~1.5 GB VRAM, 8 GB RAM; CPU ok |
| `ocr-rec-manga-ocr-2025` | recognizer | ja | 122 MB | Apache-2.0 | [jzhang533/manga-ocr-base-2025](https://huggingface.co/jzhang533/manga-ocr-base-2025)@`1e64d5be` | ~1.5 GB VRAM, 8 GB RAM; CPU ok |
| `inpaint-big-lama` | inpaint | — (script-agnostic) | 206 MB | Apache-2.0 | [Sanster/models — `add_big_lama`](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) | — (pending H1) |

Notes on the table:

- The **v6 recognisers are published for en+zh only**; on Korean pages they scored far below the v5
  Korean recogniser (evidence in `docs/PRODUCT_SPEC.md`), which is why `ocr-rec-korean-ppocrv5-mobile`
  stays the only required recognizer. No `japan`/`chinese_cht` v5 repos exist — Japanese is covered
  by the base v5 recogniser (and manga-ocr for bubble crops), Traditional Chinese by the base v5
  recogniser.
- The base v5 recogniser's card tags en+zh; the upstream PP-OCRv5 release additionally covers
  Traditional Chinese, Japanese and Chinese pinyin. The per-language v5 recognisers above are the
  mobile-size models; `latin`/`eslav`/`cyrillic`/`devanagari` are script-named models whose cards
  list no languages (noted in the catalog).
- The `*-gguf` entries are official PaddleOCR-VL builds for llama.cpp; they need the llama.cpp
  engine, which is not built yet (O1b covers the PP-OCR engines first).
- `ocr-vl` (the base release) excludes the repo's `PP-DocLayoutV2/` layout subfolder, which is not
  part of the VLM weights.
- Sizes are decimal MB rounded up from the pinned revision's files (`size_mb` in the catalog);
  hf entries verify each file's sha256 from `[model.files]`, zip entries verify the whole zip.

## Translation LLMs (Ollama)

| Model | Purpose | Upstream | Revision | Licence | Size |
|---|---|---|---|---|---|
| `llm-translategemma-12b` | translation candidate (local) | [Ollama library — `translategemma`](https://ollama.com/library/translategemma) | tag `12b` | Gemma terms | 8110 MB |
| `llm-gemma4-12b` | second translation candidate (local) | [Ollama library — `gemma4`](https://ollama.com/library/gemma4) | tag `12b` | Gemma terms | 7560 MB |
| `llm-gemma4-31b` | larger local model (24 GB+ VRAM) | [Ollama library — `gemma4`](https://ollama.com/library/gemma4) | tag `31b` | Gemma terms | 19870 MB |
| `llm-gemma4-31b-cloud` | judge and third candidate (Ollama Cloud) | [Ollama library — `gemma4`](https://ollama.com/library/gemma4) | tag `31b-cloud` | Gemma terms | 0 MB |

The local LLMs need their VRAM on the GPU or an Ollama Cloud subscription (nothing to download, but
the local daemon still proxies the requests).

## Install layout and licences

Zip models are extracted into `paths.models_dir`, so a model lives at `<models_dir>/<id>/`; the
LaMa file goes to `<models_dir>/lama/big-lama.pt`. hf models are downloaded into
`<models_dir>/<id>/` with a `.installed.json` marker (revision, verified file count and sizes).
Every download is sha256-verified against the catalog (`[model.files]` for hf entries, the zip hash
for mirrored assets). The pipeline loads each Hugging Face model from that installed folder when
present (falling back to the Hugging Face hub/cache otherwise), and `omniscan doctor` reports the
required models that are missing.

Licences: all Apache-2.0 weights keep their upstream licences (the RT-DETR-v2 detector from the
`ogkalu` Hugging Face account, PP-OCR v5/v6 and PaddleOCR-VL by PaddlePaddle, big-lama from
Sanster's `models` repo, manga-ocr from `kha-white`). The Ollama LLMs are governed by the Gemma
Terms of Use and require agreeing to them in the Ollama client.

Excluded on purpose (no catalog entries): the Paddle-inference builds of PP-OCR models (the repos
without a `_safetensors` suffix — they need the PaddlePaddle framework, which has no ROCm build), the
ONNX conversions of PP-OCR models (`*_onnx` / `PP-OCRv*_onnx*` repos — OmniScan runs PyTorch
weights; the ONNX runtime is not a supported backend), third-party (non-`PaddlePaddle`) GGUF builds
of PaddleOCR-VL, and `ogkalu/comic-text-segmenter-yolov8m` (ultralytics YOLO, AGPL).

## Model updates

The catalog must not silently fall behind upstream (card W1). The `model-watch` GitHub Action
(`.github/workflows/model-watch.yml`) runs every Monday at 06:00 UTC and compares metadata only —
it never downloads weights and never edits `config/models.toml`:

- every `hf` (and mirrored `zip`) entry's pinned `upstream_revision` is compared with the upstream
  repo's head commit → `updated`; a repo that answers 404 → `missing`;
- the listing API of the watched orgs (`PaddlePaddle`, `kha-white`, `jzhang533`, `ogkalu`, with the
  search terms and name patterns in `config/model_watch.toml`) is scanned for model repos the
  catalog does not know yet → `new`. For PP-OCR only `*_safetensors` repos are watched, since only
  those can enter the catalog (see "Excluded on purpose" above).

When something changed, the workflow posts the report (markdown + JSON) to an open issue labelled
`model-watch`, or opens that issue. What to do with a report: run
`uv run python scripts/qualify_ocr.py --only <candidate>` on a GPU machine, update
`config/models.toml` only if the qualification improves, and add repos that should never be
reported again to the `ignore` list in `config/model_watch.toml`.

Run it locally with `uv run python scripts/model_watch.py --markdown model-watch.md
--json model-watch.json --fail-on-change` (exit codes: 0 no changes, 3 changes found, 2 bad
arguments or unreadable config; `--offline` only validates both config files). An optional
`HF_TOKEN` environment variable raises the Hugging Face rate limit and is never printed.