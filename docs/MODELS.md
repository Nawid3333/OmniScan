# Models and third-party notices

`config/models.toml` is the model catalog: every model OmniScan can download, what it is for, its
size and licence. `omniscan models list` shows the entries with their install status, `omniscan
models download <id>` installs one. The shipped app stays small — the user decides what to download
(owner decision 2026-09-19).

The four vision/inpaint weights are mirrored **unchanged** as assets of the GitHub release
[`models-v1`](https://github.com/Nawid3333/OmniScan/releases/tag/models-v1) of `Nawid3333/OmniScan`
(the mirror repo is private today, so a failed download falls back to the upstream source
automatically). The large language models come from Ollama (local pull, or Ollama Cloud through the
local daemon — nothing to download, but it needs an Ollama account).

| Model | Purpose | Upstream | Revision | Licence | Size | Mirror asset |
|---|---|---|---|---|---|---|
| `detector-comic-text-bubble` | speech bubbles and free text (RT-DETR-v2) | [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) | `16e8a622f91fabc6b5b65c96d32d1183f8843546` | Apache-2.0 | 159 MB | `detector-comic-text-bubble.zip` |
| `ocr-det-ppocrv5-server` | text-line detector (PP-OCRv5 server) | [PaddlePaddle/PP-OCRv5_server_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors) | `cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb` | Apache-2.0 | 80 MB | `ocr-det-ppocrv5-server.zip` |
| `ocr-rec-korean-ppocrv5-mobile` | Korean text recognition (PP-OCRv5 mobile) | [PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors) | `6ef525a357645ce46495c7ed1fef622c4e009e7a` | Apache-2.0 | 23 MB | `ocr-rec-korean-ppocrv5-mobile.zip` |
| `inpaint-big-lama` | text removal on textured art (TorchScript LaMa) | [Sanster/models — `add_big_lama`](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) | release `add_big_lama` | Apache-2.0 | 206 MB | `big-lama.pt` |
| `llm-translategemma-12b` | translation candidate (local) | [Ollama library — `translategemma`](https://ollama.com/library/translategemma) | tag `12b` | Gemma terms | 8110 MB | — (Ollama pull) |
| `llm-gemma4-12b` | second translation candidate (local) | [Ollama library — `gemma4`](https://ollama.com/library/gemma4) | tag `12b` | Gemma terms | 7560 MB | — (Ollama pull) |
| `llm-gemma4-31b` | larger local model (24 GB+ VRAM) | [Ollama library — `gemma4`](https://ollama.com/library/gemma4) | tag `31b` | Gemma terms | 19870 MB | — (Ollama pull) |
| `llm-gemma4-31b-cloud` | judge and third candidate (Ollama Cloud) | [Ollama library — `gemma4`](https://ollama.com/library/gemma4) | tag `31b-cloud` | Gemma terms | 0 MB | — (Ollama Cloud) |

Sizes are approximate download sizes (`size_mb` in the catalog); the catalog's `bytes` fields carry
the exact mirror-asset sizes. The zip assets contain only files under a top-level `<id>/` folder and
are extracted into `paths.models_dir`, so a model lives at `<models_dir>/<id>/`; the LaMa file goes
to `<models_dir>/lama/big-lama.pt`. Every download is sha256-verified against the catalog. The
pipeline loads each Hugging Face model from that installed folder when present (falling back to the
Hugging Face hub/cache otherwise), and `omniscan doctor` reports the required models that are missing.

Licences: the Apache-2.0 weights keep their upstream licences (the RT-DETR-v2 detector from the
`ogkalu` Hugging Face account, PP-OCRv5 by PaddlePaddle, big-lama from Sanster's `models` repo).
The Ollama LLMs are governed by the Gemma Terms of Use and require agreeing to them in the Ollama
client.