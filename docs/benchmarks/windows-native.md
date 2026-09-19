# Windows-native check (2026-09-19)

Question: must development (and later the shipped app) run on Linux/WSL, or does everything work on Windows 11 natively?
Answer from measurements on this machine (Ryzen 5 7600X, RX 9070 XT gfx1201 15.9 GiB + iGPU gfx1036, Windows 11, Python
3.14.7, Windows-side Ollama 0.34.2): **everything tested works natively, with the same results as WSL.**

## What was run
| Check | WSL (Ubuntu 26.04) | Windows native | Verdict |
|---|---|---|---|
| ROCm PyTorch install of the pinned versions (`torch 2.13.0+rocm10.0.0`, `torchvision 0.28.0`, gfx1201 device packs, `rocm-sdk-*`) from `stable.repo.amd.com/rocm/whl-next/` | works | works (`win_amd64` wheels exist for cp310–cp314; pip install 207 s, venv 4.3 GB) | equal |
| `uv sync --frozen` from our `uv.lock` | works | works **after** the `triton` override (see below); 143 s | fixed |
| Full test suite | ≈ 2 000 pass, ≈ 87 s | all pass (one symlink test needed a guard), **18 s**, incl. 6 `gpu` tests on the RX 9070 XT | Windows faster |
| fp16 matmul 4096³ | 116.6 TFLOPS | 128.2 TFLOPS | equal |
| RT-DETR-v2 detector on a real page (threshold 0.2) | 10 raw / 8 after merge | 10 raw / 8 after merge (identical classes) | identical |
| Detector steady-state, 8 tiles of 640² per batch, fp32 | 71.3 tiles/s | 68.1 tiles/s | within 5 % |
| Detector steady-state, fp16 | 169.9 tiles/s | 158.4 tiles/s | within 7 %; fp16 ≈ 2.4× fp32 on both |
| PaddleOCR models via `transformers` (`scripts/paddle_models_check.py`): Korean PP-OCRv5 rec, PP-OCRv6 rec zh + ja | OK | **OK**, same recognised strings, scores 0.977 / 0.999 / 1.000 | identical |
| `omniscan doctor` | runs | runs; sees the 9070 XT and the Windows Ollama directly (no mirrored networking needed); `rocm`/`rocjpeg` rows are Linux-only checks and FAIL/WARN | needs a platform-aware doctor |

PaddleOCR is **not** a reason to use Linux: the PaddlePaddle framework is never used; the models run through PyTorch.

## Findings that need work (small)
1. **`uv.lock` was not installable on Windows.** AMD's torch metadata lists `triton` unconditionally, but only Linux
   wheels exist (and only `torch.compile` needs it). Fixed with `override-dependencies = ["triton; sys_platform == 'linux'"]`
   in `pyproject.toml` (commit `56847f9`); Linux is unchanged.
2. **GPU selection.** On Windows the integrated GPU is `cuda:0` and **crashes on the first kernel** (access violation in
   `amdhip64_7.dll`); the RX 9070 XT is `cuda:1`. WSL exposes only the 9070 XT, so `gpu.device = "cuda:0"` worked there.
   `torch.cuda.get_arch_list()` lists gfx1036 too, so it cannot be used to filter. **Fixed:** `torch` reports
   `is_integrated` per device, and `omniscan.gpu.device.resolve_device("auto")` (now the `gpu.device` default) picks the
   strongest discrete GPU, then Apple MPS, then CPU; a named device (`cuda:1`) still wins. Tests and `doctor` use it.
   (A per-device kernel smoke test is still worth adding to the future hardware-detection screen.)
3. **`omniscan doctor`** had Linux-only checks. **Fixed for `rocm`** (reports "not applicable on Windows"); the `rocjpeg`
   WARN stays informational.
4. **Symlink test** in `tests/unit/test_web_app.py` failed with WinError 1314 (creating symlinks needs Developer Mode or
   admin). Now only that case is skipped where the OS forbids it.
5. Hugging Face prints a symlink warning without Developer Mode (models are copied instead of linked; uses more disk).
   Cosmetic; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Developer Mode.
6. `scripts/omni-builder` (bash + `flock`) was Linux-only. **Fixed:** replaced by `scripts/omni_builder.py` (portable slot
   locks, junction for the shared `.venv`, card text on stdin), smoke-tested with the native Windows Claude Code CLI;
   the `/mnt/c` font path in `paddle_models_check.py` is portable too.
7. Not tested yet: the hybrid GPU codec C++ extension (would need MSVC or prebuilt wheels on Windows), PyInstaller/Nuitka
   packaging, macOS (no Mac; MPS operator coverage unknown), NVIDIA, Intel.

## Reproduce on Windows
```powershell
git clone https://github.com/Nawid3333/OmniScan.git V:\OmniScan ; cd V:\OmniScan
python -m pip install --user uv           # uv was not installed on this machine
uv sync --frozen                          # ~2.5 min the first time (downloads the ROCm wheels)
uv run pytest -q                          # no environment variables needed: gpu.device = "auto"
uv run python scripts/paddle_models_check.py
uv run omniscan doctor
```
(The throwaway environments used for this check, `C:\Users\limex\omniscan-wintest`, have been deleted.)

## Addendum (2026-09-19, later): fp16 is unreliable on this stack
The steady-state numbers above were measured at batch 8 only. Re-testing RT-DETR-v2 (`ogkalu/comic-text-and-bubble-detector`, 640² tiles) on the RX 9070 XT:

| dtype | batch 1 | batch 2 | batch 3 | batch 8 |
|---|---|---|---|---|
| fp32 | 31.5 tiles/s | 43.2 | 50.7 | 64.6 |
| fp16 | **fails** | **fails** | **fails** | 142.7 |

The failures are `MIOpen … invalid device function` / `miopenStatusUnknownError` (an FP16 group-convolution solver without a valid kernel for those shapes); the PP-OCR text detector fails in fp16 and bf16 at every shape
(`docs/benchmarks/ocr-probe.md`). Decision: run every torch vision model in fp32 (`docs/DECISIONS.md`); the "fp16 ≈ 2.4× fp32" line in the table is only true at batch 8.
