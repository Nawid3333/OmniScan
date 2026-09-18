---
name: omniscan-env
description: "Verified hardware/OS/GPU/Ollama facts for OmniScan (RX 9070 XT + iGPU, Windows-native ROCm PyTorch works, WSL retired, rocJPEG unavailable)"
metadata:
  node_type: memory
  type: project
  originSessionId: 18b96bfa-62d0-4f52-b6c0-2ca8b649a9c3
  modified: 2026-09-18T22:53:20.067Z
---

Machine: Ryzen 5 7600X (12 threads), 64 GB RAM, Windows 11, **RX 9070 XT (gfx1201, 16 GB) + Ryzen iGPU (gfx1036)**. Python 3.14.7, Node 24, git and `gh`
(logged in as Nawid3333) installed on Windows; `uv` installed for the user via `pip --user`; Claude Code CLI installed with
`npm install -g --allow-scripts=@anthropic-ai/claude-code @anthropic-ai/claude-code` (npm blocks its postinstall otherwise). Ollama 0.34.2 runs natively on
Windows at localhost:11434 (cloud plan Pro; `*-cloud` models are reachable through the local daemon).

Verified facts:
- **Windows-native ROCm PyTorch works** (`torch 2.13.0+rocm10.0.0` from AMD's `whl-next` index; `win_amd64` wheels exist for our pins; triton is Linux-only → uv
  override in pyproject). Full test suite ~18 s, detector + PaddleOCR (KO/zh/ja) models run with results identical to the old WSL setup. Details: repo
  `docs/benchmarks/windows-native.md`.
- **The iGPU is `cuda:0` on Windows and crashes on the first kernel; the RX 9070 XT is `cuda:1`.** `torch` reports `is_integrated`; the project resolves
  `gpu.device = "auto"` to the strongest discrete GPU (`omniscan.gpu.device.resolve_device`). Never bare `torch.cuda.synchronize()` without setting the device.
- rocJPEG hardware decode did not work in WSL (no /dev/dri) and does not exist on Windows either → CPU `turbo` codec today; hybrid GPU codec (C2) is an open decision.
- The WSL distro (Ubuntu-26.04, created 2026-09-16, also held an abandoned Paddle-from-source build) was **deleted on 2026-09-19** at the user's request after
  everything was moved to `V:\OmniScan`. `C:\Users\limex\.wslconfig` (mirrored networking) may still exist and is harmless.
- extract.pics webhooks can't target GitHub directly → Cloudflare Worker relay deployed by GitHub Actions (not deployed yet).
- Detector fp16 ≈ 2.4× fp32 (170 vs 71 tiles/s of 640² on this GPU); fp16 detections not yet compared with fp32.

Related: [[omniscan-project]]
