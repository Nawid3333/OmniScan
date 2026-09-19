# OCR probe (2026-09-19): PP-OCRv5 detection + Korean recognition on the GPU

Question: can text-line detection and Korean recognition both run through PyTorch/`transformers` on this machine (Windows, RX 9070 XT,
torch 2.13 + ROCm 10), and how good are they on a page with known truth? Method: a synthetic 800×1100 Korean page (four regions, seven
lines, NanumGothic 30–34 px, white ellipses and one stroked free-text line on a noisy gradient) rendered with PIL so the truth boxes are exact;
plus five real Pepper&Carrot pages (CC BY 4.0, English) for false positives. Models: `PaddlePaddle/PP-OCRv5_server_det_safetensors` (21.9 M
params), `PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors`. transformers 5.17.

| Measurement | Result |
|---|---|
| Detection, fp32, 960×704 input | **25.4 ms** per forward |
| Detection, fp16 and bf16 | **fail**: `MIOpen Error: HIP runtime error: invalid device function` / `miopenStatusUnknownError` in the stem convolution → the detector runs in **fp32** on this stack |
| Lines found (probability map > 0.3) | 7 of 7 truth lines, one component per line, no merged or split lines, IoU > 0.5 at thresholds 0.2 / 0.3 / 0.5 |
| Raw components vs the text | the map is a *shrunk* text kernel (IoU ≈ 0.55 with the text); reading the raw boxes gives CER 0.44. Expanding by DB's "unclip" (ratio 1.0–2.0) restores CER 0.048 in every case |
| Post-processing: HF `post_process_object_detection` (OpenCV, CPU) | **3.4 ms** per page; 7/7 lines, mean IoU 0.68, recognition CER **0.048** (spaces ignored; the misses are "..." and inter-word spaces, which the recognizer does not output) |
| Post-processing: own GPU connected components (iterated max-pool) | ~41 ms per page, same accuracy → **12× slower than OpenCV**, not used |
| Recognition on truth-aligned crops (4 px padding) | CER 0.048, scores 0.94–0.999 |
| Real art (Pepper&Carrot, widths 800 and 1240, tiles of the page width) | finds speech-bubble text, page numbers and credit lines; no detections on stars, checkerboard or brush strokes; one oversized box around a speech-bubble tail (dropped later by assigning lines to detected text regions) |
| Processor input | both processors accept uint8 `[3,h,w]` **GPU tensors** and keep them on the GPU |

Decisions that follow (also in `docs/DECISIONS.md`): detector in fp32; DB post-processing on the CPU with OpenCV (`opencv-python-headless`) because
it is 12× faster than a GPU implementation and costs ~0.1 s per chapter; the OCR stage runs the line detector on the same tile scheme as detection,
only on tiles that touch a detected region, and assigns lines to regions by containment.
Not measured: real Korean scans (question A1), Gaegu/handwriting fonts, vertical text, Chinese/Japanese models.
