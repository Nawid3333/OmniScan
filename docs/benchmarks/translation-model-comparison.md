# Translation model comparison (2026-09-23)

RX 9070 XT (16 GB), Ollama 0.34.2, OmniScan `e4f32a4` (num_ctx 16384, one local model resident at a time).
Every model translated the **same** `ocr.json` per chapter; scores are page chrF against the official English
text (the `omniscan eval` metric). Models ran profile-major: one model loaded, every chapter, then the next.
Script: a scratch driver over `translate.run.run_profile` and `translate.judge_chapter.judge_chapter`
(`run_ids` subsets), judge = `gemma4:31b-cloud` from `config/judge.toml`.

## Scored chapters
Pepper&Carrot KR episodes 06, 09, 12, 15, 22. Excluded: episode 30 (wordless — only the patron credits are
scorable, every model scored 0.052) and JA episode 06 (its OCR is garbage: `그E!`, `D((`).

## Single models (mean chrF over the 5 chapters, time for all 7 chapters incl. loads)

| model | chrF | time | VRAM (resident / total) |
|---|---:|---:|---|
| gemma4:31b-cloud | 0.239 | 85 s | — (cloud) |
| translategemma:12b | 0.238 * | 285 s | 8.1 / 8.1 GB |
| gemma4:12b | 0.224 | 518 s | 8.4 / 8.4 GB |
| translategemma:27b | 0.222 | 2470 s | 13.9 / 18.5 GB (CPU offload) |
| gemma4:31b (local) | 0.278 on 2 chapters (cloud: 0.284 on the same 2) | 978 s for 2 chapters, then stuck | 13.7 / 21.6 GB (CPU offload) |

\* inflated: before commit `8a92067` translategemma:12b answered punctuation-only regions with chatter
("Please provide the Korean text…"), whose English words raise chrF.

- **translategemma:27b** is 8.7x slower than the 12b and scored lower on every chapter.
- **gemma4:31b local** is the same model as `gemma4:31b-cloud`, but it doesn't fit in 16 GB and ran 50x slower. On
  episode 12, one 30-region request looped past 2,600 generated tokens and hit the 10-minute request timeout
  (then retries), so the run was stopped.

## Judged combinations (same 5 chapters)

| candidates → judge | chrF |
|---|---:|
| translategemma:27b + cloud | 0.241 |
| translategemma:12b + cloud | 0.240 |
| gemma4:12b + cloud | 0.239 |
| translategemma:12b + gemma4:12b + cloud (shipped default) | 0.238 |
| translategemma:12b + gemma4:12b (local only) | 0.228 |

Every combination that includes the cloud model lands within 0.003 of the cloud model alone (0.239). That gap
is noise at this sample size.

## What chrF misses (read by hand, episode 22)
- 혁신적이지 / 민주적이지 ("it's innovative / democratic, isn't it"): translategemma 12b **and** 27b output
  "Not innovative." / "Not democratic." (meaning reversed); both gemma4 models output the right polarity.
- translategemma ignores prompt instructions (it has its own fixed template), e.g. "Ms. Kamilla" where the
  project style keeps "-ssi".
