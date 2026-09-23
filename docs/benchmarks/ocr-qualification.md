# OCR qualification suite: which model is the default for which language (card O1c)

Question: for each release language (ko, ja, zh, en), which OCR configuration should be the pipeline
default? Answer from measurements over the Pepper&Carrot ground-truth chapters, never from model
names or single-page impressions. The suite lives in `scripts/qualify_ocr.py` (runner) +
`src/omniscan/eval/qualify.py` (library) + `config/qualification.toml` (the plan).

## Method

Every candidate (an engine plus its det/rec catalog models, `config/qualification.toml`) runs every
chapter of its language's dataset through the real pipeline stages `ingest → slice → detect → ocr`
into a dedicated qualification work root (`<work_root>_qual`; the normal work root is never
written), and the produced `ocr.json` is scored against the truth in
`data/translated-check/<series>/<chapter>/truth/<lang>/` with the same scorer the eval CLI uses
(`eval.score.score_chapter`). The GPU is held through the usual lock; each candidate releases the
VRAM manager between chapters so one candidate's memory cannot flatter or penalise the next.

Primary metrics:

- **`chrF` (page-level `ocr_chrf_mean`)** — text quality, the decision metric.
- **`char recall`** — how much of the truth text was captured at all (regions missed = recall lost).

Box-level **CER is reported but never decides**: the truth has finer boxes than our regions on
dense pages (ja Episode 06: 159 truth boxes vs 91 detected regions → micro CER 0.84 although the
text is right), so it is not comparable across engines.

Decision rule (`eval.qualify.recommend`): PROMOTE a non-default candidate only when it is the best
by mean chrF (ties: higher char recall, then less time per chapter) **and** the gain is ≥ 0.02 chrF
**and** char recall drops by ≤ 0.02. Otherwise KEEP, with the reason printed. A language whose
default has no data always answers KEEP ("no data for the default").

## Datasets (Pepper&Carrot, CC BY 4.0)

| lang | series | chapters | truth folder |
|---|---|---|---|
| ko | PepperCarrotKR | Episode 06, 09 | kr |
| ja | PepperCarrotJA | Episode 06, 09, 12, 22 | ja |
| zh | PepperCarrotCN | Episode 06, 09, 12, 22 | cn |

en shares the zh candidates but has no dataset yet — it reports "no data for the default".

## Prior evidence (2026-09-20, ja Episode 06, `docs/benchmarks/` + reports)

manga-ocr-base: chrF 0.686, char recall 0.943, 13.9 s; PP-OCRv5 multilingual: chrF 0.105 — Japanese
handwriting favours manga-ocr. That single-page result is what this suite confirms or refutes at
chapter scale, per language, including cost (s/chapter, peak VRAM GiB).

## Running

```bash
uv run python scripts/qualify_ocr.py --dry-run                 # list pairs + missing models
uv run python scripts/qualify_ocr.py --download                # fetch missing candidate models
uv run python scripts/qualify_ocr.py --json out.json --markdown out.md
uv run python scripts/qualify_ocr.py --lang ja --only manga-ocr-2025,ppocr-v5-server-multi
uv run python scripts/qualify_ocr.py --chapters 1              # one chapter per dataset, quick pass
```

Exit codes: 0 done, 1 every measurement failed, 2 bad arguments. `--default LANG=ID` overrides the
assumed current default per language (defaults: ko `ppocr-v5-ko`, everything else
`ppocr-v5-server-multi`, mirroring `OcrConfig`).

## Results

(To be filled by the director run — paste the table, one section per language, and the KEEP/PROMOTE
verdicts. Decisions then move to `docs/DECISIONS.md` and `config/models.toml` `recommended_for`.)