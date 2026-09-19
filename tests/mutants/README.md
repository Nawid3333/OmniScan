# Hand-written mutants

Each `*.py` file here defines `MUTANTS = [(file, old, new, label), ...]`: one deliberate fault per entry, in production code.
They are the durable form of the "break it on purpose" review — run them after a refactor to see whether the tests still notice.

```bash
uv run python scripts/mutate.py check tests/mutants/pipeline.py        # every `old` snippet still exists exactly once?
uv run python scripts/mutate.py run tests/mutants/pipeline.py -t tests/unit/test_pipeline_stages.py -t tests/unit/test_pipeline_runner.py
```

`KILLED` = a test failed (good), `SURVIVED` = no test noticed (write one, or decide the mutant is equivalent), `INVALID` = the code
changed and the snippet no longer matches (update or drop the entry). The list files name the tests that were used for each group:

| file | code under test | tests |
|---|---|---|
| `slicer/bands.py`, `slicer/cuts.py`, `slicer/slice_page.py`, `slicer/ingest.py` | slicer and ingest layout (card Q1) | `tests/unit/test_slicer_*.py`, `tests/unit/test_ingest*.py` |
| `pipeline.py` | `omniscan run`, stage adapters, queue executor (card R1) | `test_pipeline_*.py`, `test_queue_executor.py`, `test_queue_cli.py`, `test_cli.py` |
| `e2e.py` | codec race, page order, typeset colour, against the golden test (card E1) | `test_e2e_synthetic.py` (GPU, ~1 min per mutant) |
| `acquire/rules.py`, `acquire/download.py` | acquisition rules (drm/filters/sources) and the downloader (card Q3) | `test_acquire_*.py`, `test_acquire_mutation_gaps.py` |

Expected survivors (equivalent mutants, output cannot change): `slicer/bands.py` "chunk bound min->max" and `slicer/slice_page.py` "empty case height".
