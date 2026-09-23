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
| `models_update/rules.py`, `models_update/download.py` | model catalog/resolve/store/download and app version/GitHub/update download (card Q4) | `test_models_{catalog,resolve,store,download,cli}.py`, `test_update_{version,github,download,cli}.py`, `test_models_update_mutation_gaps.py` |
| `hw_hf/hw.py`, `hw_hf/hf.py` | hw detection/compat rules and the hf model install path + `series_config` (card Q5) | `test_hw_{assess,detect}.py`, `test_models_{catalog,resolve,store,download,cli}.py`, `test_core_series_config.py`, `test_models_update_mutation_gaps.py`, `test_hw_hf_mutation_gaps.py` |
| `slicer_strategies/strategies.py` | slicer strategies (`fixed`, `simple_gutter`, dispatcher, `_finalize`) and `compare.py` (card Q6) | `test_slicer_strategies.py`, `test_slicer_compare.py`, `test_slicer_compare_cli.py`, `test_slicer_strategies_mutation_gaps.py` |

Expected survivors (equivalent mutants, output cannot change): `slicer/bands.py` "chunk bound min->max", `slicer/slice_page.py` "empty case height", and `models_update/download.py` "missing entry -> empty digest" (the `asset.name not in checksums` raise above makes the fallback unreachable). Card Q5 (`hw_hf/*`): none.
The `models_update/*.py` and `hw_hf/*.py` files set `PYTHONDONTWRITEBYTECODE=1` on load: without it, a restore that lands in the same second as a same-size mutation can leave stale bytecode in `__pycache__` and poison later verdicts.
