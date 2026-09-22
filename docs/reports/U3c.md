# U3c — Desktop app shell: main window, library, reader, run page, `omniscan gui`

**Status: stopped at the card's stop condition, before any code was written.**
The Run page's Cancel cannot be implemented against the contracts as they stand. Per the card
("Cancel … if `run_pipeline` lacks a cancel hook, stop and describe the needed core change in the
report") and `CLAUDE.md` ("If a contract is missing or wrong, stop and describe the needed change"),
this report describes the gap and the needed change; nothing else in the card was built so the next
launch can proceed against a runner that has the hook instead of around a dead Cancel button.

## The gap

The card requires Cancel to use "the `after_stage`/`RunAbortedError` mechanism through
`run_pipeline`'s gate/report hooks". Reading the two files:

- The mechanism exists in `core/stage.py`: `run_series(..., after_stage=...)` (stage.py:210-233);
  a hook returning False raises `RunAbortedError` after the just-finished stage was recorded —
  "the run stops after the stage that was just recorded".
- `run_pipeline` uses that mechanism **only in step mode**: `_run_preview` passes its gate hook to
  `run_series(..., after_stage=hook)` (runner.py:137-141). A gate answer of False aborts the run
  (`aborted = "stopped"`), which is also how step-mode Cancel would work.
- In auto mode — the mode behind **Full, Subset and Auto**, i.e. every run long enough to be worth
  cancelling — the pass loop's two `run_series` calls (runner.py:222-224) receive **no
  `after_stage`**, and the `gate` parameter is only honoured when `mode == "step"`
  (runner.py:186-195). `report` cannot stop a run (it returns None). Verified by grep: no other
  module in `src/omniscan` passes or exposes `after_stage` (the queue executor only reads
  `result.aborted` after the fact).

So `run_pipeline` lacks a cancel hook for the three non-step modes. The card's contingency applies.

## The needed core change (pipeline/runner.py only; nothing under `src/omniscan/core/**`)

One keyword-only parameter, passed down in both phases:

1. `run_pipeline(..., gate=None, after_stage: AfterStage | None = None)` — `AfterStage` is already
   exported by `core.stage` (`type AfterStage = Callable[[ChapterContext, StageOutcome], bool]`,
   stage.py:178).
2. Auto loop: pass the hook through and fold the abort the way `_run_preview` already does:
   ```python
   for _label, pass_stages in passes:
       stage_objects = [build_stage(name, cfg, client=client) for name in pass_stages]
       try:
           pass_results = run_series(
               stage_objects, cfg, series, active, gpu=gpu, force=force, after_stage=after_stage
           )
       except RunAbortedError as error:
           for chapter, outcomes in {**error.results, error.chapter: error.outcomes}.items():
               result.outcomes.setdefault(chapter, []).extend(outcomes)
               if report is not None:
                   for outcome in outcomes:
                       report(chapter, outcome)
           result.aborted = "stopped"
           break
       _record_pass_results(result, pass_results, report)
   ```
   Note: `RunAbortedError.results` (the chapters of the interrupted pass that finished before it)
   must be merged into `result.outcomes` and reported too, alongside `error.outcomes` (the chapter
   that was being run) — otherwise a cancelled pass loses the chapters it already completed.
3. Step mode: compose the two hooks inside `_run_preview`'s `hook`, cancel first:
   ```python
   def hook(ctx: ChapterContext, outcome: StageOutcome) -> bool:
       nonlocal position
       position += 1
       if after_stage is not None and not after_stage(ctx, outcome):
           return False  # cancelled: stop like a gate "no"
       return gate(GateEvent(...))
   ```

An alternative that avoids the new parameter: honour the existing `gate` in auto mode as well
(check it after every stage and build `GateEvent`s there). I recommend `after_stage`: it matches
the card's wording ("uses the `after_stage`/`RunAbortedError` mechanism"), it keeps step mode's
preview semantics out of plain runs, and the GUI's hook is trivial (a `threading.Event` check)
without needing to build `GateEvent`s it cannot display. Either form is ~15 lines in runner.py.

Cancellation granularity is unchanged by design: Cancel takes effect **after the stage currently
running finishes** (that is the mechanism's contract — a torch/LLM stage is never killed
mid-kernel). The GUI should label the button's effect accordingly ("stops after the current stage").

## How the GUI will consume it (for the next U3c launch)

- The run worker (QThread) owns a `threading.Event`; its `after_stage` hook returns
  `not event.is_set()`. In step mode the same event is checked inside the gate (the GUI's gate
  answers False when the user pressed Cancel while paused, so the Abort path doubles as Cancel).
- `RunAbortedError` never escapes the worker: `run_pipeline` folds it into
  `PipelineResult.aborted == "stopped"`, and the worker maps that to a "cancelled" run state.
- Everything else the card needs already exists and was read end-to-end: the library service +
  strip/compare views (U3b), the models view + service (U3a), `set_user_setting`/`set_series_setting`/
  `SERIES_SECTIONS` with validation and a `path` override for tests (core/config.py:317-329),
  `describe` previews (pipeline/preview.py), and the CLI's worker-lifetime pattern to copy —
  build `OllamaClient(cfg.ollama, get_secrets())` only when a text stage is selected,
  `build_vram_manager(cfg)` only when `needs_gpu(...)`, release both in `finally`
  (cli.py:760-828).

## What was verified before stopping

- Read: `pipeline/runner.py`, `pipeline/stages.py` (`STAGE_ORDER`, `PASS_OF`, `build_stage`),
  `pipeline/preview.py`, `core/stage.py`, `core/config.py`, `core/paths.py`, `hw/detect.py`,
  `hw/assess.py`, the merged GUI code (`strip_view.py`, `compare_view.py`, `models_view.py`,
  `workers.py`, `services/library.py`, `services/models.py`), `tests/gui/*` (conventions:
  `QT_QPA_PLATFORM=offscreen` before Qt import, session-scoped `qapp`, `pytest.importorskip`),
  `scripts/gui_compare_demo.py`, `scripts/gui_models_demo.py`, the CLI `run` command.
- Confirmed `run_pipeline`'s full signature has no `after_stage`/cancel parameter and that auto
  mode never reaches a `gate` call.
- Confirmed `PySide6-Essentials` is the existing `gui` extra (pyproject.toml:38-41); no new
  dependencies are needed for the shell itself.

## Open questions

1. (blocking) Apply the runner change above (or the director's preferred variant), then relaunch
   U3c as written — no other contract gap was found.