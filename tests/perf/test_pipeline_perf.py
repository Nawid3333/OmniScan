"""Stage timings of the real pipeline on a synthetic chapter, against this device's recorded baseline.

Runs only with `pytest --perf` on a machine with the GPU and the cached model weights (like the E1 golden
test, whose pages and fake translator it uses). Two identical chapters go through all ten stages in one run;
the second is measured, so model loading and first-use warm-up (kernel compilation) are not counted.
`pytest --perf --perf-update` records the timings as this device's baseline in tests/perf/baselines.json.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from omniscan.core.config import Config, GpuConfig, PathsConfig
from tests.fixtures.korean_pages import make_korean_page
from tests.perf.budget import load_baselines, record, report, slower_stages
from tests.unit.test_e2e_synthetic import (
    SEEDS,
    TRANSLATIONS,
    DictionaryClient,
    _real_models_dir,
    _skip_without_cached_weights,
)

pytestmark = [pytest.mark.perf, pytest.mark.gpu]

SERIES = "Perf"
WARMUP, MEASURED = "Chapter 1", "Chapter 2"


def test_no_stage_got_slower(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    import torch  # deferred: the perf tests need the GPU stack

    from omniscan.gpu.device import resolve_device
    from omniscan.gpu.groups import build_vram_manager
    from omniscan.pipeline.runner import run_pipeline

    root = tmp_path
    cfg = Config(
        gpu=GpuConfig(device="auto"),
        paths=PathsConfig(
            library_root=root / "library",
            work_root=root / "work",
            output_root=root / "output",
            promo_examples=root / "promo",
            models_dir=_real_models_dir(),
        ),
    )
    pages = [make_korean_page(seed, n_bubbles=4, n_free=1, n_sfx=0) for seed in SEEDS]
    for chapter in (WARMUP, MEASURED):
        raw = cfg.paths.library_root / SERIES / chapter
        raw.mkdir(parents=True)
        for number, page in enumerate(pages, 1):
            page.image.save(raw / f"{number:03d}.jpg", format="JPEG", quality=92)
    _skip_without_cached_weights(cfg)
    gpu = build_vram_manager(cfg)
    try:
        result = run_pipeline(
            cfg, SERIES, [WARMUP, MEASURED], lama=True, client=DictionaryClient(TRANSLATIONS), gpu=gpu
        )
    except httpx.HTTPError as exc:
        pytest.skip(f"model weights not cached: {exc}")
    finally:
        gpu.release()
    assert not result.failed, f"pipeline failures: {result.failed}"
    measured = {outcome.stage: outcome.seconds for outcome in result.outcomes[MEASURED]}
    device = resolve_device("auto")
    key = f"{device.type}:{torch.cuda.get_device_name(device)}" if device.type == "cuda" else device.type
    baseline = load_baselines().get(key, {})
    print("\n" + report(key, measured, baseline))
    if request.config.getoption("--perf-update"):
        record(key, measured)
        return
    slower = slower_stages(measured, baseline)
    assert not slower, "stages slower than their budget:\n" + "\n".join(slower)
