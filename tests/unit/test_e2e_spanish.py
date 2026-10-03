"""#43: a Spanish release end to end — the real pipeline on two synthetic Korean pages, `target_lang = "es"`.

The same synthetic pages and model weights as the English golden test (test_e2e_synthetic.py); a dictionary fake
answers in Spanish. Checked: every stage runs, every translation request asks for Spanish, the final lines are the
Spanish sentences, and the lettering sets them whole, accents and ¿ ¡ ñ included, with a hyphen only between
syllables (avoiding a line that ends on an article is a preference the fitter weighs against the balloon's shape,
tested in test_typeset_hyphen.py). Skips without cached weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from omniscan.core.config import Config, GpuConfig, PathsConfig
from omniscan.core.paths import SeriesPaths
from omniscan.core.schemas import FinalArtifact, LayoutArtifact
from omniscan.pipeline.runner import PipelineResult
from omniscan.pipeline.stages import STAGE_ORDER
from omniscan.typeset.hyphen import hyphen_points
from tests.fixtures.korean_pages import KOREAN_LINES, make_korean_page
from tests.unit.test_e2e_synthetic import DictionaryClient, _real_models_dir, _skip_without_cached_weights

pytestmark = pytest.mark.gpu

SERIES = "E2E-ES"
CHAPTER = "Chapter 1"
SEEDS = (101, 102)

_SPANISH: tuple[str, ...] = (
    "¿Estás bien? ¡La mazmorra se abrió!",
    "¡Hyung, huye rápido!",
    "Eso no puede ser verdad...",
    "Ese tipo es un cazador de rango S.",
    "Ahora te protegeré.",
    "¡¿Qué?! ¡Dilo otra vez!",
    "Debemos irnos antes de que se cierre la puerta.",
    "Cuánto tiempo, Sungjin.",
    "¿Qué demonios está pasando aquí?",
    "Silencio. Alguien se acerca.",
    "Esto es solo el principio, pequeño.",
    "Te prometo que volveré.",
)
SPANISH: dict[str, str] = dict(zip(KOREAN_LINES, _SPANISH, strict=True))


@dataclass(frozen=True, slots=True)
class Release:
    """What the Spanish run left behind."""

    client: DictionaryClient
    result: PipelineResult
    final: FinalArtifact
    layout: LayoutArtifact


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Release:
    """Run every stage once over two synthetic pages of a series released in Spanish."""
    root = tmp_path_factory.mktemp("e2e_spanish")
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
    raw_dir = cfg.paths.library_root / SERIES / CHAPTER
    raw_dir.mkdir(parents=True)
    for number, seed in enumerate(SEEDS, 1):
        make_korean_page(seed, n_bubbles=4, n_free=1, n_sfx=0).image.save(
            raw_dir / f"{number:03d}.jpg", format="JPEG", quality=92
        )
    (cfg.paths.library_root / SERIES / "series.toml").write_text(
        '[translate]\ntarget_lang = "es"\n', encoding="utf-8"
    )
    _skip_without_cached_weights(cfg)

    from omniscan.gpu.groups import build_vram_manager  # deferred: imports torch
    from omniscan.pipeline.runner import run_pipeline

    client = DictionaryClient(SPANISH)
    gpu = build_vram_manager(cfg)
    try:
        result = run_pipeline(cfg, SERIES, [CHAPTER], lama=True, client=client, gpu=gpu)
    except httpx.HTTPError as exc:  # a weight turned out to be missing (e.g. offline): skip
        pytest.skip(f"model weights not cached: {exc}")
    finally:
        gpu.release()
    paths = SeriesPaths.from_config(cfg, SERIES).chapter(CHAPTER)
    return Release(
        client=client,
        result=result,
        final=FinalArtifact.load(paths.artifact("final.json")),
        layout=LayoutArtifact.load(paths.artifact("layout.json")),
    )


def test_every_stage_runs_and_asks_for_spanish(release: Release) -> None:
    outcomes = release.result.outcomes[CHAPTER]
    assert [o.stage for o in outcomes] == list(STAGE_ORDER) and all(o.status == "done" for o in outcomes)
    assert release.client.calls
    for call in release.client.calls:
        prompt = " ".join(str(m["content"]) for m in call["messages"])
        assert "Spanish" in prompt and "English release" not in prompt


def test_the_final_lines_are_the_spanish_sentences(release: Release) -> None:
    texts = {line.text for line in release.final.lines if line.text}
    assert texts and texts <= set(_SPANISH)


def test_the_lettering_sets_the_spanish_lines_whole(release: Release) -> None:
    by_region = {line.region_id: line.text for line in release.final.lines}
    assert release.layout.items
    for item in release.layout.items:
        assert not item.overflow
        joined = ""
        for index, line in enumerate(item.lines):
            if line.endswith("-"):  # a hyphen only between syllables of the word it splits
                word = line.split()[-1][:-1]
                rest = item.lines[index + 1].split()[0]
                assert len(word) in (hyphen_points(word + rest, "es") or []), (word, rest)
                joined += line[:-1]
            else:
                joined += line + " "
        assert (
            joined.strip() == by_region[item.region_id] or joined.strip() == by_region[item.region_id].upper()
        )
