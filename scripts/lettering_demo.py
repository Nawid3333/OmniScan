"""Letter synthetic pages over drawn art with the real cleaning and typesetting code, and save them to look at.

    uv run python scripts/lettering_demo.py                          # data/lettering_demo/, style auto
    uv run python scripts/lettering_demo.py --style manga --no-lama --out /tmp/demo

Each page is a synthetic Korean webtoon page (tests/fixtures/korean_pages.py) whose background is a
procedural painted panel crossed by ink lines and hatching, so text sits on art the way it does in a real
chapter. The regions are what the default OCR engine hands on (one box per region, effects arriving as free
text); the English is a fixed translation of the fixture's lines. The page then goes through the pipeline's
own code: sound-effect detection and style measurement (ocr/sfx.py), glyph-precise cleaning with LaMa
(inpaint/, weights from `paths.models_dir`, downloaded on first use), the balloon-shaped layout
(typeset/) and compositing (export/composite.py). Output per seed: `page-<seed>.png` (raw | lettered), plus
`sfx.png` with a tilted, a heavy and a vertical effect. Takes the real-GPU lock while LaMa runs on the GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # direct script run: make omniscan/ and tests/ importable
    sys.path.insert(0, str(_REPO_ROOT))

import tests.fixtures.korean_pages as korean_pages  # noqa: E402

from omniscan.core.config import Config, SfxConfig, TypesetConfig, get_config  # noqa: E402
from omniscan.core.schemas import BBox, OcrLine, Region  # noqa: E402
from omniscan.export.composite import apply_patches, blend_rgba  # noqa: E402
from omniscan.gpu.device import resolve_device  # noqa: E402
from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock  # noqa: E402
from omniscan.inpaint.lama import LamaInpainter  # noqa: E402
from omniscan.inpaint.lama_pipeline import lama_regions  # noqa: E402
from omniscan.inpaint.pipeline import inpaint_regions  # noqa: E402
from omniscan.ocr.assemble import build_ocr_regions  # noqa: E402
from omniscan.ocr.lines import LineBox  # noqa: E402
from omniscan.ocr.sfx import (  # noqa: E402
    default_sfx_text_paths,
    load_sfx_lexicon,
    measure_lettering_style,
    reclassify_sfx_regions,
)
from omniscan.typeset.plan import plan_layout  # noqa: E402
from omniscan.typeset.render import render_item  # noqa: E402

ENGLISH: dict[str, str] = {
    "괜찮아요? 던전이 열렸어!": "Are you okay? The dungeon just opened!",
    "민준 형, 빨리 도망가요!": "Minjun-hyung, run! Hurry!",
    "그럴 리가 없어...": "That can't be...",
    "저 녀석은 S급 헌터야.": "That guy is an S-rank hunter.",
    "내가 널 지켜줄게.": "I'll protect you.",
    "뭐라고?! 다시 말해 봐!": "What?! Say that again!",
    "게이트가 닫히기 전에 나가야 해.": "We have to get out before the gate closes.",
    "오랜만이야, 성진아.": "Long time no see, Seong-jin.",
    "지금 무슨 일이 일어난 거지?": "What just happened?",
    "조용히 해. 누가 오고 있어.": "Quiet. Someone's coming.",
    "이건 시작에 불과해.": "This is only the beginning.",
    "약속할게. 반드시 돌아올 거야.": "I promise. I'll definitely come back.",
    "쾅!": "BOOM!",
    "두근두근": "BA-DUMP BA-DUMP",
    "쿵!": "THUD!",
    "슈웅": "WHOOSH",
}
_DETECTOR_PAD_PX = 6  # the detector's boxes are looser than the ink


def painted_panel(rng: np.random.Generator, width: int, height: int, _kind: str = "") -> Image.Image:
    """A painted-looking panel: sky gradient, soft shapes, ink lines and hatching."""
    top, bottom = rng.integers(90, 200, 3), rng.integers(150, 250, 3)
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    base = np.repeat(top * (1 - t) + bottom * t, width, axis=1)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    shapes = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shapes)
    for _ in range(14):
        x, y, r = int(rng.integers(0, width)), int(rng.integers(0, height)), int(rng.integers(40, 160))
        colour = tuple(int(v) for v in rng.integers(150, 255, 3))
        draw.ellipse((x - r, y - r // 2, x + r, y + r // 2), fill=(*colour, 150))
    image = Image.alpha_composite(image, shapes.filter(ImageFilter.GaussianBlur(10)))
    draw = ImageDraw.Draw(image)
    for _ in range(10):
        points = [(int(rng.integers(0, width)), int(rng.integers(0, height))) for _ in range(4)]
        draw.line(points, fill=(25, 25, 35, 255), width=int(rng.integers(2, 6)), joint="curve")
    for _ in range(6):
        x0, y0 = int(rng.integers(0, width - 120)), int(rng.integers(0, height - 120))
        for k in range(0, 120, 7):
            draw.line((x0 + k, y0, x0 + k - 40, y0 + 120), fill=(40, 40, 60, 255), width=1)
    return image.convert("RGB")


def as_ocr_would(strip: torch.Tensor, truth: list[Region]) -> list[Region]:
    """The regions as the default (whole-region) OCR engine hands them on: loose boxes, one line per region,
    effects as free text, the text colour sampled from the page."""
    height, width = int(strip.shape[1]), int(strip.shape[2])
    loose = [
        region.model_copy(
            update={
                "kind": "free_text" if region.kind == "sfx" else region.kind,
                "bbox": BBox(
                    x0=max(0, region.bbox.x0 - _DETECTOR_PAD_PX),
                    y0=max(0, region.bbox.y0 - _DETECTOR_PAD_PX),
                    x1=min(width, region.bbox.x1 + _DETECTOR_PAD_PX),
                    y1=min(height, region.bbox.y1 + _DETECTOR_PAD_PX),
                ),
                "text_color": None,
            }
        )
        for region in truth
    ]
    boxes = {
        r.id: [
            LineBox(box=(float(r.bbox.x0), float(r.bbox.y0), float(r.bbox.x1), float(r.bbox.y1)), score=1.0)
        ]
        for r in loose
    }
    readings = {(r.id, 0): (r.text.replace("\n", " "), 0.95) for r in loose}
    return build_ocr_regions(
        loose, boxes, readings, strip, engine="demo", lang="ko", strip_width=width, strip_height=height
    )


def letter(
    strip: torch.Tensor, regions: list[Region], lines: dict[str, str], cfg: Config, lama: bool
) -> torch.Tensor:
    """Detect effects, measure lettering, clean (flat fills, LaMa) and letter `strip`; the lettered copy."""
    regions = reclassify_sfx_regions(regions, load_sfx_lexicon(default_sfx_text_paths()), cfg.sfx)
    regions = [measure_lettering_style(strip, r) if r.kind in ("sfx", "free_text") else r for r in regions]
    cleaned, patches, _ = inpaint_regions(strip, regions, cfg.inpaint, sfx_mode=cfg.sfx.mode)
    flat = {rid: (p.permute(1, 2, 0).cpu().numpy(), m.cpu().numpy()) for rid, (p, m) in patches.items()}
    work = strip.clone()
    apply_patches(work, cleaned.items, flat)
    erased = {item.region_id for item in cleaned.items if item.method == "flat"}
    if lama and any(item.needs_lama for item in cleaned.items):
        inpainter = LamaInpainter.load(cfg.inpaint, cfg.paths.models_dir, strip.device)
        repainted, lama_patches, _ = lama_regions(strip, cleaned, flat, inpainter, cfg.inpaint)
        apply_patches(
            work,
            repainted.items,
            {
                rid: (p.permute(1, 2, 0).cpu().numpy(), m.cpu().numpy())
                for rid, (p, m) in lama_patches.items()
            },
        )
        erased |= {item.region_id for item in repainted.items}
    fills = {item.region_id: item.fill for item in cleaned.items if item.fill is not None}
    for item in plan_layout(regions, lines, fills, cfg.typeset, sfx=cfg.sfx, erased=erased):
        glyphs = render_item(item)
        if glyphs is not None:
            blend_rgba(work, glyphs)
    return work


def sfx_scene(device: torch.device) -> tuple[torch.Tensor, list[Region], dict[str, str]]:
    """A panel with a tilted heavy outlined effect, a thin tilted one and a vertical column."""
    page = painted_panel(np.random.default_rng(7), 900, 900).convert("RGBA")
    fonts = korean_pages.FONTS_DIR
    effects = [  # text, font, size, fill, outline, stroke, tilt, centre, vertical, english
        (
            "쾅!",
            "NanumGothic-Bold.ttf",
            190,
            (20, 20, 20),
            (255, 255, 255),
            10,
            12,
            (300, 230),
            False,
            "BOOM!",
        ),
        ("휘익", "NanumGothic-Regular.ttf", 120, (40, 190, 90), None, 0, -18, (560, 560), False, "WHOOSH"),
        (
            "쿵쿵쿵",
            "NanumGothic-Bold.ttf",
            90,
            (230, 60, 40),
            (250, 240, 200),
            6,
            0,
            (800, 450),
            True,
            "THUD",
        ),
    ]
    regions: list[Region] = []
    lines: dict[str, str] = {}
    for i, (text, font, size, fill, outline, stroke, tilt, (cx, cy), vertical, english) in enumerate(effects):
        layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        typeface = ImageFont.truetype(str(fonts / font), size)
        rows = list(text) if vertical else [text]
        for k, row in enumerate(rows):
            draw.text(
                (cx, cy + (k - (len(rows) - 1) / 2) * size * 1.05),
                row,
                font=typeface,
                anchor="mm",
                fill=(*fill, 255),
                stroke_width=stroke,
                stroke_fill=(*(outline or fill), 255),
            )
        if tilt:
            layer = layer.rotate(tilt, resample=Image.Resampling.BICUBIC, center=(cx, cy))
        x0, y0, x1, y1 = layer.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox()  # type: ignore[misc]
        page.alpha_composite(layer)
        box = BBox(
            x0=x0 - _DETECTOR_PAD_PX,
            y0=y0 - _DETECTOR_PAD_PX,
            x1=x1 + _DETECTOR_PAD_PX,
            y1=y1 + _DETECTOR_PAD_PX,
        )
        rid = f"r{i + 1:04d}"
        regions.append(
            Region(
                id=rid,
                slice_index=0,
                kind="free_text",
                bbox=box,
                text=text,
                lang="ko",
                reading_order=i,
                lines=[OcrLine(bbox=box, text=text, score=0.95, engine="demo")],
            )
        )
        lines[rid] = english
    strip = torch.from_numpy(np.array(page.convert("RGB"))).permute(2, 0, 1).contiguous().to(device)
    return strip, regions, lines


def side_by_side(raw: torch.Tensor, lettered: torch.Tensor) -> Image.Image:
    """raw | lettered, 16 px apart on white."""
    left = Image.fromarray(raw.permute(1, 2, 0).cpu().numpy())
    right = Image.fromarray(lettered.permute(1, 2, 0).cpu().numpy())
    sheet = Image.new("RGB", (left.width * 2 + 16, left.height), "white")
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + 16, 0))
    return sheet


def main(argv: list[str] | None = None) -> int:
    """Render the demo pages; 0 on success."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/lettering_demo"), help="PNG output directory")
    parser.add_argument("--seeds", default="1,3,5", help="comma-separated page seeds (default: 1,3,5)")
    parser.add_argument(
        "--style", choices=("auto", "webtoon", "manga"), default="auto", help="lettering style"
    )
    parser.add_argument("--no-lama", action="store_true", help="flat fills only (effects over art stay)")
    args = parser.parse_args(argv)
    base = get_config()
    cfg = base.model_copy(
        update={
            "typeset": TypesetConfig(**{**base.typeset.model_dump(), "style": args.style}),
            "sfx": SfxConfig(),
        }
    )
    device = resolve_device(cfg.gpu.device)
    lock = acquire_gpu_lock() if device.type == "cuda" else None
    try:
        korean_pages._background = painted_panel  # text over drawn art instead of a flat background
        args.out.mkdir(parents=True, exist_ok=True)
        for seed in (int(s) for s in args.seeds.split(",")):
            page = korean_pages.make_korean_page(seed, width=800, height=1500, n_bubbles=4, n_free=1, n_sfx=1)
            strip = torch.from_numpy(np.array(page.image)).permute(2, 0, 1).contiguous().to(device)
            regions = as_ocr_would(strip, korean_pages.to_regions_artifact(page).regions)
            lines = {r.id: ENGLISH[" ".join(r.text.split())] for r in regions}
            side_by_side(strip, letter(strip, regions, lines, cfg, not args.no_lama)).save(
                args.out / f"page-{seed}.png"
            )
        strip, regions, lines = sfx_scene(device)
        side_by_side(strip, letter(strip, regions, lines, cfg, not args.no_lama)).save(args.out / "sfx.png")
    finally:
        if lock is not None:
            release_gpu_lock(lock)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
