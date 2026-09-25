"""Measure the sound-effect sweep on synthetic pages: how many drawn-in effects become sfx regions.

    uv run python scripts/sfx_sweep_check.py                       # 3 pages, the configured OCR engine
    uv run python scripts/sfx_sweep_check.py --fonts D:/fonts/ko   # also letter effects in these .ttf faces
    uv run python scripts/sfx_sweep_check.py --out data/sweep_check  # save the pages with found/missed boxes

Each page is a painted panel crossed by ink lines and hatching (scripts/lettering_demo.py) with three
speech balloons, which are handed to the sweep as detected regions, and ten Korean sound effects drawn
into the art: tilted, outlined, coloured, stacked in a column, in the repo's fonts plus any faces given
with --fonts (brush and display faces make it harder). The vision models are the configured ones
(`build_vram_manager`: CRAFT plus the OCR engine's reader, downloaded on first use), so the numbers are
the ones the pipeline gets on this machine. Prints per effect what was read, then found / total and the
regions the sweep made where no effect was drawn. Takes the real-GPU lock while the models run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # direct script run: make omniscan/, scripts/ and tests/ importable
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from lettering_demo import painted_panel  # noqa: E402

from omniscan.core.config import get_config  # noqa: E402
from omniscan.core.schemas import BBox, Region, Slice  # noqa: E402
from omniscan.detect.tiles import plan_tiles  # noqa: E402
from omniscan.gpu.groups import VISION_GROUP, build_vram_manager  # noqa: E402
from omniscan.gpu.lock import acquire_gpu_lock, release_gpu_lock  # noqa: E402
from omniscan.ocr.sfx import default_sfx_text_paths, load_sfx_lexicon  # noqa: E402
from omniscan.ocr.sweep import SweepRules, sweep  # noqa: E402
from omniscan.ocr.watermark_text import default_watermark_text_paths, load_watermark_patterns  # noqa: E402

FONTS = _REPO_ROOT / "fonts"
WIDTH, HEIGHT = 1000, 2600
DIALOGUE = ("괜찮아요? 던전이 열렸어!", "저 녀석은 S급 헌터야.", "조용히 해. 누가 오고 있어.")
RGB = tuple[int, int, int]
# text, size, fill, outline (None: none), outline px, tilt (degrees counter-clockwise), stacked in a column
EFFECTS: tuple[tuple[str, int, RGB, RGB | None, int, float, bool], ...] = (
    ("쾅!", 200, (20, 20, 20), (255, 255, 255), 10, 12.0, False),
    ("휘익", 170, (40, 160, 90), None, 0, -18.0, False),
    ("쿵쿵", 110, (220, 50, 40), (250, 240, 200), 6, 0.0, True),
    ("두근두근", 120, (200, 30, 90), None, 0, 8.0, False),
    ("스윽", 110, (30, 30, 30), None, 0, -30.0, False),
    ("파앗", 150, (255, 255, 255), (20, 20, 20), 8, 20.0, False),
    ("철컥", 90, (60, 60, 200), (255, 255, 255), 5, -8.0, False),
    ("번쩍", 160, (250, 210, 30), (40, 20, 0), 6, 0.0, False),
    ("탁", 130, (15, 15, 15), None, 0, 0.0, False),
    ("사삭", 80, (90, 70, 60), None, 0, 15.0, False),
)
Box = tuple[int, int, int, int]


def _apart(box: Box, taken: list[Box], gap: int = 20) -> bool:
    return all(
        box[2] + gap < t[0] or t[2] + gap < box[0] or box[3] + gap < t[1] or t[3] + gap < box[1]
        for t in taken
    )


def make_page(seed: int, faces: list[Path]) -> tuple[np.ndarray, list[Box], list[tuple[str, Box]]]:
    """A painted page with three balloons and the ten effects: (RGB array, balloon boxes, effects)."""
    rng = np.random.default_rng(seed)
    page = painted_panel(rng, WIDTH, HEIGHT).convert("RGBA")
    draw = ImageDraw.Draw(page)
    body = ImageFont.truetype(str(FONTS / "NanumGothic-Regular.ttf"), 30)
    taken: list[Box] = []
    for i, text in enumerate(DIALOGUE):
        cx, cy = int(rng.integers(220, WIDTH - 220)), 300 + i * 800
        balloon = (cx - 190, cy - 90, cx + 190, cy + 90)
        draw.ellipse(balloon, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=3)
        draw.text((cx, cy), text, font=body, anchor="mm", fill=(0, 0, 0, 255))
        taken.append(balloon)
    effects: list[tuple[str, Box]] = []
    for k in rng.permutation(len(EFFECTS)):
        text, size, fill, outline, stroke, tilt, column = EFFECTS[int(k)]
        face = ImageFont.truetype(str(faces[int(k) % len(faces)]), size)
        for _ in range(200):
            cx, cy = int(rng.integers(150, WIDTH - 150)), int(rng.integers(150, HEIGHT - 150))
            layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
            rows = list(text) if column else [text]
            for r, row in enumerate(rows):
                ImageDraw.Draw(layer).text(
                    (cx, cy + (r - (len(rows) - 1) / 2) * size * 1.05),
                    row,
                    font=face,
                    anchor="mm",
                    fill=(*fill, 255),
                    stroke_width=stroke,
                    stroke_fill=(*(outline or fill), 255),
                )
            if tilt:
                layer = layer.rotate(tilt, resample=Image.Resampling.BICUBIC, center=(cx, cy))
            ys, xs = np.nonzero(np.array(layer.getchannel("A")) > 127)
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if box[0] >= 0 and box[2] <= WIDTH and _apart(box, taken):
                taken.append(box)
                page.alpha_composite(layer)
                effects.append((text, box))
                break
    return np.array(page.convert("RGB")), taken[: len(DIALOGUE)], effects


def _on(region: BBox, box: Box) -> bool:
    """Whether a region's centre lies inside an effect's box."""
    cx, cy = (region.x0 + region.x1) / 2, (region.y0 + region.y1) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def main(argv: list[str] | None = None) -> int:
    """Run the check; 0 on success."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", default="1,2,3", help="comma-separated page seeds (default: 1,2,3)")
    parser.add_argument("--fonts", type=Path, help="a folder of extra .ttf faces to letter effects in")
    parser.add_argument(
        "--out", type=Path, help="save each page with found (green) / missed (red) boxes here"
    )
    args = parser.parse_args(argv)
    cfg = get_config()
    faces = [FONTS / "NanumGothic-Bold.ttf", FONTS / "NanumGothic-Regular.ttf", FONTS / "Gaegu-Regular.ttf"]
    if args.fonts is not None:
        faces += sorted(args.fonts.glob("*.ttf"))
    manager = build_vram_manager(cfg.model_copy(update={"sfx": cfg.sfx.model_copy(update={"sweep": True})}))
    lock = acquire_gpu_lock() if manager.device.type == "cuda" else None
    try:
        models = manager.acquire(VISION_GROUP)
        reader = models["recognizer"] if cfg.ocr.engine == "ppocr" else models["reader"]
        rules = SweepRules(
            words=load_sfx_lexicon(default_sfx_text_paths()).get(cfg.ocr.lang, frozenset()),
            watermark_patterns=tuple(load_watermark_patterns(default_watermark_text_paths())),
            min_score=cfg.ocr.drop_conf,
            max_chars=cfg.sfx.max_chars,
            engine=cfg.ocr.engine,
            lang=cfg.ocr.lang,
        )
        found = total = stray = 0
        for seed in (int(s) for s in args.seeds.split(",")):
            image, balloons, effects = make_page(seed, faces)
            strip = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(manager.device)
            regions = [
                Region(
                    id=f"r{i + 1:04d}",
                    slice_index=0,
                    kind="bubble_text",
                    bbox=BBox(x0=b[0] + 30, y0=b[1] + 50, x1=b[2] - 30, y1=b[3] - 50),
                    bubble_bbox=BBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3]),
                )
                for i, b in enumerate(balloons)
            ]
            tiles = plan_tiles(WIDTH, HEIGHT, cfg.ocr.tile_px, cfg.ocr.overlap)
            out, metrics = sweep(
                strip,
                regions,
                [Slice(index=0, y0=0, y1=HEIGHT)],
                tiles,
                models["sfx_sweeper"],
                reader,
                rules,
                min_px=cfg.sfx.sweep_min_px,
            )
            new = out[len(regions) :]
            print(f"page {seed}: {', '.join(f'{k} {v:g}' for k, v in metrics.items())}")
            sheet = Image.fromarray(image)
            draw = ImageDraw.Draw(sheet)
            for text, box in effects:
                hits = [r for r in new if r.kind == "sfx" and _on(r.bbox, box)]
                total += 1
                found += bool(hits)
                read = ", ".join(
                    f"{r.text!r}" + (f" (read {r.ocr_alt!r})" if r.ocr_alt else "") for r in hits
                )
                print(f"  {text:8} {'found' if hits else 'MISSED':7} {read}")
                draw.rectangle(box, outline=(0, 200, 0) if hits else (230, 0, 0), width=4)
            strays = [r for r in new if not any(_on(r.bbox, box) for _, box in effects)]
            stray += len(strays)
            for r in strays:
                print(f"  stray {r.kind} {r.text!r} at {r.bbox.x0},{r.bbox.y0}")
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                sheet.save(args.out / f"page-{seed}.png")
        print(f"found {found}/{total} effects, {stray} stray regions")
        manager.release()
    finally:
        if lock is not None:
            release_gpu_lock(lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
