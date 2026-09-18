"""Probe ogkalu/comic-text-and-bubble-detector on a page: tile it, merge, draw boxes to a PNG.

Usage: uv run python scripts/probe_detector.py <page.jpg> <out_dir> [single|tiled] [work_width]
"""

import sys
import time
from pathlib import Path

import torch
import torchvision.ops as ops
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

REPO = "ogkalu/comic-text-and-bubble-detector"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path()
page = Path(sys.argv[1])
mode = sys.argv[3] if len(sys.argv) > 3 else "single"
work_w = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
dev = torch.device("cuda:0")

t = time.time()
proc = AutoImageProcessor.from_pretrained(REPO)
model = RTDetrV2ForObjectDetection.from_pretrained(REPO).to(dev).eval()
print(
    "model load",
    round(time.time() - t, 1),
    "s; params",
    sum(p.numel() for p in model.parameters()) / 1e6,
    "M",
)
id2label = model.config.id2label
COLORS = {"bubble": (30, 120, 255), "text_bubble": (255, 60, 60), "text_free": (255, 170, 0)}

img = Image.open(page).convert("RGB")
W0, H0 = img.size
scale = work_w / W0
img_w = img.resize((work_w, round(H0 * scale)), Image.Resampling.LANCZOS)
W, H = img_w.size


@torch.no_grad()
def detect(tile: Image.Image, thr: float):
    inputs = proc(images=tile, return_tensors="pt").to(dev)
    out = model(**inputs)
    res = proc.post_process_object_detection(out, threshold=thr, target_sizes=[(tile.height, tile.width)])
    assert res is not None
    res = res[0]
    return [
        (id2label[int(label)], float(score), [float(v) for v in box])
        for score, label, box in zip(
            res["scores"].cpu(), res["labels"].cpu(), res["boxes"].cpu(), strict=True
        )
    ]


boxes = []
t = time.time()
if mode == "single":
    boxes = detect(img_w, 0.3)
else:
    tile = W  # square tiles of width x width
    step = int(tile * 0.6)
    y = 0
    n = 0
    while True:
        y1 = min(y + tile, H)
        crop = img_w.crop((0, y1 - tile if y1 - tile >= 0 else 0, W, y1))
        oy = max(0, y1 - tile)
        for name, s, b in detect(crop, 0.2):
            boxes.append((name, s, [b[0], b[1] + oy, b[2], b[3] + oy]))
        n += 1
        if y1 >= H:
            break
        y += step
    print("tiles", n)
torch.cuda.synchronize()
print("detect", round(time.time() - t, 2), "s; raw boxes", len(boxes))

# crude class-wise NMS across tiles
kept = []
for cls in COLORS:
    cb = [(s, b) for n, s, b in boxes if n == cls]
    if not cb:
        continue
    bt = torch.tensor([b for _, b in cb])
    st = torch.tensor([s for s, _ in cb])
    idx = ops.nms(bt, st, 0.5).tolist()
    kept += [(cls, cb[i][0], cb[i][1]) for i in idx]
print("after nms", len(kept), {c: sum(1 for k in kept if k[0] == c) for c in COLORS})

draw = ImageDraw.Draw(img_w)
for cls, s, b in kept:
    draw.rectangle(b, outline=COLORS[cls], width=3)
    draw.text((b[0] + 2, b[1] + 2), f"{cls[:6]} {s:.2f}", fill=COLORS[cls])
small = img_w.resize((W // 2, H // 2), Image.Resampling.LANCZOS)
outp = OUT / f"det_{page.parent.name.replace(' ', '')}_{page.stem}_{mode}.png"
small.save(outp)
print("wrote", outp, small.size)
