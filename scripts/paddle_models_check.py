"""M0 check: Paddle OCR *models* run via HF transformers (PyTorch/ROCm) on Python 3.14 — no paddlepaddle.

Renders synthetic text lines with Windows CJK fonts and recognizes them on the GPU.
"""

import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForTextRecognition

from omniscan.gpu.device import resolve_device

DEVICE = resolve_device()
if DEVICE.type == "cuda":
    torch.cuda.set_device(DEVICE)  # bare torch.cuda.* calls then mean the chosen GPU

FONTS = (
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if os.name == "nt"
    else Path("/mnt/c/Windows/Fonts")
)
CASES = [
    (
        "PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors",
        "malgun.ttf",
        "민준 형, 괜찮아요? 던전이 열렸어!",
    ),
    ("PaddlePaddle/PP-OCRv6_small_rec_safetensors", "msyh.ttc", "师兄，这个秘境太危险了！"),
    ("PaddlePaddle/PP-OCRv6_small_rec_safetensors", "YuGothM.ttc", "先輩、ちょっと待ってください！"),
]


def render(text: str, font_file: str) -> Image.Image:
    font = ImageFont.truetype(str(FONTS / font_file), 40)
    w = int(font.getlength(text)) + 40
    img = Image.new("RGB", (w, 64), "white")
    ImageDraw.Draw(img).text((20, 8), text, font=font, fill="black")
    return img


def main() -> int:
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    ok = True
    for repo, font, text in CASES:
        model = AutoModelForTextRecognition.from_pretrained(repo).to(DEVICE).eval()
        proc = AutoImageProcessor.from_pretrained(repo)
        inputs = proc(images=[render(text, font)] * 8, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            model(**inputs)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(**inputs)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
        res = proc.post_process_text_recognition(out)[0]
        match = res["text"].replace(" ", "") == text.replace(" ", "")
        ok &= match
        print(
            f"{repo.split('/')[1]:42s} {'OK  ' if match else 'DIFF'} score={res['score']:.3f} "
            f"batch8={dt:.1f}ms  got={res['text']!r}  want={text!r}"
        )
        del model
        torch.cuda.empty_cache()
    print("PADDLE MODELS CHECK", "OK" if ok else "HAS DIFFS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
