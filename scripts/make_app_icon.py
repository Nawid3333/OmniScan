"""Draw the app icon (packaging/omniscan.png and .ico): a speech balloon with "OS" in the accent colour.

    python scripts/make_app_icon.py

Run again after changing the design; the installers (scripts/build_installer.py) and the PyInstaller spec use the
files it writes.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SIZE = 512
BACKGROUND = (12, 12, 16, 255)
ACCENT = (124, 92, 255, 255)
INK = (255, 255, 255, 255)


def draw() -> Image.Image:
    """The icon at SIZE x SIZE."""
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas = ImageDraw.Draw(image)
    canvas.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=SIZE // 6, fill=BACKGROUND)
    margin = SIZE // 8
    balloon = (margin, margin, SIZE - margin, SIZE - margin - SIZE // 10)
    canvas.ellipse(balloon, fill=ACCENT)
    tail_x = SIZE // 2 - SIZE // 8
    canvas.polygon(
        [
            (tail_x, balloon[3] - SIZE // 20),
            (tail_x + SIZE // 10, balloon[3] - SIZE // 20),
            (tail_x - SIZE // 20, SIZE - margin // 2),
        ],
        fill=ACCENT,
    )
    font = ImageFont.truetype(str(ROOT / "fonts" / "Bangers-Regular.ttf"), SIZE // 3)
    centre = ((balloon[0] + balloon[2]) / 2, (balloon[1] + balloon[3]) / 2)
    canvas.text(centre, "OS", font=font, fill=INK, anchor="mm")
    return image


def main() -> None:
    """Write the PNG and the multi-size ICO."""
    image = draw()
    out = ROOT / "packaging"
    image.resize((256, 256), Image.Resampling.LANCZOS).save(out / "omniscan.png")
    image.save(out / "omniscan.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(out / "omniscan.png", out / "omniscan.ico")


if __name__ == "__main__":
    main()
