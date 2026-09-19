"""Mutants for src/omniscan/ingest/layout.py and convert.py (tests: ingest_layout, ingest_convert, ingest)."""

LA = "src/omniscan/ingest/layout.py"
CO = "src/omniscan/ingest/convert.py"

MUTANTS = [
    # ---- layout.py: dominant_width
    (LA, "return max(counts.items(), key=lambda item: (item[1], item[0]))[0]",
       "return max(counts.items(), key=lambda item: (item[0], item[1]))[0]", "tie key width/count swapped"),
    (LA, "return max(counts.items(), key=lambda item: (item[1], item[0]))[0]",
       "return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]", "tie breaks to smaller width"),
    (LA, "return max(counts.items(), key=lambda item: (item[1], item[0]))[0]",
       "return min(counts.items(), key=lambda item: (item[1], item[0]))[0]", "max -> min"),
    (LA, "return max(counts.items(), key=lambda item: (item[1], item[0]))[0]",
       "return max(counts.items(), key=lambda item: (item[1], item[0]))[1]", "return count not width"),
    (LA, "    if not widths:\n        raise ValueError(\"widths is empty\")\n    counts = Counter(widths)",
       "    counts = Counter(widths)", "drop empty guard"),
    # ---- layout.py: stack_layout
    (LA, "scale = strip_width / width", "scale = width / strip_width", "scale inverted"),
    (LA, "scaled_height = max(1, round(height * scale))", "scaled_height = min(1, round(height * scale))", "max(1,) -> min(1,)"),
    (LA, "scaled_height = max(1, round(height * scale))", "scaled_height = max(1, int(height * scale))", "round -> int (floor)"),
    (LA, "layout.append((scale, y, y + scaled_height))", "layout.append((scale, y, y + height))", "y1 uses raw height"),
    (LA, "y += scaled_height", "y += height", "y advance uses raw height"),
    (LA, "for width, height in sizes:", "for height, width in sizes:", "unpack width/height swapped"),
    (LA, "    y = 0", "    y = 1", "y starts at 1"),
    # ---- convert.py
    (CO, "_EXIF_ORIENTATION = 0x0112", "_EXIF_ORIENTATION = 0x0113", "exif tag 0x0112 -> 0x0113"),
    (CO, "return img.getexif().get(_EXIF_ORIENTATION, 1) != 1", "return img.getexif().get(_EXIF_ORIENTATION, 1) == 1", "needs_rotation != -> =="),
    (CO, "if img.format != \"JPEG\" or img.mode != \"RGB\":", "if img.format != \"JPEG\" and img.mode != \"RGB\":", "format/mode or -> and"),
    (CO, "if img.format != \"JPEG\" or img.mode != \"RGB\":", "if img.format != \"JPEG\" or img.mode == \"RGB\":", "mode != -> =="),
    (CO, "if path.suffix.lower() not in (\".jpg\", \".jpeg\"):", "if path.suffix.lower() not in (\".jpg\",):", "drop .jpeg from suffix set"),
    (CO, "if path.suffix.lower() not in (\".jpg\", \".jpeg\"):\n        return True",
       "if path.suffix.lower() not in (\".jpg\", \".jpeg\"):\n        return False", "suffix branch True -> False"),
    (CO, "if img.mode in _ALPHA_MODES or (img.mode == \"P\" and \"transparency\" in img.info):",
       "if img.mode in _ALPHA_MODES and (img.mode == \"P\" and \"transparency\" in img.info):", "alpha or -> and"),
    (CO, "if img.mode in _ALPHA_MODES or (img.mode == \"P\" and \"transparency\" in img.info):",
       "if img.mode in _ALPHA_MODES or (img.mode == \"P\" and \"transparency\" not in img.info):", "transparency in -> not in"),
    (CO, "canvas.paste(rgba, mask=rgba.getchannel(\"A\"))", "canvas.paste(rgba)", "paste drops alpha mask"),
    (CO, "canvas = Image.new(\"RGB\", rgba.size, (255, 255, 255))", "canvas = Image.new(\"RGB\", rgba.size, (0, 0, 0))", "canvas white -> black"),
    (CO, "if transposed is not None:", "if transposed is None:", "transposed is not None -> is None"),
    (CO, "if img.mode != \"RGB\":\n        return img.convert(\"RGB\")", "if img.mode == \"RGB\":\n        return img.convert(\"RGB\")", "flatten mode != -> =="),
    (CO, "dest = cache_dir / f\"{index:04d}_{src.stem}.jpg\"", "dest = cache_dir / f\"{src.stem}.jpg\"", "dest name drops index"),
    (CO, "sha256=hash_file(dest), width=out.width, height=out.height, converted=True",
       "sha256=hash_file(src), width=out.width, height=out.height, converted=True", "sha of src not dest"),
    (CO, "width=out.width, height=out.height, converted=True", "height=out.width, width=out.height, converted=True", "converted width/height swapped"),
    (CO, "quality: int = 95", "quality: int = 5", "default quality 95 -> 5"),
    (CO, "return _needs_rotation(img)", "return not _needs_rotation(img)", "negate needs_rotation"),
    (CO, "    cache_dir.mkdir(parents=True, exist_ok=True)\n    dest = cache_dir",
       "    dest = cache_dir", "drop cache_dir.mkdir"),
    (CO, "out.save(dest, format=\"JPEG\", quality=quality, subsampling=0, optimize=True)",
       "out.save(dest, format=\"PNG\", quality=quality, subsampling=0, optimize=True)", "save as PNG"),
]