import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

IM = "src/omniscan/translate/images.py"
R = "src/omniscan/translate/run.py"
C = "src/omniscan/translate/chapter.py"
ST = "src/omniscan/pipeline/stages.py"
OL = "src/omniscan/llm/ollama.py"
MUTANTS = [
    (
        R,
        "        crowded = image_of is not None and image not in shown and len(shown) >= profile.images_per_request",
        "        crowded = False",
        "requests not cut to images_per_request images",
    ),
    (
        R,
        "                images if profile.images else None,",
        "                images,",
        "profiles without images send them too",
    ),
    (
        R,
        "                images={r.id: pages[r.id] for r in missing if r.id in pages},  # only the pages still needed",
        "                images=pages,",
        "repair rounds re-send every page",
    ),
    (
        IM,
        "            return round((min(max(y, tile.y0), tile.y1) - tile.y0) * tile.scale)",
        "            return round(y * tile.scale)",
        "boxes not moved onto their tile",
    ),
    (
        IM,
        "    if not profile.images or page is None:\n        return {}\n",
        "    if page is None:\n        return {}\n",
        "a profile without images keys on the page",
    ),
    (
        IM,
        "    if not profile.images or page is None:\n        return {}\n",
        "    if not profile.images:\n        return {}\n",
        "a line sent without its page keyed as seen",
    ),
    (
        IM,
        '    return {"images": {"side": profile.image_side, "page": page}}',
        '    return {"images": {"side": profile.image_side}}',
        "a replaced page keeps the old translation",
    ),
    (
        IM,
        "    count = max(1, math.ceil(height * scale / side))",
        "    count = 1",
        "tall slices sent whole (a sliver of a page)",
    ),
    (IM, "    scale = min(1.0, side / width)", "    scale = 1.0", "wide pages sent at full size"),
    (
        IM,
        "        return layout[0], next((t for t in tiles if centre < t.y1), tiles[-1])",
        "        return layout[0], tiles[0]",
        "every region placed on its slice's first tile",
    ),
    (
        IM,
        "        layout = None if self._broken else self._slices()",
        "        layout = self._slices()",
        "unreadable pages retried for every request",
    ),
    (
        IM,
        "    return profile.model_dump(exclude=None if profile.images else set(IMAGE_FIELDS))",
        "    return profile.model_dump()",
        "the image fields change the stage hash of every profile",
    ),
    (IM, "        self.sent |= found.keys()\n", "", "no line counts as sent with its page"),
    (
        "src/omniscan/translate/incremental.py",
        '"chunk_regions", *IMAGE_FIELDS}',
        '"chunk_regions"}',
        "adding the image fields changes every existing key",
    ),
    (
        "src/omniscan/translate/prompts.py",
        '        user["images"] = [image.data for image in shown]\n',
        "        pass\n",
        "images described but not attached",
    ),
    (
        "src/omniscan/translate/suggest.py",
        "                images=images,\n            )\n        except OllamaRateLimitError:",
        "            )\n        except OllamaRateLimitError:",
        "the Studio's Translate never sees the page",
    ),
    (
        C,
        "            if region.id not in reused and region.id not in pages.sent\n",
        "            if False\n",
        "a line sent without its page keyed as seen (chapter)",
    ),
    (C, "    if pages is not None:\n        pages.begin()\n", "", "sent lines of an earlier profile count"),
    (
        ST,
        "        images = PageImages(ctx.paths) if self._images else None  # read and encoded once for every profile",
        "        images = None",
        "every profile reads and encodes the pages again",
    ),
    (
        ST,
        "        if self._images:  # what a vision model sees: the page layout and the raw pages under it",
        "        if False:",
        "a replaced page never re-runs the stage",
    ),
    (
        OL,
        "        pixels = sum(_image_tokens(image) for image in images)",
        "        pixels = 0",
        "images ignored",
    ),
    (
        OL,
        "    return math.ceil(width / _IMAGE_PATCH_PX) * math.ceil(height / _IMAGE_PATCH_PX)",
        "    return _IMAGE_TOKENS",
        "every image counted the same",
    ),
    (
        OL,
        "            needed = self._image_ctx[model] = max(needed, self._image_ctx.get(model, 0))",
        "            self._image_ctx[model] = needed",
        "num_ctx swings with the number of pages",
    ),
    (
        "src/omniscan/translate/profiles.py",
        '        if self.images and self.style != "chat_json":',
        "        if False:",
        "images accepted on a text-only style",
    ),
]
