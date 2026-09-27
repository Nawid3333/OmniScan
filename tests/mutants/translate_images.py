import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

IM = "src/omniscan/translate/images.py"
R = "src/omniscan/translate/run.py"
MUTANTS = [
    (
        R,
        "        if current and (full or (with_images and new_slice and len(slices) >= profile.images_per_request)):",
        "        if current and full:",
        "requests not cut to images_per_request pages",
    ),
    (
        R,
        "                images if profile.images else None,",
        "                images,",
        "profiles without images send them too",
    ),
    (
        IM,
        "            round((bbox.y0 - self.y0) * self.scale),",
        "            round(bbox.y0 * self.scale),",
        "boxes not moved onto their page",
    ),
    (
        IM,
        "    if not profile.images:\n        return {}\n",
        "    return {}\n",
        "switching images on keeps the old translations",
    ),
    (IM, "    scale = min(1.0, side / max(image.size))", "    scale = 1.0", "pages sent at full size"),
    (
        IM,
        "        layout = None if self._broken else self._slices()",
        "        layout = self._slices()",
        "unreadable pages retried for every request",
    ),
    (
        "src/omniscan/translate/incremental.py",
        '"chunk_regions", *IMAGE_FIELDS}',
        '"chunk_regions"}',
        "adding the image fields changes every existing key",
    ),
    (
        "src/omniscan/translate/prompts.py",
        '        user["images"] = [image.data for image in images]\n',
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
        "src/omniscan/llm/ollama.py",
        "_IMAGE_TOKENS * images + ",
        "",
        "a local model's context ignores the images",
    ),
    (
        "src/omniscan/translate/profiles.py",
        '        if self.images and self.style != "chat_json":',
        "        if False:",
        "images accepted on a text-only style",
    ),
]
