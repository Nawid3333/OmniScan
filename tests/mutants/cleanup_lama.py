import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

ST = "src/omniscan/cleanup/store.py"
LN = "src/omniscan/cleanup/lama_now.py"
WEB = "src/omniscan/web/app.py"
MUTANTS = [
    (
        ST,
        "            context_box = _grow(sbox, lama.context_px, ingest)",
        "            context_box = _grow(sbox, 0, ingest)",
        "the model sees the stroke without the page around it",
    ),
    (
        ST,
        "            if lama is None:\n                raise ValueError(",
        "            if False:\n                raise ValueError(",
        "a lama stroke without a model crashes instead of being refused",
    ),
    (
        ST,
        "            pixels = _cut(rebuilt, context_box, sbox)\n        patch = CleanupPatch(",
        "            pixels = _cut(current_crop(paths, ingest, context_box), context_box, sbox)\n        patch = CleanupPatch(",
        "the model's rebuild thrown away",
    ),
    (
        LN,
        "    page_to_strip(load_ingest(paths), page, box, mask)\n",
        "",
        "the model loads for a refused stroke",
    ),
    (WEB, '        if body.method == "lama":', "        if False:", "the web route never runs LaMa"),
    (
        WEB,
        "            except (ImportError, RuntimeError, OSError) as exc:  # no torch, a failed download, no GPU",
        "            except OSError as exc:  # no torch, a failed download, no GPU",
        "a failed LaMa load is a server error, not 'not available'",
    ),
]
