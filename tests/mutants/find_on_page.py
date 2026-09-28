import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # see the README: stale bytecode can poison verdicts

D = "src/omniscan/detect/on_demand.py"
CLI = "src/omniscan/edits/cli.py"
WEB = "src/omniscan/web/app.py"
MUTANTS = [
    (
        D,
        "        ioa(found, _box(region.bbox)) >= COVERED_IOA or ioa(_box(region.bbox), found) >= COVERED_IOA\n",
        "        ioa(found, _box(region.bbox)) >= COVERED_IOA\n",
        "a bubble found around a region is suggested again",
    ),
    (
        D,
        "        ioa(found, _box(region.bbox)) >= COVERED_IOA or ioa(_box(region.bbox), found) >= COVERED_IOA\n",
        "        ioa(_box(region.bbox), found) >= COVERED_IOA\n",
        "text a region already has is suggested again",
    ),
    (
        D,
        "        ioa(found, _box(region.bbox)) >= COVERED_IOA or ioa(_box(region.bbox), found) >= COVERED_IOA\n",
        "        ioa(found, _box(region.bbox)) > 0 or ioa(_box(region.bbox), found) > 0\n",
        "any touch counts as covered",
    ),
    (D, " and not file.filtered", "", "a promo page is searched"),
    (
        CLI,
        "        found = find_missed(paths, files[page - 1].index, scfg, threshold)",
        "        found = find_missed(paths, page - 1, scfg, threshold)",
        "the Studio's page number taken for the ingest index",
    ),
    (CLI, "        if page > len(files):\n", "        if False:\n", "a page past the chapter crashes"),
    (
        CLI,
        "        with store.edit_group(paths):  # one undo step",
        "        if True:",
        "--add is one undo step per box",
    ),
    (
        CLI,
        "                        text=item.text,\n                        bubble_bbox=item.bubble_bbox,",
        "                        bubble_bbox=item.bubble_bbox,",
        "--add drops what the OCR read",
    ),
    (
        WEB,
        "            found = page_finder(paths, page, scfg, body.threshold)",
        "            found = page_finder(paths, page, scfg, None)",
        "the web threshold ignored",
    ),
    (
        WEB,
        '        except ValueError as exc:\n            raise HTTPException(status_code=422, detail=str(exc)) from exc\n        except (ImportError, OSError, RuntimeError) as exc:  # no torch backend, no models, no GPU\n            raise HTTPException(status_code=503, detail=f"the detector',
        '        except (ImportError, OSError, RuntimeError) as exc:  # no torch backend, no models, no GPU\n            raise HTTPException(status_code=503, detail=f"the detector',
        "no such page is a server error",
    ),
]
