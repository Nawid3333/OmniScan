"""Map a Hugging Face repo id to the model manager's installed folder (card U2b).

Loaders call `local_model_source` before `from_pretrained`: a `zip` catalog entry whose
`upstream_repo` matches and whose install state is `installed` wins over the Hugging Face hub/cache,
so an installed app also works offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omniscan.models.catalog import ModelEntry, load_catalog
from omniscan.models.store import install_path, model_status


def local_model_source(
    repo: str, models_dir: Path, catalog: Sequence[ModelEntry] | None = None
) -> str | None:
    """`<models_dir>/<id>` as a str when `repo` has an installed zip entry in the catalog, else None."""
    try:
        entries = load_catalog() if catalog is None else catalog
    except OSError, ValueError:  # unreadable/corrupt catalog: use the hub, never raise (PEP 758)
        return None
    for entry in entries:
        if entry.format != "zip" or entry.upstream_repo != repo:
            continue
        path = install_path(entry, models_dir)
        if path is not None and model_status(entry, models_dir, ollama_names=None) == "installed":
            return str(path)
    return None
