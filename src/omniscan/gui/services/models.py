"""Models service: the settings screen's data source for the model catalog (Qt-free).

`rows` returns the same `ModelRow`s the CLI `models list --json` prints; `download`/`remove`/
`download_required` wrap `models.download`. Callers run these on worker threads — none of the
methods touch Qt.
"""

from __future__ import annotations

from collections.abc import Callable

from omniscan.core.config import Config
from omniscan.hw.detect import HardwareInfo, detect_hardware
from omniscan.models.catalog import ModelEntry, load_catalog
from omniscan.models.download import ModelDownloadError, download_model, remove_model
from omniscan.models.rows import ModelRow, build_rows, ollama_model_names
from omniscan.models.store import model_status

_Progress = Callable[[int, int | None], None]
_RequiredProgress = Callable[[str, int, int | None], None]


class ModelsService:
    """Rows, downloads and removals for the models settings screen."""

    def __init__(
        self,
        cfg: Config,
        *,
        ollama_names: Callable[[], set[str] | None] | None = None,
        hardware: Callable[[], HardwareInfo] | None = None,
    ) -> None:
        """Build the service; the defaults query the Ollama daemon and detect this machine."""
        self._cfg = cfg
        self._ollama_names = ollama_names or (lambda: ollama_model_names(cfg.ollama.local_url))
        self._hardware = hardware or (lambda: detect_hardware(cfg.paths.models_dir))

    def rows(
        self, *, role: str | None = None, lang: str | None = None
    ) -> tuple[list[ModelRow], HardwareInfo]:
        """Catalog rows (role/lang filtered, catalog order) plus this machine's hardware snapshot."""
        return build_rows(
            self._cfg,
            role=role,
            lang=lang,
            catalog=load_catalog(),
            hardware=self._hardware(),
            ollama_names=self._ollama_names(),
        )

    def download(self, model_id: str, on_progress: _Progress | None = None) -> str:
        """Download one catalog model; returns its source ("mirror" | "upstream" | "ollama" | "already installed")."""
        entry = self._entry(model_id)
        return download_model(
            entry,
            self._cfg.paths.models_dir,
            ollama_url=self._cfg.ollama.local_url,
            on_progress=None if on_progress is None else (lambda _id, done, total: on_progress(done, total)),
        )

    def remove(self, model_id: str) -> bool:
        """Remove an installed model (folder, file, or the Ollama daemon's copy); True when removed."""
        entry = self._entry(model_id)
        return remove_model(entry, self._cfg.paths.models_dir, ollama_url=self._cfg.ollama.local_url)

    def download_required(self, on_progress: _RequiredProgress | None = None) -> list[tuple[str, str | None]]:
        """Download every required model that is missing/corrupt, in catalog order.

        Returns `(model id, error text or None)` per attempted model; one failure does not stop
        the others.
        """
        cfg = self._cfg
        results: list[tuple[str, str | None]] = []
        for entry in load_catalog():
            if not entry.required:
                continue
            status = model_status(entry, cfg.paths.models_dir, ollama_names=self._ollama_names())
            if status not in ("missing", "corrupt"):
                continue
            try:
                download_model(
                    entry,
                    cfg.paths.models_dir,
                    ollama_url=cfg.ollama.local_url,
                    on_progress=on_progress,
                )
            except ModelDownloadError as exc:
                results.append((entry.id, str(exc)))
            else:
                results.append((entry.id, None))
        return results

    def _entry(self, model_id: str) -> ModelEntry:
        """The catalog entry with `model_id`; ValueError when it is unknown."""
        for entry in load_catalog():
            if entry.id == model_id:
                return entry
        raise ValueError(f"unknown model {model_id!r}")
