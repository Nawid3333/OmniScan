"""Hardware service: the machine snapshot plus per-model compatibility warnings (Qt-free).

The report reuses the models service's rows (catalog + `hw.assess` fit levels), so the hardware
panel shows the same per-model warnings the models list shows. Callers run `report` on a worker
thread — `detect_hardware` imports torch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omniscan.core.config import Config
from omniscan.gui.services.models import ModelsService
from omniscan.hw.detect import HardwareInfo
from omniscan.models.rows import ModelRow


@dataclass(frozen=True, slots=True)
class ModelWarning:
    """One catalog model that does not fit this machine cleanly (`assess` level and reasons)."""

    name: str
    level: str  # slow | warn | incompatible
    device: str | None
    messages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HardwareReport:
    """The machine snapshot plus every model that would run badly or not at all."""

    info: HardwareInfo
    warnings: tuple[ModelWarning, ...]


class HardwareService:
    """The settings page's hardware panel data source."""

    def __init__(
        self,
        cfg: Config,
        *,
        rows: Callable[[], tuple[list[ModelRow], HardwareInfo]] | None = None,
    ) -> None:
        """Build the service; the default rows provider is the models service (detect + catalog fit)."""
        self._rows = rows or ModelsService(cfg).rows

    def report(self) -> HardwareReport:
        """This machine's snapshot plus one warning row per model whose fit is not `ok`."""
        rows, hw = self._rows()
        return HardwareReport(
            info=hw,
            warnings=tuple(
                ModelWarning(
                    name=row.name, level=row.fit_level, device=row.fit_device, messages=row.fit_messages
                )
                for row in rows
                if row.fit_level != "ok"
            ),
        )
