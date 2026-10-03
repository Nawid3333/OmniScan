"""FoundDialog: the text areas *Find missed text* found on a page, ticked to be added as regions.

One row per area the detector found that no current region covers (detect/on_demand.py), with what the OCR
read there and both scores. Every row starts ticked; `picked()` returns the ticked ones, which the Studio adds
as hand-drawn regions (each its own undo step).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from omniscan.detect.on_demand import Found


def describe(found: Found) -> str:
    """One row of the list: kind, the text read there, its size and both scores."""
    box = found.bbox
    text = found.text or "(nothing read)"
    return (
        f"{found.kind} · {text!r} · {box.width}×{box.height} px at y {box.y0} · "
        f"detector {found.score:.2f}, OCR {found.confidence:.2f}"
    )


class FoundDialog(QDialog):
    """Tick the found text areas to add as regions."""

    def __init__(self, found: Sequence[Found], *, page: int, parent: QWidget | None = None) -> None:
        """List `found` (from raw page `page`), every row ticked."""
        super().__init__(parent)
        self.setWindowTitle("Missed text")
        self._found = list(found)
        self.list = QListWidget(self)
        for item in self._found:
            row = QListWidgetItem(describe(item), self.list)
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            row.setCheckState(Qt.CheckState.Checked)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add ticked")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Page {page + 1}: {len(self._found)} text area(s) no region covers. "
                "The ticked ones become regions with the text shown (each one undoable).",
                self,
            )
        )
        layout.addWidget(self.list, 1)
        layout.addWidget(buttons)
        self.resize(720, 360)

    def picked(self) -> list[Found]:
        """The ticked areas, in list order."""
        return [
            found
            for row, found in enumerate(self._found)
            if (item := self.list.item(row)) is not None and item.checkState() == Qt.CheckState.Checked
        ]
