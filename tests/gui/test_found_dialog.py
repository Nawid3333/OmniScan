"""FoundDialog tests (offscreen): every found area listed and ticked, unticked ones left out."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from omniscan.core.schemas import BBox
from omniscan.detect.on_demand import Found
from omniscan.gui.found_dialog import FoundDialog, describe

FOUND = [
    Found("free_text", BBox(x0=2, y0=110, x1=30, y1=140), None, 0.41, "쾅", 0.9),
    Found("bubble_text", BBox(x0=4, y0=150, x1=36, y1=190), None, 0.38, "", 0.0),
]


def test_every_area_is_listed_ticked_and_unticked_ones_are_left_out(qapp: QApplication) -> None:
    dialog = FoundDialog(FOUND, page=1)
    assert dialog.list.count() == 2
    assert dialog.list.item(0).text() == describe(FOUND[0])  # type: ignore[union-attr]
    assert describe(FOUND[0]) == "free_text · '쾅' · 28×30 px at y 110 · detector 0.41, OCR 0.90"
    assert "(nothing read)" in describe(FOUND[1])
    assert dialog.picked() == FOUND
    dialog.list.item(0).setCheckState(Qt.CheckState.Unchecked)  # type: ignore[union-attr]
    assert dialog.picked() == [FOUND[1]]
