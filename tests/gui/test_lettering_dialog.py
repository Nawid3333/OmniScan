"""LetteringDialog tests (offscreen): only ticked styles come back, pre-filled from a region's hand lettering."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from omniscan.gui.lettering_dialog import TYPESETTER, LetteringDialog

FONTS = ("Mali-Bold.ttf", "Kalam-Bold.ttf")


def test_nothing_ticked_changes_nothing(qapp: QApplication) -> None:
    dialog = LetteringDialog(FONTS)
    assert dialog.fields() == {} and not dialog.reverts()
    assert dialog.font_combo.itemText(0) == TYPESETTER and not dialog.size_spin.isEnabled()


def test_ticked_styles_come_back_as_layout_fields(qapp: QApplication) -> None:
    dialog = LetteringDialog(FONTS, count=3)
    assert dialog.windowTitle() == "Lettering of 3 regions"
    dialog.set_change("size_px")
    dialog.size_spin.setValue(32)
    dialog.set_change("font")
    dialog.font_combo.setCurrentText("Kalam-Bold.ttf")
    dialog.set_change("hidden")
    assert dialog.fields() == {"font": "Kalam-Bold.ttf", "size_px": 32, "hidden": False}
    dialog.font_combo.setCurrentText(TYPESETTER)
    assert dialog.fields()["font"] is None  # back to the typesetter's font
    dialog.revert_check.setChecked(True)
    assert dialog.reverts()


def test_prefilled_from_the_regions_hand_lettering(qapp: QApplication) -> None:
    current = {
        "font": "Mali-Bold.ttf",
        "size_px": 18,
        "color": (255, 0, 0),
        "stroke_px": 3,
        "align": "left",
        "angle": -5.0,
    }
    dialog = LetteringDialog(FONTS, current)
    assert dialog.font_combo.currentText() == "Mali-Bold.ttf" and dialog.size_spin.value() == 18
    assert dialog.color_button.color() == (255, 0, 0) and dialog.color_button.text() == "#ff0000"
    assert dialog.stroke_spin.value() == 3 and dialog.align_combo.currentText() == "left"
    assert dialog.angle_spin.value() == -5.0 and not dialog.hidden_check.isChecked()
    for key in ("color", "stroke_color", "align", "angle"):
        dialog.set_change(key)
    assert dialog.fields() == {
        "color": [255, 0, 0],
        "stroke_color": [255, 255, 255],
        "align": "left",
        "angle": -5.0,
    }
