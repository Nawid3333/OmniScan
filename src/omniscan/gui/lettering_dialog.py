"""LetteringDialog: hand-set the lettering of one or many regions at once.

Every style has a *Change* tick: only ticked styles are applied, so a dialog over a multi-selection can set
just the size (or just the colour) of twenty balloons and leave their other hand-set styles alone. The
values come back as LayoutEdit's style fields (`fields()`), ready for `StudioSession.set_layout`.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from omniscan.edits.session import LayoutFields

TYPESETTER = "(typesetter's choice)"
ALIGNMENTS = ("center", "left", "right")
RGB = tuple[int, int, int]


def _rgb(value: object) -> RGB | None:
    """A LayoutEdit colour as an (r, g, b) tuple, or None."""
    if isinstance(value, list | tuple) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    return None


class _ColorButton(QPushButton):
    """A button showing a colour; clicking it opens the colour picker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Start black."""
        super().__init__(parent)
        self._color: RGB = (0, 0, 0)
        self.clicked.connect(self._pick)
        self._show()

    def color(self) -> RGB:
        """The chosen colour."""
        return self._color

    def set_color(self, color: RGB) -> None:
        """Show `color`."""
        self._color = color
        self._show()

    def _show(self) -> None:
        r, g, b = self._color
        self.setText(f"#{r:02x}{g:02x}{b:02x}")
        self.setStyleSheet(f"background: rgb({r},{g},{b}); color: {'white' if r + g + b < 384 else 'black'};")

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(*self._color), self, "Colour")
        if chosen.isValid():
            self.set_color((chosen.red(), chosen.green(), chosen.blue()))


class LetteringDialog(QDialog):
    """Font, size, colour, outline, alignment, angle and visibility of the selected regions' lettering."""

    def __init__(
        self,
        fonts: Sequence[str],
        current: LayoutFields | None = None,
        *,
        count: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        """Offer `fonts` (file names); pre-fill from `current` (one region's hand lettering) when given."""
        super().__init__(parent)
        self.setWindowTitle("Lettering" if count == 1 else f"Lettering of {count} regions")
        current = current or {}
        self.font_combo = QComboBox(self)
        self.font_combo.addItem(TYPESETTER)
        self.font_combo.addItems(list(fonts))
        self.size_spin = QSpinBox(self)
        self.size_spin.setRange(4, 400)
        self.size_spin.setSuffix(" px")
        self.color_button = _ColorButton(self)
        self.stroke_spin = QSpinBox(self)
        self.stroke_spin.setRange(0, 40)
        self.stroke_spin.setSuffix(" px")
        self.stroke_color_button = _ColorButton(self)
        self.stroke_color_button.set_color((255, 255, 255))
        self.align_combo = QComboBox(self)
        self.align_combo.addItems(ALIGNMENTS)
        self.angle_spin = QDoubleSpinBox(self)
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setSuffix("°")
        self.hidden_check = QCheckBox("no lettering for this region", self)
        self.revert_check = QCheckBox(
            "Give the lettering back to the typesetter (drops every hand-set style)", self
        )

        self._rows: dict[str, tuple[QCheckBox, QWidget]] = {}
        grid = QGridLayout()
        for row, (key, label, editor) in enumerate(
            (
                ("font", "Font", self.font_combo),
                ("size_px", "Size", self.size_spin),
                ("color", "Colour", self.color_button),
                ("stroke_px", "Outline", self.stroke_spin),
                ("stroke_color", "Outline colour", self.stroke_color_button),
                ("align", "Alignment", self.align_combo),
                ("angle", "Angle", self.angle_spin),
                ("hidden", "Hidden", self.hidden_check),
            )
        ):
            change = QCheckBox("Change", self)
            self._rows[key] = (change, editor)
            grid.addWidget(QLabel(label, self), row, 0)
            grid.addWidget(editor, row, 1)
            grid.addWidget(change, row, 2)
            editor.setEnabled(False)
            change.toggled.connect(editor.setEnabled)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Tick the styles to change; the others keep their hand-set value.", self))
        root.addLayout(grid)
        root.addWidget(self.revert_check)
        root.addWidget(buttons)
        self._prefill(current)

    def _prefill(self, current: LayoutFields) -> None:
        """Show one region's hand lettering as the starting values (unticked: nothing changes unless asked)."""
        font = current.get("font")
        if isinstance(font, str) and self.font_combo.findText(font) >= 0:
            self.font_combo.setCurrentText(font)
        if isinstance(current.get("size_px"), int):
            self.size_spin.setValue(int(current["size_px"]))  # type: ignore[arg-type]
        else:
            self.size_spin.setValue(24)
        if (color := _rgb(current.get("color"))) is not None:
            self.color_button.set_color(color)
        if isinstance(current.get("stroke_px"), int):
            self.stroke_spin.setValue(int(current["stroke_px"]))  # type: ignore[arg-type]
        if (stroke := _rgb(current.get("stroke_color"))) is not None:
            self.stroke_color_button.set_color(stroke)
        align = current.get("align")
        if isinstance(align, str) and align in ALIGNMENTS:
            self.align_combo.setCurrentText(align)
        if isinstance(current.get("angle"), int | float):
            self.angle_spin.setValue(float(current["angle"]))  # type: ignore[arg-type]
        self.hidden_check.setChecked(bool(current.get("hidden", False)))

    def set_change(self, key: str, change: bool = True) -> None:
        """Tick (or untick) a style's *Change* box."""
        self._rows[key][0].setChecked(change)

    def reverts(self) -> bool:
        """Whether the regions go back to the typesetter entirely."""
        return self.revert_check.isChecked()

    def fields(self) -> LayoutFields:
        """The ticked styles as LayoutEdit fields (a font set to the typesetter's choice comes back as None)."""
        values: dict[str, object] = {
            "font": None if self.font_combo.currentText() == TYPESETTER else self.font_combo.currentText(),
            "size_px": self.size_spin.value(),
            "color": list(self.color_button.color()),
            "stroke_px": self.stroke_spin.value(),
            "stroke_color": list(self.stroke_color_button.color()),
            "align": self.align_combo.currentText(),
            "angle": self.angle_spin.value(),
            "hidden": self.hidden_check.isChecked(),
        }
        return {key: values[key] for key, (change, _editor) in self._rows.items() if change.isChecked()}
