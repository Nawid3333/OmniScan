"""Appearance: OLED-black (default) or light theme, a user-chosen accent colour, and quick/standard/pro modes.

The choice lives in the GUI's QSettings (`appearance/*`), not in the pipeline config: it is a per-user preference
of the desktop app. `apply_theme` restyles the running QApplication at once, so the Settings page previews live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

ThemeName = Literal["oled", "light"]
UiMode = Literal["quick", "standard", "pro"]
THEMES: tuple[ThemeName, ...] = ("oled", "light")
UI_MODES: tuple[UiMode, ...] = ("quick", "standard", "pro")
DEFAULT_ACCENT = "#7c5cff"

# sidebar pages each mode shows; "pro" shows everything (the Studio's advanced panels key off it too)
MODE_PAGES: dict[UiMode, frozenset[str]] = {
    "quick": frozenset({"Library", "Reader", "Run", "Settings", "Import"}),
    "standard": frozenset({"Library", "Reader", "Run", "Models", "Settings", "Import"}),
    "pro": frozenset({"Library", "Reader", "Run", "Models", "Settings", "Import"}),
}

# (window, base, alternate base, text, muted text, button) per theme; OLED uses true black for the large areas
_COLORS: dict[ThemeName, tuple[str, str, str, str, str, str]] = {
    "oled": ("#000000", "#000000", "#0d0d0d", "#e8e8e8", "#8a8a8a", "#141414"),
    "light": ("#f5f5f7", "#ffffff", "#f0f0f3", "#1c1c1e", "#6e6e73", "#e9e9ee"),
}


@dataclass(frozen=True, slots=True)
class Appearance:
    """The desktop app's look: theme, accent colour (#rrggbb) and how much of the app is shown."""

    theme: ThemeName = "oled"
    accent: str = DEFAULT_ACCENT
    mode: UiMode = "standard"


def load_appearance(qsettings: QSettings) -> Appearance:
    """The saved appearance; anything missing or invalid falls back to the default."""
    theme = str(qsettings.value("appearance/theme", "oled"))
    accent = str(qsettings.value("appearance/accent", DEFAULT_ACCENT))
    mode = str(qsettings.value("appearance/mode", "standard"))
    return Appearance(
        theme=cast(ThemeName, theme) if theme in THEMES else "oled",
        accent=accent if QColor.isValidColorName(accent) else DEFAULT_ACCENT,
        mode=cast(UiMode, mode) if mode in UI_MODES else "standard",
    )


def save_appearance(qsettings: QSettings, appearance: Appearance) -> None:
    """Persist the appearance."""
    qsettings.setValue("appearance/theme", appearance.theme)
    qsettings.setValue("appearance/accent", appearance.accent)
    qsettings.setValue("appearance/mode", appearance.mode)


def build_palette(appearance: Appearance) -> QPalette:
    """The QPalette for a theme with its accent as the highlight/link colour."""
    window, base, alternate, text, muted, button = (QColor(c) for c in _COLORS[appearance.theme])
    accent = QColor(appearance.accent)
    on_accent = QColor("#000000" if accent.lightnessF() > 0.6 else "#ffffff")
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, window),
        (QPalette.ColorRole.Base, base),
        (QPalette.ColorRole.AlternateBase, alternate),
        (QPalette.ColorRole.WindowText, text),
        (QPalette.ColorRole.Text, text),
        (QPalette.ColorRole.ButtonText, text),
        (QPalette.ColorRole.Button, button),
        (QPalette.ColorRole.ToolTipBase, button),
        (QPalette.ColorRole.ToolTipText, text),
        (QPalette.ColorRole.PlaceholderText, muted),
        (QPalette.ColorRole.Highlight, accent),
        (QPalette.ColorRole.HighlightedText, on_accent),
        (QPalette.ColorRole.Link, accent),
        (QPalette.ColorRole.Accent, accent),
    ):
        palette.setColor(role, color)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, muted)
    return palette


def apply_theme(app: QApplication, appearance: Appearance) -> None:
    """Restyle the whole application (Fusion style, so the palette looks the same on every OS)."""
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setPalette(build_palette(appearance))
    app.setStyleSheet(
        f"QProgressBar::chunk {{ background: {appearance.accent}; }}"
        f" QListWidget::item:selected {{ background: {appearance.accent}; }}"
    )
