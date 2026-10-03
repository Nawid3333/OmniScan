"""Appearance: OLED-black (default) or light theme, a user-chosen accent colour, and quick/standard/pro modes.

The choice lives in the GUI's QSettings (`appearance/*`), not in the pipeline config: it is a per-user preference
of the desktop app. `apply_theme` restyles the running QApplication at once, so the Settings page previews live.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

ThemeName = Literal["oled", "light"]
UiMode = Literal["quick", "standard", "pro"]
THEMES: tuple[ThemeName, ...] = ("oled", "light")
UI_MODES: tuple[UiMode, ...] = ("quick", "standard", "pro")
DEFAULT_ACCENT = "#7c5cff"

# sidebar pages each mode shows; "pro" shows everything (the Studio's advanced panels key off it too)
MODE_PAGES: dict[UiMode, frozenset[str]] = {
    "quick": frozenset({"Library", "Reader", "Run", "Settings", "Import"}),
    "standard": frozenset(
        {"Library", "Reader", "Run", "Models", "Settings", "Import", "Studio", "Queue", "Glossary"}
    ),
    "pro": frozenset(
        {"Library", "Reader", "Run", "Models", "Settings", "Import", "Studio", "Queue", "Glossary"}
    ),
}


@dataclass(frozen=True, slots=True)
class Tokens:
    """The colours one theme is drawn with; the stylesheet and the palette both come from these."""

    window: str  # page background (true black on OLED)
    surface: str  # cards, inputs, table bodies
    raised: str  # buttons, hovered rows, the sidebar's selected item background
    border: str
    text: str
    muted: str  # secondary text, placeholders, help lines
    danger: str  # inline errors


TOKENS: dict[ThemeName, Tokens] = {
    "oled": Tokens("#000000", "#000000", "#141418", "#26262c", "#ececf1", "#8e8e98", "#ff6b6b"),
    "light": Tokens("#f6f6f8", "#ffffff", "#ececf1", "#d9d9e0", "#1c1c1e", "#6e6e78", "#c62828"),
}
RADIUS_PX = 8
# translucent state tints for table cells: readable on OLED black and on the light theme alike
TINTS: dict[str, QColor] = {
    "positive": QColor(46, 160, 67, 90),
    "attention": QColor(210, 153, 34, 100),
    "warning": QColor(219, 109, 40, 100),
    "negative": QColor(218, 54, 51, 100),
}
ICONS_DIR = Path(__file__).with_name("icons")  # the stylesheet's chevron and check marks (SVG)


def _icon_url(name: str) -> str:
    """A stylesheet url() for one of the bundled SVG icons (forward slashes work on Windows too)."""
    return f"url({(ICONS_DIR / f'{name}.svg').as_posix()})"


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
    tokens = TOKENS[appearance.theme]
    window, base, alternate, text, muted, button = (
        QColor(c)
        for c in (tokens.window, tokens.surface, tokens.raised, tokens.text, tokens.muted, tokens.raised)
    )
    accent = QColor(appearance.accent)
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
        (QPalette.ColorRole.HighlightedText, QColor(on_accent(appearance.accent))),
        (QPalette.ColorRole.Link, accent),
        (QPalette.ColorRole.Accent, accent),
        (QPalette.ColorRole.Mid, QColor(tokens.border)),
    ):
        palette.setColor(role, color)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, muted)
    return palette


def on_accent(accent: str) -> str:
    """Black or white, whichever reads better on the accent colour."""
    return "#000000" if QColor(accent).lightnessF() > 0.6 else "#ffffff"


def stylesheet(appearance: Appearance) -> str:
    """The application stylesheet: flat surfaces, rounded controls, accent for focus, selection and primary actions.

    Widgets opt into roles with a dynamic property: `role="primary"` (the page's main button), `role="error"` and
    `role="muted"` (labels), `role="title"` (a page heading), `role="section"` (a group heading in a form).
    """
    t = TOKENS[appearance.theme]
    accent = appearance.accent
    hover = QColor(accent).lighter(115).name()
    pressed = QColor(accent).darker(115).name()
    chevron = _icon_url("chevron-light" if appearance.theme == "oled" else "chevron-dark")
    check = _icon_url("check-black" if on_accent(accent) == "#000000" else "check-white")
    return f"""
* {{ outline: 0; }}
QWidget {{ font-size: 10pt; }}
QMainWindow, QDialog {{ background: {t.window}; }}
QToolTip {{ background: {t.raised}; color: {t.text}; border: 1px solid {t.border}; padding: 4px 6px; }}

QLabel[role="title"] {{ font-size: 17pt; font-weight: 600; padding: 2px 0 6px 0; }}
QLabel[role="section"] {{ font-size: 11pt; font-weight: 600; padding-top: 12px; }}
QLabel[role="muted"] {{ color: {t.muted}; }}
QLabel[role="error"] {{ color: {t.danger}; }}
QLabel#brand {{ font-size: 13pt; font-weight: 700; padding: 14px 12px 10px 14px; }}

QPushButton {{
    background: {t.raised}; color: {t.text}; border: 1px solid {t.border};
    border-radius: {RADIUS_PX}px; padding: 6px 14px; min-height: 18px;
}}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {t.surface}; }}
QPushButton:disabled {{ color: {t.muted}; border-color: {t.raised}; }}
QPushButton[role="primary"] {{ background: {accent}; color: {on_accent(accent)}; border: 1px solid {accent}; }}
QPushButton[role="primary"]:hover {{ background: {hover}; border-color: {hover}; }}
QPushButton[role="primary"]:pressed {{ background: {pressed}; }}
QPushButton[role="primary"]:disabled {{ background: {t.raised}; color: {t.muted}; border-color: {t.raised}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
    border-radius: {RADIUS_PX}px; padding: 5px 8px; selection-background-color: {accent};
    selection-color: {on_accent(accent)};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus,
QTextEdit:focus {{ border-color: {accent}; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ border: 0; width: 24px; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: {chevron}; width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background: {t.surface}; border: 1px solid {t.border}; selection-background-color: {accent};
    selection-color: {on_accent(accent)}; padding: 4px;
}}
QLineEdit[role="search"] {{ padding: 7px 12px; border-radius: 16px; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px; border: 1px solid {t.border}; background: {t.surface};
}}
QCheckBox::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; image: {check}; }}
QCheckBox::indicator:disabled {{ background: {t.raised}; border-color: {t.raised}; }}

QWidget#nav {{ background: {t.window}; border-right: 1px solid {t.border}; }}
QListWidget#sidebar {{ background: transparent; border: 0; border-radius: 0; padding: 4px 8px; }}
QListWidget#sidebar::item {{ padding: 9px 12px; margin: 1px 0; border-radius: {RADIUS_PX}px; color: {t.muted}; }}
QListWidget#sidebar::item:hover {{ background: {t.surface}; color: {t.text}; }}
QListWidget#sidebar::item:selected {{ background: {t.raised}; color: {t.text}; border-left: 3px solid {accent}; }}

QTabWidget::pane {{ border: 0; border-top: 1px solid {t.border}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {t.muted}; padding: 8px 14px; margin-right: 4px;
    border: 0; border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {t.text}; }}
QTabBar::tab:selected {{ color: {t.text}; border-bottom: 2px solid {accent}; }}

QTableWidget, QTableView, QTreeWidget, QTreeView, QListWidget, QListView {{
    background: {t.surface}; alternate-background-color: {t.raised}; border: 1px solid {t.border};
    border-radius: {RADIUS_PX}px; gridline-color: {t.border};
    selection-background-color: {accent}; selection-color: {on_accent(accent)};
}}
QTableView {{ gridline-color: transparent; }}
QTableView::item, QTreeView::item {{ padding: 4px 6px; }}
QListWidget::item, QListView::item {{ padding: 6px 8px; border-radius: 6px; color: {t.text}; }}
QListWidget::item:hover, QListView::item:hover {{ background: {t.raised}; }}
QListWidget::item:selected, QListView::item:selected {{ background: {accent}; color: {on_accent(accent)}; }}
QHeaderView::section {{
    background: {t.surface}; color: {t.muted}; border: 0; border-bottom: 1px solid {t.border};
    padding: 6px 8px; font-weight: 600;
}}

QGroupBox {{ border: 1px solid {t.border}; border-radius: {RADIUS_PX}px; margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t.muted}; }}

QProgressBar {{
    background: {t.raised}; border: 0; border-radius: 4px; height: 8px; text-align: center; color: {t.text};
}}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}

QScrollArea {{ border: 0; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {t.border}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {t.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {t.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QStatusBar {{ background: {t.window}; color: {t.muted}; border-top: 1px solid {t.border}; }}
QStatusBar QLabel {{ color: {t.muted}; padding: 0 6px; }}
QMenu {{ background: {t.surface}; border: 1px solid {t.border}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {accent}; color: {on_accent(accent)}; }}
"""


def set_role(widget: QWidget, role: str) -> None:
    """Give a widget a stylesheet role (see `stylesheet`) and restyle it at once."""
    widget.setProperty("role", role)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


def apply_theme(app: QApplication, appearance: Appearance) -> None:
    """Restyle the whole application (Fusion style, so the palette looks the same on every OS)."""
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setPalette(build_palette(appearance))
    app.setStyleSheet(stylesheet(appearance))
