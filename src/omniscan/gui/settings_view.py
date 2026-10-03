"""SettingsView: the Settings page — global settings, per-series overrides, translation profiles, hardware.

Global rows come from `gui.services.settings.GLOBAL_FIELDS`; every commit goes through the
validated service writers (`set_global` / `clear_global` / `set_series_override` /
`set_profile_enabled`) and a refusal shows inline while the editor reverts. The Profiles tab also adds
your own translation model — an Ollama model, or OpenAI / Claude / an OpenAI-compatible server with your
API key, which goes into secrets.env (`add_profile`, `save_api_key`). Hardware detection runs on a pool
thread (it imports torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from omniscan.core.config import SERIES_SECTIONS, Config, SettingError
from omniscan.gui.models_view import hardware_header
from omniscan.gui.services import library, settings
from omniscan.gui.services.hardware import HardwareReport, HardwareService
from omniscan.gui.services.settings import GLOBAL_FIELDS, SECTION_TITLES, SettingField
from omniscan.gui.theme import (
    THEMES,
    UI_MODES,
    Appearance,
    apply_theme,
    load_appearance,
    save_appearance,
    set_role,
)
from omniscan.gui.workers import run_task
from omniscan.hw.tune import Plan, describe

# words that find a whole tab in the search (its own rows are matched one by one)
_TAB_KEYWORDS = {
    "series": "per-series override series.toml section key value",
    "profiles": "translation profiles model endpoint ollama cloud llm enable openai chatgpt claude anthropic api key",
    "hardware": "hardware gpu graphics card vram memory detect fit models",
}


@dataclass(slots=True)
class _SearchRow:
    """One searchable settings row: its widgets, the text it matches and the heading above it."""

    page: str
    widgets: list[QWidget]
    haystack: str
    header: QLabel | None = None


_PAGE_TITLES = {
    "global": "Global",
    "series": "Per-series",
    "profiles": "Translation",
    "hardware": "Hardware",
    "appearance": "Appearance",
}


class SettingsView(QWidget):
    """Tabbed settings editor; every successful write emits `settings_changed` once."""

    settings_changed = Signal()
    welcome_requested = Signal()  # the user wants the first-run checklist again (Hardware tab button)
    appearance_changed = Signal(object)  # the new Appearance, after it was saved and applied

    def __init__(
        self,
        cfg: Config,
        *,
        hardware: HardwareService | None = None,
        config_path: Path | None = None,
        profiles_path: Path | None = None,
        secrets_path: Path | None = None,
        qsettings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the page; services and write paths are injectable (tests never touch user files)."""
        super().__init__(parent)
        self._cfg = cfg
        self._hardware_service = hardware or HardwareService(cfg)
        self._config_path = config_path
        self._profiles_path = profiles_path
        self._secrets_path = secrets_path  # where a typed API key goes (None: ~/.config/omniscan/secrets.env)
        self._last_good: dict[tuple[str, str], Any] = {}
        self._fields: dict[tuple[str, str], SettingField] = {}
        self._editors: dict[tuple[str, str], QWidget] = {}
        self._pages: dict[str, QWidget] = {}
        self._hardware_loaded = False
        self._plan: Plan | None = None
        self._search_rows: list[_SearchRow] = []
        self._qsettings = qsettings or QSettings("OmniScan", "gui")
        self.appearance = load_appearance(self._qsettings)

        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search settings")
        self.search_edit.setClearButtonEnabled(True)
        set_role(self.search_edit, "search")
        self.search_empty = QLabel("No setting matches your search.", self)
        set_role(self.search_empty, "muted")
        self.search_empty.hide()

        self.tabs = QTabWidget(self)
        for name, title in _PAGE_TITLES.items():
            page = self._build(name)
            self._pages[name] = page
            self.tabs.addTab(page, title)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        root = QVBoxLayout(self)
        root.addWidget(self.search_edit)
        root.addWidget(self.search_empty)
        root.addWidget(self.tabs)
        self.search_edit.textChanged.connect(self.filter_settings)

    # ------------------------------------------------------------------ search

    def filter_settings(self, text: str) -> int:
        """Show only the settings matching `text` (every word, any order); returns how many rows/tabs match."""
        words = text.lower().split()
        matched_pages: set[str] = set()
        count = 0
        headers: dict[int, tuple[QLabel, bool]] = {}
        for row in self._search_rows:
            hit = all(word in row.haystack for word in words)
            for widget in row.widgets:
                widget.setVisible(hit)
            if row.header is not None:
                seen = headers.get(id(row.header), (row.header, False))[1]
                headers[id(row.header)] = (row.header, seen or hit)
            if hit:
                matched_pages.add(row.page)
                count += 1
        for header, visible in headers.values():
            header.setVisible(visible)
        for name, title in _PAGE_TITLES.items():
            tab_text = f"{title} {_TAB_KEYWORDS.get(name, '')}".lower()
            if name not in matched_pages and words and all(word in tab_text for word in words):
                matched_pages.add(name)
                count += 1
        for name, page in self._pages.items():
            self.tabs.setTabVisible(self.tabs.indexOf(page), not words or name in matched_pages)
        current = next(
            (name for name, page in self._pages.items() if page is self.tabs.currentWidget()), None
        )
        if words and current not in matched_pages and matched_pages:
            first = next(name for name in _PAGE_TITLES if name in matched_pages)
            self.tabs.setCurrentWidget(self._pages[first])
        self.search_empty.setVisible(bool(words) and not matched_pages)
        self.tabs.setVisible(not words or bool(matched_pages))
        return count

    # ------------------------------------------------------------------ reconfigure

    def reconfigure(self, cfg: Config) -> None:
        """Rebuild the tabs from a reloaded config (never emits `settings_changed`)."""
        self._cfg = cfg
        self._rebuild("global")
        self._rebuild("series")
        self._rebuild("profiles")
        self._hardware_loaded = False
        self._plan = None
        self._on_tab_changed(self.tabs.currentIndex())  # re-detect at once if Hardware is open

    def _rebuild(self, name: str) -> None:
        """Replace one tab's page in place."""
        old = self._pages[name]
        index = self.tabs.indexOf(old)
        self.tabs.removeTab(index)
        old.deleteLater()
        self._search_rows = [row for row in self._search_rows if row.page != name]
        page = self._build(name)
        self._pages[name] = page
        self.tabs.insertTab(index, page, _PAGE_TITLES[name])
        self.filter_settings(self.search_edit.text())

    def _build(self, name: str) -> QWidget:
        """Build one tab's page."""
        if name == "global":
            return self._build_global()
        if name == "series":
            return self._build_series()
        if name == "profiles":
            return self._build_profiles()
        if name == "appearance":
            return self._build_appearance()
        return self._build_hardware()

    # ------------------------------------------------------------------ global tab

    def _build_global(self) -> QWidget:
        """The `GLOBAL_FIELDS` grouped by section: plain name, editor, Reset and a help line per setting."""
        page = QWidget()
        grid = QGridLayout(page)
        grid.setHorizontalSpacing(12)
        dump = self._cfg.model_dump(mode="json")
        row = 0
        header: QLabel | None = None
        section = ""
        for field in GLOBAL_FIELDS:
            if field.section != section:
                section = field.section
                header = QLabel(SECTION_TITLES.get(section, section), page)
                set_role(header, "section")
                grid.addWidget(header, row, 0, 1, 3)
                row += 1
            value = dump[field.section][field.key]
            self._last_good[(field.section, field.key)] = value
            self._fields[(field.section, field.key)] = field
            name = QLabel(field.label or f"{field.section}.{field.key}", page)
            name.setToolTip(f"{field.section}.{field.key}")
            editor = self._editor(field, value)
            reset = QPushButton("Reset", page)
            reset.setToolTip("Remove the user override and apply the built-in default")
            reset.clicked.connect(lambda _checked=False, f=field: self._reset_global(f))
            grid.addWidget(name, row, 0)
            grid.addWidget(editor, row, 1)
            grid.addWidget(reset, row, 2)
            widgets: list[QWidget] = [name, editor, reset]
            if field.help:
                help_label = QLabel(field.help, page)
                help_label.setWordWrap(True)
                set_role(help_label, "muted")
                grid.addWidget(help_label, row + 1, 1, 1, 2)
                widgets.append(help_label)
                row += 1
            row += 1
            haystack = " ".join(
                (
                    field.label,
                    field.help,
                    field.section,
                    field.key,
                    SECTION_TITLES.get(field.section, ""),
                    *field.choices,
                )
            )
            self._search_rows.append(_SearchRow("global", widgets, haystack.lower(), header))
        self.global_error = QLabel("", page)
        set_role(self.global_error, "error")
        self.global_error.setWordWrap(True)
        grid.addWidget(self.global_error, row, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(row + 1, 1)
        return _in_scroll(page)

    def _editor(self, field: SettingField, value: Any) -> QWidget:
        """The editor widget for one field (a path row wraps editor + Browse in one row)."""
        key = (field.section, field.key)
        if field.kind == "path":
            line = QLineEdit("" if value is None else str(value))
            line.editingFinished.connect(lambda f=field, e=line: self._commit_global(f, e.text()))
            browse = QPushButton("Browse...")
            browse.clicked.connect(lambda _checked=False, f=field: self._browse_for(f))
            self._editors[key] = line
            return _row_widget(line, browse, stretch_first=True)
        if field.kind == "bool":
            editor: QWidget = QCheckBox()
            editor.setChecked(bool(value))
            editor.toggled.connect(lambda checked, f=field: self._commit_global(f, checked))
        elif field.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(field.low), int(field.high))
            editor.setValue(int(value))
            editor.editingFinished.connect(lambda f=field, e=editor: self._commit_global(f, e.value()))
        elif field.kind == "float":
            editor = QDoubleSpinBox()
            editor.setRange(field.low, field.high)
            editor.setDecimals(field.decimals)
            editor.setValue(float(value))
            editor.editingFinished.connect(lambda f=field, e=editor: self._commit_global(f, e.value()))
        elif field.kind == "model":
            editor = QComboBox()
            editor.addItem("(default)")
            editor.addItems(settings.model_ids())
            editor.setToolTip("empty means the built-in default")
            editor.activated.connect(
                lambda index, f=field, e=editor: self._commit_global(f, "" if index == 0 else e.currentText())
            )
        else:  # choice
            editor = QComboBox()
            editor.addItems(field.choices)
            editor.setEditable(field.editable)
            if field.editable:
                line = editor.lineEdit()
                if line is not None:
                    line.editingFinished.connect(
                        lambda f=field, e=editor: self._commit_global(f, e.currentText())
                    )
            editor.activated.connect(
                lambda _index, f=field, e=editor: self._commit_global(f, e.currentText())
            )
        if isinstance(editor, QComboBox):
            if field.kind == "model" and value is None:
                editor.setCurrentIndex(0)
            else:
                editor.setCurrentText("" if value is None else str(value))
        self._editors[key] = editor
        return editor

    def _browse_for(self, field: SettingField) -> None:
        """Fill the path editor from a directory dialog and commit it."""
        editor = self._editors[(field.section, field.key)]
        assert isinstance(editor, QLineEdit)  # path rows are line edits
        chosen = QFileDialog.getExistingDirectory(self, f"{field.section}.{field.key}")
        if not chosen:
            return
        editor.setText(chosen)
        self._commit_global(field, chosen)

    def _commit_global(self, field: SettingField, value: Any) -> None:
        """Write one global setting; on refusal show it inline and revert the editor."""
        key = (field.section, field.key)
        if value == self._last_good[key]:
            return
        try:
            if field.kind == "model" and value == "":
                settings.clear_global(field.section, field.key, path=self._config_path)
            else:
                settings.set_global(field.section, field.key, value, path=self._config_path)
        except SettingError as error:
            self.global_error.setText(f"{field.section}.{field.key}: {error}")
            self._set_editor_value(field, self._last_good[key])
            return
        self.global_error.setText("")
        self._last_good[key] = value
        self.settings_changed.emit()

    def _reset_global(self, field: SettingField) -> None:
        """Drop the user override so the built-in default applies again."""
        changed = settings.clear_global(field.section, field.key, path=self._config_path)
        value = Config().model_dump(mode="json")[field.section][field.key]
        self._last_good[(field.section, field.key)] = value
        self._set_editor_value(field, value)
        self.global_error.setText("")
        if changed:
            self.settings_changed.emit()

    def _set_editor_value(self, field: SettingField, value: Any) -> None:
        """Put `value` back into the editor without triggering a commit."""
        editor = self._editors[(field.section, field.key)]
        if isinstance(editor, QLineEdit):
            editor.setText("" if value is None else str(value))
        elif isinstance(editor, QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(value))
        elif isinstance(editor, QSpinBox):
            editor.setValue(int(value))
        elif isinstance(editor, QComboBox):
            if field.kind == "model" and value is None:
                editor.setCurrentIndex(0)
            else:
                editor.setCurrentText("" if value is None else str(value))

    # ------------------------------------------------------------------ per-series tab

    def _build_series(self) -> QWidget:
        """The series override editor: existing overrides plus a Set form."""
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QHBoxLayout()
        top.addWidget(QLabel("Series", page))
        self.series_combo = QComboBox(page)
        self.series_combo.addItems(library.list_series(self._cfg))
        top.addWidget(self.series_combo, 1)
        self.series_reload_button = QPushButton("Refresh", page)
        top.addWidget(self.series_reload_button)
        layout.addLayout(top)

        self.series_error = QLabel("", page)
        set_role(self.series_error, "error")
        self.series_error.setWordWrap(True)

        self.override_table = QTableWidget(0, 4, page)
        self.override_table.setHorizontalHeaderLabels(("Section", "Key", "Value", ""))
        self.override_table.verticalHeader().setVisible(False)
        header = self.override_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        add = QHBoxLayout()
        self.override_section_combo = QComboBox(page)
        self.override_section_combo.addItems(SERIES_SECTIONS)
        self.override_key_combo = QComboBox(page)
        self.override_key_combo.addItems(settings.section_keys(self.override_section_combo.currentText()))
        self.override_value_edit = QLineEdit(page)
        self.override_value_edit.setPlaceholderText("value (bool/int/float/list via JSON, else text)")
        self.override_set_button = QPushButton("Set", page)
        add.addWidget(self.override_section_combo)
        add.addWidget(self.override_key_combo, 1)
        add.addWidget(self.override_value_edit, 2)
        add.addWidget(self.override_set_button)
        layout.addLayout(add)

        layout.addWidget(self.override_table, 1)
        layout.addWidget(self.series_error)

        self.series_combo.currentTextChanged.connect(self._reload_overrides)
        self.series_reload_button.clicked.connect(self._reload_overrides)
        self.override_section_combo.currentTextChanged.connect(self._reload_override_keys)
        self.override_set_button.clicked.connect(self._on_override_set)

        self._reload_overrides()
        return page

    def _series_dir(self) -> Any:
        """The current series' library directory (its series.toml lives there)."""
        return self._cfg.paths.library_root / self.series_combo.currentText()

    def _reload_override_keys(self, section: str) -> None:
        """Refill the key combo from the section's config keys."""
        self.override_key_combo.clear()
        self.override_key_combo.addItems(settings.section_keys(section))

    def _reload_overrides(self) -> None:
        """Fill the overrides table for the chosen series."""
        self.override_table.setRowCount(0)
        if not self.series_combo.currentText():
            return
        try:
            overrides = settings.series_overrides(self._series_dir())
        except OSError as error:
            self.series_error.setText(str(error))
            return
        for (section, key), value in _flatten(overrides):
            row = self.override_table.rowCount()
            self.override_table.insertRow(row)
            self.override_table.setItem(row, 0, QTableWidgetItem(section))
            self.override_table.setItem(row, 1, QTableWidgetItem(key))
            self.override_table.setItem(row, 2, QTableWidgetItem(_display(value)))
            remove = QPushButton("Remove", self.override_table)
            remove.clicked.connect(lambda _checked=False, s=section, k=key: self._on_override_remove(s, k))
            self.override_table.setCellWidget(row, 3, remove)

    def _on_override_set(self) -> None:
        """Write one per-series override (validated); inline error on refusal."""
        section = self.override_section_combo.currentText()
        key = self.override_key_combo.currentText()
        if not self.series_combo.currentText() or not section or not key:
            return
        try:
            value = settings.parse_value(self.override_value_edit.text())
            settings.set_series_override(self._series_dir(), section, key, value)
        except (ValueError, SettingError) as error:
            self.series_error.setText(f"{section}.{key}: {error}")
            return
        self.series_error.setText("")
        self.override_value_edit.setText("")
        self._reload_overrides()
        self.settings_changed.emit()

    def _on_override_remove(self, section: str, key: str) -> None:
        """Drop one override; the machine value applies again."""
        try:
            changed = settings.remove_series_override(self._series_dir(), section, key)
        except OSError as error:
            self.series_error.setText(str(error))
            return
        self.series_error.setText("")
        if changed:
            self._reload_overrides()
            self.settings_changed.emit()

    # ------------------------------------------------------------------ translation tab

    def _build_profiles(self) -> QWidget:
        """The translation profiles with per-profile `enabled` toggles, and a form to add your own model."""
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Profiles come from config/translation_profiles.toml (repo) and the user's\n"
            "translation_profiles.toml (a same-named user entry overrides; toggling writes that file).\n"
            "Enabled profiles are the ones the translator may use.",
            page,
        )
        layout.addWidget(note)
        self.profiles_error = QLabel("", page)
        set_role(self.profiles_error, "error")
        self.profiles_error.setWordWrap(True)
        layout.addWidget(self.profiles_error)
        self._profile_rows: list[QWidget] = []
        self._profile_boxes: list[QCheckBox] = []
        self._profiles_page, self._profiles_layout = page, layout
        layout.addWidget(self._build_add_model(page))
        layout.addStretch(1)
        self._reload_profiles(page, layout)
        return page

    def _build_add_model(self, page: QWidget) -> QWidget:
        """The "Use your own model" form: provider, model, server, key, profile name."""
        box = QWidget(page)
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 12, 0, 0)
        title = QLabel("Use your own model", box)
        set_role(title, "section")
        self.provider_combo = QComboBox(box)
        for label, *_rest in settings.PROVIDERS:
            self.provider_combo.addItem(label)
        self.model_combo = QComboBox(box)
        self.model_combo.setEditable(True)
        self.model_combo.setToolTip(
            "The model's name as the provider writes it (gpt-5.4-mini, claude-sonnet-5-5, gemma4:12b)"
        )
        self.list_models_button = QPushButton("List Ollama models", box)
        self.base_url_edit = QLineEdit(box)
        self.base_url_edit.setPlaceholderText("https://openrouter.ai/api/v1")
        self.key_env_edit = QLineEdit(box)
        self.key_env_edit.setPlaceholderText("OPENROUTER_API_KEY")
        self.key_edit = QLineEdit(box)
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("paste your API key (saved in secrets.env, never shown again)")
        self.key_state = QLabel("", box)
        self.profile_name_edit = QLineEdit(box)
        self.only_check = QCheckBox("Use only this model (switch the other profiles off)", box)
        self.only_check.setChecked(True)
        self.add_profile_button = QPushButton("Add model", box)
        set_role(self.add_profile_button, "primary")
        self.add_model_status = QLabel("", box)
        self.add_model_status.setWordWrap(True)
        rows: list[tuple[str, QWidget]] = [
            ("Where it runs", self.provider_combo),
            ("Model", _row_widget(self.model_combo, self.list_models_button, stretch_first=True)),
            ("Server address", self.base_url_edit),
            ("Key variable", self.key_env_edit),
            ("API key", _row_widget(self.key_edit, self.key_state, stretch_first=True)),
            ("Profile name", self.profile_name_edit),
        ]
        grid.addWidget(title, 0, 0, 1, 2)
        self._provider_rows: dict[str, tuple[QLabel, QWidget]] = {}
        for row, (label, widget) in enumerate(rows, start=1):
            caption = QLabel(label, box)
            grid.addWidget(caption, row, 0)
            grid.addWidget(widget, row, 1)
            self._provider_rows[label] = (caption, widget)
        grid.addWidget(self.only_check, len(rows) + 1, 1)
        grid.addWidget(
            _row_widget(self.add_profile_button, self.add_model_status, stretch_first=False), len(rows) + 2, 1
        )
        grid.setColumnStretch(1, 1)
        self.provider_combo.currentIndexChanged.connect(lambda _index: self._on_provider())
        self.model_combo.currentTextChanged.connect(lambda _text: self._suggest_profile_name())
        self.key_env_edit.textChanged.connect(lambda _text: self._show_key_state())
        self.list_models_button.clicked.connect(self.list_ollama_models)
        self.add_profile_button.clicked.connect(self.add_model)
        self._on_provider()
        return box

    def _provider(self) -> tuple[str, str | None, str | None]:
        """(endpoint, key variable or None, base URL or None) of the picked provider and the typed fields."""
        _label, endpoint, key_env, base_url = settings.PROVIDERS[self.provider_combo.currentIndex()]
        if key_env == "":  # an OpenAI-compatible server: its address and key variable are typed
            key_env = self.key_env_edit.text().strip() or None
        if base_url == "":
            base_url = self.base_url_edit.text().strip() or None
        return endpoint, key_env, base_url

    def _on_provider(self) -> None:
        """Show the fields the picked provider needs."""
        _label, endpoint, key_env, base_url = settings.PROVIDERS[self.provider_combo.currentIndex()]
        api = endpoint != "local"
        for label, visible in (
            ("Server address", base_url == ""),
            ("Key variable", key_env == ""),
            ("API key", api),
        ):
            caption, widget = self._provider_rows[label]
            caption.setVisible(visible)
            widget.setVisible(visible)
        self.list_models_button.setVisible(not api)
        self._suggest_profile_name()
        self._show_key_state()

    def _suggest_profile_name(self) -> None:
        """Name the new profile after its model (until the user types a name of their own)."""
        endpoint, _key_env, _base_url = self._provider()
        self.profile_name_edit.setText(
            settings.suggest_profile_name(endpoint, self.model_combo.currentText())
        )

    def _show_key_state(self) -> None:
        """Say whether the provider's key is already set (never its value)."""
        _endpoint, key_env, _base_url = self._provider()
        if key_env is None:
            self.key_state.setText("")
            return
        self.key_state.setText(
            f"{key_env} is set"
            if settings.has_api_key(key_env, env_file=self._secrets_path)
            else f"{key_env} not set yet"
        )

    def list_ollama_models(self) -> list[str]:
        """Fill the model list with what the local Ollama has; the error inline when it does not answer."""
        try:
            names = settings.ollama_models(self._cfg)
        except SettingError as error:
            self.add_model_status.setText(str(error))
            return []
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(names)
        if current:
            self.model_combo.setCurrentText(current)
        self.add_model_status.setText(f"{len(names)} model(s) installed")
        return names

    def add_model(self) -> bool:
        """Save the typed API key (if any) and add the profile; False with the reason inline when refused."""
        endpoint, key_env, base_url = self._provider()
        _label, _endpoint, default_env, _base = settings.PROVIDERS[self.provider_combo.currentIndex()]
        name = self.profile_name_edit.text().strip()
        try:
            if endpoint != "local" and key_env is None:
                raise SettingError("name the variable that holds this server's key (e.g. OPENROUTER_API_KEY)")
            if self.key_edit.text().strip() and key_env is not None:
                settings.save_api_key(key_env, self.key_edit.text(), env_file=self._secrets_path)
            settings.add_profile(
                name,
                endpoint=endpoint,
                model=self.model_combo.currentText(),
                base_url=base_url,
                api_key_env=key_env if key_env != default_env else None,
                only=self.only_check.isChecked(),
                user_path=self._profiles_path,
            )
        except SettingError as error:
            self.add_model_status.setText(str(error))
            return False
        self.key_edit.clear()
        self._show_key_state()
        missing = key_env is not None and not settings.has_api_key(key_env, env_file=self._secrets_path)
        self.add_model_status.setText(
            f"added {name}" + (f" — set {key_env} before translating" if missing else "")
        )
        self._reload_profiles(self._profiles_page, self._profiles_layout)
        self.settings_changed.emit()
        return True

    def _reload_profiles(self, page: QWidget, layout: QVBoxLayout) -> None:
        """(Re)fill the profile checkboxes from the service; a profile you added also gets a Remove button."""
        for row in self._profile_rows:
            layout.removeWidget(row)
            row.deleteLater()
        self._profile_rows.clear()
        self._profile_boxes.clear()
        try:
            profiles = settings.translation_profiles(settings.profile_paths(self._profiles_path))
            yours = settings.user_profile_names(self._profiles_path)
        except ValueError as error:  # pydantic/TOML errors: a broken user profile file
            self.profiles_error.setText(f"cannot load translation profiles: {error}")
            return
        self.profiles_error.setText("")
        for index, profile in enumerate(profiles):
            box = QCheckBox(f"{profile.name} — {profile.model} @ {profile.endpoint}", page)
            box.setChecked(profile.enabled)
            box.toggled.connect(
                lambda checked, name=profile.name, b=box: self._commit_profile(name, checked, b)
            )
            self._profile_boxes.append(box)
            widgets: list[QWidget] = [box]
            if profile.name in yours:
                remove = QPushButton("Remove", page)
                remove.clicked.connect(lambda _checked=False, name=profile.name: self.remove_profile(name))
                widgets.append(remove)
            row = _row_widget(*widgets, stretch_first=True)
            self._profile_rows.append(row)
            layout.insertWidget(2 + index, row)  # after the note and the error line

    def remove_profile(self, name: str) -> bool:
        """Remove a profile you added; False with the reason inline when refused."""
        try:
            settings.remove_profile(name, user_path=self._profiles_path)
        except SettingError as error:
            self.profiles_error.setText(str(error))
            return False
        self._reload_profiles(self._profiles_page, self._profiles_layout)
        self.settings_changed.emit()
        return True

    def _commit_profile(self, name: str, enabled: bool, box: QCheckBox) -> None:
        """Enable/disable one profile in the user's translation_profiles.toml."""
        try:
            settings.set_profile_enabled(name, enabled, user_path=self._profiles_path)
        except SettingError as error:
            self.profiles_error.setText(str(error))
            box.blockSignals(True)
            box.setChecked(not enabled)
            box.blockSignals(False)
            return
        self.profiles_error.setText("")
        self.settings_changed.emit()

    # ------------------------------------------------------------------ hardware tab

    def _build_hardware(self) -> QWidget:
        """The machine snapshot (worker thread) plus the per-model warnings.

        Detection is deferred to the first time this tab is shown (it imports torch), so
        opening the Settings page alone never pays for it.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        self.hardware_header_label = QLabel("hardware: not detected yet", page)
        self.hardware_header_label.setWordWrap(True)
        self.hardware_button = QPushButton("Re-detect", page)
        self.plan_label = QLabel("", page)
        self.plan_label.setWordWrap(True)
        self.optimise_button = QPushButton("Optimise for this PC", page)
        self.optimise_button.setToolTip(
            "Write the plan's device, memory budget, OCR engine, batch sizes and LaMa default to the config"
        )
        self.optimise_button.setEnabled(False)
        self.plan_status_label = QLabel("", page)
        set_role(self.plan_status_label, "muted")
        layout.addWidget(self.hardware_header_label)
        buttons = QHBoxLayout()
        self.welcome_button = QPushButton("Setup checklist…", page)
        self.welcome_button.setToolTip("The first-start checklist: hardware plan, models, Ollama")
        buttons.addWidget(self.hardware_button)
        buttons.addWidget(self.optimise_button)
        buttons.addWidget(self.welcome_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addWidget(self.plan_label)
        layout.addWidget(self.plan_status_label)

        self.hardware_table = QTableWidget(0, 4, page)
        self.hardware_table.setHorizontalHeaderLabels(("Model", "Fit", "Device", "Messages"))
        self.hardware_table.verticalHeader().setVisible(False)
        header = self.hardware_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.hardware_table, 1)

        self.hardware_button.clicked.connect(self._load_hardware)
        self.optimise_button.clicked.connect(self.apply_plan)
        self.welcome_button.clicked.connect(self.welcome_requested)
        return page

    def apply_plan(self) -> int:
        """Write the tuning plan's settings to the user config (one `settings_changed`); how many."""
        plan = self._plan
        if plan is None:
            return 0
        try:
            for (section, key), value in plan.overrides.items():
                settings.set_global(section, key, value, path=self._config_path)
        except settings.SettingError as error:
            self.plan_status_label.setText(f"refused: {error}")
            return 0
        self.plan_status_label.setText(f"wrote {len(plan.overrides)} setting(s); the pages reload them")
        self.settings_changed.emit()
        return len(plan.overrides)

    def _on_tab_changed(self, index: int) -> None:
        """Detect the first time the Hardware tab is opened (the snapshot costs a torch import)."""
        if self.tabs.widget(index) is self._pages.get("hardware") and not self._hardware_loaded:
            self._load_hardware()

    def _load_hardware(self) -> None:
        """Detect on a pool thread (torch import); the result fills the tab."""
        self._hardware_loaded = True
        self.hardware_header_label.setText("detecting hardware...")
        signals = run_task(lambda _progress: self._hardware_service.report())
        signals.finished.connect(self._on_hardware_ready)
        signals.failed.connect(self._on_hardware_failed)

    def _on_hardware_ready(self, report: HardwareReport) -> None:
        """Show the snapshot line and one row per model warning."""
        self.hardware_header_label.setText(hardware_header(report.info))
        self._plan = report.plan
        self.plan_label.setText("\n".join(describe(report.plan)) if report.plan is not None else "")
        self.optimise_button.setEnabled(report.plan is not None)
        self.hardware_table.setRowCount(len(report.warnings))
        for row, warning in enumerate(report.warnings):
            self.hardware_table.setItem(row, 0, QTableWidgetItem(warning.name))
            self.hardware_table.setItem(row, 1, QTableWidgetItem(warning.level))
            self.hardware_table.setItem(row, 2, QTableWidgetItem(warning.device or ""))
            self.hardware_table.setItem(row, 3, QTableWidgetItem(" — ".join(warning.messages)))

    def _on_hardware_failed(self, text: str) -> None:
        """Detection failed (e.g. torch import broken): show it on the tab."""
        self.hardware_header_label.setText(f"hardware detection failed: {text}")

    # --------------------------------------------------------------------- module-level helpers

    # ------------------------------------------------------------------ appearance tab

    def _build_appearance(self) -> QWidget:
        """Theme (OLED black / light), accent colour and quick / standard / pro mode, applied live."""
        page = QWidget()
        grid = QGridLayout(page)
        self.theme_combo = QComboBox(page)
        self.theme_combo.addItems(THEMES)
        self.theme_combo.setCurrentText(self.appearance.theme)
        self.accent_button = QPushButton(self.appearance.accent, page)
        self.mode_combo = QComboBox(page)
        self.mode_combo.addItems(UI_MODES)
        self.mode_combo.setCurrentText(self.appearance.mode)
        rows = (
            ("Theme", self.theme_combo, "OLED black or light", "theme dark black oled light white"),
            ("Accent colour", self.accent_button, "Buttons, selection and highlights", "accent color colour"),
            (
                "Mode",
                self.mode_combo,
                "quick shows the basics; pro shows every tool",
                "mode quick standard pro simple",
            ),
        )
        for index, (label, widget, help_text, words) in enumerate(rows):
            name = QLabel(label, page)
            help_label = QLabel(help_text, page)
            set_role(help_label, "muted")
            grid.addWidget(name, 2 * index, 0)
            grid.addWidget(widget, 2 * index, 1)
            grid.addWidget(help_label, 2 * index + 1, 1)
            haystack = f"{label} {help_text} {words} appearance look".lower()
            self._search_rows.append(_SearchRow("appearance", [name, widget, help_label], haystack))
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(2 * len(rows), 1)
        self.theme_combo.currentTextChanged.connect(lambda _text: self._commit_appearance())
        self.mode_combo.currentTextChanged.connect(lambda _text: self._commit_appearance())
        self.accent_button.clicked.connect(self._pick_accent)
        return page

    def _pick_accent(self) -> None:
        """Ask for an accent colour and apply it."""
        color = QColorDialog.getColor(QColor(self.appearance.accent), self, "Accent colour")
        if color.isValid():
            self.set_accent(color.name())

    def set_accent(self, accent: str) -> None:
        """Use `accent` (#rrggbb) as the accent colour."""
        self.accent_button.setText(accent)
        self._commit_appearance()

    def _commit_appearance(self) -> None:
        """Save the tab's choices, restyle the app and tell the window (mode changes its pages)."""
        appearance = Appearance(
            theme=THEMES[self.theme_combo.currentIndex()],
            accent=self.accent_button.text(),
            mode=UI_MODES[self.mode_combo.currentIndex()],
        )
        if appearance == self.appearance:
            return
        self.appearance = appearance
        save_appearance(self._qsettings, appearance)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, appearance)
        self.appearance_changed.emit(appearance)


def _flatten(overrides: dict[str, dict[str, Any]]) -> list[tuple[tuple[str, str], Any]]:
    """(section, key) -> value rows for the overrides table."""
    return [((section, key), value) for section, table in overrides.items() for key, value in table.items()]


def _display(value: Any) -> str:
    """The value column text (None shows as the empty string)."""
    return "" if value is None else str(value)


def _in_scroll(page: QWidget) -> QScrollArea:
    """Wrap a form page so narrow windows scroll instead of clipping."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    return scroll


def _row_widget(*widgets: QWidget, stretch_first: bool = False) -> QWidget:
    """A horizontal row widget; `stretch_first` lets the first widget take the spare width."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if stretch_first and index == 0 else 0)
    if not stretch_first:
        layout.addStretch(1)
    return row
