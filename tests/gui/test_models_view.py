"""ModelsView tests (offscreen): header, cells, filters, details, worker-backed download/remove."""
# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QPushButton

from omniscan.gui.models_view import ModelsView
from omniscan.hw.detect import HardwareInfo
from omniscan.models.download import ModelDownloadError
from omniscan.models.rows import ModelRow

WAIT_S = 5.0
HEADER_EXPECTED = "No GPU found — CPU only — RAM 32 GB — disk 500 GB free — torch cpu"


def hardware() -> HardwareInfo:
    """A CPU-only machine snapshot: RAM 32 GB, 500 GB free, torch cpu."""
    return HardwareInfo(
        os="windows",
        arch="x64",
        cpu_name="AMD Ryzen 9",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_gb=32.0,
        gpus=(),
        torch_build="cpu",
        best_device="cpu",
        onnxruntime_providers=("CPUExecutionProvider",),
        disk_free_gb=500.0,
    )


def row(
    model_id: str,
    name: str,
    *,
    status: str = "missing",
    fit: str = "ok",
    required: bool = False,
    fmt: str = "zip",
    size_mb: int = 5,
    langs: tuple[str, ...] = (),
    role: str = "detector",
    family: str = "",
    size_class: str = "",
    notes: str = "",
    messages: tuple[str, ...] = (),
    installed_path: str | None = None,
    description: str = "does things",
) -> ModelRow:
    """One hand-made ModelRow."""
    return ModelRow(
        id=model_id,
        name=name,
        kind="vision",
        role=role,
        family=family or None,
        size_class=size_class or None,
        size_mb=size_mb,
        required=required,
        format=fmt,
        license="Apache-2.0",
        description=description,
        langs=langs,
        recommended_for=(),
        notes=notes,
        used_by=(),
        status=status,
        installed_path=installed_path,
        fit_level=fit,
        fit_device=None if fit == "incompatible" else "cpu",
        fit_messages=messages,
    )


def default_rows() -> list[ModelRow]:
    """ok installed, ok missing, slow, warn, incompatible, cloud, required-missing."""
    return [
        row("ok-inst", "Ok Installed", status="installed", installed_path="C:/models/ok-inst"),
        row("ok-missing", "Ok Missing", size_mb=4096),  # 4.0 GB
        row("slow", "Slow One", role="inpaint", fit="slow", messages=("runs on the CPU (slow)",)),
        row(
            "warn",
            "Warn One",
            role="recognizer",
            fit="warn",
            size_mb=100,
            messages=("needs 0.1 GB of free disk space, 0 GB free",),
        ),
        row(
            "incomp",
            "Incompatible One",
            role="inpaint",
            fit="incompatible",
            family="lama",
            size_class="base",
            notes="big",
            messages=("needs 24 GB of GPU memory, your GPU has 16 GB",),
        ),
        row("cloud", "Cloud One", role="llm", fmt="cloud", status="cloud", size_mb=0),
        row("req", "Required One", required=True, fit="slow", messages=("runs on the CPU (slow)",)),
    ]


class FakeService:
    """Fixed rows and a recording confirm hook; download/remove/required can flip statuses."""

    def __init__(self, rows: list[ModelRow]) -> None:
        self._rows = rows
        self._hardware = hardware()
        self.confirm_log: list[tuple[str, str]] = []
        self.confirm_answer = True
        self.rows_calls = 0
        self.rows_error: Exception | None = None
        self.download_ids: list[str] = []
        self.download_threads: list[int] = []
        self.download_events: list[tuple[int, int | None]] = []
        self.download_release = threading.Event()
        self.download_result = "mirror"
        self.download_error: str | None = None
        self.download_flips = False
        self.remove_ids: list[str] = []
        self.remove_result = True
        self.remove_error: str | None = None
        self.remove_release = threading.Event()
        self.remove_flips = False
        self.required_plan: list[tuple[str, str | None]] = []
        self.required_release = threading.Event()
        self.required_progress: list[str] = []
        self.required_calls = 0

    def confirm(self, title: str, text: str) -> bool:
        """Record the question and return the canned answer."""
        self.confirm_log.append((title, text))
        return self.confirm_answer

    def rows(self, *, role: str | None = None, lang: str | None = None) -> Any:
        self.rows_calls += 1
        if self.rows_error is not None:
            raise self.rows_error
        return list(self._rows), self._hardware

    def download(self, model_id: str, on_progress: Any = None) -> str:
        self.download_ids.append(model_id)
        self.download_threads.append(threading.get_ident())
        for event in self.download_events:
            on_progress(*event)
        self.download_release.wait(WAIT_S)
        if self.download_error is not None:
            raise ModelDownloadError(self.download_error)  # must reach `failed` as its message
        if self.download_flips:
            self.flip(model_id)
        return self.download_result

    def remove(self, model_id: str) -> bool:
        self.remove_ids.append(model_id)
        self.remove_release.wait(WAIT_S)
        if self.remove_error is not None:
            raise RuntimeError(self.remove_error)
        if self.remove_flips:
            self.flip(model_id, status="missing")
        return self.remove_result

    def download_required(self, on_progress: Any = None) -> list[tuple[str, str | None]]:
        self.required_calls += 1
        results: list[tuple[str, str | None]] = []
        for index, (model_id, error) in enumerate(self.required_plan):
            if on_progress is not None:
                on_progress(model_id, 50, 100)
            self.required_progress.append(model_id)
            if index == 0:
                self.required_release.wait(WAIT_S)  # let the test observe the (1 of n) label
            if error is None:
                self.flip(model_id)
            results.append((model_id, error))
        return results

    def flip(self, model_id: str, *, status: str = "installed") -> None:
        """Swap one row for the same row with `status` (the refresh then shows the new state)."""
        for index, old in enumerate(self._rows):
            if old.id == model_id:
                installed = status == "installed"
                self._rows[index] = replace(
                    old,
                    status=status,
                    installed_path=f"C:/models/{old.id}" if installed else None,
                )


def make_fake(**kwargs: Any) -> FakeService:
    fake = FakeService(default_rows())
    for key, value in kwargs.items():
        setattr(fake, key, value)
    return fake


def view_of(qapp: QApplication, fake: FakeService) -> ModelsView:
    view = ModelsView(fake, confirm=fake.confirm)
    view.resize(1000, 600)
    qapp.processEvents()
    return view


def pump(qapp: QApplication, predicate: Any, timeout_s: float = WAIT_S) -> None:
    """Process events until `predicate()` holds (worker signals arrive as queued events)."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("timed out waiting for the view to settle")
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def buttons(qapp: QApplication, view: ModelsView) -> list[QPushButton]:
    """Every action button currently in the table."""
    qapp.processEvents()
    return [
        widget
        for index in range(view.table.rowCount())
        if isinstance(widget := view.table.cellWidget(index, 6), QPushButton)
    ]


def captions(view: ModelsView) -> dict[str, str | None]:
    """Action-cell button text per model id (None when the row has no button)."""
    result: dict[str, str | None] = {}
    for index, model_row in enumerate(view._view_rows):
        widget = view.table.cellWidget(index, 6)
        result[model_row.id] = widget.text() if isinstance(widget, QPushButton) else None
    return result


# ---------------------------------------------------------------- header and cells


def test_header_text_is_exact(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    assert view.hardware_label.text() == HEADER_EXPECTED


def test_table_columns_headers_and_row_order(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    headers = [view.table.horizontalHeaderItem(c).text() for c in range(7)]
    assert headers == ["Model", "Role", "Languages", "Size", "Fit", "Status", ""]
    assert [view.table.item(r, 0).text() for r in range(7)] == [
        "Ok Installed",
        "Ok Missing",
        "Slow One",
        "Warn One",
        "Incompatible One",
        "Cloud One",
        "Required One (required)",
    ]


def test_cell_texts_tooltips_and_backgrounds(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    table = view.table
    # ok installed row
    assert table.item(0, 1).text() == "detector"
    assert table.item(0, 2).text() == "-"
    assert table.item(0, 3).text() == "5 MB"
    assert table.item(0, 4).text() == "ok"
    assert table.item(0, 4).background().style() == Qt.BrushStyle.NoBrush
    assert table.item(0, 5).text() == "installed"
    assert table.item(0, 5).font().bold() is True
    # missing row: not bold, GB size
    assert table.item(1, 0).text() == "Ok Missing"
    assert table.item(1, 0).toolTip() == "ok-missing"
    assert table.item(1, 3).text() == "4.0 GB"
    assert table.item(1, 5).font().bold() is False
    # slow row: light yellow background, fit tooltip
    assert table.item(2, 4).text() == "slow"
    assert table.item(2, 4).toolTip() == "runs on the CPU (slow)"
    color = table.item(2, 4).background().color()
    assert (color.red(), color.green(), color.blue()) == (255, 249, 196)
    # warn row: light orange
    color = table.item(3, 4).background().color()
    assert (color.red(), color.green(), color.blue()) == (255, 224, 178)
    # incompatible row: light red, its messages as tooltip
    assert table.item(4, 4).text() == "incompatible"
    assert table.item(4, 4).toolTip() == "needs 24 GB of GPU memory, your GPU has 16 GB"
    assert table.item(4, 4).background().color() == QColor(255, 205, 210)
    # a clean ok row has an empty fit tooltip
    assert table.item(0, 4).toolTip() == ""


def test_required_suffix_and_button_captions(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    table = view.table
    assert table.item(6, 0).text() == "Required One (required)"
    assert table.item(5, 3).text() == "cloud"
    assert captions(view) == {
        "ok-inst": "Remove",
        "ok-missing": "Download",
        "slow": "Download",
        "warn": "Download",
        "incomp": "Download",
        "cloud": None,  # cloud models have no button
        "req": "Download",
    }


# ---------------------------------------------------------------- filters


def test_role_filter_changes_row_count(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    combo = view.role_combo
    assert combo.itemText(0) == "All roles"
    assert [combo.itemText(i) for i in range(1, combo.count())] == [
        "detector",
        "inpaint",
        "llm",
        "recognizer",
    ]
    combo.setCurrentIndex(combo.findText("inpaint"))
    qapp.processEvents()
    assert [view.table.item(r, 0).text() for r in range(view.table.rowCount())] == [
        "Slow One",
        "Incompatible One",
    ]


def test_lang_filter_changes_row_count(qapp: QApplication) -> None:
    fake = make_fake()
    fake._rows[1] = row("ok-missing", "Ok Missing", langs=("ko",), size_mb=4096)
    fake._rows[5] = row(
        "cloud", "Cloud One", role="llm", fmt="cloud", status="cloud", size_mb=0, langs=("zh",)
    )
    view = view_of(qapp, fake)
    combo = view.lang_combo
    assert combo.itemText(0) == "All languages"
    assert [combo.itemText(i) for i in range(1, combo.count())] == ["ko", "zh"]
    combo.setCurrentIndex(combo.findText("ko"))
    qapp.processEvents()
    assert view.table.rowCount() == 1 and view.table.item(0, 0).text() == "Ok Missing"
    combo.setCurrentIndex(combo.findText("zh"))
    qapp.processEvents()
    assert view.table.rowCount() == 1 and view.table.item(0, 0).text() == "Cloud One"


def test_installed_only_and_search_filters(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    view.installed_only.setChecked(True)
    qapp.processEvents()
    assert view.table.rowCount() == 1 and view.table.item(0, 0).text() == "Ok Installed"
    view.installed_only.setChecked(False)
    view.search.setText("INCOMPATIBLE")  # case-insensitive substring of the name
    qapp.processEvents()
    assert view.table.rowCount() == 1 and view.table.item(0, 0).text() == "Incompatible One"
    view.search.setText("no such model")
    qapp.processEvents()
    assert view.table.rowCount() == 0
    view.search.setText("ok-missing")  # search hits the id too
    qapp.processEvents()
    assert view.table.rowCount() == 1


def test_filters_combine(qapp: QApplication) -> None:
    fake = make_fake()
    fake._rows[1] = row("ok-missing", "Ok Missing", role="recognizer", langs=("ko",), size_mb=4096)
    fake._rows[3] = row("warn", "Warn One", role="recognizer", fit="warn", size_mb=100)
    view = view_of(qapp, fake)
    view.role_combo.setCurrentIndex(view.role_combo.findText("recognizer"))
    qapp.processEvents()
    assert view.table.rowCount() == 2
    view.lang_combo.setCurrentIndex(view.lang_combo.findText("ko"))
    qapp.processEvents()
    assert view.table.rowCount() == 1 and view.table.item(0, 0).text() == "Ok Missing"
    view.installed_only.setChecked(True)
    qapp.processEvents()
    assert view.table.rowCount() == 0


# ---------------------------------------------------------------- details


def test_details_for_selected_incompatible_row(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    view.table.selectRow(4)
    qapp.processEvents()
    text = view.details.toPlainText()
    assert text.startswith("does things")
    assert "license: Apache-2.0" in text
    assert "family: lama" in text and "size class: base" in text and "notes: big" in text
    assert "needs 24 GB of GPU memory, your GPU has 16 GB" in text
    assert "This model cannot run on this machine." in text


def test_details_installed_row_shows_path(qapp: QApplication) -> None:
    view = view_of(qapp, make_fake())
    view.table.selectRow(0)
    qapp.processEvents()
    assert "installed at: C:/models/ok-inst" in view.details.toPlainText()


# ---------------------------------------------------------------- download flow


def test_download_runs_in_worker_and_refreshes_after_install(qapp: QApplication) -> None:
    fake = make_fake(download_events=[(1024, None), (2_000_000, 4_000_000)], download_flips=True)
    view = view_of(qapp, fake)
    view.show()
    qapp.processEvents()
    main_thread = threading.get_ident()
    view.table.cellWidget(1, 6).click()
    pump(qapp, lambda: len(fake.download_ids) == 1)
    assert fake.download_ids == ["ok-missing"]
    assert fake.download_threads[0] != main_thread  # ran on a worker thread
    assert view.status_label.text() == "Downloading Ok Missing …"
    pump(qapp, lambda: view.progress.maximum() == 4_000_000)  # determinate as fed
    assert view.progress.value() == 2_000_000 and not view.progress.isHidden()
    assert all(not button.isEnabled() for button in buttons(qapp, view))
    assert not view.required_button.isEnabled()
    fake.download_release.set()
    pump(qapp, lambda: view.status_label.text() == "Ok Missing: installed (mirror)")
    assert view.progress.isHidden()
    assert fake.rows_calls >= 2  # refreshed after the download finished
    index = next(i for i, r in enumerate(view._view_rows) if r.id == "ok-missing")
    assert view.table.cellWidget(index, 6).text() == "Remove"  # the fake flipped the status


def test_download_indeterminate_while_total_unknown(qapp: QApplication) -> None:
    fake = make_fake(download_events=[(10, None)])
    view = view_of(qapp, fake)
    view.table.cellWidget(1, 6).click()
    pump(qapp, lambda: len(fake.download_ids) == 1)
    pump(qapp, lambda: not view.progress.isHidden())
    assert view.progress.maximum() == 0  # busy 0..0 while the total is unknown
    fake.download_release.set()
    pump(qapp, lambda: view.status_label.text() == "Ok Missing: installed (mirror)")


def test_download_failure_shows_red_error(qapp: QApplication) -> None:
    fake = make_fake(download_error="mirror down, upstream down")
    view = view_of(qapp, fake)
    view.table.cellWidget(1, 6).click()
    fake.download_release.set()
    pump(qapp, lambda: "ModelDownloadError" in view.status_label.text())
    assert view.status_label.text() == "Ok Missing: ModelDownloadError: mirror down, upstream down"
    assert "red" in view.status_label.styleSheet()
    assert all(button.isEnabled() for button in buttons(qapp, view))


def test_download_incompatible_confirm_gate(qapp: QApplication) -> None:
    fake = make_fake()
    view = view_of(qapp, fake)
    fake.confirm_answer = False
    view.table.cellWidget(4, 6).click()
    qapp.processEvents()
    assert fake.download_ids == []
    assert fake.confirm_log == [
        (
            "Download anyway?",
            "Incompatible One is not compatible with this machine:\n"
            "needs 24 GB of GPU memory, your GPU has 16 GB",
        )
    ]
    fake.confirm_answer = True
    view.table.cellWidget(4, 6).click()
    pump(qapp, lambda: len(fake.download_ids) == 1)
    assert fake.download_ids == ["incomp"]
    fake.download_release.set()
    pump(qapp, lambda: view.status_label.text() == "Incompatible One: installed (mirror)")


def test_download_warn_model_has_no_question(qapp: QApplication) -> None:
    fake = make_fake()
    view = view_of(qapp, fake)
    view.table.cellWidget(3, 6).click()  # the warn row
    pump(qapp, lambda: len(fake.download_ids) == 1)
    assert fake.confirm_log == []
    fake.download_release.set()  # let the worker finish inside the test, no leaked thread
    pump(qapp, lambda: view.status_label.text() == "Warn One: installed (mirror)")


# ---------------------------------------------------------------- remove flow


def test_remove_with_confirm_true_flips_row(qapp: QApplication) -> None:
    fake = make_fake(remove_result=True, remove_flips=True)
    view = view_of(qapp, fake)
    view.table.cellWidget(0, 6).click()  # Remove on the installed row
    pump(qapp, lambda: len(fake.remove_ids) == 1)
    assert fake.remove_ids == ["ok-inst"]
    assert view.status_label.text() == "Removing Ok Installed …"
    fake.remove_release.set()
    pump(qapp, lambda: view.status_label.text() == "Ok Installed: removed")
    index = next(i for i, r in enumerate(view._view_rows) if r.id == "ok-inst")
    assert view.table.cellWidget(index, 6).text() == "Download"  # refreshed to missing


def test_remove_declined_does_nothing(qapp: QApplication) -> None:
    fake = make_fake(confirm_answer=False)
    view = view_of(qapp, fake)
    view.table.cellWidget(0, 6).click()
    qapp.processEvents()
    assert fake.remove_ids == []
    assert fake.confirm_log == [("Remove model", "Delete Ok Installed from disk?")]


def test_remove_nothing_to_remove(qapp: QApplication) -> None:
    fake = make_fake(remove_result=False)
    view = view_of(qapp, fake)
    view.table.cellWidget(0, 6).click()
    fake.remove_release.set()
    pump(qapp, lambda: view.status_label.text() == "Ok Installed: nothing to remove")


# ---------------------------------------------------------------- required flow


def test_required_button_enabled_and_summary(qapp: QApplication) -> None:
    fake = make_fake(required_plan=[("req", None)])
    view = view_of(qapp, fake)
    assert view.required_button.text() == "Download required models"
    assert view.required_button.isEnabled()
    view.required_button.click()
    pump(qapp, lambda: view.status_label.text() == "Downloading req (1 of 1) …")
    fake.required_release.set()
    pump(qapp, lambda: view.status_label.text() == "1 installed, 0 failed")
    assert "Failed downloads:" not in view.details.toPlainText()  # no failures listed
    assert not view.required_button.isEnabled()  # req is installed now, nothing required left


def test_required_button_disabled_without_missing_required(qapp: QApplication) -> None:
    fake = FakeService([r for r in default_rows() if r.id != "req"])
    view = view_of(qapp, fake)
    assert not view.required_button.isEnabled()


def test_required_failure_listed_in_details(qapp: QApplication) -> None:
    fake = make_fake(required_plan=[("req", "req: mirror down")])
    view = view_of(qapp, fake)
    view.required_button.click()
    fake.required_release.set()
    pump(qapp, lambda: view.status_label.text() == "0 installed, 1 failed")
    assert "red" in view.status_label.styleSheet()
    text = view.details.toPlainText()
    assert "Failed downloads:" in text and "req: req: mirror down" in text


def test_required_progress_label_counts_steps(qapp: QApplication) -> None:
    fake = FakeService([*default_rows(), row("req2", "Required Two", required=True)])
    fake.required_plan = [("req", None), ("req2", "req2: refused")]
    view = view_of(qapp, fake)
    view.required_button.click()
    pump(qapp, lambda: view.status_label.text() == "Downloading req (1 of 2) …")
    fake.required_release.set()
    pump(qapp, lambda: view.status_label.text() == "1 installed, 1 failed")
    assert fake.required_progress == ["req", "req2"]


# ---------------------------------------------------------------- busy state and errors


def test_all_action_buttons_disabled_while_a_task_runs(qapp: QApplication) -> None:
    fake = make_fake()
    view = view_of(qapp, fake)
    view.table.cellWidget(1, 6).click()
    pump(qapp, lambda: len(fake.download_ids) == 1)
    assert all(not button.isEnabled() for button in buttons(qapp, view))
    assert not view.required_button.isEnabled()
    fake.download_release.set()
    pump(qapp, lambda: all(button.isEnabled() for button in buttons(qapp, view)))
    assert view.required_button.isEnabled()  # req is still missing


def test_exception_in_rows_shows_in_status_label(qapp: QApplication) -> None:
    fake = make_fake()
    fake.rows_error = RuntimeError("boom")
    view = view_of(qapp, fake)
    assert view.status_label.text() == "RuntimeError: boom"
    assert "red" in view.status_label.styleSheet()
    assert view.table.rowCount() == 0
