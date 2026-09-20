"""Fixtures for the GUI tests: headless Qt (offscreen) and the shared QApplication."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before any Qt import

from typing import TYPE_CHECKING  # noqa: E402

import pytest  # noqa: E402

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> "QApplication":
    """The one shared offscreen QApplication for all GUI tests."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])