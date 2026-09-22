"""`python -m omniscan.gui`: the desktop app entry point (exit code 2 without PySide6)."""

from omniscan.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
