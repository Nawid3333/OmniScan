"""Entry script of the packaged desktop app (OmniScan.exe / OmniScan.app / OmniScan): the GUI."""

import sys

from omniscan.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
