"""Entry script of the packaged desktop app (OmniScan.exe / OmniScan.app / OmniScan): the GUI.

The GPU runtime the user downloaded (`omniscan runtime install`) is activated first, before anything imports torch.
"""

import sys

from omniscan.runtime import activate

activate()

from omniscan.gui.app import main  # noqa: E402 — after the runtime is active

if __name__ == "__main__":
    sys.exit(main())
