"""Entry script of the packaged command line (omniscan.exe / omniscan): the same CLI as `uv run omniscan`.

The GPU runtime the user downloaded (`omniscan runtime install`) is activated first, before anything imports torch.
"""

from omniscan.runtime import activate

activate()

from omniscan.cli import app  # noqa: E402 — after the runtime is active

if __name__ == "__main__":
    app()
