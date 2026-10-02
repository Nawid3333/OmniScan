"""Entry script of the packaged command line (omniscan.exe / omniscan): the same CLI as `uv run omniscan`."""

from omniscan.cli import app

if __name__ == "__main__":
    app()
