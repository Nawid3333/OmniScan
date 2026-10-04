# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the OmniScan desktop app: one folder with two programs sharing one runtime.

`OmniScan` (windowed) opens the desktop app, `omniscan` (console) is the command line. scripts/build_app.py
runs this spec, then copies `config/`, `fonts/` and the built web UI next to the programs — the code finds
them through REPO_ROOT (`Path(__file__).parents[3]`), which in the bundle is the folder holding the programs,
exactly like the repository root is in a checkout. Torch comes from whatever backend extra the building
environment has (the release builds use `cpu`: it runs everywhere); `omniscan runtime install` downloads the GPU
build later, at the version recorded here with torch's own metadata (src/omniscan/runtime/).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH is set by PyInstaller

# transformers and torch load model classes by name: the hooks-contrib hooks collect most of it, the rest is
# listed here so a frozen pipeline run finds the same modules the checkout does
hidden = [
    *collect_submodules("omniscan"),
    *collect_submodules("transformers.models.auto"),
    "PIL.ImageQt",
    "pyspellchecker",
    "spellchecker",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]
datas = [
    *collect_data_files("spellchecker"),  # the English dictionary of the typo check
    *collect_data_files("transformers", include_py_files=False),
    *copy_metadata("torch"),  # the version `omniscan runtime install` downloads the GPU build at
    *copy_metadata("torchvision"),
]
ICON = str(ROOT / "packaging" / "omniscan.ico")

gui_a = Analysis(
    [str(ROOT / "packaging" / "launch_gui.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=["tkinter", "matplotlib", "IPython", "notebook", "pytest"],
    noarchive=False,
)
cli_a = Analysis(
    [str(ROOT / "packaging" / "launch_cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=["tkinter", "matplotlib", "IPython", "notebook", "pytest"],
    noarchive=False,
)
MERGE((gui_a, "launch_gui", "OmniScan"), (cli_a, "launch_cli", "omniscan"))

gui_pyz = PYZ(gui_a.pure)
cli_pyz = PYZ(cli_a.pure)
gui_exe = EXE(
    gui_pyz,
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name="OmniScan",
    icon=ICON,
    debug=False,
    strip=False,
    upx=False,
    console=False,  # a windowed program: no console window behind the app
)
cli_exe = EXE(
    cli_pyz,
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name="omniscan",
    icon=ICON,
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
COLLECT(
    gui_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_exe,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=False,
    name="omniscan",
)
