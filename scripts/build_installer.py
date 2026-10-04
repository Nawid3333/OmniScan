"""Build the installer for this OS from the packaged app folder (scripts/build_app.py), and smoke-test it (#44).

    python scripts/build_installer.py [--bundle dist/omniscan] [--out dist] [--smoke]

- Windows: an Inno Setup installer, `OmniScan-Setup-windows-x64.exe` (packaging/omniscan.iss; needs ISCC.exe,
  found on PATH, in `$ISCC` or in Inno Setup 6's default folder). The smoke test installs it silently into a
  temporary folder, runs the installed command line and uninstalls it again.
- macOS: a disk image, `OmniScan-macos-<arch>.dmg`, holding the app folder (drag it to Applications). The smoke
  test mounts it and runs the command line from it.
- Linux: an AppImage, `OmniScan-linux-x64.AppImage` (appimagetool is downloaded once into the output folder). It
  starts the desktop app; `--cli ARGS` runs the command line instead. The smoke test runs `--cli --help`.

All of them are unsigned until the code-signing question is answered (docs/OPEN_QUESTIONS.md B8). Only the
standard library is used, so any Python runs this script.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPIMAGETOOL = (
    "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-{arch}.AppImage"
)
_ARCH = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}
APP_RUN = """#!/bin/sh
# OmniScan AppImage entry: the desktop app, or the command line with --cli
here="$(dirname "$(readlink -f "$0")")"
if [ "$1" = "--cli" ]; then
  shift
  exec "$here/usr/lib/omniscan/omniscan" "$@"
fi
exec "$here/usr/lib/omniscan/OmniScan" "$@"
"""
DESKTOP = """[Desktop Entry]
Type=Application
Name=OmniScan
Comment=Translate manhwa and manga chapters
Exec=OmniScan
Icon=omniscan
Categories=Graphics;Office;
Terminal=false
"""
DMG_README = """OmniScan

Drag the OmniScan folder to Applications, then open OmniScan inside it.
The app is not signed yet: the first time, right-click OmniScan and choose Open.
The command line is the `omniscan` program in the same folder.
"""


def version() -> str:
    """The app version (src/omniscan/__init__.py)."""
    text = (ROOT / "src" / "omniscan" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else "0.0.0"


def arch() -> str:
    """This machine's architecture as the release assets name it (x64, arm64)."""
    return _ARCH.get(platform.machine().lower(), platform.machine().lower())


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 3600) -> str:
    """Run a command; its stdout. A failure stops the build with the command's output."""
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def windows_installer(bundle: Path, out: Path) -> Path:
    """The Inno Setup installer of the bundle."""
    iscc = os.environ.get("ISCC") or shutil.which("ISCC") or shutil.which("iscc")
    default = (
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe"
    )
    if iscc is None and default.is_file():
        iscc = str(default)
    if iscc is None:
        raise SystemExit("ISCC.exe (Inno Setup 6) not found: install it or set $ISCC")
    name = f"OmniScan-Setup-windows-{arch()}"
    run(
        [
            iscc,
            "/Q",
            f"/DSourceDir={bundle}",
            f"/DOutputDir={out}",
            f"/DOutputName={name}",
            f"/DAppVersion={version()}",
            f"/DIconFile={ROOT / 'packaging' / 'omniscan.ico'}",
            str(ROOT / "packaging" / "omniscan.iss"),
        ]
    )
    return out / f"{name}.exe"


def mac_dmg(bundle: Path, out: Path) -> Path:
    """A compressed disk image holding the app folder and a short read-me."""
    target = out / f"OmniScan-macos-{arch()}.dmg"
    with tempfile.TemporaryDirectory() as staging:
        shutil.copytree(bundle, Path(staging) / "OmniScan", symlinks=True)
        (Path(staging) / "README.txt").write_text(DMG_README, encoding="utf-8")
        target.unlink(missing_ok=True)
        run(
            [
                "hdiutil",
                "create",
                "-volname",
                "OmniScan",
                "-srcfolder",
                staging,
                "-ov",
                "-format",
                "UDZO",
                str(target),
            ]
        )
    return target


def linux_appimage(bundle: Path, out: Path) -> Path:
    """An AppImage: the app folder under usr/lib/omniscan, started by AppRun."""
    machine = platform.machine().lower()
    tool = out / f"appimagetool-{machine}.AppImage"
    if not tool.is_file():
        urllib.request.urlretrieve(APPIMAGETOOL.format(arch=machine), tool)
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    target = out / f"OmniScan-linux-{arch()}.AppImage"
    with tempfile.TemporaryDirectory() as staging:
        appdir = Path(staging) / "OmniScan.AppDir"
        shutil.copytree(bundle, appdir / "usr" / "lib" / "omniscan", symlinks=True)
        app_run = appdir / "AppRun"
        app_run.write_text(APP_RUN, encoding="utf-8")
        app_run.chmod(0o755)
        (appdir / "omniscan.desktop").write_text(DESKTOP, encoding="utf-8")
        shutil.copy2(ROOT / "packaging" / "omniscan.png", appdir / "omniscan.png")
        env = {**os.environ, "ARCH": machine, "APPIMAGE_EXTRACT_AND_RUN": "1"}
        target.unlink(missing_ok=True)
        run([str(tool), str(appdir), str(target)], env=env)
    return target


def smoke(installer: Path) -> None:
    """Install or mount the installer and run the command line it put there."""
    if sys.platform == "win32":
        with tempfile.TemporaryDirectory() as folder:
            run(
                [
                    str(installer),
                    "/VERYSILENT",
                    "/SUPPRESSMSGBOXES",
                    "/NORESTART",
                    "/CURRENTUSER",
                    f"/DIR={folder}",
                ]
            )
            print(run([str(Path(folder) / "omniscan.exe"), "--help"])[:200])
            run([str(Path(folder) / "unins000.exe"), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"])
    elif sys.platform == "darwin":
        with tempfile.TemporaryDirectory() as mount:
            run(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", mount, str(installer)])
            try:
                print(run([str(Path(mount) / "OmniScan" / "omniscan"), "--help"])[:200])
            finally:
                run(["hdiutil", "detach", mount])
    else:
        env = {**os.environ, "APPIMAGE_EXTRACT_AND_RUN": "1"}
        print(run([str(installer), "--cli", "--help"], env=env)[:200])
    print(f"smoke ok: {installer.name}")


def main(argv: list[str] | None = None) -> int:
    """Build this OS's installer (and smoke-test it); prints its path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist" / "omniscan", help="the app folder")
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="output folder (default: dist)")
    parser.add_argument("--smoke", action="store_true", help="install the result and run its command line")
    args = parser.parse_args(argv)
    bundle, out = args.bundle.resolve(), args.out.resolve()
    if not bundle.is_dir():
        raise SystemExit(f"no app folder at {bundle}: run scripts/build_app.py first")
    out.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        installer = windows_installer(bundle, out)
    elif sys.platform == "darwin":
        installer = mac_dmg(bundle, out)
    else:
        installer = linux_appimage(bundle, out)
    if args.smoke:
        smoke(installer)
    print(installer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
