"""Build the packaged OmniScan app: PyInstaller (packaging/omniscan.spec), the data folders, one zip.

    uv run --with pyinstaller python scripts/build_app.py [--out dist] [--no-zip] [--smoke]

Writes `dist/omniscan/` (the folder a user unzips: `OmniScan` the desktop app, `omniscan` the command line,
`config/`, `fonts/`, the web UI when `webui/dist` was built) and `dist/omniscan-<os>-<arch>.zip`, the asset
name `omniscan update` looks for in a release. `--smoke` runs the packaged command line (`--help`,
`hardware --simulate cpu-laptop`) before zipping, so a bundle that cannot start never becomes a release.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "omniscan.spec"
DATA_DIRS = ("config", "fonts")
_OS_NAMES = {"win32": "windows", "darwin": "macos", "linux": "linux"}
_ARCH_NAMES = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}


def platform_key() -> str:
    """This machine as `<os>-<arch>` (what omniscan.update.github.platform_key reports)."""
    return f"{_OS_NAMES[sys.platform]}-{_ARCH_NAMES[platform.machine().lower()]}"


def run_pyinstaller(out: Path) -> Path:
    """Run the spec; returns the bundle folder."""
    work = out / "build"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(out),
            "--workpath",
            str(work),
            str(SPEC),
        ],
        check=True,
        cwd=ROOT,
    )
    bundle = out / "omniscan"
    if not bundle.is_dir():
        raise SystemExit(f"PyInstaller did not produce {bundle}")
    return bundle


def copy_data(bundle: Path) -> None:
    """Put the config, fonts and web UI next to the programs (where REPO_ROOT-relative code finds them)."""
    for name in DATA_DIRS:
        target = bundle / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / name, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    webui = ROOT / "webui" / "dist"
    if webui.is_dir():
        target = bundle / "webui" / "dist"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(webui, target)


def program(bundle: Path, name: str) -> Path:
    """The packaged program's path on this OS."""
    return bundle / (f"{name}.exe" if sys.platform == "win32" else name)


def smoke(bundle: Path) -> None:
    """Start the packaged command line twice; a failure stops the build."""
    cli = program(bundle, "omniscan")
    for args in (
        ["--help"],
        ["hardware", "--simulate", "cpu-laptop"],
        ["tune", "--simulate", "rtx3060", "--json"],
    ):
        result = subprocess.run([str(cli), *args], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise SystemExit(
                f"smoke test failed: omniscan {' '.join(args)}\n{result.stdout}\n{result.stderr}"
            )
        print(f"smoke ok: omniscan {' '.join(args)} ({len(result.stdout)} bytes)")


def zip_bundle(bundle: Path, out: Path) -> Path:
    """Zip the bundle as omniscan-<os>-<arch>.zip (file modes kept, so the programs stay executable)."""
    target = out / f"omniscan-{platform_key()}.zip"
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(bundle.rglob("*")):
            archive.write(path, path.relative_to(out))
    return target


def main(argv: list[str] | None = None) -> int:
    """Build, copy the data, optionally smoke-test, zip; prints the zip path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="output folder (default: dist)")
    parser.add_argument("--no-zip", action="store_true", help="leave the folder only")
    parser.add_argument("--smoke", action="store_true", help="run the packaged command line before zipping")
    args = parser.parse_args(argv)
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle = run_pyinstaller(out)
    copy_data(bundle)
    if args.smoke:
        smoke(bundle)
    if args.no_zip:
        print(bundle)
        return 0
    target = zip_bundle(bundle, out)
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
