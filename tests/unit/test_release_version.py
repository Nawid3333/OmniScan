"""scripts/release_version.py: the bump level from commit messages, the next version, writing it."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_version.py"
_spec = importlib.util.spec_from_file_location("release_version", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
sys.modules["release_version"] = script
_spec.loader.exec_module(script)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("messages", "level"),
    [
        ([], "patch"),
        (["detect: find text the detector missed", "ci: parallel tests"], "patch"),
        (["fix: a crash", "feat: reading mode"], "minor"),
        (["feat(studio): line status"], "minor"),
        (["edits: per-line status\n\n[minor]"], "minor"),
        (["web: drop the old route\n\nBREAKING CHANGE: /api/v1 is gone"], "major"),
        (["feat!: new project format"], "major"),
        (["feat: one", "core: new manifest [major]"], "major"),
        (["ocr: mention feat: in a body line"], "patch"),
    ],
)
def test_bump_level(messages: list[str], level: str) -> None:
    assert script.bump_level(messages) == level


def test_bump() -> None:
    assert script.bump("0.1.9", "patch") == "0.1.10"
    assert script.bump("1.2.3", "minor") == "1.3.0"
    assert script.bump("1.2.3", "major") == "2.0.0"
    with pytest.raises(ValueError, match=r"not a MAJOR\.MINOR\.PATCH"):
        script.bump("1.2", "patch")


def _repo(tmp_path: Path, version: str = "0.1.0") -> Path:
    (tmp_path / "src" / "omniscan").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "omniscan"\nversion = "{version}"\n\n[tool.x]\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "omniscan" / "__init__.py").write_text(
        f'"""OmniScan."""\n\n__version__ = "{version}"\n', encoding="utf-8"
    )
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    _commit(tmp_path, "initial")
    return tmp_path


def _commit(root: Path, message: str) -> None:
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-am", message], cwd=root, check=True)


def _tag(root: Path, tag: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    _commit(root, f"release: {tag}")
    subprocess.run(["git", "tag", tag], cwd=root, check=True)


def test_first_release_keeps_the_version(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit(root, "feat: something")
    assert script.next_version(root=root) == "0.1.0"


def test_next_version_after_a_tag(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _tag(root, "v0.1.0")
    subprocess.run(["git", "tag", "models-v1"], cwd=root, check=True)
    _commit(root, "detect: tweak")
    assert script.next_version(root=root) == "0.1.1"
    _commit(root, "feat: new tool")
    assert script.next_version(root=root) == "0.2.0"
    assert script.next_version("major", root=root) == "1.0.0"
    assert script.next_version("patch", root=root) == "0.1.1"


def test_a_hand_raised_version_is_released_as_it_is(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _tag(root, "v0.1.0")
    script.write_version("0.5.0", root)
    _commit(root, "raise to 0.5.0")
    assert script.next_version(root=root) == "0.5.0"


def test_write_version(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    script.write_version("0.2.0", root)
    assert script.read_version(root) == "0.2.0"
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "9.9.9"' in pyproject  # only the [project] line
    assert '__version__ = "0.2.0"' in (root / "src" / "omniscan" / "__init__.py").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"not a MAJOR\.MINOR\.PATCH"):
        script.write_version("v0.3.0", root)


def test_repo_versions_agree() -> None:
    """pyproject.toml and __version__ carry the same version, so a release never names two."""
    init = (REPO_ROOT / "src" / "omniscan" / "__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{script.read_version(REPO_ROOT)}"' in init


def test_release_workflow() -> None:
    raw = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    triggers = data.get("on", data.get(True))  # YAML 1.1 parses the bare `on` key as boolean True
    assert triggers["workflow_run"]["workflows"] == ["ci"]  # the `name:` of ci.yml
    steps_text = "\n".join(str(step.get("run", "")) for step in data["jobs"]["release"]["steps"])
    assert "scripts/release_version.py next" in steps_text
    assert "uv lock" in steps_text
    assert "[skip ci]" in steps_text  # the release commit must not start CI (and a release) again
    assert "SHA256SUMS" in steps_text  # omniscan update verifies downloads against it
