"""Filesystem-safe filenames for packaged chapter files."""

from __future__ import annotations

_UNSAFE = frozenset('<>:"/\\|?*')


def safe_filename(name: str) -> str:
    """Replace every character in <>:"/\\|?* and every ASCII control character (code 0-31) with '_', then strip
    leading/trailing spaces and dots. If the result is empty return '_'. Nothing else is changed (Unicode such as
    Korean is kept as-is)."""
    cleaned = "".join("_" if ch in _UNSAFE or ord(ch) < 32 else ch for ch in name)
    cleaned = cleaned.strip(" .")
    return cleaned or "_"
