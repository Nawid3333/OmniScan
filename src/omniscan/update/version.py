"""Version parsing and SemVer comparison for the updater (card U6)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
_APP_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")


@dataclass(frozen=True, slots=True, order=False)
class Version:
    """A parsed app version; `pre` is () for a final release."""

    major: int
    minor: int
    patch: int
    pre: tuple[str | int, ...]

    def __str__(self) -> str:
        """The plain version text, e.g. "1.2.3" or "1.2.3-beta.2"."""
        core = f"{self.major}.{self.minor}.{self.patch}"
        return core if not self.pre else f"{core}-{'.'.join(str(identifier) for identifier in self.pre)}"


def parse_version(text: str) -> Version:
    """Parse "v1.2.3", "1.2.3" or "1.2.3-beta.2"; ValueError for anything else."""
    match = _VERSION_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"not a version: {text!r}")
    pre: tuple[str | int, ...] = ()
    if match.group(4) is not None:
        pre = tuple(
            int(identifier) if identifier.isascii() and identifier.isdigit() else identifier
            for identifier in match.group(4).split(".")
        )
    return Version(int(match.group(1)), int(match.group(2)), int(match.group(3)), pre)


def is_app_tag(tag: str) -> bool:
    """True only for strict app tags ^vMAJOR.MINOR.PATCH[-prerelease]$ ("models-v1", "1.2.3" -> False)."""
    return _APP_TAG_RE.fullmatch(tag) is not None


def _pre_key(pre: tuple[str | int, ...]) -> tuple[tuple[int, str | int], ...]:
    """Sort key for a `pre` part: numeric identifiers rank below alphanumeric ones."""
    return tuple((0, identifier) if isinstance(identifier, int) else (1, identifier) for identifier in pre)


def _version_key(version: Version) -> tuple[int, int, int, int, tuple[tuple[int, str | int], ...]]:
    """Total order with SemVer precedence; a release with `pre` is lower than the same core without."""
    return (version.major, version.minor, version.patch, 0 if version.pre else 1, _pre_key(version.pre))


def compare_versions(a: Version, b: Version) -> int:
    """-1, 0 or 1 with SemVer precedence."""
    key_a, key_b = _version_key(a), _version_key(b)
    return (key_a > key_b) - (key_a < key_b)


def current_version() -> Version:
    """The installed omniscan version; 0.0.0 when the package metadata is missing."""
    try:
        return parse_version(_installed_version("omniscan"))
    except PackageNotFoundError:
        return Version(0, 0, 0, ())
