"""Hand-written mutants for the rule logic of the model manager and the updater (card Q4).

Covers `models/catalog.py` + `models/resolve.py` (indices 0-14) and `update/version.py` +
`update/github.py` (indices 15-43).
Tests: `test_models_catalog.py`, `test_models_resolve.py`, `test_models_store.py`,
`test_models_cli.py`, `test_update_version.py`, `test_update_github.py`,
`test_update_download.py`, `test_update_cli.py`.
"""

MUTANTS = [
    # ---------------------------------------------------------------- catalog.py (0-7)
    (
        "src/omniscan/models/catalog.py",
        "missing = [f for f in _REQUIRED_FIELDS[self.format] if getattr(self, f) is None]",
        "missing = [f for f in _REQUIRED_FIELDS[self.format] if getattr(self, f) is not None]",
        "catalog: flip required-field check",
    ),
    (
        "src/omniscan/models/catalog.py",
        "needs {missing[0]}",
        "needs {missing[-1]}",
        "catalog: report last missing field",
    ),
    (
        "src/omniscan/models/catalog.py",
        'if self.format in ("zip", "file") and (self.sha256 is None or not _SHA256_RE.fullmatch(self.sha256)):',
        'if self.format in ("zip", "file") or (self.sha256 is None or not _SHA256_RE.fullmatch(self.sha256)):',
        "catalog: and -> or in sha-format gate",
    ),
    (
        "src/omniscan/models/catalog.py",
        '_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")',
        '_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")',
        "catalog: sha regex accepts uppercase",
    ),
    (
        "src/omniscan/models/catalog.py",
        "paths = (path,) if path is not None else (default_catalog_path(), machine_catalog_path())",
        "paths = (path,) if path is None else (default_catalog_path(), machine_catalog_path())",
        "catalog: swap explicit-path branch",
    ),
    (
        "src/omniscan/models/catalog.py",
        'tables = data.get("model", [])',
        'tables = data.get("models", [])',
        "catalog: wrong toml key",
    ),
    (
        "src/omniscan/models/catalog.py",
        "if entry.id in seen:\n                raise ValueError(f\"{file_path}: duplicate model id {entry.id!r}\")",
        "if entry.id not in seen:\n                raise ValueError(f\"{file_path}: duplicate model id {entry.id!r}\")",
        "catalog: flip duplicate-id check",
    ),
    (
        "src/omniscan/models/catalog.py",
        "entries[entry.id] = entry  # replacing a key keeps its position; new ids go to the end",
        "entries.setdefault(entry.id, entry)  # replacing a key keeps its position; new ids go to the end",
        "catalog: machine override ignored",
    ),
    # ---------------------------------------------------------------- resolve.py (8-14)
    (
        "src/omniscan/models/resolve.py",
        'if entry.format != "zip" or entry.upstream_repo != repo:\n            continue',
        'if entry.format != "zip" and entry.upstream_repo != repo:\n            continue',
        "resolve: or -> and in entry filter",
    ),
    (
        "src/omniscan/models/resolve.py",
        "entry.upstream_repo != repo",
        "entry.upstream_repo == repo",
        "resolve: flip repo match",
    ),
    (
        "src/omniscan/models/resolve.py",
        'path is not None and model_status(entry, models_dir, ollama_names=None) == "installed"',
        'path is not None or model_status(entry, models_dir, ollama_names=None) == "installed"',
        "resolve: and -> or in installed gate",
    ),
    (
        "src/omniscan/models/resolve.py",
        "return str(path)",
        "return str(path.parent)",
        "resolve: return parent folder",
    ),
    (
        "src/omniscan/models/resolve.py",
        "entries = load_catalog() if catalog is None else catalog",
        "entries = load_catalog() if catalog is not None else catalog",
        "resolve: swap catalog source",
    ),
    (
        "src/omniscan/models/resolve.py",
        "except OSError, ValueError:  # unreadable/corrupt catalog: use the hub, never raise (PEP 758)",
        "except OSError:  # unreadable/corrupt catalog: use the hub, never raise (PEP 758)",
        "resolve: narrow catalog error catch",
    ),
    (
        "src/omniscan/models/resolve.py",
        'entry.format != "zip"',
        'entry.format == "zip"',
        "resolve: flip format filter",
    ),
    # ---------------------------------------------------------------- version.py (15-26)
    (
        "src/omniscan/update/version.py",
        'return core if not self.pre else f"{core}-{'
        "'.'"
        '.join(str(identifier) for identifier in self.pre)}"',
        'return core if self.pre else f"{core}-{'
        "'.'"
        '.join(str(identifier) for identifier in self.pre)}"',
        "version: flip prerelease test in __str__",
    ),
    (
        "src/omniscan/update/version.py",
        r'_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")',
        r'_VERSION_RE = re.compile(r"^v?(\d)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")',
        r"version: major \d+ -> \d",
    ),
    (
        "src/omniscan/update/version.py",
        "int(identifier) if identifier.isascii() and identifier.isdigit() else identifier",
        "int(identifier) if identifier.isascii() or identifier.isdigit() else identifier",
        "version: and -> or in numeric id check",
    ),
    (
        "src/omniscan/update/version.py",
        "return tuple((0, identifier) if isinstance(identifier, int) else (1, identifier) for identifier in pre)",
        "return tuple((1, identifier) if isinstance(identifier, int) else (0, identifier) for identifier in pre)",
        "version: numeric pre ranks above string",
    ),
    (
        "src/omniscan/update/version.py",
        "return (version.major, version.minor, version.patch, 0 if version.pre else 1, _pre_key(version.pre))",
        "return (version.major, version.minor, version.patch, 1 if version.pre else 0, _pre_key(version.pre))",
        "version: flip release-above-prerelease flag",
    ),
    (
        "src/omniscan/update/version.py",
        "return (key_a > key_b) - (key_a < key_b)",
        "return (key_a < key_b) - (key_a > key_b)",
        "version: invert comparison sign",
    ),
    (
        "src/omniscan/update/version.py",
        "return _APP_TAG_RE.fullmatch(tag) is not None",
        "return _APP_TAG_RE.fullmatch(tag) is None",
        "version: invert app-tag verdict",
    ),
    (
        "src/omniscan/update/version.py",
        r'_APP_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")',
        r'_APP_TAG_RE = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")',
        "version: app tag no longer requires v",
    ),
    (
        "src/omniscan/update/version.py",
        "return Version(0, 0, 0, ())",
        "return Version(0, 0, 1, ())",
        "version: fallback 0.0.0 -> 0.0.1",
    ),
    (
        "src/omniscan/update/version.py",
        "if match.group(4) is not None:",
        "if match.group(1) is not None:",
        "version: prerelease taken from major group",
    ),
    (
        "src/omniscan/update/version.py",
        "return Version(int(match.group(1)), int(match.group(2)), int(match.group(3)), pre)",
        "return Version(int(match.group(2)), int(match.group(1)), int(match.group(3)), pre)",
        "version: swap major/minor groups",
    ),
    (
        "src/omniscan/update/version.py",
        "'.'.join(str(identifier) for identifier in self.pre)",
        "'-'.join(str(identifier) for identifier in self.pre)",
        "version: join prerelease with dash",
    ),
    # ---------------------------------------------------------------- github.py (27-43)
    (
        "src/omniscan/update/github.py",
        "if response.status_code == 404:",
        "if response.status_code != 404:",
        "github: flip 404 check",
    ),
    (
        "src/omniscan/update/github.py",
        'if response.status_code in (403, 429) and response.headers.get("X-RateLimit-Remaining") == "0":',
        'if response.status_code in (403, 429) or response.headers.get("X-RateLimit-Remaining") == "0":',
        "github: and -> or in rate-limit gate",
    ),
    (
        "src/omniscan/update/github.py",
        'if response.status_code >= 400:\n        raise UpdateError(f"GitHub answered HTTP {response.status_code}")',
        'if response.status_code > 400:\n        raise UpdateError(f"GitHub answered HTTP {response.status_code}")',
        "github: >= 400 -> > 400",
    ),
    (
        "src/omniscan/update/github.py",
        'if not isinstance(entry, dict) or entry.get("draft"):\n        return False',
        'if not isinstance(entry, dict) and entry.get("draft"):\n        return False',
        "github: or -> and in draft filter",
    ),
    (
        "src/omniscan/update/github.py",
        "releases.sort(key=cmp_to_key(lambda a, b: compare_versions(a.version, b.version)), reverse=True)",
        "releases.sort(key=cmp_to_key(lambda a, b: compare_versions(a.version, b.version)), reverse=False)",
        "github: flip release sort direction",
    ),
    (
        "src/omniscan/update/github.py",
        'notes=entry.get("body") or "",',
        'notes=entry.get("body") and "",',
        "github: or -> and in notes fallback",
    ),
    (
        "src/omniscan/update/github.py",
        'prerelease=bool(entry.get("prerelease")),',
        'prerelease=not bool(entry.get("prerelease")),',
        "github: flip prerelease flag",
    ),
    (
        "src/omniscan/update/github.py",
        'if reset is None or not reset.isdigit():\n        return "a while"',
        'if reset is None and not reset.isdigit():\n        return "a while"',
        "github: or -> and in reset check",
    ),
    (
        "src/omniscan/update/github.py",
        'return datetime.fromtimestamp(int(reset), tz=UTC).strftime("%H:%M UTC")',
        'return datetime.fromtimestamp(int(reset), tz=UTC).strftime("%H:%M")',
        "github: drop UTC suffix from retry hint",
    ),
    (
        "src/omniscan/update/github.py",
        'candidates = [r for r in releases if channel == "beta" or (not r.prerelease and not r.version.pre)]',
        'candidates = [r for r in releases if channel == "beta" and (not r.prerelease and not r.version.pre)]',
        "github: or -> and in channel filter",
    ),
    (
        "src/omniscan/update/github.py",
        "return max(candidates, key=cmp_to_key(lambda a, b: compare_versions(a.version, b.version)))",
        "return min(candidates, key=cmp_to_key(lambda a, b: compare_versions(a.version, b.version)))",
        "github: max -> min",
    ),
    (
        "src/omniscan/update/github.py",
        'if asset.name == f"omniscan-{key}.zip":',
        'if asset.name != f"omniscan-{key}.zip":',
        "github: flip asset name match",
    ),
    (
        "src/omniscan/update/github.py",
        'if asset.name == "SHA256SUMS":',
        'if asset.name != "SHA256SUMS":',
        "github: flip checksum asset match",
    ),
    (
        "src/omniscan/update/github.py",
        "os_name = _OS_NAMES.get(sys.platform)",
        "os_name = _ARCH_NAMES.get(sys.platform)",
        "github: platform table used for os",
    ),
    (
        "src/omniscan/update/github.py",
        'return f"{os_name}-{arch}"',
        'return f"{arch}-{os_name}"',
        "github: swap os-arch in platform key",
    ),
    (
        "src/omniscan/update/github.py",
        'headers["Authorization"] = f"Bearer {token}"',
        'headers["Authorization"] = token',
        "github: token without Bearer prefix",
    ),
    (
        "src/omniscan/update/github.py",
        'params={"per_page": "100"},',
        'params={"per_page": "10"},',
        "github: per_page 100 -> 10",
    ),
]