"""Hand-written mutants for the hf install path and `series_config` (card Q5).

Covers `models/store.py` (hf branch of `model_status`), `models/download.py`
(`_download_hf`, `remove_model`, `verify_models`), `models/catalog.py` (the hf
validation + descriptive fields) and `core/config.py` (`series_config`, merge).
Tests: `test_models_store.py`, `test_models_download.py`, `test_models_catalog.py`,
`test_models_resolve.py`, `test_models_cli.py`, `test_core_series_config.py`,
`test_models_update_mutation_gaps.py` (plus `test_hw_hf_mutation_gaps.py`).
The zip/file download paths are covered by `tests/mutants/models_update/` (card Q4).
"""

import os

# The harness runs pytest on the mutated files; without this a restore that lands in the same
# second as the mutation (same-size mutants) can leave stale bytecode in `__pycache__`.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

MUTANTS = [
    # ---------------------------------------------------------------- store.py (hf status)
    (
        "src/omniscan/models/store.py",
        'if entry.format in ("zip", "hf"):\n        return models_dir / entry.id',
        'if entry.format in ("zip",):\n        return models_dir / entry.id',
        "store: hf install path dropped",
    ),
    (
        "src/omniscan/models/store.py",
        'if data.get("revision") != entry.upstream_revision:',
        'if data.get("revision") == entry.upstream_revision:',
        "store: flip hf revision equality",
    ),
    (
        "src/omniscan/models/store.py",
        'sizes = data.get("file_sizes")',
        'sizes = data.get("file_size")',
        "store: wrong marker key for file_sizes",
    ),
    (
        "src/omniscan/models/store.py",
        "if not isinstance(sizes, dict):",
        "if isinstance(sizes, list):",
        "store: file_sizes type check narrowed",
    ),
    (
        "src/omniscan/models/store.py",
        "or file_path.stat().st_size != recorded",
        "or file_path.stat().st_size == recorded",
        "store: flip hf file-size comparison",
    ),
    (
        "src/omniscan/models/store.py",
        "if deep and file_sha256(file_path) != sha256:",
        "if deep or file_sha256(file_path) != sha256:",
        "store: and -> or in deep rehash gate",
    ),
    (
        "src/omniscan/models/store.py",
        "for name, sha256 in entry.files.items():",
        "for name, sha256 in list(entry.files.items())[:1]:",
        "store: hf status checks one file only",
    ),
    # ---------------------------------------------------------------- download.py (hf)
    (
        "src/omniscan/models/download.py",
        "if repo is None or revision is None:  # unreachable for a validated catalog entry",
        "if repo is None and revision is None:  # unreachable for a validated catalog entry",
        "download: or -> and in hf repo/revision guard",
    ),
    (
        "src/omniscan/models/download.py",
        "hf_download(repo, revision=revision, local_dir=str(folder), allow_patterns=list(entry.files))",
        "hf_download(repo, revision=revision, local_dir=str(folder))",
        "download: drop allow_patterns",
    ),
    (
        "src/omniscan/models/download.py",
        'if not file_path.is_file():\n            bad.append(f"{name} (missing)")',
        'if file_path.is_file():\n            bad.append(f"{name} (missing)")',
        "download: flip hf missing-file check",
    ),
    (
        "src/omniscan/models/download.py",
        "elif file_sha256(file_path) != sha256:",
        "elif file_sha256(file_path) == sha256:",
        "download: flip hf file hash comparison",
    ),
    (
        "src/omniscan/models/download.py",
        "if bad:",
        "if not bad:",
        "download: flip hf verification verdict",
    ),
    (
        "src/omniscan/models/download.py",
        "shutil.rmtree(folder, ignore_errors=True)  # never keep a half-verified install",
        "pass  # never keep a half-verified install",
        "download: keep the half-verified hf folder",
    ),
    (
        "src/omniscan/models/download.py",
        '"files_verified": len(entry.files),',
        '"files_verified": len(entry.files) + 1,',
        "download: marker files_verified off by one",
    ),
    (
        "src/omniscan/models/download.py",
        '"file_sizes": {name: (folder / name).stat().st_size for name in entry.files},',
        '"file_sizes": {name: 0 for name in entry.files},',
        "download: marker records zero sizes",
    ),
    (
        "src/omniscan/models/download.py",
        '"source": "upstream",\n        "revision": entry.upstream_revision,\n        "installed_at": datetime.now(tz=UTC).isoformat(),\n        "files_verified": len(entry.files),',
        '"source": "upstream",\n        "revision": entry.id,\n        "installed_at": datetime.now(tz=UTC).isoformat(),\n        "files_verified": len(entry.files),',
        "download: hf marker revision is the id",
    ),
    (
        "src/omniscan/models/download.py",
        "ollama_names=ollama_names, deep=deep) for entry in entries",
        "ollama_names=ollama_names, deep=False) for entry in entries",
        "download: verify drops the deep flag",
    ),
    (
        "src/omniscan/models/download.py",
        "ollama_names=ollama_names, deep=deep) for entry in entries",
        "deep=deep) for entry in entries",
        "download: verify drops ollama_names",
    ),
    (
        "src/omniscan/models/download.py",
        'if entry.format in ("zip", "hf"):\n        shutil.rmtree(path)',
        'if entry.format in ("zip",):\n        shutil.rmtree(path)',
        "download: hf removal unlinks the folder",
    ),
    (
        "src/omniscan/models/download.py",
        "if parent.resolve() != models_dir.resolve() and not any(parent.iterdir()):",
        "if parent.resolve() != models_dir.resolve() and any(parent.iterdir()):",
        "download: remove non-empty parent folder",
    ),
    # ---------------------------------------------------------------- catalog.py (hf validation)
    (
        "src/omniscan/models/catalog.py",
        '_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")',
        '_REVISION_RE = re.compile(r"^[0-9a-f]{39}$")',
        "catalog: revision regex off by one",
    ),
    (
        "src/omniscan/models/catalog.py",
        "if not _REVISION_RE.fullmatch(self.upstream_revision):",
        "if _REVISION_RE.fullmatch(self.upstream_revision):",
        "catalog: flip revision-format gate",
    ),
    (
        "src/omniscan/models/catalog.py",
        "if not self.files:",
        "if self.files:",
        "catalog: flip empty-files gate",
    ),
    (
        "src/omniscan/models/catalog.py",
        "bad = next((name for name, sha in self.files.items() if not _SHA256_RE.fullmatch(sha)), None)",
        "bad = next((name for name, sha in self.files.items() if _SHA256_RE.fullmatch(sha)), None)",
        "catalog: flip per-file sha gate",
    ),
    (
        "src/omniscan/models/catalog.py",
        "if self.size_mb <= 0:",
        "if self.size_mb < 0:",
        "catalog: size_mb <= 0 -> < 0",
    ),
    (
        "src/omniscan/models/catalog.py",
        'if self.format == "hf":\n            assert self.upstream_revision is not None  # checked above',
        'if self.format != "hf":\n            assert self.upstream_revision is not None  # checked above',
        "catalog: flip hf validation gate",
    ),
    (
        "src/omniscan/models/catalog.py",
        'family: str = ""  # ppocrv6 / ppocrv5 / paddleocr-vl / manga-ocr / comic-detector / lama / llm',
        'family: str = "ppocrv5"  # ppocrv6 / ppocrv5 / paddleocr-vl / manga-ocr / comic-detector / lama / llm',
        "catalog: family default ppocrv5",
    ),
    # ---------------------------------------------------------------- config.py (series_config)
    (
        "src/omniscan/core/config.py",
        'path = series_dir / "series.toml"',
        'path = series_dir / "series2.toml"',
        "config: wrong series file name",
    ),
    (
        "src/omniscan/core/config.py",
        "if not path.is_file():\n        return cfg",
        "if path.is_file():\n        return cfg",
        "config: flip series-file gate",
    ),
    (
        "src/omniscan/core/config.py",
        "unknown = sorted(set(data) - set(SERIES_SECTIONS))",
        "unknown = sorted(set(data) & set(SERIES_SECTIONS))",
        "config: unknown = allowed sections",
    ),
    (
        "src/omniscan/core/config.py",
        "if unknown:",
        "if not unknown:",
        "config: flip unknown-section gate",
    ),
    (
        "src/omniscan/core/config.py",
        "f\"{path}: sections not allowed per series: {', '.join(unknown)} (allowed: {', '.join(SERIES_SECTIONS)})\"",
        "f\"{path}: sections not allowed per series: {', '.join(SERIES_SECTIONS)} (allowed: {', '.join(unknown)})\"",
        "config: swap sections in error text",
    ),
    (
        "src/omniscan/core/config.py",
        "return Config(**_deep_merge(cfg.model_dump(), data))",
        "return Config(**_deep_merge(data, cfg.model_dump()))",
        "config: series overrides lose to machine",
    ),
    (
        "src/omniscan/core/config.py",
        "if isinstance(value, dict) and isinstance(out.get(key), dict):",
        "if isinstance(value, dict) or isinstance(out.get(key), dict):",
        "config: deep merge replaces sections",
    ),
    (
        "src/omniscan/core/config.py",
        'except ValidationError as exc:\n        raise SeriesConfigError(f"{path}: {exc}") from exc',
        "except ValidationError as exc:\n        raise exc",
        "config: plain ValidationError escapes",
    ),
    (
        "src/omniscan/core/config.py",
        'except tomllib.TOMLDecodeError as exc:\n        raise SeriesConfigError(f"{path}: {exc}") from exc',
        'except OSError as exc:\n        raise SeriesConfigError(f"{path}: {exc}") from exc',
        "config: invalid TOML not caught",
    ),
    (
        "src/omniscan/core/config.py",
        '    "typeset",\n    "export",\n)  # machine-level sections are not per series',
        '    "typeset",\n)  # machine-level sections are not per series',
        "config: export not allowed per series",
    ),
    # ---------------------------------------------------------------- round 2 (around the survivors)
    (
        "src/omniscan/models/catalog.py",
        '    "hf": ("upstream_repo", "upstream_revision", "files"),',
        '    "hf": ("upstream_revision", "files"),',
        "catalog: hf needs no upstream_repo",
    ),
    (
        "src/omniscan/models/download.py",
        'bad.append(f"{name} (sha256 mismatch)")',
        'bad.append(f"{name} (size mismatch)")',
        "download: wrong hf hash message",
    ),
    (
        "src/omniscan/models/download.py",
        "if path is None or not path.exists():\n        return False",
        "if path is not None or not path.exists():\n        return False",
        "download: remove always reports False",
    ),
    (
        "src/omniscan/core/config.py",
        'except tomllib.TOMLDecodeError as exc:\n        raise SeriesConfigError(f"{path}: {exc}") from exc',
        "except tomllib.TOMLDecodeError as exc:\n        raise SeriesConfigError(str(exc)) from exc",
        "config: TOML error without the path",
    ),
    (
        "src/omniscan/models/store.py",
        'if not marker.is_file():\n            return "corrupt"\n        try:\n            data = json.loads(marker.read_text(encoding="utf-8"))\n        except OSError, ValueError:\n            return "corrupt"\n        if data.get("revision") != entry.upstream_revision:',
        'if not marker.is_file():\n            return "missing"\n        try:\n            data = json.loads(marker.read_text(encoding="utf-8"))\n        except OSError, ValueError:\n            return "corrupt"\n        if data.get("revision") != entry.upstream_revision:',
        "store: hf marker missing is not corrupt",
    ),
    (
        "src/omniscan/models/store.py",
        'if entry.format == "hf":\n        if not path.is_dir():\n            return "missing"',
        'if entry.format == "zip":\n        if not path.is_dir():\n            return "missing"',
        "store: hf branch becomes the zip branch",
    ),
    (
        "src/omniscan/models/download.py",
        "hf_download(repo, revision=revision, local_dir=str(folder), allow_patterns=list(entry.files))",
        "hf_download(repo, revision=revision, local_dir=str(models_dir), allow_patterns=list(entry.files))",
        "download: hf fetch ignores the model folder",
    ),
]
