"""Hand-written mutants for the download logic of the model manager and the updater (card Q4).

Covers `models/store.py` + `models/download.py` (indices 0-34) and `update/download.py`
(indices 35-54).
Tests: `test_models_store.py`, `test_models_download.py`, `test_models_resolve.py`,
`test_models_cli.py`, `test_update_download.py`, `test_update_cli.py`.
"""

import os

# The harness runs pytest on the mutated files; without this a restore that lands in the same
# second as the mutation (same-size mutants) can leave stale bytecode in `__pycache__`.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

MUTANTS = [
    # ---------------------------------------------------------------- models/store.py (0-14)
    (
        "src/omniscan/models/store.py",
        "return models_dir / entry.id",
        'return models_dir / (entry.id + "-wrong")',
        "store: zip install folder renamed",
    ),
    (
        "src/omniscan/models/store.py",
        'if entry.format == "file" and entry.install_path is not None:',
        'if entry.format == "file" and entry.install_path is None:',
        "store: flip install_path guard",
    ),
    (
        "src/omniscan/models/store.py",
        'if entry.format == "cloud":',
        'if entry.format != "cloud":',
        "store: flip cloud branch",
    ),
    (
        "src/omniscan/models/store.py",
        'if ollama_names is None:\n            return "unknown"',
        'if ollama_names is None:\n            return "missing"',
        "store: unknown -> missing (daemon unreachable)",
    ),
    (
        "src/omniscan/models/store.py",
        'return "installed" if tag is not None and tag in ollama_names else "missing"',
        'return "installed" if tag is not None or tag in ollama_names else "missing"',
        "store: and -> or in ollama tag check",
    ),
    (
        "src/omniscan/models/store.py",
        'if not path.is_dir():\n            return "missing"',
        'if path.is_dir():\n            return "missing"',
        "store: flip is_dir check (zip)",
    ),
    (
        "src/omniscan/models/store.py",
        'return "installed" if data.get("sha256") == entry.sha256 else "corrupt"',
        'return "installed" if data.get("sha256") != entry.sha256 else "corrupt"',
        "store: flip marker sha comparison",
    ),
    (
        "src/omniscan/models/store.py",
        'if not path.is_file():\n        return "missing"',
        'if path.is_file():\n        return "missing"',
        "store: flip is_file check (file)",
    ),
    (
        "src/omniscan/models/store.py",
        "if entry.sha256 is None or entry.bytes is None or path.stat().st_size != entry.bytes:",
        "if entry.sha256 is None or entry.bytes is None or path.stat().st_size == entry.bytes:",
        "store: flip size comparison (file)",
    ),
    (
        "src/omniscan/models/store.py",
        'return "installed" if file_sha256(path) == entry.sha256 else "corrupt"',
        'return "installed" if file_sha256(path) != entry.sha256 else "corrupt"',
        "store: flip file hash comparison",
    ),
    (
        "src/omniscan/models/store.py",
        "e.id == entry_id",
        "e.id != entry_id",
        "store: flip entry-id lookup",
    ),
    (
        "src/omniscan/models/store.py",
        'if path is None or model_status(entry, cfg.paths.models_dir, ollama_names=None) != "installed":',
        'if path is None or model_status(entry, cfg.paths.models_dir, ollama_names=None) == "installed":',
        "store: flip installed gate in resolve_model_path",
    ),
    (
        "src/omniscan/models/store.py",
        "entries = catalog if catalog is not None else load_catalog()",
        "entries = catalog if catalog is None else load_catalog()",
        "store: swap explicit-catalog branch",
    ),
    (
        "src/omniscan/models/store.py",
        'if path is None:  # unreachable for a validated catalog entry\n        return "corrupt"',
        'if path is not None:  # unreachable for a validated catalog entry\n        return "corrupt"',
        "store: flip no-path branch",
    ),
    (
        "src/omniscan/models/store.py",
        "except OSError, ValueError:",
        "except OSError:",
        "store: narrow marker error catch",
    ),
    # ---------------------------------------------------------------- models/download.py (15-34)
    (
        "src/omniscan/models/download.py",
        'if entry.format == "cloud":\n        raise ModelDownloadError(f"{entry.id} runs on Ollama Cloud, nothing to download")',
        'if entry.format != "cloud":\n        raise ModelDownloadError(f"{entry.id} runs on Ollama Cloud, nothing to download")',
        "download: flip cloud guard",
    ),
    (
        "src/omniscan/models/download.py",
        'if model_status(entry, models_dir, ollama_names=None) == "installed":\n        return "already installed"',
        'if model_status(entry, models_dir, ollama_names=None) != "installed":\n        return "already installed"',
        "download: flip already-installed check",
    ),
    (
        "src/omniscan/models/download.py",
        'for source in ("mirror", "upstream"):\n            try:',
        'for source in ("mirror", "mirror"):\n            try:',
        "download: zip upstream source dropped",
    ),
    (
        "src/omniscan/models/download.py",
        'for source in ("mirror", "upstream"):\n            url = entry.mirror_url if source == "mirror" else entry.upstream_url',
        'for source in ("upstream", "mirror"):\n            url = entry.mirror_url if source == "mirror" else entry.upstream_url',
        "download: file sources upstream-first",
    ),
    (
        "src/omniscan/models/download.py",
        "_stream_to_part(entry, http, entry.mirror_url, part, on_progress)",
        "_stream_to_part(entry, http, entry.upstream_url, part, on_progress)",
        "download: zip mirror URL -> upstream_url (None)",
    ),
    (
        "src/omniscan/models/download.py",
        'return "mirror"',
        'return "upstream"',
        "download: zip mirror source label",
    ),
    (
        "src/omniscan/models/download.py",
        'if response.status_code != 200:\n                raise _SourceError(f"HTTP {response.status_code}")',
        'if response.status_code == 200:\n                raise _SourceError(f"HTTP {response.status_code}")',
        "download: flip HTTP-200 gate",
    ),
    (
        "src/omniscan/models/download.py",
        "if length is not None and length.isdigit():\n                total = int(length)",
        "if length is not None or length.isdigit():\n                total = int(length)",
        "download: and -> or in content-length check",
    ),
    (
        "src/omniscan/models/download.py",
        "if entry.sha256 is not None and digest.hexdigest() != entry.sha256:",
        "if entry.sha256 is not None or digest.hexdigest() != entry.sha256:",
        "download: and -> or in sha gate",
    ),
    (
        "src/omniscan/models/download.py",
        "if entry.bytes is not None and done != entry.bytes:",
        "if entry.bytes is not None and done == entry.bytes:",
        "download: flip size comparison",
    ),
    (
        "src/omniscan/models/download.py",
        "done += len(chunk)",
        "done += 1",
        "download: progress counts chunks not bytes",
    ),
    (
        "src/omniscan/models/download.py",
        "if dest != root and root not in dest.parents:",
        "if dest != root or root not in dest.parents:",
        "download: and -> or in zip-slip containment",
    ),
    (
        "src/omniscan/models/download.py",
        "for member in zf.infolist():  # validate everything before extracting anything\n                _check_zip_member(entry, member, models_dir)",
        "for member in zf.infolist():  # validate everything before extracting anything\n                pass",
        "download: drop zip-slip member scan",
    ),
    (
        "src/omniscan/models/download.py",
        "zf.extractall(models_dir)",
        "zf.extractall(root)",
        "download: extract into model subfolder",
    ),
    (
        "src/omniscan/models/download.py",
        "hf_download(entry.upstream_repo, revision=entry.upstream_revision, local_dir=str(folder))",
        "hf_download(entry.upstream_repo, local_dir=str(folder))",
        "download: drop upstream revision",
    ),
    (
        "src/omniscan/models/download.py",
        'if "error" in data:',
        'if "error" not in data:',
        "download: flip ollama error-line check",
    ),
    (
        "src/omniscan/models/download.py",
        "if on_progress is not None and completed is not None:",
        "if on_progress is not None or completed is not None:",
        "download: and -> or in ollama progress gate",
    ),
    (
        "src/omniscan/models/download.py",
        "return response.status_code == 200",
        "return response.status_code != 200",
        "download: flip ollama delete verdict",
    ),
    (
        "src/omniscan/models/download.py",
        "if parent.resolve() != models_dir.resolve() and not any(parent.iterdir()):",
        "if parent.resolve() == models_dir.resolve() and not any(parent.iterdir()):",
        "download: flip empty-parent cleanup guard",
    ),
    (
        "src/omniscan/models/download.py",
        "part.unlink(missing_ok=True)  # never leave a .part behind, also on failure",
        "pass  # never leave a .part behind, also on failure",
        "download: drop .part cleanup on failure",
    ),
    # ---------------------------------------------------------------- update/download.py (35-54)
    (
        "src/omniscan/update/download.py",
        "checksums[match.group(2)] = match.group(1).lower()",
        "checksums[match.group(1)] = match.group(2).lower()",
        "update dl: swap name and digest",
    ),
    (
        "src/omniscan/update/download.py",
        'if not stripped or stripped.startswith("#"):\n            continue',
        'if not stripped and stripped.startswith("#"):\n            continue',
        "update dl: or -> and in comment skip",
    ),
    (
        "src/omniscan/update/download.py",
        "sha256 = checksums[asset.name]",
        'sha256 = checksums.get(asset.name, "")',
        "update dl: missing entry -> empty digest",
    ),
    (
        "src/omniscan/update/download.py",
        r'_CHECKSUM_LINE_RE = re.compile(r"^([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$")',
        r'_CHECKSUM_LINE_RE = re.compile(r"^([0-9A-Fa-f]{63})\s+\*?(.+?)\s*$")',
        "update dl: digest length 64 -> 63",
    ),
    (
        "src/omniscan/update/download.py",
        "if release is None or compare_versions(release.version, current) <= 0:",
        "if release is None or compare_versions(release.version, current) < 0:",
        "update dl: flip <= to < on update check",
    ),
    (
        "src/omniscan/update/download.py",
        "current = current_version() if current is None else current",
        "current = current_version() if current is not None else current",
        "update dl: swap current-version default",
    ),
    (
        "src/omniscan/update/download.py",
        "if not (target.is_file() and _file_sha256(target) == sha256):",
        "if not (target.is_file() or _file_sha256(target) == sha256):",
        "update dl: and -> or in reuse check",
    ),
    (
        "src/omniscan/update/download.py",
        "tag_dir = updates_dir / release.tag",
        "tag_dir = updates_dir / str(release.version)",
        "update dl: stage under version not tag",
    ),
    (
        "src/omniscan/update/download.py",
        "if asset.name not in checksums:",
        "if asset.name in checksums:",
        "update dl: flip missing-entry check",
    ),
    (
        "src/omniscan/update/download.py",
        'with client.stream("GET", asset.url, follow_redirects=True) as response:\n            if response.status_code >= 400:',
        'with client.stream("GET", asset.url, follow_redirects=True) as response:\n            if response.status_code > 400:',
        "update dl: >= 400 -> > 400 in staging",
    ),
    (
        "src/omniscan/update/download.py",
        "if digest.hexdigest() != expected:",
        "if digest.hexdigest() == expected:",
        "update dl: flip checksum verdict",
    ),
    (
        "src/omniscan/update/download.py",
        "part.replace(tag_dir / asset.name)  # os.replace: atomic on the same volume",
        'part.replace(tag_dir / (asset.name + ".staged"))  # os.replace: atomic on the same volume',
        "update dl: stage under wrong name",
    ),
    (
        "src/omniscan/update/download.py",
        "part.unlink(missing_ok=True)  # .part files never remain",
        "pass  # .part files never remain",
        "update dl: drop .part cleanup",
    ),
    (
        "src/omniscan/update/download.py",
        "on_progress(done, total)",
        "on_progress(total, done)",
        "update dl: swap progress args",
    ),
    (
        "src/omniscan/update/download.py",
        "total = int(length) if length is not None and length.isdigit() else None",
        "total = int(length) if length is not None or length.isdigit() else None",
        "update dl: and -> or in content-length check",
    ),
    (
        "src/omniscan/update/download.py",
        'raise UpdateError(f"cannot reach GitHub: {exc}") from exc\n    if response.status_code >= 400:\n        raise UpdateError(f"HTTP {response.status_code} while downloading {asset.name}")',
        'raise UpdateError(f"cannot reach GitHub: {exc}") from exc\n    if response.status_code <= 400:\n        raise UpdateError(f"HTTP {response.status_code} while downloading {asset.name}")',
        "update dl: flip fetch status check",
    ),
    (
        "src/omniscan/update/download.py",
        "fh.write(chunk)\n                    digest.update(chunk)\n                    done += len(chunk)",
        "fh.write(chunk)\n                    pass\n                    done += len(chunk)",
        "update dl: drop digest update",
    ),
    (
        "src/omniscan/update/download.py",
        '"version": str(release.version),',
        '"version": release.tag,',
        "update dl: staged.json version from tag",
    ),
    (
        "src/omniscan/update/download.py",
        "version=release.version, tag=release.tag, path=target, sha256=sha256, asset=asset.name",
        "version=release.version, tag=release.tag, path=target, sha256=asset.name, asset=sha256",
        "update dl: swap sha256/asset in result",
    ),
    (
        "src/omniscan/update/download.py",
        "key = platform_key() if key is None else key",
        "key = platform_key() if key is not None else key",
        "update dl: swap platform-key default",
    ),
    # ---------------------------------------------------------------- round 2 (55-59)
    (
        "src/omniscan/models/download.py",
        'url = entry.mirror_url if source == "mirror" else entry.upstream_url',
        'url = entry.upstream_url if source == "mirror" else entry.mirror_url',
        "download: file mirror/upstream URLs swapped",
    ),
    (
        "src/omniscan/models/download.py",
        '"source": source,\n        "revision": entry.upstream_revision,',
        '"source": source,\n        "revision": None,',
        "download: marker drops revision",
    ),
    (
        "src/omniscan/models/download.py",
        'finally:\n        if client is None:\n            http.close()\n    return "ollama"',
        'finally:\n        if client is None:\n            http.close()\n    return "unknown"',
        "download: ollama stream end -> unknown",
    ),
    (
        "src/omniscan/models/store.py",
        "return digest.hexdigest()",
        "return digest.hexdigest().upper()",
        "store: file hash upper-cased",
    ),
    (
        "src/omniscan/update/download.py",
        'tag_dir / "staged.json"',
        'tag_dir / "staged.jsonx"',
        "update dl: staged.json renamed",
    ),
]
