"""Hand-written mutants for the downloader (card Q3): src/omniscan/acquire/download.py."""

from __future__ import annotations

MUTANTS: list[tuple[str, str, str, str]] = [
    # status codes, retries, backoff, Retry-After
    (
        "src/omniscan/acquire/download.py",
        "        if response.status_code == 200:",
        "        if response.status_code != 200:",
        "flip == 200",
    ),
    (
        "src/omniscan/acquire/download.py",
        "        retryable = response.status_code == 429 or response.status_code >= 500",
        "        retryable = response.status_code == 429 and response.status_code >= 500",
        "flip or to and (retryable)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "        retryable = response.status_code == 429 or response.status_code >= 500",
        "        retryable = response.status_code == 429 or response.status_code >= 400",
        "widen 5xx to 4xx",
    ),
    (
        "src/omniscan/acquire/download.py",
        "        retryable = response.status_code == 429 or response.status_code >= 500",
        "        retryable = response.status_code == 429 or response.status_code > 500",
        "narrow >= to > (5xx)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "            if attempt >= retries:",
        "            if attempt > retries:",
        "transport retry off-by-one",
    ),
    (
        "src/omniscan/acquire/download.py",
        "            sleep(backoff_s * 2**attempt)\n            attempt += 1\n            continue",
        "            sleep(backoff_s * 2**(attempt + 1))\n            attempt += 1\n            continue",
        "backoff exponent off-by-one (transport)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "        else:\n            sleep(backoff_s * 2**attempt)",
        "        else:\n            sleep(backoff_s + 2**attempt)",
        "multiply becomes add (5xx)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "                sleep(min(retry_after, 30))",
        "                sleep(max(retry_after, 30))",
        "swap min/max (Retry-After cap)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    except ValueError:\n        return None",
        "    except ValueError:\n        return 0",
        "unparsable Retry-After becomes 0",
    ),
    (
        "src/omniscan/acquire/download.py",
        "            if retry_after is None:",
        "            if retry_after is not None:",
        "flip Retry-After guard",
    ),
    (
        "src/omniscan/acquire/download.py",
        "        if response.status_code == 429:",
        "        if response.status_code != 429:",
        "flip 429 branch",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    retries: int = 3,",
        "    retries: int = 2,",
        "retries default 3 to 2",
    ),
    # content checks and decoding
    (
        "src/omniscan/acquire/download.py",
        "        if len(data) > MAX_IMAGE_BYTES:",
        "        if len(data) >= MAX_IMAGE_BYTES:",
        "flip > to >= (size limit)",
    ),
    (
        "src/omniscan/acquire/download.py",
        '        if len(data) > MAX_IMAGE_BYTES:\n            raise _RefError("too large")',
        '        if False:\n            raise _RefError("too large")',
        "drop too-large check",
    ),
    (
        "src/omniscan/acquire/download.py",
        "            probe.verify()\n            fmt = probe.format",
        "            fmt = probe.format",
        "drop probe.verify()",
    ),
    (
        "src/omniscan/acquire/download.py",
        "            fmt = probe.format",
        "            pass",
        "drop format capture",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    if fmt not in SUPPORTED_FORMATS:",
        "    if fmt in SUPPORTED_FORMATS:",
        "flip format check",
    ),
    (
        "src/omniscan/acquire/download.py",
        "        ok, reason = image_looks_like_page(width, height, len(data))\n        if not ok:\n            return _Rejected(reason)",
        "        ok, reason = image_looks_like_page(width, height, len(data))\n        if ok:\n            return _Rejected(reason)",
        "flip not ok (content filter)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "        ok, reason = image_looks_like_page(width, height, len(data))",
        "        ok, reason = image_looks_like_page(height, width, len(data))",
        "swap width/height (content filter)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    if apply_filters:\n        ok, reason = image_looks_like_page(width, height, len(data))",
        "    if True:\n        ok, reason = image_looks_like_page(width, height, len(data))",
        "drop apply_filters guard (content)",
    ),
    # position numbering
    (
        "src/omniscan/acquire/download.py",
        "    digits = max(3, len(str(total)))",
        "    digits = min(3, len(str(total)))",
        "max to min (digits)",
    ),
    (
        "src/omniscan/acquire/download.py",
        '        "file": f"{position:0{digits}d}{SUPPORTED_FORMATS[fmt]}",',
        '        "file": f"{position:{digits}d}{SUPPORTED_FORMATS[fmt]}",',
        "drop zero padding",
    ),
    (
        "src/omniscan/acquire/download.py",
        "            for position, ref in enumerate(refs, start=1):",
        "            for position, ref in enumerate(refs, start=0):",
        "position start 1 to 0",
    ),
    # resume and filters-off behaviour
    (
        "src/omniscan/acquire/download.py",
        "                if carried is not None and _file_matches(dest, carried):",
        "                if carried is not None or _file_matches(dest, carried):",
        "flip and to or (resume)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "                carried = old_images.get(key)",
        "                carried = old_rejected.get(key)",
        "read carried from wrong dict",
    ),
    (
        "src/omniscan/acquire/download.py",
        '        return (dest / str(entry["file"])).stat().st_size == int(entry["bytes"])',
        '        return (dest / str(entry["file"])).stat().st_size < int(entry["bytes"])',
        "flip == to < (resume size)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    except OSError:\n        return False",
        "    except OSError:\n        return True",
        "missing file counts as matching",
    ),
    (
        "src/omniscan/acquire/download.py",
        "                elif (\n                    apply_filters and key in old_rejected",
        "                elif (\n                    apply_filters or key in old_rejected",
        "flip and to or (carried rejection)",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    if not apply_filters:\n        return None",
        "    if False:\n        return None",
        "drop filters-off guard (url)",
    ),
    # manifest and result
    (
        "src/omniscan/acquire/download.py",
        '        "images": sorted(images, key=lambda entry: entry["position"]),',
        '        "images": list(images),',
        "unsorted manifest images",
    ),
    (
        "src/omniscan/acquire/download.py",
        '    files = [str(entry["file"]) for entry in sorted(images, key=lambda entry: entry["position"])]',
        '    files = [str(entry["file"]) for entry in images]',
        "unsorted result files",
    ),
    (
        "src/omniscan/acquire/download.py",
        '        "rejected": sorted(rejected, key=lambda entry: entry["position"]),',
        '        "rejected": list(rejected),',
        "unsorted manifest rejected",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    return DownloadResult(files=files, downloaded=downloaded, skipped=skipped, rejected=rejected_count)",
        "    return DownloadResult(files=files, downloaded=skipped, skipped=downloaded, rejected=rejected_count)",
        "swap downloaded/skipped",
    ),
    (
        "src/omniscan/acquire/download.py",
        '                elif isinstance(outcome, _Rejected):\n                    rejected.append({"position": position, "url": ref.url, "reason": outcome.reason})\n                    rejected_count += 1',
        '                elif isinstance(outcome, _Rejected):\n                    rejected.append({"position": position, "url": ref.url, "reason": outcome.reason})',
        "drop rejected count",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    part.replace(dest / name)",
        "    pass",
        "skip the .part replace",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    part.replace(path)",
        "    pass",
        "skip the manifest replace",
    ),
    # second round (Q3 review)
    (
        "src/omniscan/acquire/download.py",
        "    backoff_s: float = 0.5,",
        "    backoff_s: float = 0.05,",
        "backoff default 0.5 to 0.05",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    dest.mkdir(parents=True, exist_ok=True)",
        "    pass",
        "drop dest.mkdir",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    progress: Callable[[int, int], None] = on_progress if on_progress is not None else _noop",
        "    progress: Callable[[int, int], None] = _noop",
        "drop on_progress wiring",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    for ref in refs:\n        check_allowed(ref.url)",
        "    for ref in refs:\n        pass",
        "drop the DRM scan",
    ),
    (
        "src/omniscan/acquire/download.py",
        '        raise AcquireError(f"{len(failures)} of {total} image(s) failed: {detail}")',
        '        raise AcquireError(f"{len(failures)} of {len(failures)} image(s) failed: {detail}")',
        "total becomes len(failures) in the error",
    ),
    (
        "src/omniscan/acquire/download.py",
        '        detail = "; ".join(f"#{position} {message}" for position, message in sorted(failures))',
        '        detail = "; ".join(f"#{position} {message}" for position, message in failures)',
        "unsorted failure detail",
    ),
    (
        "src/omniscan/acquire/download.py",
        '    if ref.referer:\n        headers["Referer"] = ref.referer',
        '    if True:\n        headers["Referer"] = ref.referer',
        "drop referer guard",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    old_images, old_rejected = _load_manifest(manifest_path)",
        "    old_rejected, old_images = _load_manifest(manifest_path)",
        "swap images/rejected on load",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    total = len(refs)",
        "    total = len(refs) + 1",
        "total + 1",
    ),
    (
        "src/omniscan/acquire/download.py",
        "    if failures:",
        "    if not failures:",
        "flip failures guard",
    ),
]
