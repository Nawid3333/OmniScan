"""Hand-written mutants for the acquisition rules group (card Q3): drm.py, filters.py, sources.py."""

from __future__ import annotations

MUTANTS: list[tuple[str, str, str, str]] = [
    # src/omniscan/acquire/drm.py
    (
        "src/omniscan/acquire/drm.py",
        '        if host == domain or host.endswith("." + domain):',
        '        if host == domain and host.endswith("." + domain):',
        "flip or to and (platform match)",
    ),
    (
        "src/omniscan/acquire/drm.py",
        '        if host == domain or host.endswith("." + domain):\n            return name',
        "        if host == domain or host.endswith(domain):\n            return name",
        "suffix match drops the dot",
    ),
    (
        "src/omniscan/acquire/drm.py",
        "    host = _host(url)\n    if host is None:\n        return None\n    for domain, name in DRM_PLATFORMS.items():",
        "    host = _host(url)\n    for domain, name in DRM_PLATFORMS.items():",
        "drop None-host guard",
    ),
    (
        "src/omniscan/acquire/drm.py",
        '    "bomtoon.com": "Bomtoon",\n',
        "",
        "drop a platform entry",
    ),
    (
        "src/omniscan/acquire/drm.py",
        "    name = drm_platform(url)\n    if name is not None:",
        "    name = drm_platform(url)\n    if name is None:",
        "flip refusal guard",
    ),
    (
        "src/omniscan/acquire/drm.py",
        "import files you own with `omniscan import` instead",
        "import files you own with `omniscan-import` instead",
        "change the import hint",
    ),
    (
        "src/omniscan/acquire/drm.py",
        '    return host.removeprefix("www.")',
        '    return host.removeprefix("www")',
        "strip www without the dot",
    ),
    (
        "src/omniscan/acquire/drm.py",
        "            return name",
        "            return domain",
        "return the wrong variable",
    ),
    # src/omniscan/acquire/filters.py
    (
        "src/omniscan/acquire/filters.py",
        "    if width < 300:",
        "    if width <= 300:",
        "flip < to <= (width)",
    ),
    (
        "src/omniscan/acquire/filters.py",
        "    if height < 100:",
        "    if height <= 100:",
        "flip < to <= (height)",
    ),
    (
        "src/omniscan/acquire/filters.py",
        "    if width / height > 6.0:",
        "    if width / height >= 6.0:",
        "flip > to >= (ratio)",
    ),
    (
        "src/omniscan/acquire/filters.py",
        "    if width / height > 6.0:",
        "    if height / width > 6.0:",
        "swap width and height (ratio)",
    ),
    (
        "src/omniscan/acquire/filters.py",
        '    if size_bytes < 8000:\n        return (False, "tiny file")\n    return (True, "")',
        '    return (True, "")',
        "drop tiny-file guard",
    ),
    (
        "src/omniscan/acquire/filters.py",
        '    if urlsplit(url).scheme not in ("http", "https"):',
        '    if urlsplit(url).scheme in ("http", "https"):',
        "flip scheme guard",
    ),
    (
        "src/omniscan/acquire/filters.py",
        "    path = urlsplit(url).path.lower()",
        "    path = urlsplit(url).path",
        "drop .lower() on path",
    ),
    (
        "src/omniscan/acquire/filters.py",
        "        if token in path:",
        "        if token in url.lower():",
        "token check spans the query",
    ),
    (
        "src/omniscan/acquire/filters.py",
        '    "logo",\n    "banner",',
        '    "banner",\n    "logo",',
        "swap token order",
    ),
    (
        "src/omniscan/acquire/filters.py",
        'not in ("http", "https")',
        'not in ("https",)',
        "drop http from allowed schemes",
    ),
    # src/omniscan/acquire/sources.py
    (
        "src/omniscan/acquire/sources.py",
        'enumerate(data.get("chapter", []), start=1)',
        'enumerate(data.get("chapter", []), start=2)',
        "enumerate start 1 to 2",
    ),
    (
        "src/omniscan/acquire/sources.py",
        'data.get("chapter", [])',
        'data.get("chapter")',
        "missing-chapter default drops",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "    if first > last:",
        "    if first >= last:",
        "flip > to >= (template bounds)",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "    if last - first + 1 > MAX_TEMPLATE_CHAPTERS:",
        "    if last - first + 1 >= MAX_TEMPLATE_CHAPTERS:",
        "flip > to >= (chapter cap)",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "    if not isinstance(first, int) or not isinstance(last, int):",
        "    if not isinstance(first, int) and not isinstance(last, int):",
        "flip or to and (int check)",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "    for number in range(first, last + 1):",
        "    for number in range(first, last):",
        "loop bound drops + 1",
    ),
    (
        "src/omniscan/acquire/sources.py",
        '    name_template = template.get("name", "Chapter {n}")',
        '    name_template = template.get("name")',
        "default name template drops",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "        sources.append(ChapterSource(name=name, url=url.format(n=number)))",
        "        sources.append(ChapterSource(name=name, url=url))",
        "drop url formatting",
    ),
    (
        "src/omniscan/acquire/sources.py",
        '        url = _check_url(url, f"chapter #{index}")',
        '        url = _check_url(url, "chapter")',
        "error label drops index",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "        name = _check_name(name)\n        _check_duplicate(name, seen)",
        "        name = _check_name(name)",
        "drop explicit duplicate check",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "    seen.add(name)",
        "    pass",
        "drop seen.add",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "    if template:\n        sources.extend(_template_sources(template, seen))",
        "    if template is not None:\n        sources.extend(_template_sources(template, seen))",
        "flip template truthiness",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "        sources.extend(_template_sources(template, seen))",
        "        sources = _template_sources(template, seen)",
        "extend becomes assignment",
    ),
    (
        "src/omniscan/acquire/sources.py",
        '    return library_root / series / "sources.toml"',
        '    return library_root / "sources.toml"',
        "sources_path drops series",
    ),
    (
        "src/omniscan/acquire/sources.py",
        '    for key in ("url", "first", "last"):',
        '    for key in ("url", "first"):',
        "required-keys tuple drops last",
    ),
    (
        "src/omniscan/acquire/sources.py",
        "    if not isinstance(name_template, str):",
        "    if isinstance(name_template, str):",
        "flip name-template isinstance",
    ),
]
