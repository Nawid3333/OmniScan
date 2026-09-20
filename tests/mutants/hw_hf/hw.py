"""Hand-written mutants for `hw/assess.py` + `hw/detect.py` (card Q5).

Covers the level/device/messages rules, the disk-space warning, the vendor/integrated
heuristics, GPU sorting, torch-build detection and the disk probe.
Tests: `test_hw_assess.py`, `test_hw_detect.py` (plus `test_hw_hf_mutation_gaps.py`).
"""

import os

# The harness runs pytest on the mutated files; without this a restore that lands in the same
# second as the mutation (same-size mutants) can leave stale bytecode in `__pycache__`.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

MUTANTS = [
    # ---------------------------------------------------------------- assess.py
    (
        "src/omniscan/hw/assess.py",
        '_LEVEL_ORDER: dict[Level, int] = {"ok": 0, "slow": 1, "warn": 2, "incompatible": 3}',
        '_LEVEL_ORDER: dict[Level, int] = {"ok": 0, "slow": 1, "warn": 1, "incompatible": 3}',
        "assess: level order warn==slow",
    ),
    (
        "src/omniscan/hw/assess.py",
        '_CPU_LEVEL: dict[str, Level] = {\n    "fast": "ok",\n    "ok": "slow",',
        '_CPU_LEVEL: dict[str, Level] = {\n    "fast": "slow",\n    "ok": "ok",',
        "assess: swap first two CPU levels",
    ),
    (
        "src/omniscan/hw/assess.py",
        '"unusable": "incompatible",',
        '"unusable": "warn",',
        "assess: unusable CPU maps to warn",
    ),
    (
        "src/omniscan/hw/assess.py",
        'if entry.format == "cloud":  # served by Ollama Cloud, never bound to this machine',
        'if entry.format != "cloud":  # served by Ollama Cloud, never bound to this machine',
        "assess: flip cloud gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if entry.notes:\n            messages.append(entry.notes)",
        "if not entry.notes:\n            messages.append(entry.notes)",
        "assess: flip cloud-notes gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if integrated_only and candidate is not None:",
        "if integrated_only or candidate is not None:",
        "assess: or in integrated-message gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        "candidate.vram_gb >= entry.min_vram_gb",
        "candidate.vram_gb > entry.min_vram_gb",
        "assess: flip >= to > on VRAM",
    ),
    (
        "src/omniscan/hw/assess.py",
        'device = "ollama" if entry.format == "ollama" else candidate.device',
        'device = "ollama" if entry.format == "ollama" else "cpu"',
        "assess: ok device reported as cpu",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if candidate is not None:  # the GPU is there but too small",
        "if candidate is None:  # the GPU is there but too small",
        "assess: flip too-small gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        'f"needs {entry.min_vram_gb:g} GB of GPU memory, your {candidate.name} has {candidate.vram_gb:g} GB"',
        'f"needs {entry.min_vram_gb:g} GB of GPU memory, your {candidate.name} has {entry.min_vram_gb:g} GB"',
        "assess: VRAM message shows the requirement",
    ),
    (
        "src/omniscan/hw/assess.py",
        'if (entry.backends and "cpu" not in entry.backends) or not entry.cpu_ok:',
        'if (entry.backends and "cpu" not in entry.backends) and not entry.cpu_ok:',
        "assess: or -> and in CPU-refusal gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        'if candidate is None:\n            messages.append("needs a supported GPU (cuda/rocm/mps); none found")',
        'if candidate is not None:\n            messages.append("needs a supported GPU (cuda/rocm/mps); none found")',
        "assess: flip none-found guard",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if entry.min_ram_gb is not None and hw.ram_gb < entry.min_ram_gb:",
        "if entry.min_ram_gb is not None and hw.ram_gb <= entry.min_ram_gb:",
        "assess: flip < to <= on RAM",
    ),
    (
        "src/omniscan/hw/assess.py",
        "level = _CPU_LEVEL[entry.cpu_speed]",
        'level = _CPU_LEVEL["slow"]',
        "assess: CPU level ignores the entry",
    ),
    (
        "src/omniscan/hw/assess.py",
        'device = "ollama" if entry.format == "ollama" else "cpu"',
        'device = "ollama" if entry.format == "ollama" else "ollama"',
        "assess: CPU device reported as ollama",
    ),
    (
        "src/omniscan/hw/assess.py",
        'if level == "incompatible":\n        return Compatibility(level, None, (*messages, message))',
        'if level != "incompatible":\n        return Compatibility(level, None, (*messages, message))',
        "assess: flip unusable-CPU gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        "allowed = set(entry.backends) or None",
        "allowed = set(entry.backends) and None",
        "assess: backend filter dropped",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if not gpu.integrated and (allowed is None or gpu.backend in allowed):",
        "if not gpu.integrated or (allowed is None or gpu.backend in allowed):",
        "assess: or in discrete-GPU gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        "return gpu, False",
        "return gpu, True",
        "assess: discrete GPU flagged integrated",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if gpu.integrated and (allowed is None or gpu.backend in allowed):",
        "if not gpu.integrated and (allowed is None or gpu.backend in allowed):",
        "assess: flip integrated fallback gate",
    ),
    (
        "src/omniscan/hw/assess.py",
        "needed_gb = entry.size_mb / 1024 * 1.2",
        "needed_gb = entry.size_mb / 1024 * 1.0",
        "assess: disk factor 1.2 -> 1.0",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if hw.disk_free_gb >= needed_gb:",
        "if hw.disk_free_gb > needed_gb:",
        "assess: flip >= to > on free disk",
    ),
    (
        "src/omniscan/hw/assess.py",
        'level = compat.level if _LEVEL_ORDER[compat.level] >= _LEVEL_ORDER["warn"] else "warn"',
        'level = compat.level if _LEVEL_ORDER[compat.level] < _LEVEL_ORDER["warn"] else "warn"',
        "assess: incompatible raised to warn",
    ),
    # ---------------------------------------------------------------- detect.py
    (
        "src/omniscan/hw/detect.py",
        '("nvidia", re.compile(r"nvidia|geforce|rtx|gtx|quadro|tesla", re.IGNORECASE)),',
        '("nvidia", re.compile(r"geforce|rtx|gtx|quadro|tesla", re.IGNORECASE)),',
        "detect: drop nvidia from vendor regex",
    ),
    (
        "src/omniscan/hw/detect.py",
        '_AMD_INTEGRATED = re.compile(r"radeon\\s*\\d{3}m")',
        '_AMD_INTEGRATED = re.compile(r"radeon\\s*\\d{4}m")',
        "detect: AMD integrated needs 4 digits",
    ),
    (
        "src/omniscan/hw/detect.py",
        '_INTEL_INTEGRATED = re.compile(r"uhd|iris")',
        '_INTEL_INTEGRATED = re.compile(r"uhd")',
        "detect: drop iris from Intel integrated",
    ),
    (
        "src/omniscan/hw/detect.py",
        'if vendor == "apple":\n        return False',
        'if vendor == "apple":\n        return True',
        "detect: Apple GPU counted integrated",
    ),
    (
        "src/omniscan/hw/detect.py",
        "ram_gb = round(psutil.virtual_memory().total / GIB, 1)",
        "ram_gb = round(psutil.virtual_memory().total / GIB, 0)",
        "detect: RAM rounded to whole GB",
    ),
    (
        "src/omniscan/hw/detect.py",
        "cpu_cores_logical=psutil.cpu_count(logical=True) or 1,",
        "cpu_cores_logical=psutil.cpu_count(logical=False) or 1,",
        "detect: logical cores are physical",
    ),
    (
        "src/omniscan/hw/detect.py",
        "key=lambda gpu: (gpu.integrated, -gpu.vram_gb, gpu.index)",
        "key=lambda gpu: (gpu.integrated, gpu.vram_gb, gpu.index)",
        "detect: GPUs sorted by smallest VRAM",
    ),
    (
        "src/omniscan/hw/detect.py",
        'build: TorchBuild = "cpu"',
        'build: TorchBuild = "cuda"',
        "detect: default torch build cuda",
    ),
    (
        "src/omniscan/hw/detect.py",
        'backend: Backend = "rocm" if build == "rocm" else "cuda"',
        'backend: Backend = "cuda" if build == "rocm" else "rocm"',
        "detect: swap cuda/rocm backend pick",
    ),
    (
        "src/omniscan/hw/detect.py",
        'if build == "cpu":\n                build = "mps"',
        'if build == "mps":\n                build = "mps"',
        "detect: MPS build never promoted",
    ),
    (
        "src/omniscan/hw/detect.py",
        'except Exception:  # missing or broken: the app stays usable on the CPU paths\n        return [], "cpu"',
        'except Exception:  # missing or broken: the app stays usable on the CPU paths\n        return [], "rocm"',
        "detect: broken torch reports rocm",
    ),
    (
        "src/omniscan/hw/detect.py",
        'return str(resolve_device("auto"))',
        'return "cpu"',
        "detect: best device always cpu",
    ),
    # ---------------------------------------------------------------- round 2 (around the survivors)
    (
        "src/omniscan/hw/detect.py",
        '("amd", re.compile(r"amd|radeon|instinct", re.IGNORECASE)),',
        '("amd", re.compile(r"radeon|instinct", re.IGNORECASE)),',
        "detect: drop amd from vendor regex",
    ),
    (
        "src/omniscan/hw/detect.py",
        '_INTEL_INTEGRATED = re.compile(r"uhd|iris")',
        '_INTEL_INTEGRATED = re.compile(r"uhd|iris|arc")',
        "detect: Arc counted integrated",
    ),
    (
        "src/omniscan/hw/assess.py",
        "if hw.disk_free_gb >= needed_gb:",
        "if hw.disk_free_gb < needed_gb:",
        "assess: flip disk-gate direction",
    ),
    (
        "src/omniscan/hw/detect.py",
        "props = torch.cuda.get_device_properties(index)\n    name = str(props.name)\n    vram_gb = round(props.total_memory / GIB, 1)",
        "props = torch.cuda.get_device_properties(index)\n    name = str(props.name)\n    vram_gb = round(props.total_memory / GIB, 0)",
        "detect: GPU VRAM rounded to whole GB",
    ),
    (
        "src/omniscan/hw/detect.py",
        '        return "macos"',
        '        return "darwin"',
        "detect: darwin reported as darwin",
    ),
    (
        "src/omniscan/hw/detect.py",
        '    if machine in ("arm64", "aarch64"):\n        return "arm64"',
        '    if machine in ("arm64",):\n        return "arm64"',
        "detect: aarch64 not mapped to arm64",
    ),
    (
        "src/omniscan/hw/assess.py",
        '        messages.append(f"needs {entry.min_ram_gb:g} GB of RAM, you have {hw.ram_gb:g} GB")\n        return Compatibility("incompatible", None, tuple(messages))',
        '        messages.append(f"needs {entry.min_ram_gb:g} GB of RAM, you have {hw.ram_gb:g} GB")',
        "assess: low RAM no longer incompatible",
    ),
    (
        "src/omniscan/hw/detect.py",
        'device=f"cuda:{index}",',
        'device=f"cuda:{index + 1}",',
        "detect: cuda device index shifted",
    ),
    (
        "src/omniscan/hw/assess.py",
        "            return gpu, True\n    return None, False",
        "            return gpu, False\n    return None, False",
        "assess: integrated fallback flagged discrete",
    ),
]
