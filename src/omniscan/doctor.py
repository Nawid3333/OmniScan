"""`omniscan doctor` — machine checks: Python, GPU/torch, ROCm, rocJPEG, Ollama, secrets, paths, models."""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from omniscan.core.config import Config, Secrets
from omniscan.models.catalog import load_catalog
from omniscan.models.store import model_status

Status = Literal["OK", "WARN", "FAIL"]

REQUIRED_OLLAMA_MODELS: tuple[str, ...] = (
    "translategemma:12b",
    "gemma4:12b",
    "gemma4:31b-cloud",
    "glm-5.3-flash:cloud",
)

_ROCJPEG_CANDIDATES = (
    "/opt/rocm/core-10.0/lib/librocjpeg.so.1",
    "/opt/rocm/lib/librocjpeg.so.1",
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: Status
    detail: str


def check_python() -> CheckResult:
    """Check the interpreter is Python 3.14."""
    if sys.version_info[:2] == (3, 14):
        return CheckResult("python", "OK", ".".join(map(str, sys.version_info[:3])))
    version = ".".join(map(str, sys.version_info[:3]))
    return CheckResult("python", "FAIL", f"{version} (need 3.14)")


def check_torch_gpu() -> CheckResult:
    """Check ROCm torch sees the GPU."""
    try:
        import torch
    except Exception as exc:
        return CheckResult("torch_gpu", "FAIL", f"{type(exc).__name__}: {exc}")
    if "rocm" not in torch.__version__:
        return CheckResult("torch_gpu", "FAIL", f"torch {torch.__version__} is not a ROCm build")
    if not torch.cuda.is_available():
        return CheckResult("torch_gpu", "FAIL", f"torch {torch.__version__} but cuda is not available")
    from omniscan.core.config import get_config
    from omniscan.gpu.device import resolve_device

    device = resolve_device(get_config().gpu.device)
    if device.type != "cuda":
        return CheckResult("torch_gpu", "FAIL", f"no usable discrete GPU (device resolves to {device})")
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    name = props.name
    arch = getattr(props, "gcnArchName", "?")
    free, total = torch.cuda.mem_get_info(index)
    free_gib, total_gib = free / 2**30, total / 2**30
    detail = f"{torch.__version__} | {name} | {arch} | {free_gib:.1f}/{total_gib:.1f} GiB free | cuda:{index}"
    return CheckResult("torch_gpu", "OK", detail)


def check_rocm() -> CheckResult:
    """Check `rocminfo` reports a gfx1201 agent (Linux/WSL only; on Windows torch_gpu covers the GPU)."""
    if sys.platform == "win32":
        return CheckResult("rocm", "OK", "not applicable on Windows (torch_gpu covers the GPU)")
    try:
        proc = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return CheckResult("rocm", "FAIL", "rocminfo not found")
    except subprocess.TimeoutExpired:
        return CheckResult("rocm", "FAIL", "rocminfo timed out after 30s")
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        detail = stderr[0] if stderr else f"rocminfo exited {proc.returncode}"
        return CheckResult("rocm", "FAIL", detail)
    agents = [line.strip() for line in proc.stdout.splitlines() if "gfx" in line and "Name:" in line]
    if any("gfx1201" in line for line in proc.stdout.splitlines()):
        return CheckResult("rocm", "OK", "gfx1201 agent found")
    if agents:
        return CheckResult("rocm", "WARN", agents[0].removeprefix("Name:").strip())
    return CheckResult("rocm", "WARN", "no gfx agent")


def check_rocjpeg() -> CheckResult:
    """Check the rocJPEG hardware decoder (WARN when unavailable — WSL has no VCN access today)."""
    path = next((p for p in _ROCJPEG_CANDIDATES if Path(p).exists()), None)
    if path is None:
        path = ctypes.util.find_library("rocjpeg")
    if not path:
        return CheckResult("rocjpeg", "WARN", "librocjpeg not found")
    try:
        lib = ctypes.CDLL(path)
        lib.rocJpegCreate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        lib.rocJpegCreate.restype = ctypes.c_int
        lib.rocJpegDestroy.argtypes = [ctypes.c_void_p]
        handle = ctypes.c_void_p()
        status = lib.rocJpegCreate(0, 0, ctypes.byref(handle))
        if status == 0:
            lib.rocJpegDestroy(handle)
            return CheckResult("rocjpeg", "OK", "hardware decoder available")
        return CheckResult(
            "rocjpeg",
            "WARN",
            f"hardware backend unavailable (status {status}); hybrid codec will be used",
        )
    except OSError as exc:
        return CheckResult("rocjpeg", "WARN", f"librocjpeg not loadable: {exc}")


def check_ollama_local(cfg: Config, http: httpx.Client) -> CheckResult:
    """Check the local Ollama server answers /api/version."""
    url = f"{cfg.ollama.local_url}/api/version"
    try:
        resp = http.get(url)
        resp.raise_for_status()
        version = resp.json()["version"]
    except httpx.HTTPError as exc:
        return CheckResult("ollama_local", "FAIL", f"unreachable at {url}: {exc}")
    except (KeyError, ValueError) as exc:
        return CheckResult("ollama_local", "FAIL", f"unexpected response at {url}: {exc}")
    return CheckResult("ollama_local", "OK", f"ollama {version}")


def check_ollama_models(cfg: Config, http: httpx.Client) -> CheckResult:
    """Check the local Ollama has every required model."""
    url = f"{cfg.ollama.local_url}/api/tags"
    try:
        resp = http.get(url)
        resp.raise_for_status()
        models = resp.json()["models"]
    except httpx.HTTPError as exc:
        return CheckResult("ollama_models", "FAIL", f"unreachable at {url}: {exc}")
    except (KeyError, ValueError) as exc:
        return CheckResult("ollama_models", "FAIL", f"unexpected response at {url}: {exc}")
    present = {model["name"] for model in models}
    missing = [name for name in REQUIRED_OLLAMA_MODELS if name not in present]
    if missing:
        return CheckResult("ollama_models", "WARN", f"missing: {', '.join(missing)}")
    return CheckResult("ollama_models", "OK", f"{len(models)} models, all required present")


def check_ollama_cloud(cfg: Config, secrets: Secrets, http: httpx.Client) -> CheckResult:
    """Check the Ollama cloud API key (only when one is set)."""
    key = secrets.ollama_api_key
    if key is None:
        return CheckResult(
            "ollama_cloud",
            "WARN",
            "OLLAMA_API_KEY not set (cloud models still work through the local signed-in Ollama)",
        )
    url = f"{cfg.ollama.cloud_url}/api/tags"
    headers = {"Authorization": f"Bearer {key.get_secret_value()}"}
    try:
        resp = http.get(url, headers=headers)
        if resp.status_code in (401, 403):
            return CheckResult("ollama_cloud", "FAIL", "cloud API key rejected")
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return CheckResult("ollama_cloud", "WARN", f"{type(exc).__name__}: {exc}")
    return CheckResult("ollama_cloud", "OK", "cloud API key valid")


def check_secrets(secrets: Secrets) -> CheckResult:
    """Check optional secrets; never prints values."""
    if secrets.ollama_api_key is None:
        return CheckResult("secrets", "WARN", "not set: OLLAMA_API_KEY")
    return CheckResult("secrets", "OK", "all optional secrets set")


def check_paths(cfg: Config) -> CheckResult:
    """Check pipeline roots exist; WARN on the ones that will be created on first use."""
    roots = {
        "library_root": cfg.paths.library_root,
        "work_root": cfg.paths.work_root,
        "output_root": cfg.paths.output_root,
    }
    missing = [name for name, path in roots.items() if not path.exists()]
    if missing:
        result = CheckResult(
            "paths",
            "WARN",
            f"will be created on first use: {', '.join(missing)}",
        )
    else:
        result = CheckResult("paths", "OK", ", ".join(f"{n}={p}" for n, p in roots.items()))
    anchor = roots["work_root"]
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free / 2**30
    return CheckResult(
        result.name,
        result.status,
        f"{result.detail}; {free:.0f} GiB free",
    )


def check_models(cfg: Config) -> CheckResult:
    """Check every required catalog model is installed in models_dir (the pipeline prefers those)."""
    try:
        entries = [entry for entry in load_catalog() if entry.required]
        missing = [
            entry
            for entry in entries
            if model_status(entry, cfg.paths.models_dir, ollama_names=None) != "installed"
        ]
    except (OSError, ValueError) as exc:
        return CheckResult("models", "WARN", f"{type(exc).__name__}: {exc}")
    if not missing:
        return CheckResult("models", "OK", f"{len(entries)} required models installed")
    total_mb = sum(entry.size_mb for entry in missing)
    ids = ", ".join(entry.id for entry in missing)
    return CheckResult(
        "models",
        "WARN",
        f"{len(missing)} required model(s) not installed ({total_mb} MB): {ids}; "
        'run "omniscan models download --required" (until then they are loaded from the Hugging Face hub/cache)',
    )


def check_codec(cfg: Config) -> CheckResult:
    """Report the configured codec (real probe comes in card C2)."""
    return CheckResult("codec", "OK", f"configured: {cfg.gpu.codec}")


_CHECKS = (
    ("python", lambda cfg, secrets, http: check_python()),
    ("torch_gpu", lambda cfg, secrets, http: check_torch_gpu()),
    ("rocm", lambda cfg, secrets, http: check_rocm()),
    ("rocjpeg", lambda cfg, secrets, http: check_rocjpeg()),
    ("ollama_local", lambda cfg, secrets, http: check_ollama_local(cfg, http)),
    ("ollama_models", lambda cfg, secrets, http: check_ollama_models(cfg, http)),
    ("ollama_cloud", lambda cfg, secrets, http: check_ollama_cloud(cfg, secrets, http)),
    ("secrets", lambda cfg, secrets, http: check_secrets(secrets)),
    ("paths", lambda cfg, secrets, http: check_paths(cfg)),
    ("models", lambda cfg, secrets, http: check_models(cfg)),
    ("codec", lambda cfg, secrets, http: check_codec(cfg)),
)


def run_all_checks(cfg: Config, secrets: Secrets, http: httpx.Client | None = None) -> list[CheckResult]:
    """Run every check in the order listed above. If `http` is None, create one httpx.Client(timeout=5.0)
    and close it at the end. A check that raises an unexpected exception becomes FAIL with detail
    f"{type(exc).__name__}: {exc}" — run_all_checks itself never raises."""
    own_http = http is None
    client = httpx.Client(timeout=5.0) if own_http else http
    results: list[CheckResult] = []
    try:
        for name, fn in _CHECKS:
            try:
                results.append(fn(cfg, secrets, client))
            except Exception as exc:
                results.append(CheckResult(name, "FAIL", f"{type(exc).__name__}: {exc}"))
    finally:
        if own_http and client is not None:
            client.close()
    return results
