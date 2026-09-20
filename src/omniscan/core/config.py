"""OmniScan configuration contract.

Load order (later wins): built-in defaults -> config/default.toml (repo)
-> ~/.config/omniscan/config.toml (user) -> env vars `OMNISCAN_<SECTION>__<KEY>`.
Secrets come only from the environment or ~/.config/omniscan/secrets.env.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
USER_CONFIG_DIR = Path.home() / ".config" / "omniscan"
DEFAULT_TOML = REPO_ROOT / "config" / "default.toml"
USER_TOML = USER_CONFIG_DIR / "config.toml"
SECRETS_ENV = USER_CONFIG_DIR / "secrets.env"


class PathsConfig(BaseModel):
    library_root: Path = Path.home() / "omniscan" / "library"
    work_root: Path = Path.home() / "omniscan" / "work"
    output_root: Path = Path.home() / "omniscan" / "output"
    promo_examples: Path = Path.home() / "omniscan" / "promo_examples"
    models_dir: Path = REPO_ROOT / "models"


class GpuConfig(BaseModel):
    device: str = "auto"  # auto | cpu | mps | cuda | cuda:N (auto = strongest discrete GPU)
    vram_budget_gib: float = 14.5
    codec: Literal["auto", "rocjpeg", "hybrid", "turbo"] = "auto"
    warmup: bool = (
        True  # initialise the GPU libraries (convolution, FFT, GEMM) on a background thread at start-up
    )


class SlicerConfig(BaseModel):
    band_min_px: int = 50
    target_height: int = 3000
    min_height: int = 1500
    max_height: int = 6000
    hard_max_height: int = 15000
    uniform_tol: int = 10
    max_drift: float = 2.0
    strategy: Literal["smart", "page", "fixed", "simple_gutter"] = (
        "smart"  # see docs/PRODUCT_SPEC.md section 3
    )
    gutter_variance: float = (
        5.0  # simple_gutter: a row is a gutter when its per-channel variance is below this
    )
    gutter_min_rows: int = 8  # simple_gutter: gutters thinner than this many rows are ignored


class DetectConfig(BaseModel):
    repo: str = "ogkalu/comic-text-and-bubble-detector"  # RT-DETR-v2: bubble / text_bubble / text_free
    revision: str | None = None  # pin a HF commit hash once a model has been validated
    threshold: float = 0.3  # minimum detector score kept before merging
    tile_px: int = 1280  # tile side in strip pixels (capped at the strip width); resized to 640 for the model
    overlap: float = (
        0.5  # fraction of a tile shared with its neighbour (every object <= tile*overlap fits whole)
    )
    batch_size: int = 8  # tiles per forward pass
    nms_iou: float = 0.5  # same-class IoU above which the lower-scored box is dropped
    contain_thr: float = 0.85  # a tile-edge box mostly inside a same-class box is dropped
    edge_penalty: float = 0.15  # score penalty for boxes cut by an internal tile edge
    merge_bubble_text: bool = True  # several text boxes inside one bubble become one region
    reading_direction: Literal["ltr", "rtl"] = "ltr"  # order of regions within a row


class OllamaConfig(BaseModel):
    local_url: str = "http://localhost:11434"
    cloud_url: str = "https://ollama.com"
    request_timeout_s: float = 600.0


class RelayConfig(BaseModel):
    url: str = ""  # e.g. https://omniscan-relay.<account>.workers.dev


class OcrConfig(BaseModel):
    det_repo: str = "PaddlePaddle/PP-OCRv5_server_det_safetensors"  # text-line detector (fp32 only, see docs/DECISIONS.md)
    det_revision: str | None = None  # pin a HF commit hash once validated
    rec_repo: str = (
        "PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors"  # recognition model of the source language
    )
    rec_revision: str | None = None
    tile_px: int = 1280  # tile side in strip pixels (capped at the strip width); the detector's processor downsizes to <= 960
    overlap: float = (
        0.25  # fraction of a tile shared with its neighbour (a line no taller than tile*overlap fits whole)
    )
    det_threshold: float = 0.3  # probability-map binarisation threshold
    box_threshold: float = 0.6  # minimum mean probability inside a detected line
    unclip_ratio: float = 1.5  # DB box expansion (the map is a shrunk text kernel)
    min_size: int = 3  # smallest accepted line side, in tile pixels
    rec_batch_size: int = 64  # line crops per recognition forward pass
    line_pad_px: int = 3  # padding around a line box before recognition
    assign_min_ioa: float = 0.5  # share of a line that must lie inside a region's (padded) text box
    region_pad_px: int = 8  # padding of a region's text box when assigning lines
    nms_iou: float = 0.5  # same-line IoU above which the lower-scored box is dropped (overlapping tiles)
    low_conf: float = 0.85  # a region whose confidence is below this counts as low-confidence
    drop_conf: float = (
        0.5  # regions with an OCR confidence below this (or without any text) are dropped from ocr.json
    )
    lang: Literal["ko", "zh", "ja", "en"] = "ko"  # language written into regions
    engine: Literal["ppocr", "manga_ocr", "paddleocr_vl"] = (
        "ppocr"  # ppocr = line detector + recogniser; the others read region crops
    )
    det_model: str | None = None  # catalog id of the text-line detector (ppocr); None = det_repo
    rec_model: str | None = (
        None  # catalog id of the recogniser / crop reader; None = rec_repo (ppocr) or the engine's default
    )
    crop_pad_px: int = (
        6  # padding around a region before a crop-reading engine (manga_ocr, paddleocr_vl) reads it
    )
    crop_batch_size: int = 16  # region crops per forward pass of a crop-reading engine
    vl_max_new_tokens: int = 192  # paddleocr_vl: longest text it may generate for one region


class InpaintConfig(BaseModel):
    pad_px: int = 8  # patch = union of a region's line boxes grown by this
    mask_dilate_px: int = 3  # line boxes are grown by this to form the text mask
    flat_tol: float = 8.0  # 90th-percentile colour deviation of the ring under which a flat fill is accepted
    min_ring_px: int = 48  # fewer ring pixels than this -> no flat fill
    lama_url: str = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"  # TorchScript LaMa (Apache-2.0)
    lama_sha256: str = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"  # checked after every download and load
    lama_file: str = "big-lama.pt"  # file name inside <models_dir>/lama/
    lama_window: int = (
        512  # ONE fixed window size (every new input shape costs a 10-25 s warm-up); a multiple of 8
    )
    lama_dilate_px: int = 4  # extra growth of the flat-fill mask for LaMa (generous masks inpaint better)
    lama_context_px: int = (
        32  # a region needs at least this much context inside the window on every side, else it is skipped
    )


class TypesetConfig(BaseModel):
    min_px: int = 14  # smallest font size tried
    max_px: int = 48  # largest font size tried
    line_spacing: float = 1.15  # line pitch = size * this
    margin_px: int = 6  # inset of the bubble's inscribed rectangle
    free_grow: float = 0.10  # free text / SFX boxes are grown by this fraction on every side
    stroke_free_px: int = 3  # outline of free-standing text
    stroke_sfx_px: int = 5  # outline of sound effects


class ExportConfig(BaseModel):
    jpeg_quality: int = 95
    subsampling: Literal["444", "422", "420"] = "444"  # chroma subsampling of the exported slices


class Secrets(BaseSettings):
    """Secrets only come from the environment or ~/.config/omniscan/secrets.env — never from TOML."""

    model_config = SettingsConfigDict(env_file=SECRETS_ENV, env_file_encoding="utf-8", extra="ignore")

    ollama_api_key: SecretStr | None = Field(default=None, alias="OLLAMA_API_KEY")
    extractpics_api_key: SecretStr | None = Field(default=None, alias="EXTRACTPICS_API_KEY")
    relay_client_token: SecretStr | None = Field(default=None, alias="OMNISCAN_RELAY_CLIENT_TOKEN")


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNISCAN_", env_nested_delimiter="__", extra="ignore")

    paths: PathsConfig = PathsConfig()
    gpu: GpuConfig = GpuConfig()
    slicer: SlicerConfig = SlicerConfig()
    detect: DetectConfig = DetectConfig()
    ollama: OllamaConfig = OllamaConfig()
    relay: RelayConfig = RelayConfig()
    ocr: OcrConfig = OcrConfig()
    inpaint: InpaintConfig = InpaintConfig()
    typeset: TypesetConfig = TypesetConfig()
    export: ExportConfig = ExportConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(*extra_tomls: Path) -> Config:
    """Build the effective config; env vars override TOML values."""
    data: dict[str, Any] = {}
    for path in (DEFAULT_TOML, USER_TOML, *extra_tomls):
        data = _deep_merge(data, _read_toml(path))
    # pydantic-settings: init kwargs have the highest priority, so apply env on top explicitly.
    env_only = Config().model_dump(exclude_unset=True)
    return Config(**_deep_merge(data, env_only))


SERIES_SECTIONS = (
    "slicer",
    "detect",
    "ocr",
    "inpaint",
    "typeset",
    "export",
)  # machine-level sections are not per series


class SeriesConfigError(ValueError):
    """`<series>/series.toml` is unreadable, invalid or overrides a machine-level section."""


def series_config(cfg: Config, series_dir: Path) -> Config:
    """`cfg` with `<series_dir>/series.toml` merged on top (only SERIES_SECTIONS may be overridden)."""
    path = series_dir / "series.toml"
    if not path.is_file():
        return cfg
    try:
        data = _read_toml(path)
    except tomllib.TOMLDecodeError as exc:
        raise SeriesConfigError(f"{path}: {exc}") from exc
    unknown = sorted(set(data) - set(SERIES_SECTIONS))
    if unknown:
        raise SeriesConfigError(
            f"{path}: sections not allowed per series: {', '.join(unknown)} (allowed: {', '.join(SERIES_SECTIONS)})"
        )
    try:
        return Config(**_deep_merge(cfg.model_dump(), data))
    except ValidationError as exc:
        raise SeriesConfigError(f"{path}: {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached config."""
    return load_config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Process-wide cached secrets (never log these)."""
    return Secrets()
