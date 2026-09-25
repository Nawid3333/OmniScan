"""OmniScan configuration contract.

Load order (later wins): built-in defaults -> config/default.toml (repo)
-> ~/.config/omniscan/config.toml (user) -> env vars `OMNISCAN_<SECTION>__<KEY>`.
Secrets come only from the environment or ~/.config/omniscan/secrets.env.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
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
    # Context (num_ctx) sent with every local-model request, raised for prompts that need more (see
    # OllamaClient). Leaving it to the server is unsafe: the Ollama app's context slider applies to every
    # model, and at 256K it gave translategemma:12b an 8 GiB KV cache that pushed 11 of its 49 layers to
    # the CPU (measured: 2.6x slower per request). 0 = send nothing, use the server's setting.
    num_ctx: int = Field(default=16384, ge=0)


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
    glyph_mask: bool = True  # remove only the text's own ink (+ outline, anti-aliasing), not whole line boxes
    glyph_grow: float = 1.0  # the ink is grown by this many estimated stroke widths
    glyph_grow_sfx: float = 1.6  # ... for sound effects, whose outlines are thicker
    glyph_grow_min_px: int = 2  # growth is clamped to [min, max] pixels
    glyph_grow_max_px: int = 12
    glyph_grow_max_sfx_px: int = 32  # sound effects are big and their outlines thick
    glyph_ring_px: int = 4  # band around the glyphs that must be flat for a flat fill of just the glyphs


class TypesetConfig(BaseModel):
    min_px: int = 14  # smallest font size tried
    max_px: int = 48  # largest font size tried
    line_spacing: float = 1.15  # line pitch = size * this
    margin_px: int = 6  # smallest gap between lettering and the balloon's edge
    free_grow: float = 0.10  # free text / SFX boxes are grown by this fraction on every side
    stroke_free_px: int = 3  # outline of free-standing text
    stroke_sfx_px: int = 5  # thinnest outline of sound effects (large ones scale it up)
    # Lettering style: fonts and capitalisation. auto = manga for Japanese sources, webtoon otherwise.
    style: Literal["auto", "webtoon", "manga"] = "auto"
    uppercase: Literal["auto", "always", "never"] = "auto"  # auto = the style's convention (manga: capitals)
    # Per-role font overrides: a file in the fonts folder or an absolute path ("" = the style's preset),
    # e.g. a licensed CC Wild Words for font_dialogue.
    font_dialogue: str = ""
    font_thought: str = ""
    font_shout: str = ""
    font_narration: str = ""
    font_free: str = ""
    font_sfx: str = ""  # "" = picked per sound effect from the original's stroke weight
    bubble_padding: float = 0.12  # share of a balloon's width / height kept free on each side
    size_spread: float = 1.15  # dialogue may be at most this x the chapter's typical size (0 = no limit)
    shout_spread: float = 1.35  # the same limit for shouted lines
    hyphenate: bool = True  # split a word that fits no line even at min_px instead of overflowing
    sfx_max_px: int = 240  # largest sound-effect lettering


class SfxConfig(BaseModel):
    """Sound effects: finding them among free text after OCR, and how the English version replaces them."""

    detect: bool = True  # free text that reads as onomatopoeia (config/sfx_text.toml) becomes kind "sfx"
    max_chars: int = 8  # longer text (letters only, punctuation ignored) is never taken for an SFX
    lexicon_size_ratio: float = 1.0  # a lexicon match needs glyphs >= this x the chapter's dialogue glyphs
    size_ratio: float = 2.0  # without a lexicon match, glyphs >= this x the dialogue glyphs ...
    size_max_chars: int = 3  # ... on text of at most this many letters also count as an SFX
    mode: Literal["replace", "subtitle", "keep"] = (
        "replace"  # replace: erase and redraw in English; subtitle: keep the art, add a small translation; keep: untouched
    )


class ExportConfig(BaseModel):
    jpeg_quality: int = 95
    subsampling: Literal["444", "422", "420"] = "444"  # chroma subsampling of the exported slices


class FilterConfig(BaseModel):
    """Promo / credit filtering (tier 2 = pHash against the user's example images; see docs/PRODUCT_SPEC.md section 4)."""

    enabled: bool = True
    threshold: float = 0.90  # dHash similarity at or above which a file/slice counts as a match


class Secrets(BaseSettings):
    """Secrets only come from the environment or ~/.config/omniscan/secrets.env — never from TOML."""

    model_config = SettingsConfigDict(env_file=SECRETS_ENV, env_file_encoding="utf-8", extra="ignore")

    ollama_api_key: SecretStr | None = Field(default=None, alias="OLLAMA_API_KEY")


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNISCAN_", env_nested_delimiter="__", extra="ignore")

    paths: PathsConfig = PathsConfig()
    gpu: GpuConfig = GpuConfig()
    slicer: SlicerConfig = SlicerConfig()
    detect: DetectConfig = DetectConfig()
    ollama: OllamaConfig = OllamaConfig()
    ocr: OcrConfig = OcrConfig()
    inpaint: InpaintConfig = InpaintConfig()
    typeset: TypesetConfig = TypesetConfig()
    sfx: SfxConfig = SfxConfig()
    export: ExportConfig = ExportConfig()
    filter: FilterConfig = FilterConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# Per-language OCR engine defaults (cards O1c/O1d; measured in docs/benchmarks/ocr-qualification-results.md).
# Applied wherever a TOML source sets ocr.lang without itself naming an engine/det_model/rec_model, so a
# series only has to declare its language to get the best-measured engine instead of silently inheriting
# whatever engine an earlier layer (repo default, user config) happened to set for a different language.
_LANG_OCR_ENGINE_DEFAULTS: dict[str, dict[str, str]] = {
    "ko": {"engine": "paddleocr_vl"},  # chrF 0.567 vs 0.475 (ppocr-v5-ko); -0.056 char recall, accepted
    "zh": {"engine": "paddleocr_vl"},  # chrF 0.741 vs 0.642 (ppocr-v5-server-multi)
    "ja": {"engine": "ppocr", "det_model": "ocr-det-ppocrv6-medium", "rec_model": "ocr-rec-ppocrv6-medium"},
}


def _apply_lang_ocr_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Fill `ocr.engine`/`det_model`/`rec_model` from `_LANG_OCR_ENGINE_DEFAULTS[ocr.lang]` when `data`
    sets `ocr.lang` but none of those three keys itself; an explicit value for any of them always wins."""
    ocr = data.get("ocr")
    if not isinstance(ocr, dict):
        return data
    lang = ocr.get("lang")
    defaults = _LANG_OCR_ENGINE_DEFAULTS.get(lang) if isinstance(lang, str) else None
    if defaults is None or ({"engine", "det_model", "rec_model"} & ocr.keys()):
        return data
    return {**data, "ocr": {**defaults, **ocr}}


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
    data = _apply_lang_ocr_defaults(data)
    # pydantic-settings: init kwargs have the highest priority, so apply env on top explicitly.
    env_only = Config().model_dump(exclude_unset=True)
    return Config(**_deep_merge(data, env_only))


SERIES_SECTIONS = (
    "slicer",
    "detect",
    "ocr",
    "inpaint",
    "typeset",
    "sfx",
    "export",
    "filter",
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
    data = _apply_lang_ocr_defaults(data)
    try:
        return Config(**_deep_merge(cfg.model_dump(), data))
    except ValidationError as exc:
        raise SeriesConfigError(f"{path}: {exc}") from exc


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Serialise a dict of scalars, lists of scalars and one level of tables to TOML (comments are not kept)."""

    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, (str, Path)):
            return json.dumps(
                str(value), ensure_ascii=False
            )  # JSON string escapes are valid TOML basic strings
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(scalar(item) for item in value) + "]"
        raise ValueError(f"cannot write {type(value).__name__} value {value!r} to TOML")

    lines: list[str] = []
    tables: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {scalar(value)}")
    for name, table in tables:
        if lines:
            lines.append("")
        lines.append(f"[{name}]")
        for key, value in table.items():
            if isinstance(value, Mapping):
                raise ValueError(f"nested table {name}.{key} is not supported")
            lines.append(f"{key} = {scalar(value)}")
    return "\n".join(lines) + "\n"


class SettingError(ValueError):
    """A settings write was refused: unknown section/key, or a value the config model rejects."""


def _check_setting(section: str, key: str, value: Any, *, allowed_sections: Sequence[str]) -> None:
    """Refuse unknown sections/keys and values that do not validate against the config model."""
    if section not in allowed_sections:
        raise SettingError(
            f"unknown or not allowed section {section!r} (allowed: {', '.join(allowed_sections)})"
        )
    model = Config.model_fields[section].annotation
    fields = getattr(model, "model_fields", {})
    if key not in fields:
        raise SettingError(f"unknown key {section}.{key} (known: {', '.join(fields)})")
    try:
        Config.model_validate({section: {**Config().model_dump()[section], key: value}})
    except ValidationError as exc:
        raise SettingError(f"{section}.{key}: {exc.errors()[0]['msg']}") from exc


def _write_setting(path: Path, section: str, key: str, value: Any) -> Path:
    data = _read_toml(path)
    data.setdefault(section, {})[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps_toml(data), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


def set_user_setting(section: str, key: str, value: Any, *, path: Path | None = None) -> Path:
    """Write `[section] key = value` to the user config (default `~/.config/omniscan/config.toml`), validated first.

    Only scalar and list values are supported; comments in an existing file are not kept. Returns the file path.
    """
    _check_setting(section, key, value, allowed_sections=tuple(Config.model_fields))
    return _write_setting(path or USER_TOML, section, key, value)


def set_series_setting(series_dir: Path, section: str, key: str, value: Any) -> Path:
    """Write `[section] key = value` to `<series_dir>/series.toml` (only SERIES_SECTIONS), validated first."""
    _check_setting(section, key, value, allowed_sections=SERIES_SECTIONS)
    return _write_setting(series_dir / "series.toml", section, key, value)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached config."""
    return load_config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Process-wide cached secrets (never log these)."""
    return Secrets()
