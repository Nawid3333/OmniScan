"""Crop-reading OCR engines: whole region crops -> (text, score), no line detection (cards O1b, O1d).

`MangaOcrReader` runs the manga-ocr VisionEncoderDecoder over region crops (uint8 `[3, h, w]`, any
device) in fp32; the crops are resized to 224x224 by the ViT processor, so no width batching is
needed. Decoding uses the model's `vocab.txt` directly (transformers' tokenizer needs `fugashi`).
`PaddleOcrVlReader` runs PaddleOCR-VL (a 0.9 B vision-language model) over the same crops — a
chat-formatted `generate` over `cfg.crop_batch_size` crops at a time, in order, which reads all CJK
scripts and Latin.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image

from omniscan.core.config import OcrConfig
from omniscan.ocr.engines import engine_rec_model, load_kwargs, model_source
from omniscan.ocr.model import _check_model_role, _to_device

log = logging.getLogger(__name__)

_SPECIAL_TOKENS = frozenset(("[PAD]", "[UNK]", "[CLS]", ""))
_UNUSED_RE = re.compile(r"<unused\d+>")
# printable ASCII (0x21-0x7E) minus the two characters the card keeps half-width: `.` and `~`
_FULLWIDTH_FROM = ord("!")
_FULLWIDTH_TO = ord("~")
_FULLWIDTH_KEEP = frozenset((".", "~"))


class TextReader(Protocol):
    """Anything that reads text from uint8 `[3, h, w]` crops; LineRecognizer already satisfies it."""

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]: ...


def decode_wordpieces(ids: Sequence[int], vocab: Sequence[str]) -> str:
    """Join WordPiece ids to text: specials and `<unusedN>` are skipped, `##` continues a token.

    Decoding stops at the first `[SEP]`; an id outside the vocab raises IndexError.
    """
    parts: list[str] = []
    for token_id in ids:
        index = int(token_id)
        if not 0 <= index < len(vocab):
            raise IndexError(f"token id {index} outside the vocab ({len(vocab)} entries)")
        token = vocab[index]
        if token == "[SEP]":
            break
        if token in _SPECIAL_TOKENS or _UNUSED_RE.fullmatch(token):
            continue
        parts.append(token[2:] if token.startswith("##") else token)
    return "".join(parts)


def clean_manga_text(text: str) -> str:
    """Upstream manga-ocr post-processing (no jaconv): squeeze whitespace, tidy dots, widen ASCII."""
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda match: "." * len(match.group()), text)
    return "".join(
        chr(ord(c) + 0xFEE0) if _FULLWIDTH_FROM <= ord(c) <= _FULLWIDTH_TO and c not in _FULLWIDTH_KEEP else c
        for c in text
    )


class MangaOcrReader:
    """manga-ocr crop reader (`cfg.rec_model` or the engine default): crops -> (text, score)."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        vocab: Sequence[str],
        device: torch.device,
        *,
        batch_size: int,
    ) -> None:
        self.model = model
        self.processor = processor
        self.vocab = vocab
        self._device = device
        self.batch_size = batch_size

    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device, models_dir: Path | None = None) -> MangaOcrReader:
        """Download (first use) and load manga-ocr, fp32 everywhere (no `.half()` on this stack)."""
        rec_model = engine_rec_model(cfg)
        if rec_model is None:
            raise ValueError("manga_ocr needs ocr.rec_model")
        source = model_source(rec_model, models_dir)
        path_or_repo, kwargs = load_kwargs(source, models_dir=models_dir)
        if device.type == "cuda":
            # MIOpen launches kernels against the current device (the iGPU here), not the tensors' device
            torch.cuda.set_device(device)
        from transformers import AutoImageProcessor, VisionEncoderDecoderModel

        model = _to_device(VisionEncoderDecoderModel.from_pretrained(path_or_repo, **kwargs), device).eval()
        processor = AutoImageProcessor.from_pretrained(path_or_repo, **kwargs)
        return cls(model, processor, _load_vocab(source), device, batch_size=cfg.crop_batch_size)

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]:
        """(text, score) per crop in the given order; the ViT processor resizes every crop to 224x224."""
        if not crops:
            return []
        readings: list[tuple[str, float]] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = list(crops[start : start + self.batch_size])
            pixel_values = self._pixel_values(chunk).to(self._device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            readings.extend(self._readings(outputs))
        return readings

    def _pixel_values(self, crops: list[torch.Tensor]) -> torch.Tensor:
        """The processor's pixel_values for one chunk; GPU crops stay on the GPU when possible."""
        try:
            processed = self.processor(images=crops, return_tensors="pt")
        except Exception:  # the processor wants host data: one small host round trip per crop
            from torchvision.transforms.functional import to_pil_image

            processed = self.processor(images=[to_pil_image(crop) for crop in crops], return_tensors="pt")
        return processed["pixel_values"]

    def _readings(self, outputs: Any) -> list[tuple[str, float]]:
        """(text, score) per row of `outputs.sequences`: exp of the mean generated-token log-prob."""
        try:
            transition = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
            )
        except Exception:
            log.debug("manga-ocr transition scores unavailable; scoring 1.0", exc_info=True)
            transition = None
        pad_id = getattr(getattr(self.model, "config", None), "pad_token_id", None)
        if pad_id is None:
            pad_id = 0  # [PAD] in the manga-ocr vocab
        readings: list[tuple[str, float]] = []
        for index, sequence in enumerate(outputs.sequences):
            text = clean_manga_text(decode_wordpieces(sequence.tolist(), self.vocab))
            if not text:
                readings.append(("", 0.0))
                continue
            if transition is None:
                score = 1.0
            else:
                generated = sequence[len(sequence) - transition.shape[1] :]
                log_probs = transition[index][generated != pad_id]
                score = (
                    round(min(max(math.exp(float(log_probs.mean())), 0.0), 1.0), 4)
                    if log_probs.numel()
                    else 0.0
                )
            readings.append((text, score))
        return readings


def _load_vocab(source: Any) -> list[str]:
    """vocab.txt of the model: from the installed folder, else the hub — one token per line."""
    if source.local is not None:
        path = Path(source.local) / "vocab.txt"
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(source.repo, "vocab.txt", revision=source.revision))
    vocab = path.read_text(encoding="utf-8").split("\n")
    if vocab and vocab[-1] == "":
        vocab.pop()  # the file's trailing newline is not a token
    return vocab


VL_PROMPT = "OCR:"
VL_MIN_SIDE = 28  # the vision tower works on 28-px patches: smaller sides must be upscaled first
VL_MAX_PIXELS = 1280 * 28 * 28  # the processor's longest_edge: a region crop never needs more


def prepare_vl_image(crop: torch.Tensor) -> Image.Image:
    """A crop (uint8 `[3, h, w]`, any device) as an RGB PIL image, upscaled so min(w, h) >= 28."""
    from torchvision.transforms.functional import to_pil_image

    image = to_pil_image(crop.detach().cpu()).convert("RGB")
    width, height = image.size
    smallest = min(width, height)
    if smallest >= VL_MIN_SIDE:
        return image
    scale = VL_MIN_SIDE / smallest
    return image.resize((math.ceil(width * scale), math.ceil(height * scale)), Image.Resampling.LANCZOS)


def clean_vl_text(text: str) -> str:
    """One space per whitespace run (the model writes newlines between the lines); nothing else."""
    return " ".join(text.split())


class PaddleOcrVlReader:
    """PaddleOCR-VL crop reader (`cfg.rec_model` or the engine default): `cfg.crop_batch_size` crops per `generate`."""

    def __init__(
        self, model: Any, processor: Any, device: torch.device, *, max_new_tokens: int, batch_size: int
    ) -> None:
        self.model = model
        self.processor = processor
        self._device = device
        self._max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    @classmethod
    def load(cls, cfg: OcrConfig, device: torch.device, models_dir: Path | None = None) -> PaddleOcrVlReader:
        """Download (first use) and load PaddleOCR-VL fp32 (~3.4 GiB — the accuracy mode)."""
        rec_model = engine_rec_model(cfg)
        if rec_model is None:
            raise ValueError("paddleocr_vl needs ocr.rec_model")
        _check_model_role(rec_model, "vlm_ocr")
        source = model_source(rec_model, models_dir)
        path_or_repo, kwargs = load_kwargs(source, models_dir=models_dir)
        if device.type == "cuda":
            # MIOpen launches kernels against the current device (the iGPU here), not the tensors' device
            torch.cuda.set_device(device)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = _to_device(
            AutoModelForImageTextToText.from_pretrained(path_or_repo, dtype=torch.float32, **kwargs), device
        ).eval()
        processor = AutoProcessor.from_pretrained(path_or_repo, **kwargs)
        return cls(
            model, processor, device, max_new_tokens=cfg.vl_max_new_tokens, batch_size=cfg.crop_batch_size
        )

    def read(self, crops: Sequence[torch.Tensor]) -> list[tuple[str, float]]:
        """(text, score) per crop, in order, `cfg.crop_batch_size` crops per batched `generate`.

        The model reads a whole region with the fixed "OCR:" chat prompt, greedy, scored by the mean
        generated-token log-probability. Batches are left-padded: every prompt in a chunk then ends
        at the same index, so one shared prompt length offsets the generated tail of every row.
        """
        if not crops:
            return []
        readings: list[tuple[str, float]] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = list(crops[start : start + self.batch_size])
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": prepare_vl_image(crop)},
                            {"type": "text", "text": VL_PROMPT},
                        ],
                    }
                ]
                for crop in chunk
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                images_kwargs={
                    "size": {
                        "shortest_edge": self.processor.image_processor.size["shortest_edge"],
                        "longest_edge": VL_MAX_PIXELS,
                    }
                },
            ).to(self._device)
            n_prompt = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            readings.extend(self._readings(out, n_prompt))
        return readings

    def _readings(self, out: Any, n_prompt: int) -> list[tuple[str, float]]:
        """(text, score) per row of the batch: exp of the mean non-pad generated-token log-prob."""
        try:
            transition = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
        except Exception:
            log.debug("paddleocr-vl transition scores unavailable; scoring 1.0", exc_info=True)
            transition = None
        pad_id = getattr(getattr(self.model, "config", None), "pad_token_id", None)
        if pad_id is None:
            pad_id = 0  # the PaddleOCR-VL vocab pads with <unk> (id 0)
        readings: list[tuple[str, float]] = []
        for index, sequence in enumerate(out.sequences):
            text = clean_vl_text(self.processor.decode(sequence[n_prompt:], skip_special_tokens=True))
            if not text:
                readings.append(("", 0.0))
                continue
            if transition is None:
                score = 1.0
            else:
                generated = sequence[len(sequence) - transition.shape[1] :]
                log_probs = transition[index][generated != pad_id]
                score = (
                    round(min(max(math.exp(float(log_probs.mean())), 0.0), 1.0), 4)
                    if log_probs.numel()
                    else 0.0
                )
            readings.append((text, score))
        return readings
