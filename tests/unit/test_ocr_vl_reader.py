"""prepare_vl_image / clean_vl_text / PaddleOcrVlReader against fake model+processor (card O1d, tests 1-4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from omniscan.core.config import OcrConfig
from omniscan.models.catalog import ModelEntry
from omniscan.models.store import MARKER_NAME
from omniscan.ocr.crop_readers import (
    VL_MAX_PIXELS,
    VL_PROMPT,
    PaddleOcrVlReader,
    clean_vl_text,
    prepare_vl_image,
)

REVISION = "c" * 40
VL_ID = "ocr-vl-1.6"
VL_REPO = "PaddlePaddle/PaddleOCR-VL-1.6"


def vl_entry(model_id: str = VL_ID, repo: str = VL_REPO) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        name="PaddleOCR-VL",
        kind="ocr",
        format="hf",
        size_mb=1931,
        license="Apache-2.0",
        description="d",
        upstream_repo=repo,
        upstream_revision=REVISION,
        files={"model.safetensors": hashlib.sha256(b"weights").hexdigest()},
        role="vlm_ocr",
    )


def install_fake_vl(models_dir: Path, model_id: str = VL_ID) -> Path:
    folder = models_dir / model_id
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_text("weights", encoding="utf-8")
    marker = {
        "id": model_id,
        "source": "upstream",
        "revision": REVISION,
        "files_verified": 1,
        "file_sizes": {"model.safetensors": 7},  # len(b"weights")
    }
    (folder / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    return folder


# ---------------------------------------------------------------- prepare_vl_image (test 1)


def test_prepare_vl_image_keeps_big_crops_and_their_pixels() -> None:
    crop = torch.arange(3 * 60 * 100, dtype=torch.uint8).reshape(3, 60, 100) % 251

    image = prepare_vl_image(crop)

    assert image.size == (100, 60) and image.mode == "RGB"
    assert image.tobytes() == crop.permute(1, 2, 0).contiguous().numpy().tobytes()


def test_prepare_vl_image_upscales_below_the_28px_patch_side() -> None:
    assert prepare_vl_image(torch.zeros(3, 60, 20, dtype=torch.uint8)).size == (28, 84)  # 20x60
    assert prepare_vl_image(torch.zeros(3, 50, 1, dtype=torch.uint8)).size == (28, 1400)  # 1x50
    assert prepare_vl_image(torch.zeros(3, 28, 28, dtype=torch.uint8)).size == (28, 28)  # exact 28


def test_prepare_vl_image_takes_any_tensor_without_device_assumptions() -> None:
    crop = torch.full((3, 40, 80), 7, dtype=torch.uint8)
    crop.requires_grad_(False)  # the CUDA-like path: a plain non-gradient tensor on the host

    image = prepare_vl_image(crop)

    assert image.size == (80, 40) and image.getpixel((0, 0)) == (7, 7, 7)


# ---------------------------------------------------------------- clean_vl_text (test 2)


def test_clean_vl_text_joins_the_lines_with_single_spaces() -> None:
    assert clean_vl_text("きっと...\nうっかりして\n寝ちゃった") == "きっと... うっかりして 寝ちゃった"
    assert clean_vl_text("a \t b\n\nc  d") == "a b c d"
    assert clean_vl_text("…? ") == "…?"  # punctuation is untouched (no manga-ocr folding here)


def test_clean_vl_text_empty_and_whitespace_only() -> None:
    assert clean_vl_text("") == ""
    assert clean_vl_text("  \t\n ") == ""


# ---------------------------------------------------------------- fakes (test 3)


class FakeInputs(dict):
    """apply_chat_template's return value: a dict that records `.to()` and unpacks into generate."""

    def to(self, device: torch.device) -> FakeInputs:
        self.to_device = device
        return self


class FakeVLProcessor:
    """Records apply_chat_template calls; decodes one scripted text per crop."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict[str, Any]] = []
        self.inputs: list[FakeInputs] = []
        self.decoded: list[list[int]] = []
        self.image_processor = SimpleNamespace(size={"shortest_edge": 448})

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> FakeInputs:
        self.calls.append({"messages": messages, **kwargs})
        inputs = FakeInputs({"input_ids": torch.arange(5).reshape(1, -1)})
        self.inputs.append(inputs)
        return inputs

    def decode(self, tokens: Any, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded.append([int(t) for t in tokens])
        return self.texts[len(self.decoded) - 1]


class FakeVLModel:
    """Scripted generate: prompt + N generated tokens, transition scores from the scripted logs."""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.generate_calls: list[dict[str, Any]] = []
        self.score_calls = 0
        self.to_devices: list[torch.device] = []
        self.evaled = False

    def to(self, device: torch.device) -> FakeVLModel:
        self.to_devices.append(device)
        return self

    def eval(self) -> FakeVLModel:
        self.evaled = True
        return self

    def generate(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        call = self.calls[len(self.generate_calls) - 1]
        generated = (torch.arange(len(call["logs"]), dtype=torch.long) + 100).reshape(1, -1)
        sequences = torch.cat([kwargs["input_ids"], generated], dim=1)
        return SimpleNamespace(sequences=sequences, scores=tuple(object() for _ in call["logs"]))

    def compute_transition_scores(self, sequences: Any, scores: Any, normalize_logits: bool) -> torch.Tensor:
        assert normalize_logits is True
        call = self.calls[len(self.generate_calls) - 1]
        if call.get("raise"):
            raise RuntimeError("no transition scores")
        return torch.tensor([call["logs"]])


def make_reader(
    calls: list[dict[str, Any]], texts: list[str], *, cfg: OcrConfig | None = None
) -> tuple[PaddleOcrVlReader, FakeVLModel, FakeVLProcessor]:
    cfg = cfg or OcrConfig(engine="paddleocr_vl", vl_max_new_tokens=99)
    model = FakeVLModel(calls)
    processor = FakeVLProcessor(texts)
    reader = PaddleOcrVlReader(model, processor, torch.device("cpu"), max_new_tokens=cfg.vl_max_new_tokens)
    return reader, model, processor


# ---------------------------------------------------------------- read (test 3)


def test_read_sends_the_card_messages_and_generate_arguments() -> None:
    calls = [{"logs": [-0.1, -0.3]}, {"logs": [-1.0]}]
    reader, model, processor = make_reader(calls, ["first", "second"])

    out = reader.read([torch.zeros(3, 40, 60, dtype=torch.uint8), torch.zeros(3, 30, 30, dtype=torch.uint8)])

    assert out == [("first", 0.8187), ("second", 0.3679)]  # exp(-0.2), exp(-1.0)
    assert len(processor.calls) == 2 and len(model.generate_calls) == 2  # one generate per crop, in order
    for call, inputs in zip(processor.calls, processor.inputs, strict=True):
        image = call["messages"][0]["content"][0]["image"]
        assert isinstance(image, Image.Image)
        assert call["messages"] == [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": VL_PROMPT}],
            }
        ]
        assert call["add_generation_prompt"] is True
        assert call["tokenize"] is True
        assert call["return_dict"] is True
        assert call["return_tensors"] == "pt"
        assert call["images_kwargs"] == {"size": {"shortest_edge": 448, "longest_edge": VL_MAX_PIXELS}}
        assert inputs.to_device == torch.device("cpu")
    for kwargs in model.generate_calls:
        assert kwargs["max_new_tokens"] == 99  # cfg.vl_max_new_tokens of make_reader's cfg
        assert kwargs["do_sample"] is False
        assert kwargs["output_scores"] is True
        assert kwargs["return_dict_in_generate"] is True


def test_read_decodes_only_the_generated_tail() -> None:
    calls = [{"logs": [-0.2, -0.4, -0.6]}]
    reader, _model, processor = make_reader(calls, ["decoded"])

    reader.read([torch.zeros(3, 8, 8, dtype=torch.uint8)])

    # the prompt is arange(5); the generated tail of the fake sequence is 100, 101, 102
    assert processor.decoded == [[100, 101, 102]]


def test_read_cleans_the_decoded_text() -> None:
    calls = [{"logs": [-0.1]}]
    reader, _model, _processor = make_reader(calls, ["きっと...\nうっかりして\n寝ちゃったんだ!"])

    assert reader.read([torch.zeros(3, 20, 20, dtype=torch.uint8)]) == [
        ("きっと... うっかりして 寝ちゃったんだ!", 0.9048)  # exp(-0.1)
    ]


def test_read_clamps_the_score_to_one() -> None:
    calls = [{"logs": [0.5, 0.2]}]  # a mean log-prob above 0 cannot happen; the fake can
    reader, _model, _processor = make_reader(calls, ["text"])

    assert reader.read([torch.zeros(3, 20, 20, dtype=torch.uint8)]) == [("text", 1.0)]


def test_read_transition_scores_raising_scores_1() -> None:
    calls = [{"logs": [-0.1, -0.2], "raise": True}]
    reader, _model, _processor = make_reader(calls, ["text"])

    assert reader.read([torch.zeros(3, 20, 20, dtype=torch.uint8)]) == [("text", 1.0)]


def test_read_empty_decode_scores_0() -> None:
    calls = [{"logs": [-0.1]}]
    reader, model, _processor = make_reader(calls, [""])

    assert reader.read([torch.zeros(3, 20, 20, dtype=torch.uint8)]) == [("", 0.0)]
    assert len(model.generate_calls) == 1  # the crop was still read; only scoring is skipped


def test_read_no_crops_never_calls_the_model() -> None:
    reader, model, processor = make_reader([], [])

    assert reader.read([]) == []
    assert model.generate_calls == [] and processor.calls == []


def test_read_upscales_a_crop_below_28px_before_the_processor() -> None:
    calls = [{"logs": [-0.1]}]
    reader, _model, processor = make_reader(calls, ["text"])

    reader.read([torch.zeros(3, 60, 20, dtype=torch.uint8)])

    image = processor.calls[0]["messages"][0]["content"][0]["image"]
    assert image.size == (28, 84)


# ---------------------------------------------------------------- load (test 4)


@pytest.fixture
def patched_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], FakeVLModel, FakeVLProcessor]:
    """Recorders on the real Auto classes' `from_pretrained` (module-attribute patches are undone
    by transformers' lazy module as soon as a second attribute is resolved)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model, processor, seen = FakeVLModel([]), FakeVLProcessor([]), {}

    def fake_model_from(path: Any, **kwargs: Any) -> Any:
        seen["model"] = {"path": path, **kwargs}
        return model

    def fake_processor_from(path: Any, **kwargs: Any) -> Any:
        seen["processor"] = {"path": path, **kwargs}
        return processor

    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", fake_model_from)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", fake_processor_from)
    return seen, model, processor


def test_load_installed_model_uses_its_folder_fp32(monkeypatch, tmp_path, patched_transformers) -> None:
    folder = install_fake_vl(tmp_path / "models")
    monkeypatch.setattr("omniscan.ocr.engines.load_catalog", lambda: [vl_entry()])
    seen, model, processor = patched_transformers

    reader = PaddleOcrVlReader.load(
        OcrConfig(engine="paddleocr_vl"), torch.device("cpu"), tmp_path / "models"
    )

    assert seen["model"] == {"path": str(folder), "dtype": torch.float32, "local_files_only": True}
    assert seen["processor"] == {"path": str(folder), "local_files_only": True}
    assert model.to_devices == [torch.device("cpu")] and model.evaled
    assert reader.model is model and reader.processor is processor
    assert reader._max_new_tokens == 192  # cfg.vl_max_new_tokens default


def test_load_missing_model_falls_back_to_the_pinned_hub(
    monkeypatch, tmp_path, patched_transformers, caplog
) -> None:
    monkeypatch.setattr("omniscan.ocr.engines.load_catalog", lambda: [vl_entry()])
    seen, _model, _processor = patched_transformers

    PaddleOcrVlReader.load(OcrConfig(engine="paddleocr_vl"), torch.device("cpu"), tmp_path / "models")

    assert seen["model"] == {"path": VL_REPO, "dtype": torch.float32, "revision": REVISION}
    warnings = [r for r in caplog.records if r.name == "omniscan.ocr.engines"]
    assert len(warnings) == 1 and "omniscan models download ocr-vl-1.6" in warnings[0].message


def test_load_without_models_dir_uses_the_hub_quietly(monkeypatch, patched_transformers) -> None:
    monkeypatch.setattr("omniscan.ocr.engines.load_catalog", lambda: [vl_entry()])
    seen, _model, _processor = patched_transformers

    PaddleOcrVlReader.load(OcrConfig(engine="paddleocr_vl"), torch.device("cpu"))

    assert seen["model"] == {"path": VL_REPO, "dtype": torch.float32, "revision": REVISION}


def test_load_rejects_a_model_of_the_wrong_role(monkeypatch, tmp_path, patched_transformers) -> None:
    monkeypatch.setattr(
        "omniscan.ocr.engines.load_catalog", lambda: [vl_entry().model_copy(update={"role": "recognizer"})]
    )

    with pytest.raises(ValueError, match="has role 'recognizer', expected 'vlm_ocr'"):
        PaddleOcrVlReader.load(OcrConfig(engine="paddleocr_vl"), torch.device("cpu"), tmp_path / "models")


def test_load_without_a_rec_model_raises() -> None:
    with pytest.raises(ValueError, match=r"paddleocr_vl needs ocr\.rec_model"):
        PaddleOcrVlReader.load(OcrConfig(engine="ppocr"), torch.device("cpu"))


def test_load_explicit_rec_model_wins_over_the_default(monkeypatch, patched_transformers) -> None:
    entries = [vl_entry(), vl_entry("ocr-vl-1.5", "PaddlePaddle/PaddleOCR-VL-1.5")]
    monkeypatch.setattr("omniscan.ocr.engines.load_catalog", lambda: entries)
    seen, _model, _processor = patched_transformers

    PaddleOcrVlReader.load(OcrConfig(engine="paddleocr_vl", rec_model="ocr-vl-1.5"), torch.device("cpu"))

    assert seen["model"]["path"] == "PaddlePaddle/PaddleOCR-VL-1.5"
