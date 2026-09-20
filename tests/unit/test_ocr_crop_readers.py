"""decode_wordpieces / clean_manga_text / MangaOcrReader against fake model+processor (card O1b, tests 4-5)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from omniscan.core.config import OcrConfig
from omniscan.ocr.crop_readers import MangaOcrReader, clean_manga_text, decode_wordpieces

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "", "あ", "い", "##う", "、"]


# ---------------------------------------------------------------- decode_wordpieces (test 4)


def test_decode_wordpieces_goldens() -> None:
    assert decode_wordpieces([2, 5, 6, 7, 8, 3, 5], VOCAB) == "あいう、"
    assert decode_wordpieces([2, 3], VOCAB) == ""


def test_decode_wordpieces_edge_cases() -> None:
    assert decode_wordpieces([7, 5], VOCAB) == "うあ"  # `##` at the start contributes its remainder
    assert decode_wordpieces([1, 4, 0, 2, 5], VOCAB) == "あ"  # specials and  skipped
    assert decode_wordpieces([5, 3, 6], VOCAB) == "あ"  # decoding stops at the first [SEP]
    assert decode_wordpieces([], VOCAB) == ""


def test_decode_wordpieces_out_of_range_id_raises() -> None:
    with pytest.raises(IndexError):
        decode_wordpieces([99], VOCAB)
    with pytest.raises(IndexError):
        decode_wordpieces([-1], VOCAB)


# ---------------------------------------------------------------- clean_manga_text (test 4)


def test_clean_manga_text_goldens() -> None:
    assert clean_manga_text("あれ、 しまった…") == "あれ、しまった..."
    assert clean_manga_text("・・・強い風の音...") == "...強い風の音..."
    assert clean_manga_text("コモナ!") == "コモナ！"
    assert clean_manga_text("ABC 123") == "ＡＢＣ１２３"
    assert clean_manga_text("…") == "..."


def test_clean_manga_text_edges() -> None:
    assert clean_manga_text("") == ""
    assert clean_manga_text("   \t\n ") == ""  # whitespace only, tabs and newlines included
    assert clean_manga_text("あ\tい\nう") == "あいう"  # tab and newline are whitespace
    assert clean_manga_text("ＡＢab12") == "ＡＢａｂ１２"  # already full width stays, half width widens
    assert clean_manga_text("x.y~") == "ｘ.ｙ~"  # `.` and `~` stay half width, letters widen
    assert clean_manga_text("・.") == ".."  # a mixed run of two becomes two dots


# ---------------------------------------------------------------- fakes (test 5)


class FakeGenerateModel(torch.nn.Module):
    """One scripted result per generate call: sequences, transition scores, optional raise."""

    def __init__(self, calls: list[dict[str, Any]], *, pad_id: int | None = 0) -> None:
        super().__init__()
        self.calls = calls
        self.pixel_shapes: list[tuple[int, ...]] = []
        self.inference_flags: list[bool] = []
        self.config = SimpleNamespace(pad_token_id=pad_id)

    def generate(self, *, pixel_values: torch.Tensor, **kwargs: Any) -> Any:
        assert kwargs["output_scores"] is True and kwargs["return_dict_in_generate"] is True
        self.pixel_shapes.append(tuple(pixel_values.shape))
        self.inference_flags.append(torch.is_inference_mode_enabled())
        call = self.calls[len(self.pixel_shapes) - 1]
        return SimpleNamespace(
            sequences=torch.tensor(call["sequences"]),
            scores=(object(),),
            beam_indices=torch.zeros(len(call["sequences"]), len(call["logs"][0]), dtype=torch.long),
        )

    def compute_transition_scores(
        self, sequences: Any, scores: Any, beam_indices: Any, normalize_logits: bool
    ) -> torch.Tensor:
        assert normalize_logits is False
        call = self.calls[len(self.pixel_shapes) - 1]
        if call.get("raise"):
            raise RuntimeError("no transition scores")
        return torch.tensor(call["logs"])


class FakeProcessor:
    """Records the chunk it receives; every crop becomes a 224x224 pixel_values row."""

    def __init__(self, *, need_pil: bool = False) -> None:
        self.images: list[Any] = []
        self._need_pil = need_pil

    def __call__(self, images: Any, return_tensors: str | None = None, **kwargs: Any) -> dict[str, Any]:
        assert return_tensors == "pt"
        if self._need_pil:
            if not isinstance(images[0], Image.Image):
                raise TypeError("PIL images required")
        else:
            assert all(isinstance(image, torch.Tensor) for image in images)
        self.images.append(list(images))
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}


def make_reader(
    calls: list[dict[str, Any]], *, batch_size: int = 2, pad_id: int | None = 0, processor: Any | None = None
) -> tuple[MangaOcrReader, FakeGenerateModel]:
    model = FakeGenerateModel(calls, pad_id=pad_id)
    reader = MangaOcrReader(
        model, processor or FakeProcessor(), VOCAB, torch.device("cpu"), batch_size=batch_size
    )
    return reader, model


def crops(n: int) -> list[torch.Tensor]:
    return [torch.full((3, 10 + i, 20), i, dtype=torch.uint8) for i in range(n)]


# ---------------------------------------------------------------- MangaOcrReader.read (test 5)


def test_read_batches_in_order_and_scores_mean_log_probs() -> None:
    calls = [
        {
            "sequences": [[2, 5, 6, 3], [2, 5, 3, 0]],
            "logs": [[-0.2, -0.3, -0.4], [-0.1, -0.2, -0.3]],
        },
        {
            "sequences": [[2, 8, 3, 0], [2, 3, 0, 0]],
            "logs": [[-0.05, -0.1, -0.2], [-0.4, -0.4, -0.4]],
        },
        {"sequences": [[2, 5, 0, 0]], "logs": [[-0.25, -0.9, -0.9]]},
    ]
    reader, model = make_reader(calls, batch_size=2)

    out = reader.read(crops(5))

    assert out == [
        ("あい", 0.7408),  # exp(-(0.2+0.3+0.4)/3)
        ("あ", 0.8607),  # exp(-(0.1+0.2)/2): the trailing [PAD] is not a generated position
        ("、", 0.9277),  # exp(-(0.05+0.1)/2)
        ("", 0.0),  # empty text scores 0.0
        ("あ", 0.7788),  # exp(-0.25): only the first generated position survived
    ]
    assert model.pixel_shapes == [(2, 3, 224, 224), (2, 3, 224, 224), (1, 3, 224, 224)]
    assert all(model.inference_flags)


def test_read_no_crops_never_calls_the_model() -> None:
    reader, model = make_reader([])
    assert reader.read([]) == []
    assert model.pixel_shapes == []


def test_read_transition_scores_raising_scores_1() -> None:
    calls = [{"sequences": [[2, 5, 6, 3], [2, 3, 0, 0]], "logs": [[0.0, 0.0, 0.0]], "raise": True}]
    reader, _model = make_reader(calls, batch_size=2)

    assert reader.read(crops(2)) == [("あい", 1.0), ("", 0.0)]


def test_read_1x1_crop_does_not_crash() -> None:
    calls = [{"sequences": [[2, 5, 3, 0]], "logs": [[-0.5, -0.5, -0.5]]}]
    reader, _model = make_reader(calls)

    assert reader.read([torch.zeros(3, 1, 1, dtype=torch.uint8)]) == [("あ", 0.6065)]


def test_read_falls_back_to_pil_when_the_processor_refuses_tensors() -> None:
    calls = [{"sequences": [[2, 5, 3, 0]], "logs": [[-0.1, -0.2, -0.3]]}]
    processor = FakeProcessor(need_pil=True)
    reader, _model = make_reader(calls, processor=processor)

    assert reader.read([torch.zeros(3, 4, 8, dtype=torch.uint8)]) == [("あ", 0.8607)]
    assert all(isinstance(image, Image.Image) for image in processor.images[0])


def test_load_without_rec_model_raises() -> None:
    with pytest.raises(ValueError, match=r"manga_ocr needs ocr\.rec_model"):
        MangaOcrReader.load(OcrConfig(engine="ppocr"), torch.device("cpu"))
