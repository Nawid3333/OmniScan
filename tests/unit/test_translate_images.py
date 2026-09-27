"""Tests for page images as context for the translation model (translate/images.py and its use; torch-free)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import omniscan.translate.images as images_module
from omniscan.core.config import OllamaConfig, Secrets
from omniscan.core.paths import ChapterPaths
from omniscan.core.schemas import (
    BBox,
    GlossaryEntry,
    IngestArtifact,
    Region,
    Slice,
    SlicesArtifact,
    SourceFile,
)
from omniscan.llm.ollama import ChatResponse, OllamaClient
from omniscan.translate.images import PageImages
from omniscan.translate.incremental import translation_key
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import chat_json_messages
from omniscan.translate.run import run_profile
from omniscan.translate.suggest import suggest

WIDTH, PAGE_H = 400, 900  # two raw pages, three slices of 600 rows


class EchoClient:
    """Records every request and answers each region with EN(<source>)."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **_: Any) -> ChatResponse:
        self.calls.append(messages)
        regions = json.loads(messages[-1]["content"].split("Regions (reading order):\n", 1)[1])
        answer = {"translations": [{"id": r["id"], "text": f"EN({r['text']})"} for r in regions]}
        return ChatResponse(json.dumps(answer, ensure_ascii=False), model, True, None, 1, 1, {})


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    paths = ChapterPaths(
        series="S",
        chapter="Chapter 1",
        raw_dir=tmp_path / "lib" / "S" / "Chapter 1",
        work_dir=tmp_path / "work" / "S" / "Chapter 1",
        output_dir=tmp_path / "out" / "S" / "Chapter 1",
        filtered_dir=tmp_path / "out" / "S" / "_filtered" / "Chapter 1",
    )
    paths.raw_dir.mkdir(parents=True)
    for i, shade in enumerate((60, 180)):
        pixels = np.full((PAGE_H, WIDTH, 3), shade, np.uint8)
        pixels[100:160, 50:350] = 255  # a bright band marks where the page is
        Image.fromarray(pixels).save(paths.raw_dir / f"{i + 1:03d}.jpg", quality=95)
    IngestArtifact(
        series="S",
        chapter="Chapter 1",
        strip_width=WIDTH,
        strip_height=2 * PAGE_H,
        files=[
            SourceFile(
                index=i,
                name=f"{i + 1:03d}.jpg",
                sha256="0" * 64,
                width=WIDTH,
                height=PAGE_H,
                y0=i * PAGE_H,
                y1=(i + 1) * PAGE_H,
            )
            for i in range(2)
        ],
    ).save(paths.artifact("ingest.json"))
    SlicesArtifact(
        strip_width=WIDTH,
        strip_height=2 * PAGE_H,
        bands=[],
        slices=[Slice(index=i, y0=600 * i, y1=600 * (i + 1)) for i in range(3)],
    ).save(paths.artifact("slices.json"))
    return paths


def region(rid: str, slice_index: int, y0: int, text: str) -> Region:
    return Region(
        id=rid,
        slice_index=slice_index,
        kind="bubble_text",
        bbox=BBox(x0=40, y0=y0, x1=240, y1=y0 + 80),
        text=text,
    )


REGIONS = [
    region("r0001", 0, 100, "가자"),
    region("r0002", 0, 400, "어디로?"),
    region("r0003", 1, 700, "게이트로"),
    region("r0004", 2, 1300, "늦었어"),
]


def profile(**fields: Any) -> TranslationProfile:
    return TranslationProfile(
        name="vision", endpoint="local", model="gemma4:12b", style="chat_json", **fields
    )


def decoded(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


def test_requests_carry_the_pages_their_regions_are_on(paths: ChapterPaths) -> None:
    client = EchoClient()
    run = run_profile(
        client,
        profile(images=True, image_side=300, images_per_request=2),
        REGIONS,
        [],
        images=PageImages(paths),
    )
    assert [c.text for c in run.candidates] == ["EN(가자)", "EN(어디로?)", "EN(게이트로)", "EN(늦었어)"]
    first, second = client.calls  # slices 0 and 1 in one request, slice 2 in the next
    images = [decoded(data) for data in first[-1]["images"]]
    assert [image.size for image in images] == [(200, 300), (200, 300)]  # 400 x 600 slices, long side 300
    assert decoded(second[-1]["images"][0]).size == (200, 300)
    assert np.asarray(images[0].convert("L"))[65, 100] > 200  # the bright band of page 1, scaled by 0.5
    sent = json.loads(first[-1]["content"].split("Regions (reading order):\n", 1)[1])
    assert [(r["id"], r["image"], r["box"]) for r in sent] == [
        ("r0001", 0, [20, 50, 120, 90]),
        ("r0002", 0, [20, 200, 120, 240]),
        ("r0003", 1, [20, 50, 120, 90]),  # strip row 700 is row 100 of slice 1
    ]
    assert "Page images: the attached images show the pages" in first[-1]["content"]
    assert "images" not in first[0]  # only the user message carries them


def test_a_profile_without_images_sends_exactly_what_it_sent_before(paths: ChapterPaths) -> None:
    client = EchoClient()
    run_profile(client, profile(chunk_regions=3), REGIONS, [], images=PageImages(paths))
    assert [
        len(json.loads(c[-1]["content"].split("Regions (reading order):\n", 1)[1])) for c in client.calls
    ] == [3, 1]
    assert client.calls[0] == chat_json_messages(REGIONS[:3], [])  # no images, no image fields, same chunks
    assert all("images" not in message for call in client.calls for message in call)


def test_keys_change_only_for_profiles_that_send_images() -> None:
    line = Region(
        id="r0001",
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        text="진우야, 가자!",
    )
    entries = [GlossaryEntry(id=1, source="진우", target="Jinwoo", status="locked")]
    plain = TranslationProfile(name="p", endpoint="cloud", model="gemma4:31b-cloud", style="chat_json")
    before_images_existed = "546526f4c58d71995eb38a2e50cc9b65b566096558d207aec86c083c5558512e"
    assert translation_key(line, entries, plain) == before_images_existed
    assert (
        translation_key(line, entries, plain.model_copy(update={"image_side": 2048})) == before_images_existed
    )
    seeing = plain.model_copy(update={"images": True})
    assert translation_key(line, entries, seeing) != before_images_existed
    assert translation_key(line, entries, seeing.model_copy(update={"image_side": 2048})) != translation_key(
        line, entries, seeing
    )


def test_unreadable_pages_translate_without_images(
    paths: ChapterPaths, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    for page in paths.raw_dir.iterdir():
        page.unlink()
    reads: list[int] = []
    real = images_module.strip_crop

    def counted(*args: Any) -> Any:
        reads.append(1)
        return real(*args)

    monkeypatch.setattr(images_module, "strip_crop", counted)
    client = EchoClient()
    run = run_profile(
        client, profile(images=True, images_per_request=1), REGIONS, [], images=PageImages(paths)
    )
    assert len(run.candidates) == 4 and len(client.calls) == 3  # still cut to one page per request
    assert all("images" not in message for call in client.calls for message in call)
    assert sum("translating without page images" in r.message for r in caplog.records) == 1  # said once
    assert len(reads) == 1  # given up after the first failed read, not retried per request


def test_studio_suggestions_see_the_page_too(paths: ChapterPaths) -> None:
    client = EchoClient()
    found = suggest(client, [profile(images=True)], REGIONS, ["r0004"], [], {}, images=PageImages(paths))
    assert [s.text for s in found] == ["EN(늦었어)"]
    (call,) = client.calls
    assert len(call[-1]["images"]) == 1 and '"image": 0' in call[-1]["content"]


def test_images_need_a_chat_json_profile() -> None:
    with pytest.raises(ValidationError, match="images need style"):
        TranslationProfile(
            name="tg", endpoint="local", model="translategemma:12b", style="translategemma", images=True
        )


def test_a_local_model_gets_room_for_the_images() -> None:
    client = OllamaClient(OllamaConfig(num_ctx=4096), Secrets())
    text = [{"role": "user", "content": "x" * 100}]
    with_images = [{"role": "user", "content": "x" * 100, "images": ["a", "b", "c"]}]
    plain = client._with_num_ctx("gemma4:12b", text, None, cloud=False)
    seeing = client._with_num_ctx("gemma4:12b", with_images, None, cloud=False)
    assert plain is not None and seeing is not None and seeing["num_ctx"] > plain["num_ctx"]
