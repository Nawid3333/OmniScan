"""Release languages other than English (`[translate] target_lang`, issue #43): prompts, keys, config, metadata."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, get_args

import pytest

from omniscan.core.config import Config, PathsConfig, SeriesConfigError, series_config
from omniscan.core.paths import ChapterPaths, SeriesPaths
from omniscan.core.schemas import BBox, CandidateRun, Region, RegionsArtifact, TargetLang
from omniscan.llm.ollama import ChatResponse
from omniscan.packaging.cbz import pack_cbz
from omniscan.pipeline.stages import JudgeStage, TranslateStage
from omniscan.qa.typos import TypoChecker, release_language
from omniscan.translate.chapter import translate_chapter
from omniscan.translate.incremental import judge_key, translation_key
from omniscan.translate.judge_chapter import judge_chapter
from omniscan.translate.judge_config import JudgeConfig
from omniscan.translate.judge_prompts import judge_system
from omniscan.translate.languages import target_language
from omniscan.translate.profiles import TranslationProfile
from omniscan.translate.prompts import chat_json_system, translategemma_prompt

PROFILE = TranslationProfile(name="p", endpoint="local", model="m", style="chat_json")
JUDGE = JudgeConfig(model="judge-model")


def region(rid: str, text: str, order: int) -> Region:
    return Region(
        id=rid,
        slice_index=0,
        kind="bubble_text",
        bbox=BBox(x0=0, y0=order * 50, x1=10, y1=order * 50 + 10),
        reading_order=order,
        text=text,
    )


class EchoClient:
    """chat_json: answers `<tag>:<source>` per region; judge: picks candidate A. Records the system prompts."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.systems: list[str] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        self.systems.append(messages[0]["content"])
        items = json.loads(messages[-1]["content"].split("Regions (reading order):\n", 1)[1])
        if "judgements" in json.dumps(kwargs.get("format")):
            answer = {"judgements": [{"id": r["id"], "decision": "pick", "pick": "A"} for r in items]}
        else:
            answer = {"translations": [{"id": r["id"], "text": f"{self.tag}:{r['text']}"} for r in items]}
        return ChatResponse(json.dumps(answer, ensure_ascii=False), model, True, None, 1, 1, {})


@pytest.fixture
def paths(tmp_path: Path) -> ChapterPaths:
    return ChapterPaths("S", "Chapter 1", tmp_path / "r", tmp_path / "w", tmp_path / "o", tmp_path / "f")


def _cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            library_root=tmp_path / "lib",
            work_root=tmp_path / "work",
            output_root=tmp_path / "out",
            promo_examples=tmp_path / "promo",
            models_dir=tmp_path / "models",
        )
    )


def test_every_release_language_has_prompt_names() -> None:
    for code in get_args(TargetLang):
        assert target_language(code).code == code
    with pytest.raises(ValueError, match="unknown target language"):
        target_language("xx")


@pytest.mark.parametrize("system", [chat_json_system, judge_system])
def test_a_german_release_asks_for_german_only(system: Any) -> None:
    prompt = system("ko", "de")
    assert "official German release of a Korean manhwa" in prompt
    assert "English" not in prompt
    assert "BUMM" in prompt and "BOOM" not in prompt  # sound effects the release letters in German
    assert "unless that reads badly in German" in prompt  # the honorifics sentence names the release too
    assert system("ko") == system("ko", "en")  # English stays the default


def test_translategemma_names_both_languages() -> None:
    prompt = translategemma_prompt("안녕", "ko", "es")
    assert "Korean (ko) to Spanish (es) translator" in prompt
    assert "English" not in prompt
    assert prompt.endswith("\n\n\n안녕")


def test_keys_of_an_english_release_are_unchanged_and_other_languages_differ() -> None:
    r = region("r0001", "철수가 왔다", 0)
    runs = {"p": {"r0001": "Cheolsu came"}}
    assert translation_key(r, [], PROFILE) == translation_key(r, [], PROFILE, target="en")
    assert translation_key(r, [], PROFILE, target="de") != translation_key(r, [], PROFILE)
    assert translation_key(r, [], PROFILE, target="de") != translation_key(r, [], PROFILE, target="es")
    assert judge_key(r, runs, [], JUDGE) == judge_key(r, runs, [], JUDGE, "en")
    assert judge_key(r, runs, [], JUDGE, "de") != judge_key(r, runs, [], JUDGE)


def test_switching_the_release_language_translates_and_judges_again(paths: ChapterPaths) -> None:
    RegionsArtifact(regions=[region("r0001", "철수가 왔다", 0), region("r0002", "가자", 1)]).save(
        paths.artifact("ocr.json")
    )
    translate_chapter(EchoClient("EN"), paths, PROFILE, [], force=True, reuse=True)
    german = EchoClient("DE")
    translate_chapter(german, paths, PROFILE, [], force=True, reuse=True, target="de")
    assert len(german.systems) == 1  # nothing was reused from the English run
    assert "official German release" in german.systems[0]
    run = CandidateRun.load(paths.artifact("translations/p.json"))
    assert [c.text for c in run.candidates] == ["DE:철수가 왔다", "DE:가자"]

    judge = EchoClient("J")
    _status, final, _stats = judge_chapter(judge, paths, JUDGE, [], force=True, reuse=True, target="de")
    assert final is not None
    assert all("official German release" in system for system in judge.systems)
    again = EchoClient("J")
    judge_chapter(again, paths, JUDGE, [], force=True, reuse=True, target="de")
    assert again.systems == []  # same language, same candidates: every judged line is kept


def test_series_toml_sets_the_release_language(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.translate.target_lang == "en"
    series_dir = tmp_path / "lib" / "S"
    series_dir.mkdir(parents=True)
    (series_dir / "series.toml").write_text('[translate]\ntarget_lang = "de"\n', encoding="utf-8")
    assert series_config(cfg, series_dir).translate.target_lang == "de"
    (series_dir / "series.toml").write_text('[translate]\ntarget_lang = "xx"\n', encoding="utf-8")
    with pytest.raises(SeriesConfigError):
        series_config(cfg, series_dir)


def test_stage_hashes_change_only_for_another_release_language(tmp_path: Path) -> None:
    english = _cfg(tmp_path)
    german = english.model_copy(
        update={"translate": english.translate.model_copy(update={"target_lang": "de"})}
    )
    translate = TranslateStage(object(), [PROFILE])  # type: ignore[arg-type]
    judge = JudgeStage(object(), JUDGE)  # type: ignore[arg-type]
    for stage in (translate, judge):
        assert "target_lang" not in stage.config_subset(english)  # English hashes as before
        assert stage.config_subset(german)["target_lang"] == "de"


def test_cbz_metadata_names_the_release_language(tmp_path: Path) -> None:
    page = tmp_path / "0001.jpg"
    page.write_bytes(b"\xff\xd8\xff\xd9")
    dest = tmp_path / "c.cbz"
    pack_cbz([page], dest, language="de")
    with zipfile.ZipFile(dest) as zf:
        assert b"<LanguageISO>de</LanguageISO>" in zf.read("ComicInfo.xml")


def test_the_typo_check_reads_the_release_language(tmp_path: Path) -> None:
    german = TypoChecker([], "de")
    assert [t.word for t in german.check("Das ist großartig, aber falsh.")] == [
        "falsh"
    ]  # ß and umlauts are letters
    assert "falsch" in german.check("Das ist falsh.")[0].suggestions
    spanish = TypoChecker([], "es")
    assert [t.word for t in spanish.check("¡Encontré la espda! Es increíble.")] == ["espda"]
    assert [t.word for t in TypoChecker().check("I found teh sword.")] == ["teh"]  # English as before

    series = SeriesPaths.from_config(_cfg(tmp_path), "S")
    series.library_dir.mkdir(parents=True)
    assert release_language(series) == "en"
    (series.library_dir / "series.toml").write_text('[translate]\ntarget_lang = "es"\n', encoding="utf-8")
    assert release_language(series) == "es"
    (series.library_dir / "series.toml").write_text("[nonsense]\n", encoding="utf-8")
    assert release_language(series) == "en"  # a broken series.toml is the stages' to report
