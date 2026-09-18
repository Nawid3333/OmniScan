"""Tests for omniscan.translate.parse."""

from __future__ import annotations

import pytest

from omniscan.translate.parse import extract_translations, parse_translations

IDS = ["r0001", "r0002", "r0003"]

PLAIN = '{"translations":[{"id":"r0001","text":" Hello "},{"id":"r0002","text":"Hi"}]}'


def test_extract_plain_json() -> None:
    items = extract_translations(PLAIN)
    assert items == [{"id": "r0001", "text": " Hello "}, {"id": "r0002", "text": "Hi"}]


def test_extract_json_fence() -> None:
    content = f"```json\n{PLAIN}\n```"
    assert extract_translations(content) == extract_translations(PLAIN)


def test_extract_bare_fence() -> None:
    content = f"```\n{PLAIN}\n```"
    assert extract_translations(content) == extract_translations(PLAIN)


def test_extract_prose_around_json() -> None:
    content = f"Sure! Here are the translations:\n\n{PLAIN}\n\nHope that helps."
    assert extract_translations(content) == [
        {"id": "r0001", "text": " Hello "},
        {"id": "r0002", "text": "Hi"},
    ]


def test_extract_second_object_with_translations_wins() -> None:
    content = 'Some preamble {"unrelated": true} mid {"translations":[{"id":"r0001","text":"A"}]}'
    assert extract_translations(content) == [{"id": "r0001", "text": "A"}]


def test_extract_translations_not_a_list_returns_none() -> None:
    assert extract_translations('{"translations": {"id": "r0001"}}') is None


def test_extract_garbage_returns_none() -> None:
    assert extract_translations("no json at all") is None


def test_extract_empty_string_returns_none() -> None:
    assert extract_translations("") is None
    assert extract_translations("   \n  ") is None


def test_parse_strips_and_drops_unknown_and_malformed() -> None:
    content = (
        '{"translations": ['
        '{"id": "r0001", "text": "  A  "},'
        '{"id": "r0009", "text": "extra"},'
        '{"id": "r0002"},'
        '{"id": 3, "text": "no"},'
        '{"id": "r0003", "text": 42},'
        '"junk",'
        '{"id": "r0002", "text": "Second wins? No — first usable wins"}'
        "]}"
    )
    result = parse_translations(content, IDS)
    assert result.texts == {"r0001": "A", "r0002": "Second wins? No — first usable wins"}


def test_parse_whitespace_only_text_is_missing() -> None:
    result = parse_translations(
        '{"translations":[{"id":"r0001","text":"  \\n "},{"id":"r0002","text":"ok"}]}', IDS
    )
    assert result.texts == {"r0002": "ok"}
    assert result.missing == ["r0001", "r0003"]


def test_parse_missing_order_follows_expected_ids() -> None:
    result = parse_translations(PLAIN, ["r0003", "r0001", "r0002"])
    assert result.missing == ["r0003"]


def test_parse_unparseable_everything_missing() -> None:
    result = parse_translations("total garbage", IDS)
    assert result.texts == {}
    assert result.missing == IDS


def test_parse_duplicated_id_first_wins() -> None:
    content = '{"translations":[{"id":"r0001","text":"first"},{"id":"r0001","text":"second"}]}'
    result = parse_translations(content, IDS)
    assert result.texts == {"r0001": "first"}


@pytest.mark.parametrize("content", ["[]", '{"other": 1}'])
def test_extract_non_qualifying_json(content: str) -> None:
    assert extract_translations(content) is None


def test_extract_empty_translations_list_is_still_a_list() -> None:
    assert extract_translations('{"translations": []}') == []
