"""Tests for omniscan.story.prompts."""

from __future__ import annotations

from omniscan.story.prompts import (
    MAX_SUMMARY_CHARS,
    SUMMARY_SCHEMA,
    SUMMARY_SYSTEM,
    parse_summary_reply,
    summary_messages,
)

LONG = "x" * (MAX_SUMMARY_CHARS + 10)


def test_summary_messages_is_a_system_user_pair_with_every_line() -> None:
    messages = summary_messages(["첫 문장", "둘째 문장"])
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": SUMMARY_SYSTEM}
    assert messages[1] == {"role": "user", "content": "Chapter lines (reading order):\n첫 문장\n둘째 문장"}


def test_summary_schema_shape() -> None:
    assert SUMMARY_SCHEMA == {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }


def test_parse_clean_json() -> None:
    assert parse_summary_reply('{"summary": "Minjun enters the gate."}') == "Minjun enters the gate."


def test_parse_fenced_json_with_tag() -> None:
    reply = '```json\n{"summary": "Minjun enters the gate."}\n```'
    assert parse_summary_reply(reply) == "Minjun enters the gate."


def test_parse_fenced_json_without_tag() -> None:
    reply = '```\n{"summary": "Minjun enters the gate."}\n```'
    assert parse_summary_reply(reply) == "Minjun enters the gate."


def test_parse_fenced_bare_prose_returns_the_prose() -> None:
    reply = "```\nMinjun enters the gate. A guard follows.\n```"
    assert parse_summary_reply(reply) == "Minjun enters the gate. A guard follows."


def test_parse_bare_prose_without_any_json_returns_truncated_prose() -> None:
    prose = "Minjun enters the gate. A guard follows."
    assert parse_summary_reply(prose) == prose
    assert parse_summary_reply(LONG) == "x" * MAX_SUMMARY_CHARS


def test_parse_whitespace_only_summary_falls_back_to_the_raw_text() -> None:
    raw = '{"summary": "  "}'
    assert parse_summary_reply(raw) == raw


def test_parse_empty_reply_is_none() -> None:
    assert parse_summary_reply("") is None
    assert parse_summary_reply("   \n  ") is None


def test_long_summary_is_truncated_in_every_branch() -> None:
    assert parse_summary_reply('{"summary": "' + LONG + '"}') == "x" * MAX_SUMMARY_CHARS
    assert parse_summary_reply("```json\n" + LONG + "\n```") == "x" * MAX_SUMMARY_CHARS
    assert parse_summary_reply(LONG) == "x" * MAX_SUMMARY_CHARS
