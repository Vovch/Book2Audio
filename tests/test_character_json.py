"""Tests for character JSON parsing (no LLM / TTS)."""

import json

import pytest

from book2audio.audiobook_pipeline import (
    build_character_research_user_prompt,
    character_rows_to_editor_json,
    characters_from_external_llm_response,
    format_character_research_messages_for_external_llm,
    parse_character_json,
)


def test_parse_array_roundtrip():
    rows_in = [
        {"name": "A", "role": "r", "summary": "s", "voice_gender": "male"},
        {"name": "B", "role": "n", "summary": "m"},
    ]
    expected = [
        {"name": "A", "role": "r", "summary": "s", "voice_gender": "male"},
        {"name": "B", "role": "n", "summary": "m", "voice_gender": "male"},
    ]
    text = character_rows_to_editor_json(rows_in)
    out = parse_character_json(text)
    assert out == expected


def test_parse_drops_empty_optional_fields():
    rows_in = [{"name": "Z", "role": "", "summary": ""}]
    text = character_rows_to_editor_json(rows_in)
    out = parse_character_json(text)
    assert out == [{"name": "Z", "voice_gender": "male"}]


def test_parse_wrapped_characters_key():
    raw = json.dumps({"characters": [{"name": "X", "role": "", "summary": "hero"}]})
    out = parse_character_json(raw)
    assert len(out) == 1 and out[0]["name"] == "X" and out[0]["voice_gender"] == "male"


def test_parse_markdown_fenced_json():
    inner = json.dumps({"characters": [{"name": "Fence", "role": "", "summary": ""}]})
    raw = f"Here you go:\n```json\n{inner}\n```\n"
    out = parse_character_json(raw)
    assert len(out) == 1 and out[0]["name"] == "Fence" and out[0]["voice_gender"] == "male"


def test_parse_skips_empty_names():
    raw = '[{"name": "", "role": "n"}, {"name": "Z", "role": ""}]'
    out = parse_character_json(raw)
    assert len(out) == 1 and out[0]["name"] == "Z" and out[0]["voice_gender"] == "male"


def test_parse_empty_string_returns_empty():
    assert parse_character_json("") == []
    assert parse_character_json("   ") == []


def test_parse_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_character_json("not json")


def test_parse_not_a_list():
    with pytest.raises(ValueError, match="array"):
        parse_character_json('"hello"')


def test_external_character_prompt_skips_web_blob():
    _sys, user, copy_paste = format_character_research_messages_for_external_llm(
        "Pride and Prejudice", "Jane Austen"
    )
    assert "Web research" not in user
    assert "Web research" not in copy_paste
    assert "Pride and Prejudice" in user
    assert "research or browsing tools" in user
    assert "Use these as" not in copy_paste
    assert "===" not in copy_paste


def test_local_character_prompt_includes_web_blob():
    user = build_character_research_user_prompt("T", "A", "SNIPPET_LINE_X")
    assert "Web research" in user
    assert "SNIPPET_LINE_X" in user


def test_external_llm_wrapped_characters():
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "Pat",
                    "role": "Lead",
                    "summary": "Pilot",
                    "voice_gender": "female",
                    "voice_age": "young adult",
                    "voice_characteristics": "crisp",
                    "voice_accent": "Canadian",
                }
            ]
        }
    )
    rows = characters_from_external_llm_response(raw)
    assert len(rows) == 1
    assert rows[0]["name"] == "Pat"
    assert rows[0]["voice_accent"] == "Canadian"


def test_external_llm_top_level_array():
    raw = json.dumps([{"name": "Kim", "role": "x", "summary": "y"}])
    rows = characters_from_external_llm_response(raw)
    assert len(rows) == 1 and rows[0]["name"] == "Kim" and rows[0]["voice_gender"] == "male"


def test_external_llm_markdown_fence():
    inner = json.dumps({"characters": [{"name": "Fence", "role": "", "summary": ""}]})
    raw = f"Here you go:\n```json\n{inner}\n```\n"
    rows = characters_from_external_llm_response(raw)
    assert len(rows) == 1 and rows[0]["name"] == "Fence" and rows[0]["voice_gender"] == "male"


def test_external_llm_unknown_gender_becomes_male():
    raw = json.dumps({"characters": [{"name": "U", "voice_gender": "unknown", "role": "", "summary": ""}]})
    rows = characters_from_external_llm_response(raw)
    assert rows[0]["voice_gender"] == "male"


def test_external_llm_empty_paste():
    with pytest.raises(ValueError, match="empty"):
        characters_from_external_llm_response("   ")


def test_voice_fields_roundtrip():
    rows = [
        {
            "name": "Dr Watson",
            "role": "Narrator figure",
            "summary": "Physician",
            "voice_gender": "male",
            "voice_age": "middle-aged",
            "voice_characteristics": "measured, thoughtful",
            "voice_accent": "British",
        }
    ]
    text = character_rows_to_editor_json(rows)
    out = parse_character_json(text)
    assert out == rows
