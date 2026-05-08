"""Voice age from character JSON resolves to OmniVoice age tokens."""

from __future__ import annotations

from book2audio.audiobook_pipeline import (
    _normalize_character_row,
    _omnivoice_instruct_from_row,
    _voice_age_fragment_for_row,
    resolve_voice_age_token,
)


def test_resolve_voice_age_token_exact_and_aliases() -> None:
    assert resolve_voice_age_token("") == "middle-aged"
    assert resolve_voice_age_token("elderly") == "elderly"
    assert resolve_voice_age_token("Elderly") == "elderly"
    assert resolve_voice_age_token("teenager") == "teenager"
    assert resolve_voice_age_token("Teenager") == "teenager"
    assert resolve_voice_age_token("octogenarian") == "elderly"
    assert resolve_voice_age_token("adult") == "young adult"


def test_resolve_voice_age_token_unicode_dash_middle_aged() -> None:
    # en dash U+2013 normalized in _normalize_age_text
    assert resolve_voice_age_token("middle\u2013aged") == "middle-aged"


def test_resolve_voice_age_token_no_false_young_adult_on_middle_aged_adult() -> None:
    assert resolve_voice_age_token("middle-aged adult") == "middle-aged"


def test_voice_age_fragment_prefers_voice_age_over_age() -> None:
    row = {"voice_age": "child", "age": "elderly"}
    assert _voice_age_fragment_for_row(row) == "child"


def test_voice_age_fragment_falls_back_to_age() -> None:
    assert _voice_age_fragment_for_row({"age": "elderly"}) == "elderly"


def test_omnivoice_instruct_uses_age_key_when_no_voice_age() -> None:
    row = {
        "voice_gender": "male",
        "age": "elderly",
        "voice_characteristics": "",
        "voice_accent": "american",
    }
    instr = _omnivoice_instruct_from_row(row)
    assert "male" in instr
    assert "elderly" in instr


def test_omnivoice_instruct_child_omits_pitch():
    row = {
        "voice_gender": "female",
        "voice_age": "child",
        "voice_characteristics": "bright, high pitched",
        "voice_accent": "american",
    }
    instr = _omnivoice_instruct_from_row(row)
    assert "child" in instr
    assert "pitch" not in instr.lower()


def test_omnivoice_instruct_teenager_omits_pitch():
    row = {
        "voice_gender": "male",
        "voice_age": "teenager",
        "voice_characteristics": "deep voice",
        "voice_accent": "british",
    }
    instr = _omnivoice_instruct_from_row(row)
    assert "teenager" in instr
    assert "pitch" not in instr.lower()


def test_omnivoice_instruct_breathy_adds_whisper_not_as_pitch() -> None:
    row = {
        "voice_gender": "female",
        "voice_age": "young adult",
        "voice_characteristics": "breathy, bright",
        "voice_accent": "american",
    }
    instr = _omnivoice_instruct_from_row(row)
    assert "high pitch" in instr
    assert "whisper" in instr
    assert instr.index("high pitch") < instr.index("whisper")


def test_normalize_character_row_copies_age_into_voice_age() -> None:
    row = _normalize_character_row(
        {
            "name": "Zed",
            "age": "teenager",
            "voice_gender": "female",
        }
    )
    assert row is not None
    assert row.get("voice_age") == "teenager"
