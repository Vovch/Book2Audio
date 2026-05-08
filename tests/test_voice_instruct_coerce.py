"""Voice instruct coercion for OmniVoice (no model load)."""

from book2audio.synthesis import coerce_voice_instruct_for_omnivoice


def test_coerce_drops_invalid_llm_phrases():
    messy = (
        "male, middle-aged, high-pitched, fussy, breathless, anxious, "
        "speaks quickly, constantly late, afraid, reprimanded, RP British"
    )
    out = coerce_voice_instruct_for_omnivoice(messy)
    assert "fussy" not in out.lower()
    assert "high-pitched" not in out.lower()
    assert "high pitch" in out.lower()
    assert "british accent" in out.lower()
    assert "male" in out.lower()
    assert "middle-aged" in out.lower()


def test_coerce_empty_returns_fallback_shape():
    out = coerce_voice_instruct_for_omnivoice("")
    assert "male" in out and "american accent" in out


def test_resolved_en_to_chinese_voice_design_drops_accent_and_maps_age():
    from book2audio import synthesis as syn

    zh = syn._resolved_english_instruct_to_chinese_voice_design(
        "male, elderly, moderate pitch, american accent"
    )
    assert "american accent" not in zh
    assert "\u8001\u5e74" in zh  # 老年


def test_resolved_en_to_chinese_voice_design_child_drops_pitch():
    from book2audio import synthesis as syn

    zh = syn._resolved_english_instruct_to_chinese_voice_design(
        "female, child, high pitch, american accent"
    )
    assert "\u9ad8\u97f3\u8c03" not in zh  # 高音调
    assert "\u513f\u7ae5" in zh  # 儿童


def test_coerce_strips_pitch_for_child_and_teenager():
    out_c = coerce_voice_instruct_for_omnivoice(
        "female, child, high pitch, british accent"
    )
    assert "child" in out_c
    assert "high pitch" not in out_c.lower()
    assert "british accent" in out_c.lower()
    out_t = coerce_voice_instruct_for_omnivoice(
        "male, teenager, very low pitch, american accent"
    )
    assert "teenager" in out_t
    assert "pitch" not in out_t.lower()


def test_omnivoice_resolve_maps_zh_age_to_english_when_accent_present():
    from omnivoice.models.omnivoice import _resolve_instruct

    out = _resolve_instruct("male, \u8001\u5e74, moderate pitch, american accent")
    assert "elderly" in out
