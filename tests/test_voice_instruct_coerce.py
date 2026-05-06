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
