"""Speaker label → voice clone resolution (no TTS)."""

from book2audio.audiobook_pipeline import (
    _normalize_segment_speaker_label,
    resolve_clone_prompt_for_speaker,
)


def _p():
    return object()


def test_resolve_exact_key():
    p = _p()
    d = {"Alice Mercer": p}
    got, key = resolve_clone_prompt_for_speaker(d, "Alice Mercer")
    assert got is p and key == "Alice Mercer"


def test_resolve_case_insensitive():
    p = _p()
    d = {"Alice Mercer": p}
    got, key = resolve_clone_prompt_for_speaker(d, "alice mercer")
    assert got is p and key == "Alice Mercer"


def test_normalize_strips_label_punctuation():
    assert _normalize_segment_speaker_label(' "Bob": ') == "Bob"


def test_fallback_narrator():
    pn, pa = _p(), _p()
    d = {"Narrator": pn, "Sam": pa}
    got, key = resolve_clone_prompt_for_speaker(d, "not-in-cast", fallback_key="Narrator")
    assert got is pn and key == "Narrator"


def test_fallback_case_insensitive_narrator():
    pn = _p()
    d = {"narrator": pn}
    got, key = resolve_clone_prompt_for_speaker(d, "Not anyone", fallback_key="Narrator")
    assert got is pn and key == "narrator"
