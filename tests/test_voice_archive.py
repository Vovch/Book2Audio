"""Voice archive JSON + WAV pack (no OmniVoice unless patched)."""

import io
import json
import wave
import zipfile
from unittest.mock import patch

import numpy as np

from book2audio.audiobook_pipeline import (
    VOICE_ARCHIVE_JSON,
    AudiobookPlan,
    CharacterCard,
    import_voice_archive_from_zip_bytes,
    read_wav_bytes_mono_float32,
    safe_voice_archive_basename,
    voice_archive_dict_from_plan,
)


def _pcm_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24_000)
        wf.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


def test_read_wav_roundtrip():
    raw = _pcm_wav_bytes()
    sr, mono = read_wav_bytes_mono_float32(raw)
    assert sr == 24_000
    assert mono.shape == (160,)
    assert np.max(np.abs(mono)) == 0.0


def test_voice_archive_dict_contains_research_and_rows():
    plan = AudiobookPlan()
    plan.raw_character_rows = [{"name": "Pat", "role": "Lead", "summary": "Pilot", "voice_gender": "female"}]
    plan.characters = [
        CharacterCard(
            name="Pat",
            role="Lead",
            summary="Pilot",
            voice_instruct="female, young adult, moderate pitch, american accent",
        )
    ]
    plan.character_research_blob = "snippet: foo"
    plan.voice_samples["Pat"] = (24_000, np.zeros(10, dtype=np.float32))

    d = voice_archive_dict_from_plan(plan)
    assert d["book2audio_archive_version"] == 1
    assert d["character_research_blob"] == "snippet: foo"
    assert d["voice_files"]["Pat"] == f"{safe_voice_archive_basename('Pat')}.wav"
    assert len(d["character_cards"]) == 1
    assert d["character_cards"][0]["name"] == "Pat"


def test_zip_layout_has_json_and_wav_names():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(VOICE_ARCHIVE_JSON, '{"x": 1}')
        zf.writestr("a.wav", _pcm_wav_bytes())
    names = zipfile.ZipFile(io.BytesIO(buf.getvalue()), "r").namelist()
    assert VOICE_ARCHIVE_JSON in names
    assert any(n.endswith(".wav") for n in names)


@patch(
    "book2audio.audiobook_pipeline.build_voice_clone_from_reference_audio",
    return_value=(object(), np.zeros(8, dtype=np.float32), 24_000),
)
def test_import_wav_only_zip_builds_cast_from_stems(_mock_clone):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("Zora.wav", _pcm_wav_bytes())
        zf.writestr("Amy.wav", _pcm_wav_bytes())
    plan = import_voice_archive_from_zip_bytes(zbuf.getvalue())
    assert {c.name for c in plan.characters} == {"Amy", "Zora"}
    assert set(plan.clone_prompts.keys()) == {"Amy", "Zora"}


@patch(
    "book2audio.audiobook_pipeline.build_voice_clone_from_reference_audio",
    return_value=(object(), np.zeros(8, dtype=np.float32), 24_000),
)
def test_import_full_archive_with_manifest(_mock_clone):
    wav_name = f"{safe_voice_archive_basename('Pat')}.wav"
    manifest = {
        "book2audio_archive_version": 1,
        "raw_character_rows": [
            {"name": "Pat", "role": "x", "summary": "y", "voice_gender": "female"}
        ],
        "character_cards": [
            {
                "name": "Pat",
                "role": "x",
                "summary": "y",
                "voice_instruct": "female, young adult, moderate pitch, american accent",
            }
        ],
        "narrator_instruct": "male, middle-aged, moderate pitch, american accent",
        "character_research_blob": "WEB",
        "voice_files": {"Pat": wav_name},
    }
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr(VOICE_ARCHIVE_JSON, json.dumps(manifest))
        zf.writestr(wav_name, _pcm_wav_bytes())
    plan = import_voice_archive_from_zip_bytes(zbuf.getvalue())
    assert plan.character_research_blob == "WEB"
    assert plan.raw_character_rows[0]["name"] == "Pat"
    assert "Pat" in plan.clone_prompts
