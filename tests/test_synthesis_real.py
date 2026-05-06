"""Real OmniVoice inference + WAV write (slow; needs HF cache of ``k2-fsa/OmniVoice``).

WAV files are written under pytest's ``tmp_path`` (a unique folder under the OS temp
directory, e.g. ``%TEMP%\\pytest-of-...`` on Windows). That directory is **removed after
the test session**—there is no separate cleanup step in this repo; pytest handles it.
To inspect output, run a single test with ``pytest --basetemp=.\_pytest_artifacts`` (optional).
"""

from __future__ import annotations

import wave

import pytest
import soundfile as sf

from book2audio.synthesis import clear_model_cache, synthesize_to_numpy

pytestmark = pytest.mark.integration

# Fewer diffusion steps keeps CI/local runs tolerable; output quality is lower.
_NUM_STEP = 8


def test_synthesize_to_numpy_writes_real_wav(tmp_path) -> None:
    clear_model_cache()
    phrase = "Synthetic test phrase for Book2Audio."
    out = synthesize_to_numpy(phrase, "", num_step=_NUM_STEP)

    assert out is not None
    sample_rate, audio = out
    assert sample_rate == 24_000
    assert audio.ndim == 1
    assert audio.size > 2_000

    wav_path = tmp_path / "pytest_synthesis_real.wav"
    sf.write(str(wav_path), audio, sample_rate)
    assert wav_path.is_file()
    assert wav_path.stat().st_size > 1_000

    # Sanity: readable RIFF/WAVE with expected frame count
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 24_000
        assert w.getnframes() == int(audio.shape[0])


def test_synthesize_empty_returns_none() -> None:
    assert synthesize_to_numpy("", "") is None
    assert synthesize_to_numpy("   \n", "") is None
