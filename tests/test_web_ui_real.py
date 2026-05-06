"""Gradio handler :func:`_synthesize_for_ui` exercises real ``synthesis`` (integration).

Output WAVs use ``tmp_path`` (see :mod:`test_synthesis_real` docstring): ephemeral, deleted
after pytest finishes unless you use ``--basetemp``.
"""

from __future__ import annotations

from functools import partial
from unittest.mock import patch

import pytest
import soundfile as sf

import book2audio.synthesis as synthesis_mod
from book2audio.web_ui import _synthesize_for_ui

pytestmark = pytest.mark.integration

_NUM_STEP = 8


@pytest.fixture(autouse=True)
def _fast_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep handler path aligned with synthesis while using fewer diffusion steps."""
    monkeypatch.setattr(
        "book2audio.web_ui.synthesize_to_numpy",
        partial(synthesis_mod.synthesize_to_numpy, num_step=_NUM_STEP),
    )


@patch("book2audio.web_ui.gr.Warning")
def test_ui_empty_text_calls_warning_and_returns_none(mock_warning) -> None:
    assert _synthesize_for_ui("", "") is None
    mock_warning.assert_called_once()

    mock_warning.reset_mock()
    assert _synthesize_for_ui("  \t", "female") is None
    mock_warning.assert_called_once()


def test_ui_handler_produces_real_wav(tmp_path) -> None:
    """Handler runs the same stack as production Gradio (real inference)."""
    text = "Handler should write valid speech output."
    out = _synthesize_for_ui(text, "")

    assert out is not None
    rate, audio = out
    assert rate == 24_000
    assert audio.ndim == 1
    assert audio.size > 2_000

    ui_wav = tmp_path / "pytest_web_ui_real.wav"
    sf.write(str(ui_wav), audio, rate)
    assert ui_wav.stat().st_size > 1_000


def test_ui_handler_passes_through_instruct(tmp_path) -> None:
    out = _synthesize_for_ui("Short line.", "male")
    assert out is not None
    rate, audio = out
    assert rate == 24_000
    assert audio.size > 500
    sf.write(str(tmp_path / "pytest_web_ui_instruct.wav"), audio, rate)
