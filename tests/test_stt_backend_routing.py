"""STT backend routing (ONNX vs HTTP) without loading models."""

from __future__ import annotations

import numpy as np
import pytest

from book2audio.stt_parakeet import transcribe_reference_float32


def test_auto_uses_http_when_onnx_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOK2AUDIO_STT_BACKEND", raising=False)
    monkeypatch.setattr("book2audio.stt_parakeet._onnx_importable", lambda: False)
    calls: list[bytes] = []

    def fake_http(wav: bytes, **kwargs) -> str:
        calls.append(wav)
        return "from-http"

    monkeypatch.setattr("book2audio.stt_parakeet._transcribe_http_bytes", fake_http)
    out = transcribe_reference_float32(np.zeros(64, dtype=np.float32), 16_000)
    assert out == "from-http"
    assert calls and calls[0].startswith(b"RIFF")


def test_auto_prefers_onnx_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOK2AUDIO_STT_BACKEND", raising=False)
    monkeypatch.setattr("book2audio.stt_parakeet._onnx_importable", lambda: True)
    monkeypatch.setattr(
        "book2audio.stt_parakeet._transcribe_onnx_wav_bytes", lambda b: "from-onnx"
    )

    def no_http(*_a, **_k):
        raise AssertionError("HTTP should not run when ONNX succeeds")

    monkeypatch.setattr("book2audio.stt_parakeet._transcribe_http_bytes", no_http)
    out = transcribe_reference_float32(np.zeros(64, dtype=np.float32), 16_000)
    assert out == "from-onnx"


def test_auto_falls_back_to_http_when_onnx_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOK2AUDIO_STT_BACKEND", raising=False)
    monkeypatch.setattr("book2audio.stt_parakeet._onnx_importable", lambda: True)

    def bad_onnx(_b: bytes) -> str:
        raise RuntimeError("onnx load failed")

    monkeypatch.setattr("book2audio.stt_parakeet._transcribe_onnx_wav_bytes", bad_onnx)
    monkeypatch.setattr(
        "book2audio.stt_parakeet._transcribe_http_bytes", lambda wav, **kw: "fallback-http"
    )
    out = transcribe_reference_float32(np.zeros(64, dtype=np.float32), 16_000)
    assert out == "fallback-http"


def test_http_mode_skips_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOK2AUDIO_STT_BACKEND", "http")

    def no_onnx(_b: bytes) -> str:
        raise AssertionError("ONNX should not run in http mode")

    monkeypatch.setattr("book2audio.stt_parakeet._transcribe_onnx_wav_bytes", no_onnx)
    monkeypatch.setattr(
        "book2audio.stt_parakeet._transcribe_http_bytes", lambda wav, **kw: "http-only"
    )
    out = transcribe_reference_float32(np.zeros(64, dtype=np.float32), 16_000)
    assert out == "http-only"
