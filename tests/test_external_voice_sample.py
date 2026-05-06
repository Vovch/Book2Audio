"""Tests for Gradio → mono conversion used by external reference clips."""

import numpy as np

from book2audio.synthesis import SAMPLE_RATE, _mono_float32_from_gradio, _resample_mono_linear


def test_mono_from_gradio_stereo_samples_first():
    sr = 44_100
    stereo = np.zeros((500, 2), dtype=np.float32)
    stereo[:, 0] = 0.4
    stereo[:, 1] = 0.2
    out = _mono_float32_from_gradio((sr, stereo))
    assert out is not None
    r, mono = out
    assert r == sr
    assert mono.shape == (500,)
    assert np.allclose(mono, 0.3)


def test_mono_from_gradio_channels_first():
    sr = 16_000
    x = np.zeros((2, 300), dtype=np.float32)
    x[0, :] = 1.0
    x[1, :] = -1.0
    out = _mono_float32_from_gradio((sr, x))
    assert out is not None
    r, mono = out
    assert r == sr
    assert mono.shape == (300,)
    assert np.allclose(mono, 0.0)


def test_resample_identity():
    w = np.linspace(-1, 1, 100, dtype=np.float32)
    y = _resample_mono_linear(w, SAMPLE_RATE, SAMPLE_RATE)
    assert y.shape == w.shape
    assert np.allclose(y, w)


def test_resample_changes_length():
    w = np.ones(2400, dtype=np.float32)
    y = _resample_mono_linear(w, 24_000, 12_000)
    assert abs(y.shape[0] - 1200) <= 1
