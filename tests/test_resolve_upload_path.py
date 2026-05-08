"""Gradio File inputs: path resolution for archive/WAV import."""

from __future__ import annotations

import pytest

from book2audio.web_ui import _resolve_upload_path


def test_resolve_none():
    assert _resolve_upload_path(None) is None


def test_resolve_plain_str():
    assert _resolve_upload_path("  C:/tmp/x.wav  ") == "C:/tmp/x.wav"


def test_resolve_gradio_filedata():
    gradio = pytest.importorskip("gradio")
    FileData = gradio.data_classes.FileData
    fd = FileData(path=r"C:\app\upload\clip.wav")
    assert _resolve_upload_path(fd) == r"C:\app\upload\clip.wav"


def test_resolve_filedata_dict():
    assert _resolve_upload_path({"path": "/tmp/a.wav", "meta": {"_type": "gradio.FileData"}}) == "/tmp/a.wav"


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


def test_resolve_named_temp_style():
    assert _resolve_upload_path(_Named("/old/tmp.wav")) == "/old/tmp.wav"
