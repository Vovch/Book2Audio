"""Unit tests for audiobook text splitting (no LLM / no TTS)."""

from book2audio.audiobook_pipeline import split_into_chapters


def test_split_no_chapter_headings_returns_single_bucket():
    text = "Just prose.\n\nMore prose."
    parts = split_into_chapters(text)
    assert len(parts) == 1
    assert parts[0][0] == "Document"


def test_split_detects_chapter_markers():
    text = """Chapter 1
The beginning.

Chapter 2
The end."""
    parts = split_into_chapters(text)
    assert len(parts) == 2
    assert "Chapter 1" in parts[0][0]
    assert "beginning" in parts[0][1]
    assert "end" in parts[1][1]


def test_split_chapter_uppercase():
    text = "CHAPTER III\n\nContent here.\n\nCHAPTER IV\n\nMore."
    parts = split_into_chapters(text)
    assert len(parts) == 2
