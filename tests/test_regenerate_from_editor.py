"""Regenerate voice sample picks up manual character JSON edits."""

from __future__ import annotations

import json

import numpy as np
import pytest

from book2audio.audiobook_pipeline import (
    AudiobookPlan,
    CharacterCard,
    _omnivoice_instruct_from_row,
    regenerate_single_voice_sample,
)


def test_regenerate_applies_editor_voice_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = AudiobookPlan()
    plan.characters = [
        CharacterCard(
            name="Alice",
            role="Old role",
            summary="Old summary",
            voice_instruct="female, middle-aged, moderate pitch, american accent",
        ),
        CharacterCard(
            name="Bob",
            role="B",
            summary="Sb",
            voice_instruct="male, middle-aged, moderate pitch, american accent",
        ),
    ]
    plan.raw_character_rows = [{"name": "Alice", "role": "r", "summary": "s"}]
    plan.voice_samples = {"Alice": (24_000, np.zeros(100, dtype=np.float32))}
    plan.clone_prompts = {"Alice": object()}

    editor_obj = [
        {
            "name": "Alice",
            "role": "New role",
            "summary": "New summary",
            "voice_gender": "female",
            "voice_age": "young adult",
            "voice_characteristics": "bright",
            "voice_accent": "British",
        }
    ]
    editor = json.dumps(editor_obj, ensure_ascii=False)
    expected_instruct = _omnivoice_instruct_from_row(editor_obj[0])

    built: list = []

    def fake_build(cards, **kwargs):
        built.extend(cards)
        name = cards[0].name
        return {name: object()}, {name: (24_000, np.ones(8, dtype=np.float32))}

    assign_calls: list[int] = []

    def no_assign(_rows):
        assign_calls.append(1)
        raise AssertionError("regenerate should not call assign_omnivoice_profiles")

    monkeypatch.setattr(
        "book2audio.audiobook_pipeline.assign_omnivoice_profiles",
        no_assign,
    )
    monkeypatch.setattr(
        "book2audio.audiobook_pipeline.build_clone_prompts_for_cast",
        fake_build,
    )

    regenerate_single_voice_sample(
        plan,
        "alice",
        sample_steps=8,
        sample_line=None,
        editor_text=editor,
    )

    assert not assign_calls
    alice = next(c for c in plan.characters if c.name == "Alice")
    assert alice.role == "New role"
    assert alice.summary == "New summary"
    assert alice.voice_instruct == expected_instruct
    assert "british accent" in alice.voice_instruct.lower()
    assert built and built[0].voice_instruct == alice.voice_instruct


def test_regenerate_without_editor_keeps_previous_card(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = AudiobookPlan()
    plan.characters = [
        CharacterCard(
            name="Alice",
            role="R",
            summary="S",
            voice_instruct="female, middle-aged, moderate pitch, american accent",
        )
    ]
    plan.voice_samples = {"Alice": (24_000, np.zeros(10, dtype=np.float32))}
    plan.clone_prompts = {"Alice": object()}

    assign_calls: list[int] = []

    def fake_assign(_rows):
        assign_calls.append(1)
        raise AssertionError("should not refresh from LLM when editor_text omitted")

    monkeypatch.setattr(
        "book2audio.audiobook_pipeline.assign_omnivoice_profiles",
        fake_assign,
    )

    def fake_build(cards, **kwargs):
        name = cards[0].name
        return {name: object()}, {name: (24_000, np.ones(4, dtype=np.float32))}

    monkeypatch.setattr(
        "book2audio.audiobook_pipeline.build_clone_prompts_for_cast",
        fake_build,
    )

    regenerate_single_voice_sample(
        plan,
        "Alice",
        sample_steps=8,
        sample_line=None,
        editor_text=None,
    )
    assert not assign_calls
