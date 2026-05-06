"""Pytest configuration: stable device choice before ``book2audio`` imports."""

from __future__ import annotations

import os

# Must run before tests import book2audio (deterministic, avoids slow/stuck DirectML in CI/local runs).
os.environ.setdefault("BOOK2AUDIO_DEVICE", "cpu")
os.environ.setdefault("BOOK2AUDIO_SKIP_DML", "1")
