"""OmniVoice loading and text-to-speech synthesis (no UI dependencies)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from omnivoice import OmniVoice

from book2audio.device import pick_device_and_dtype

logger = logging.getLogger(__name__)

_MODEL_CACHE: OmniVoice | None = None
_MODEL_ID = "k2-fsa/OmniVoice"


def get_or_load_model() -> OmniVoice:
    """Lazy-load the shared OmniVoice model (idempotent)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    device_map, dtype, label = pick_device_and_dtype()
    logger.info("Loading OmniVoice on %s (%s)", label, dtype)
    _MODEL_CACHE = OmniVoice.from_pretrained(
        _MODEL_ID,
        device_map=device_map,
        dtype=dtype,
    )
    return _MODEL_CACHE


def synthesize_to_numpy(
    text: str,
    instruct: str = "",
    *,
    num_step: int = 16,
) -> tuple[int, np.ndarray] | None:
    """Return ``(sample_rate, mono_float32_or_int16_array)`` or ``None`` if *text* is empty after strip."""
    phrase = (text or "").strip()
    if not phrase:
        return None

    model = get_or_load_model()
    kwargs: dict = {"text": phrase, "num_step": num_step}
    hint = (instruct or "").strip()
    if hint:
        kwargs["instruct"] = hint

    audio = model.generate(**kwargs)
    chunk = audio[0]
    if not isinstance(chunk, np.ndarray):
        chunk = np.asarray(chunk)
    return 24_000, chunk


def clear_model_cache() -> None:
    """Drop the cached model (releases weights; next call reloads)."""
    global _MODEL_CACHE
    _MODEL_CACHE = None
