"""OmniVoice loading and text-to-speech synthesis (no UI dependencies)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import VoiceClonePrompt

from book2audio.device import pick_device_and_dtype

logger = logging.getLogger(__name__)
_MODEL_CACHE: OmniVoice | None = None
MODEL_ID = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24_000


def get_or_load_model() -> OmniVoice:
    """Lazy-load the shared OmniVoice model (idempotent)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    device_map, dtype, label = pick_device_and_dtype()
    logger.info("Loading OmniVoice on %s (%s)", label, dtype)
    _MODEL_CACHE = OmniVoice.from_pretrained(
        MODEL_ID,
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
    return SAMPLE_RATE, chunk


def build_voice_clone_from_instruct(
    instruct: str,
    sample_text: str | None = None,
    *,
    num_step: int = 12,
    language: str = "English",
) -> tuple[VoiceClonePrompt, np.ndarray]:
    """Synthesize a short line with *instruct*, build :class:`VoiceClonePrompt`, return prompt + mono waveform."""
    phrase = (sample_text or "").strip() or (
        "Hello — this is my voice for this story."
    )
    model = get_or_load_model()
    audio_list = model.generate(
        text=phrase,
        instruct=instruct,
        language=language,
        num_step=num_step,
    )
    wav = np.asarray(audio_list[0], dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    tensor = torch.from_numpy(wav).unsqueeze(0)
    vcp = model.create_voice_clone_prompt(
        ref_audio=(tensor, SAMPLE_RATE),
        ref_text=phrase,
        preprocess_prompt=True,
    )
    return vcp, wav


def synthesize_voice_clone_to_numpy(
    text: str,
    voice_clone_prompt: VoiceClonePrompt,
    *,
    num_step: int = 16,
    language: str = "English",
    speed: float | None = None,
    instruct: str | None = None,
) -> tuple[int, np.ndarray] | None:
    """Synthesize using a pre-built :class:`VoiceClonePrompt` (per-character voice)."""
    phrase = (text or "").strip()
    if not phrase:
        return None
    model = get_or_load_model()
    kwargs: dict = {
        "text": phrase,
        "voice_clone_prompt": voice_clone_prompt,
        "language": language,
        "num_step": num_step,
    }
    if speed is not None:
        kwargs["speed"] = float(speed)
    hint = (instruct or "").strip()
    if hint:
        kwargs["instruct"] = hint
    audio = model.generate(**kwargs)
    chunk = audio[0]
    if not isinstance(chunk, np.ndarray):
        chunk = np.asarray(chunk)
    return SAMPLE_RATE, chunk


def clear_model_cache() -> None:
    """Drop the cached model (releases weights; next call reloads)."""
    global _MODEL_CACHE
    _MODEL_CACHE = None
