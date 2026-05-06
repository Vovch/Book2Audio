"""OmniVoice loading and text-to-speech synthesis (no UI dependencies)."""

from __future__ import annotations

import logging
import re

import numpy as np
import torch
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import VoiceClonePrompt, _resolve_instruct
from omnivoice.utils.voice_design import _INSTRUCT_MUTUALLY_EXCLUSIVE, _INSTRUCT_VALID_EN

from book2audio.device import pick_device_and_dtype

logger = logging.getLogger(__name__)
_MODEL_CACHE: OmniVoice | None = None
MODEL_ID = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24_000

# Half-baked LLM phrases -> single OmniVoice token (see omnivoice.utils.voice_design).
_TOKEN_ALIASES: dict[str, str] = {
    "high-pitched": "high pitch",
    "high pitched": "high pitch",
    "low-pitched": "low pitch",
    "low pitched": "low pitch",
    "middle aged": "middle-aged",
    "middleaged": "middle-aged",
    "very high-pitched": "very high pitch",
    "very low-pitched": "very low pitch",
    "rp british": "british accent",
    "rp english": "british accent",
    "english rp": "british accent",
    "neutral american": "american accent",
    "american neutral": "american accent",
    "west coast": "american accent",
    "us accent": "american accent",
    "uk accent": "british accent",
    "indian english": "indian accent",
}


def _normalize_voice_instruct_token(segment: str) -> str | None:
    """Map one comma-separated phrase to a valid English instruct token, or ``None`` to drop."""
    t = segment.strip().lower()
    if not t:
        return None
    t = _TOKEN_ALIASES.get(t, t)
    if t in _INSTRUCT_VALID_EN:
        return t
    if "rp" in t and "british" in t:
        return "british accent"
    if t == "british" or (t.startswith("british") and "accent" not in t and len(t) < 12):
        return "british accent"
    if t in ("american", "us"):
        return "american accent"
    return None


def _filter_mutually_exclusive(tokens: list[str]) -> list[str]:
    """Keep first token per OmniVoice mutex category (gender, age, pitch, …)."""
    out: list[str] = []
    used_cat: set[int] = set()
    for tok in tokens:
        cat_i: int | None = None
        for ci, cat in enumerate(_INSTRUCT_MUTUALLY_EXCLUSIVE):
            if tok in cat:
                cat_i = ci
                break
        if cat_i is not None and cat_i in used_cat:
            continue
        if cat_i is not None:
            used_cat.add(cat_i)
        out.append(tok)
    return out


def _ensure_core_english_tags(tokens: list[str]) -> list[str]:
    """Ensure gender, age, pitch, and one accent so OmniVoice always gets a usable design."""
    present: set[int] = set()
    for tok in tokens:
        for ci, cat in enumerate(_INSTRUCT_MUTUALLY_EXCLUSIVE):
            if tok in cat:
                present.add(ci)
                break
    defaults: list[tuple[int, str]] = [
        (0, "male"),
        (1, "middle-aged"),
        (2, "moderate pitch"),
        (4, "american accent"),
    ]
    out = list(tokens)
    for ci, default_tok in defaults:
        if ci not in present:
            out.append(default_tok)
    return _filter_mutually_exclusive(out)


def coerce_voice_instruct_for_omnivoice(
    instruct: str | None,
    *,
    fallback: str = "male, middle-aged, moderate pitch, american accent",
) -> str:
    """Strip invalid tags, map common aliases, and validate via OmniVoice's resolver.

    LLMs often emit free-text (e.g. *fussy*, *RP British*) that :meth:`OmniVoice.generate`
    rejects. This keeps only allowed English tokens and fills missing core categories.
    """
    raw = (instruct or "").strip()
    if not raw:
        try:
            r = _resolve_instruct(fallback)
            return r or fallback
        except ValueError:
            return fallback

    parts = re.split(r"\s*[,，]\s*", raw)
    toks: list[str] = []
    for p in parts:
        n = _normalize_voice_instruct_token(p)
        if n:
            toks.append(n)
        else:
            logger.debug("Dropped unsupported voice instruct fragment: %r", p)

    toks = _filter_mutually_exclusive(toks)
    toks = _ensure_core_english_tags(toks)
    if not toks:
        try:
            r = _resolve_instruct(fallback)
            return r or fallback
        except ValueError:
            return fallback

    joined = ", ".join(toks)
    try:
        r = _resolve_instruct(joined)
        return r or joined
    except ValueError:
        logger.warning(
            "Voice instruct still invalid after coercion %r -> using fallback",
            instruct,
        )
        try:
            r = _resolve_instruct(fallback)
            return r or fallback
        except ValueError:
            return fallback


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
        kwargs["instruct"] = coerce_voice_instruct_for_omnivoice(hint)

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
    resolved = coerce_voice_instruct_for_omnivoice(instruct)
    audio_list = model.generate(
        text=phrase,
        instruct=resolved,
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
        kwargs["instruct"] = coerce_voice_instruct_for_omnivoice(hint)
    audio = model.generate(**kwargs)
    chunk = audio[0]
    if not isinstance(chunk, np.ndarray):
        chunk = np.asarray(chunk)
    return SAMPLE_RATE, chunk


def clear_model_cache() -> None:
    """Drop the cached model (releases weights; next call reloads)."""
    global _MODEL_CACHE
    _MODEL_CACHE = None
