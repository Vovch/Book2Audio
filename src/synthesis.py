"""OmniVoice loading and text-to-speech synthesis (no UI dependencies)."""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import torch
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import VoiceClonePrompt, _resolve_instruct
from omnivoice.utils.voice_design import (
    _INSTRUCT_EN_TO_ZH,
    _INSTRUCT_MUTUALLY_EXCLUSIVE,
    _INSTRUCT_VALID_EN,
    _INSTRUCT_VALID_ZH,
)

from book2audio.device import pick_device_and_dtype

logger = logging.getLogger(__name__)
_MODEL_CACHE: OmniVoice | None = None
MODEL_ID = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24_000

# Short line for Chinese voice-design preview when ``BOOK2AUDIO_ZH_VOICE_DESIGN_SAMPLE`` is set.
_ZH_VOICE_SAMPLE_LINE = "您好，这是我在这个故事里的声音。"

# OmniVoice often misfires when **child** / **teenager** are combined with explicit pitch tags; omit pitch.
OMNIVOICE_AGE_OMIT_PITCH_EN: frozenset[str] = frozenset({"child", "teenager"})
OMNIVOICE_AGE_OMIT_PITCH_ZH: frozenset[str] = frozenset(
    _INSTRUCT_EN_TO_ZH[a] for a in OMNIVOICE_AGE_OMIT_PITCH_EN
)
OMNIVOICE_PITCH_EN: frozenset[str] = frozenset(
    {
        "very low pitch",
        "low pitch",
        "moderate pitch",
        "high pitch",
        "very high pitch",
    }
)
OMNIVOICE_PITCH_ZH: frozenset[str] = frozenset(
    _INSTRUCT_EN_TO_ZH[p] for p in OMNIVOICE_PITCH_EN
)


def omnivoice_age_should_omit_pitch(age_en: str) -> bool:
    """Whether resolved English age should not carry an explicit pitch token."""
    return (age_en or "").strip() in OMNIVOICE_AGE_OMIT_PITCH_EN


def _tokens_include_child_or_teen_age(tokens: list[str]) -> bool:
    for t in tokens:
        if t in OMNIVOICE_AGE_OMIT_PITCH_EN or t in OMNIVOICE_AGE_OMIT_PITCH_ZH:
            return True
    return False


def _strip_pitch_tokens_for_young_voice(tokens: list[str]) -> list[str]:
    if not _tokens_include_child_or_teen_age(tokens):
        return tokens
    drop = OMNIVOICE_PITCH_EN | OMNIVOICE_PITCH_ZH
    return [t for t in tokens if t not in drop]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _resolved_english_instruct_to_chinese_voice_design(resolved_en: str) -> str:
    """Drop English accent tokens and map voice-design tags to Chinese.

    OmniVoice ``_resolve_instruct`` maps Chinese age → English whenever any ``* accent`` is
    present (English-audio path). For the **short clone-preview** pass only, we omit accent
    and use Chinese sample text so ``use_zh`` stays true and tags such as 老年 stay Chinese.
    """
    parts = [p.strip() for p in (resolved_en or "").split(",") if p.strip()]
    zh_parts: list[str] = []
    for p in parts:
        if " accent" in p:
            continue
        zh_parts.append(_INSTRUCT_EN_TO_ZH.get(p, p))
    if OMNIVOICE_AGE_OMIT_PITCH_ZH & set(zh_parts):
        zh_parts = [z for z in zh_parts if z not in OMNIVOICE_PITCH_ZH]
    if not zh_parts:
        zh_parts = ["男", "中年", "中音调"]
    return "，".join(zh_parts)


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
    """Map one comma-separated phrase to a valid instruct token, or ``None`` to drop."""
    raw = segment.strip()
    if not raw:
        return None
    if raw in _INSTRUCT_VALID_ZH:
        return raw
    t = raw.lower()
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
    """Ensure gender, age, and one accent; pitch only when age is not child/teenager."""
    present: set[int] = set()
    for tok in tokens:
        for ci, cat in enumerate(_INSTRUCT_MUTUALLY_EXCLUSIVE):
            if tok in cat:
                present.add(ci)
                break
    young = _tokens_include_child_or_teen_age(tokens)
    defaults: list[tuple[int, str]] = [
        (0, "male"),
        (1, "middle-aged"),
        (4, "american accent"),
    ]
    if not young:
        defaults.insert(2, (2, "moderate pitch"))
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

    **Child / teenager:** explicit pitch tags are dropped and default pitch is not injected —
    OmniVoice is unreliable when combining those ages with pitch controls.
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
    toks = _strip_pitch_tokens_for_young_voice(toks)
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
    """Lazy-load the shared OmniVoice model (idempotent).

    Reference transcription for voice cloning uses local **onnx-asr** Parakeet (optional) or HTTP STT —
    OmniVoice’s optional built-in speech-to-text stack is not loaded (``load_asr=False``).
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    device_map, dtype, label = pick_device_and_dtype()
    logger.info("Loading OmniVoice on %s (%s)", label, dtype)
    _MODEL_CACHE = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        dtype=dtype,
        load_asr=False,
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


def _mono_float32_from_gradio(
    audio: tuple[int, np.ndarray] | None,
) -> tuple[int, np.ndarray] | None:
    """Normalize Gradio ``Audio(type='numpy')`` to ``(sample_rate, mono float32)`` or ``None``."""
    if audio is None:
        return None
    sr, data = audio
    if data is None:
        return None
    raw = np.asarray(data)
    if raw.size == 0:
        return None
    if raw.dtype.kind in "iu":
        x = (raw.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    else:
        x = raw.astype(np.float32, copy=False)
        peak = float(np.nanmax(np.abs(x))) if x.size else 0.0
        if peak > 1.5:
            x = np.clip(x / 32767.0, -1.0, 1.0)
    if x.ndim == 1:
        mono = x
    elif x.ndim == 2:
        # Gradio uses (samples, channels); OmniVoice uses (channels, samples) sometimes.
        mono = np.mean(x, axis=1) if x.shape[0] >= x.shape[1] else np.mean(x, axis=0)
    else:
        return None
    mono = np.clip(mono.reshape(-1), -1.0, 1.0)
    return int(sr), mono


def _resample_mono_linear(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Cheap linear resample for UI preview (matches :data:`SAMPLE_RATE` used elsewhere)."""
    if sr_in == sr_out:
        return np.asarray(wav, dtype=np.float32).reshape(-1)
    w = np.asarray(wav, dtype=np.float32).reshape(1, 1, -1)
    t = torch.from_numpy(w)
    new_len = max(1, int(round(wav.shape[-1] * sr_out / sr_in)))
    t2 = torch.nn.functional.interpolate(t, size=new_len, mode="linear", align_corners=True)
    return t2.squeeze().numpy().astype(np.float32)


def build_voice_clone_from_reference_audio(
    audio: tuple[int, np.ndarray] | None,
    ref_text: str | None = None,
    *,
    preprocess_prompt: bool = True,
) -> tuple[VoiceClonePrompt, np.ndarray, int]:
    """Build a :class:`VoiceClonePrompt` from an uploaded clip; preview at :data:`SAMPLE_RATE`.

    *ref_text*: transcript of the clip. If unset, uses **local Parakeet ONNX** via ``onnx-asr``
    when installed (``pip install "book2audio[parakeet-stt]"`` — Hugging Face download on first use), or
    HTTP STT if ``BOOK2AUDIO_STT_BACKEND=http``. See ``book2audio.stt_parakeet`` for all env vars, or paste
    the transcript to skip STT.
    """
    pair = _mono_float32_from_gradio(audio)
    if pair is None:
        raise ValueError("No audio: upload a non-empty reference clip.")
    sr_in, mono = pair
    model = get_or_load_model()
    tensor = torch.from_numpy(mono).unsqueeze(0)
    rt = (ref_text or "").strip() or None
    if rt is None:
        from book2audio.stt_parakeet import transcribe_reference_float32

        rt = transcribe_reference_float32(mono, sr_in)
        logger.info("Reference transcript (Parakeet kit): %s", rt[:160])
    vcp = model.create_voice_clone_prompt(
        ref_audio=(tensor, sr_in),
        ref_text=rt,
        preprocess_prompt=preprocess_prompt,
    )
    preview = _resample_mono_linear(mono, sr_in, SAMPLE_RATE)
    return vcp, preview, SAMPLE_RATE


def build_voice_clone_from_instruct(
    instruct: str,
    sample_text: str | None = None,
    *,
    num_step: int = 12,
    language: str = "English",
) -> tuple[VoiceClonePrompt, np.ndarray]:
    """Synthesize a short line with *instruct*, build :class:`VoiceClonePrompt`, return prompt + mono waveform.

    Set environment variable ``BOOK2AUDIO_ZH_VOICE_DESIGN_SAMPLE`` to ``1`` to build the preview
    clip with **Chinese** voice-design tags (no English ``* accent`` token) and a short Chinese
    phrase. OmniVoice otherwise converts Chinese age to English whenever an English accent is
    present in the same ``instruct`` string; this mode is a workaround for age-sensitive previews.
    Chapter TTS remains English + clone prompts built from the preview audio.
    """
    use_zh_sample = _env_flag("BOOK2AUDIO_ZH_VOICE_DESIGN_SAMPLE")
    if use_zh_sample:
        phrase = (sample_text or "").strip() or _ZH_VOICE_SAMPLE_LINE
        lang_for_gen = "Chinese"
    else:
        phrase = (sample_text or "").strip() or (
            "Hello — this is my voice for this story."
        )
        lang_for_gen = language
    model = get_or_load_model()
    resolved_en = coerce_voice_instruct_for_omnivoice(instruct)
    resolved = (
        _resolved_english_instruct_to_chinese_voice_design(resolved_en)
        if use_zh_sample
        else resolved_en
    )
    audio_list = model.generate(
        text=phrase,
        instruct=resolved,
        language=lang_for_gen,
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
