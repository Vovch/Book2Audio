"""Qwen 3.5 planner for character extraction, voice tags, and TTS prep (lazy-loaded).

Speech synthesis stays on OmniVoice; web search stays separate. Override the checkpoint with
``BOOK2AUDIO_LLM_MODEL`` only if you use another Hugging Face model (non–Qwen-3.5 ids load via
:class:`~transformers.AutoModelForCausalLM`).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from book2audio.device import pick_device_and_dtype

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"

_TOKENIZER: Any | None = None
_MODEL: Any | None = None


def _use_image_text_to_text_loader(model_id: str) -> bool:
    """Qwen3.5 Hub checkpoints use ``Qwen3_5ForConditionalGeneration`` / image-text-to-text."""
    m = model_id.lower().replace("_", ".")
    return "qwen3.5" in m


def default_llm_model_id() -> str:
    return (os.environ.get("BOOK2AUDIO_LLM_MODEL") or _DEFAULT_MODEL).strip()


def _resolve_llm_device_str() -> str:
    forced = (os.environ.get("BOOK2AUDIO_LLM_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda", "mps", "cuda:0"):
        return forced
    device_map, _dtype, _label = pick_device_and_dtype()
    s = device_map if isinstance(device_map, str) else str(device_map)
    if "privateuseone" in s:
        return "cpu"
    return s


def _load() -> tuple[Any, Any]:
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    model_id = default_llm_model_id()
    dev = _resolve_llm_device_str()
    llm_dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    logger.info("Loading planner LLM %s on %s (%s)", model_id, dev, llm_dtype)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if _use_image_text_to_text_loader(model_id):
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=llm_dtype,
        )
    else:
        # Optional ``BOOK2AUDIO_LLM_MODEL``: non–Qwen-3.5 checkpoints (CausalLM).
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=llm_dtype,
        )
    model.to(dev)
    model.eval()
    _TOKENIZER, _MODEL = tok, model
    return tok, model


def chat_complete(system: str, user: str, *, max_new_tokens: int = 1024) -> str:
    tokenizer, model = _load()
    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # If ``apply_chat_template`` fails (unusual templates / overrides).
        _end = "<|" + "im_end" + "|>"
        prompt = (
            f"<|im_start|>system\n{system.strip()}{_end}\n"
            f"<|im_start|>user\n{user.strip()}{_end}\n"
            "<|im_start|>assistant\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.35,
        "top_p": 0.9,
    }
    if pad_id is not None:
        gen_kw["pad_token_id"] = pad_id
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            **gen_kw,
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from model output (handles markdown fences)."""
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
    if fence:
        raw = fence.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)
