"""LLM + search orchestration for multi-character audiobook prep (Qwen + OmniVoice)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from book2audio.llm_qwen import chat_complete, extract_json_object
from book2audio.search_web import format_snippets_for_llm, search_snippets
from book2audio.synthesis import (
    SAMPLE_RATE,
    build_voice_clone_from_instruct,
    synthesize_voice_clone_to_numpy,
)
from omnivoice.models.omnivoice import VoiceClonePrompt

logger = logging.getLogger(__name__)

_INSTRUCT_RULES = """OmniVoice English voice tags (comma + space separated). Use only these tokens:
male, female, child, teenager, young adult, middle-aged, elderly,
very low pitch, low pitch, moderate pitch, high pitch, very high pitch,
whisper,
american accent, british accent, australian accent, chinese accent, canadian accent,
indian accent, korean accent, portuguese accent, russian accent, japanese accent."""

CHARACTER_RESEARCH_SYSTEM_PROMPT = (
    "You extract literary casts with **how each character should sound** for audiobook casting. "
    "Reply with one JSON object only, no markdown."
)

_EXAMPLE_CHARACTER_ROWS: list[dict[str, str]] = [
    {
        "name": "Narrator",
        "role": "Narration",
        "summary": "Neutral storyteller for third-person prose.",
        "voice_gender": "male",
        "voice_age": "middle-aged",
        "voice_characteristics": "clear, steady pace, warm but neutral timbre",
        "voice_accent": "neutral American",
    },
    {
        "name": "Example Hero",
        "role": "Protagonist",
        "summary": "Curious engineer from the coast.",
        "voice_gender": "female",
        "voice_age": "young adult",
        "voice_characteristics": "bright, slightly fast, enthusiastic",
        "voice_accent": "American West Coast",
    },
]
EXAMPLE_CHARACTERS_EDITOR_JSON = json.dumps(
    _EXAMPLE_CHARACTER_ROWS,
    ensure_ascii=False,
    indent=2,
)

_CHARACTER_ROW_KEYS = (
    "role",
    "summary",
    "voice_gender",
    "voice_age",
    "voice_characteristics",
    "voice_accent",
)


def _normalize_character_row(item: dict[str, Any]) -> dict[str, str] | None:
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    out: dict[str, str] = {"name": name}
    for key in _CHARACTER_ROW_KEYS:
        raw = item.get(key)
        if raw is None:
            continue
        val = str(raw).strip()
        if val:
            out[key] = val
    raw_g = out.get("voice_gender", "")
    gl = str(raw_g).strip().lower()
    if not gl or gl == "unknown":
        out["voice_gender"] = "male"
    elif gl == "female":
        out["voice_gender"] = "female"
    elif gl == "male":
        out["voice_gender"] = "male"
    return out


@dataclass
class CharacterCard:
    name: str
    role: str
    summary: str
    voice_instruct: str


@dataclass
class TTSSegment:
    speaker: str
    text: str
    speed: float | None = None
    extra_instruct: str | None = None


@dataclass
class AudiobookPlan:
    """Server-side session object (keep in Gradio ``State``)."""

    raw_character_rows: list[dict[str, Any]] = field(default_factory=list)
    characters: list[CharacterCard] = field(default_factory=list)
    clone_prompts: dict[str, VoiceClonePrompt] = field(default_factory=dict)
    #: Short OmniVoice samples used to build clone prompts — for UI playback only.
    voice_samples: dict[str, tuple[int, np.ndarray]] = field(default_factory=dict)
    narrator_instruct: str = (
        "male, middle-aged, moderate pitch, american accent"
    )
    last_segments: list[TTSSegment] = field(default_factory=list)
    diagnostics: str = ""


def character_rows_to_editor_json(rows: list[dict[str, Any]]) -> str:
    """Format character dicts for the Gradio editor (story + voice metadata)."""
    clean: list[dict[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        one = _normalize_character_row(r)
        if one:
            clean.append(one)
    return json.dumps(clean, ensure_ascii=False, indent=2)


def _decode_character_list_from_text(raw: str) -> list[Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = extract_json_object(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
    if isinstance(data, dict) and "characters" in data:
        items = data["characters"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(
            'Expected a JSON array of objects, or {"characters": [...]}'
        )
    if not isinstance(items, list):
        raise ValueError("characters must be a JSON array")
    return items


def _rows_normalized_from_items(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            one = _normalize_character_row(item)
            if one:
                rows.append(one)
    return rows


def parse_character_json(text: str) -> list[dict[str, Any]]:
    """Parse the editable field; accepts LLM markdown fences and ``{"characters":[...]}``."""
    raw = (text or "").strip()
    if not raw:
        return []
    items = _decode_character_list_from_text(raw)
    return _rows_normalized_from_items(items)


def characters_from_external_llm_response(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Paste is empty")
    items = _decode_character_list_from_text(raw)
    rows = _rows_normalized_from_items(items)
    if not rows:
        raise ValueError("No valid characters (each needs a non-empty 'name')")
    return rows


def split_into_chapters(book_text: str) -> list[tuple[str, str]]:
    """Split *book_text* into [(heading, body), ...] using light heuristics."""
    raw = (book_text or "").strip()
    if not raw:
        return []

    pattern = re.compile(
        r"(?m)^(?P<h>(?:Chapter|CHAPTER|Book)\s+[^\n]{1,120})\s*\n+",
    )
    matches = list(pattern.finditer(raw))
    if not matches:
        return [("Document", raw)]

    chapters: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group("h").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if body:
            chapters.append((heading, body))
    return chapters or [("Document", raw)]


def _character_research_queries(title: str, auth: str) -> list[str]:
    return [
        f"{title} {auth} full characters list and descriptions",
        f"{title} {auth} main characters summary",
        f"{title} novel characters plot",
    ]


def gather_character_research_blob(
    book_title: str,
    author: str,
    *,
    on_progress: Callable[[float, str], None] | None = None,
) -> str:
    def _p(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, frac)), msg)

    title = (book_title or "").strip() or "Unknown title"
    auth = (author or "").strip() or "Unknown author"
    queries = _character_research_queries(title, auth)
    snippets: list[str] = []
    nq = len(queries)
    _p(0.02, "Gathering web snippets…")
    for i, q in enumerate(queries):
        _p(0.05 + 0.35 * (i / max(nq, 1)), f"Web search ({i + 1}/{nq})…")
        snippets.extend(search_snippets(q, num_results=5))
    return format_snippets_for_llm(snippets)


def _character_research_schema_and_field_guide() -> str:
    return """Return JSON exactly in this shape (every character MUST include voice fields — infer cautiously if needed):
{{"characters":[
  {{
    "name": string,
    "role": string,
    "summary": string,
    "voice_gender": string,
    "voice_age": string,
    "voice_characteristics": string,
    "voice_accent": string
  }}
]}}

Field guide:
- **summary**: who they are in the story (not voice).
- **voice_gender**: exactly **male** or **female**. If unclear, use **male** (do not use "unknown" for gender).
- **voice_age**: one of: child, teenager, young adult, middle-aged, elderly, unknown — or a short phrase (e.g. "very old", "about twelve").
- **voice_characteristics**: how they *sound*: pitch (deep/light), tempo, timbre (gravelly, smooth, nasal), energy, emotion typical of speech, quirks. Plain English.
- **voice_accent**: region or style of speech if known or strongly implied (e.g. "American South", "RP British", "neutral American"); use "unknown" if not inferable.

Try to find and describe all the characters in the book, starting from the major characters."""


def build_character_research_user_prompt(
    book_title: str,
    author: str,
    research_blob: str | None,
    *,
    book_excerpt: str = "",
) -> str:
    title = (book_title or "").strip() or "Unknown title"
    auth = (author or "").strip() or "Unknown author"
    tail = _character_research_schema_and_field_guide()
    if research_blob is not None:
        tail += (
            " If the book is poorly covered online, return fewer entries; use **male** when gender is unclear; "
            "for **voice_age** or **voice_accent** you may use **unknown** rather than inventing plot or locale detail."
        )
        ex = (book_excerpt or "").strip()
        excerpt_blk = f"\n\nBook excerpt (may be empty; cross-check research):\n{ex[:8000]}\n" if ex else ""
        return f"""Book title: {title}
Author: {auth}

Web research (may be incomplete or wrong):
{research_blob}
{excerpt_blk}
{tail}"""

    tail += (
        " Use your **training knowledge** and any **research or browsing tools** your environment "
        "offers to identify the cast. If you are unsure, return fewer entries; use **male** when gender is unclear; for **voice_age** or "
        "**voice_accent** you may use **unknown** rather than inventing plot or locale detail."
    )
    return f"""Book title: {title}
Author: {auth}

{tail}"""


def format_character_research_messages_for_external_llm(
    book_title: str,
    author: str,
) -> tuple[str, str, str]:
    """Return ``(system_prompt, user_prompt, copy_paste_text)``.

    *copy_paste_text* is a **single** message: system instructions, blank line, then user body—no API wrapper
    or ``=== SYSTEM ===`` headers, so the user can paste it verbatim into any LLM chat.
    """
    system = CHARACTER_RESEARCH_SYSTEM_PROMPT
    user = build_character_research_user_prompt(book_title, author, research_blob=None)
    copy_paste = f"{system}\n\n{user}"
    return system, user, copy_paste


def research_characters(
    book_title: str,
    author: str,
    excerpt: str = "",
    *,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Web search + Qwen: return parsed JSON with ``characters`` list."""

    def _p(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, frac)), msg)

    _p(0.02, "Starting character extraction…")
    blob = gather_character_research_blob(book_title, author, on_progress=on_progress)
    _p(0.42, "Building research prompt…")
    user = build_character_research_user_prompt(
        book_title,
        author,
        blob,
        book_excerpt=excerpt,
    )
    _p(
        0.48,
        "Running Qwen (first run loads weights — often 1–3+ min on CPU; wait for this step)…",
    )
    raw_llm = chat_complete(CHARACTER_RESEARCH_SYSTEM_PROMPT, user, max_new_tokens=1600)
    _p(0.92, "Parsing model response…")
    data = extract_json_object(raw_llm)
    if not isinstance(data.get("characters"), list):
        raise ValueError("LLM did not return characters[]")
    _p(1.0, "Character extraction finished.")
    return data


def assign_omnivoice_profiles(rows: list[dict[str, Any]]) -> list[CharacterCard]:
    """Map rough character rows to OmniVoice `voice_instruct` strings (validated by OmniVoice later)."""
    system = (
        "You assign TTS voice design tags to story characters. "
        "Honor **voice_gender**, **voice_age**, **voice_characteristics**, and **voice_accent** from each row "
        "when building `voice_instruct`. **voice_gender** is only male or female; if missing or unclear, treat as **male**. "
        "Output JSON only. "
        + _INSTRUCT_RULES
    )
    payload = {"characters": rows, "narrator_default": AudiobookPlan().narrator_instruct}
    user = f"""{json.dumps(payload, ensure_ascii=False)}

Return JSON:
{{"voices":[{{"name":string,"role":string,"summary":string,"voice_instruct":string}}]}}

Include one entry named exactly "Narrator" for omniscient prose (male or female ok).
voice_instruct must use only allowed tags, comma-separated English."""

    raw = chat_complete(system, user, max_new_tokens=1200)
    data = extract_json_object(raw)
    voices = data.get("voices")
    if not isinstance(voices, list):
        raise ValueError("Expected voices[]")
    out: list[CharacterCard] = []
    for v in voices:
        if not isinstance(v, dict):
            continue
        out.append(
            CharacterCard(
                name=str(v.get("name", "Unknown")).strip() or "Unknown",
                role=str(v.get("role", "")).strip(),
                summary=str(v.get("summary", "")).strip(),
                voice_instruct=str(v.get("voice_instruct", "")).strip()
                or "moderate pitch, american accent",
            )
        )
    return out


def build_clone_prompts_for_cast(
    cards: list[CharacterCard],
    *,
    sample_steps: int = 10,
    sample_line: str | None = None,
) -> tuple[dict[str, VoiceClonePrompt], dict[str, tuple[int, np.ndarray]]]:
    """Synthesize a short clip per character, build clone prompts, return prompts + preview audio."""
    line = (sample_line or "").strip() or (
        "This is my voice for this story — listen carefully."
    )
    prompts: dict[str, VoiceClonePrompt] = {}
    samples: dict[str, tuple[int, np.ndarray]] = {}
    for c in cards:
        key = c.name.strip()
        if not key:
            continue
        logger.info("Voice sample + clone prompt for %s", key)
        vcp, wav = build_voice_clone_from_instruct(
            c.voice_instruct,
            sample_text=line,
            num_step=sample_steps,
            language="English",
        )
        prompts[key] = vcp
        samples[key] = (SAMPLE_RATE, np.asarray(wav, dtype=np.float32).reshape(-1))
    return prompts, samples


def prepare_chapter_segments(
    chapter_body: str,
    character_names: list[str],
) -> list[TTSSegment]:
    """Qwen rewrites one chapter into labeled segments for TTS (IPA, emotion hints)."""
    names = ", ".join(character_names)
    system = (
        "You prepare novel text for expressive text-to-speech. JSON only. "
        "Speakers must be from the provided list."
    )
    body = (chapter_body or "").strip()[:14_000]
    user = f"""Characters: {names}

Chapter text:
{body}

Return JSON:
{{"segments":[
  {{"speaker":string,"text":string,"speed":number|null,"extra_instruct":string|null}}
]}}

Rules:
- Use speaker "Narrator" for narration / inner monologue unless a named character speaks dialogue.
- For tricky names, add IPA once, e.g. Hermione (hɜːˈmaɪəni).
- You may prefix [emotion] tags like [warmly], [tense], [sorrowful] inside `text` where useful.
- `extra_instruct` rarely: only a single extra allowed tag such as whisper (otherwise null).
- Split into natural beats (dialogue vs narration). Keep each text under ~400 characters when possible."""

    raw = chat_complete(system, user, max_new_tokens=2000)
    data = extract_json_object(raw)
    segs = data.get("segments")
    if not isinstance(segs, list):
        raise ValueError("Expected segments[]")
    out: list[TTSSegment] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        sp = str(s.get("speaker", "Narrator")).strip() or "Narrator"
        tx = str(s.get("text", "")).strip()
        if not tx:
            continue
        spd = s.get("speed")
        speed = float(spd) if isinstance(spd, (int, float)) else None
        extra = s.get("extra_instruct")
        ex = str(extra).strip() if extra else None
        out.append(TTSSegment(speaker=sp, text=tx, speed=speed, extra_instruct=ex))
    return out


def synthesize_segments(
    segments: list[TTSSegment],
    clone_prompts: dict[str, VoiceClonePrompt],
    *,
    fallback_key: str = "Narrator",
    pause_ms: int = 280,
    num_step: int = 16,
) -> tuple[int, np.ndarray]:
    """Concatenate segment audio. Unknown speakers use *fallback_key* clone."""
    if not segments:
        return SAMPLE_RATE, np.array([], dtype=np.float32)

    pause = np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for seg in segments:
        vcp = clone_prompts.get(seg.speaker) or clone_prompts.get(fallback_key)
        if vcp is None:
            logger.warning("No clone for speaker %r — skipping line", seg.speaker)
            continue
        res = synthesize_voice_clone_to_numpy(
            seg.text,
            vcp,
            num_step=num_step,
            language="English",
            speed=seg.speed,
            instruct=seg.extra_instruct,
        )
        if res is None:
            continue
        _sr, wav = res
        w = np.asarray(wav, dtype=np.float32).reshape(-1)
        if chunks:
            chunks.append(pause)
        chunks.append(w)
    if not chunks:
        return SAMPLE_RATE, np.array([], dtype=np.float32)
    return SAMPLE_RATE, np.concatenate(chunks, axis=0)
