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

# Summarized from OmniVoice docs (voice-design mode).
# Source: https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md
OMNIVOICE_VOICE_DESIGN_LLM_DOC = """
OmniVoice **voice design** (downstream TTS): the app will turn each character into an
`instruct` string — comma + space separated attributes in **English** (case-insensitive).
Each attribute belongs to **one category** (gender, age, pitch, style, accent, or Chinese dialect).
**Only one token per category.** Do not use free-text mood words as tags (e.g. not “anxious”,
“breathless”); describe prose in **voice_* fields** so they can be mapped to the tokens below.

**Gender:** male | female

**Age:** child | teenager | young adult | middle-aged | elderly

**Pitch:** very low pitch | low pitch | moderate pitch | high pitch | very high pitch

**Style:** whisper

**English accents** (use when book/audio is English; pick **one**):
american accent, british accent, australian accent, canadian accent, indian accent,
chinese accent, korean accent, japanese accent, portuguese accent, russian accent

**Chinese dialects** (only when synthesis text is Chinese; full-width comma in pure-Chinese strings):
河南话, 陕西话, 四川话, 贵州话, 云南话, 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话, 青岛话, 东北话

**Writing tips (from OmniVoice docs):** combine one token from each category you care about,
e.g. `female, young adult, high pitch, british accent`. English uses half-width commas + space.
Accent applies to English speech; dialect applies to Chinese speech — do not mix accent+dialect
in the same design string. Omit categories you don’t care about; a minimal design is still valid.

Doc link: https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md
""".strip()

_INSTRUCT_RULES = (
    "OmniVoice `voice_instruct` must use **only** allowed English tokens, comma + space, "
    "**one per category** (gender, age, pitch, optional whisper, accent for English). "
    "Never output raw mood words or hyphenated non-tokens (e.g. use `high pitch` not `high-pitched`).\n\n"
    + OMNIVOICE_VOICE_DESIGN_LLM_DOC
)

CHARACTER_RESEARCH_SYSTEM_PROMPT = (
    "You extract literary casts with **how each character should sound** for audiobook casting with **OmniVoice** "
    "(downstream TTS uses fixed voice-design tokens from the user message — align voice_* fields with those lists). "
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


def _omnivoice_instruct_from_row(row: dict[str, Any]) -> str:
    """Build OmniVoice tag string from editor voice_* fields when the LLM returns a weak instruct."""
    g = str(row.get("voice_gender", "male")).strip().lower()
    gender = "female" if g == "female" else "male"
    age_raw = str(row.get("voice_age", "middle-aged")).strip().lower()
    age_aliases = [
        ("child", "child"),
        ("teenager", "teenager"),
        ("young adult", "young adult"),
        ("young", "young adult"),
        ("middle-aged", "middle-aged"),
        ("middle aged", "middle-aged"),
        ("elderly", "elderly"),
    ]
    age = "middle-aged"
    for needle, tag in age_aliases:
        if needle in age_raw or age_raw == needle.replace(" ", "-"):
            age = tag
            break
    ch = str(row.get("voice_characteristics", "")).lower()
    if "whisper" in ch:
        pitch = "whisper"
    elif any(x in ch for x in ("very low", "deep", "gravel")):
        pitch = "very low pitch"
    elif any(x in ch for x in ("low pitch", "low,", " low ", "deep")):
        pitch = "low pitch"
    elif any(x in ch for x in ("very high", "shrill")):
        pitch = "very high pitch"
    elif any(x in ch for x in ("high pitch", "bright", "light")):
        pitch = "high pitch"
    else:
        pitch = "moderate pitch"

    acc_raw = str(row.get("voice_accent", "")).strip().lower()
    accent = "american accent"
    if any(x in acc_raw for x in ("british", "uk ", " uk", "rp ", " england")):
        accent = "british accent"
    elif "australian" in acc_raw:
        accent = "australian accent"
    elif "canadian" in acc_raw:
        accent = "canadian accent"
    elif "indian" in acc_raw:
        accent = "indian accent"
    elif any(x in acc_raw for x in ("chinese", "mandarin", "cantonese")):
        accent = "chinese accent"
    elif "japanese" in acc_raw:
        accent = "japanese accent"
    elif "korean" in acc_raw:
        accent = "korean accent"
    elif "russian" in acc_raw:
        accent = "russian accent"
    elif "portuguese" in acc_raw or "brazil" in acc_raw:
        accent = "portuguese accent"

    return f"{gender}, {age}, {pitch}, {accent}"


def _instruct_from_llm_usable(instruct: str, *, default: str = "moderate pitch, american accent") -> bool:
    s = (instruct or "").strip()
    if len(s) < 12:
        return False
    if s.lower() == default.lower():
        return False
    low = s.lower()
    return ("male" in low or "female" in low) and ("accent" in low or "pitch" in low or "whisper" in low)


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


def _normalize_segment_speaker_label(raw: str) -> str:
    """Strip wrappers/punctuation Qwen sometimes adds around dialogue labels."""
    t = (raw or "").strip()
    while t.endswith((":", ".", ",", ";")):
        t = t[:-1].strip()
    t = t.strip('"').strip("'").strip()
    while t.endswith((":", ".", ",", ";")):
        t = t[:-1].strip()
    return t


def resolve_clone_prompt_for_speaker(
    clone_prompts: dict[str, VoiceClonePrompt],
    speaker: str,
    *,
    fallback_key: str = "Narrator",
) -> tuple[VoiceClonePrompt | None, str]:
    """Match segment ``speaker`` to a per-character clone (case-insensitive on cast keys).

    OmniVoice only receives the resolved ``VoiceClonePrompt`` (built per cast member in Step 2)
    plus segment ``text`` / optional ``extra_instruct``. This mapping is therefore what ties
    dialogue lines to the right voice.
    """
    if not clone_prompts:
        return None, ""

    label = _normalize_segment_speaker_label(speaker)
    if not label:
        label = fallback_key

    if label in clone_prompts:
        return clone_prompts[label], label

    by_lower: dict[str, tuple[str, VoiceClonePrompt]] = {
        k.lower(): (k, v) for k, v in clone_prompts.items()
    }
    hit = by_lower.get(label.lower())
    if hit:
        canon, vcp = hit
        return vcp, canon

    if fallback_key in clone_prompts:
        return clone_prompts[fallback_key], fallback_key

    fb = by_lower.get(fallback_key.lower())
    if fb:
        canon, vcp = fb
        return vcp, canon

    return None, ""


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
    return f"""Return JSON exactly in this shape (every character MUST include voice fields — infer cautiously if needed):
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
- **voice_age**: align with OmniVoice when possible: child, teenager, young adult, middle-aged, elderly.
- **voice_characteristics**: how they *sound* in plain English (timbre, tempo, energy). Only use OmniVoice tokens.
- **voice_accent**: free-text locale or style that maps to **one** English accent name from the OmniVoice list (e.g. "RP British" → british accent); use "unknown" if not inferable.

OmniVoice reference (your descriptions feed Step 2 `voice_instruct`):
{OMNIVOICE_VOICE_DESIGN_LLM_DOC}

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
    """Map editor rows to :class:`CharacterCard`; Qwen supplies ``voice_instruct`` when helpful.

    **role** and **summary** always come from the pasted editor JSON so OmniVoice samples
    and clone prompts stay aligned with the user's cast. Weak or generic LLM tags are
    replaced with :func:`_omnivoice_instruct_from_row`.
    """
    system = (
        "You assign OmniVoice **voice design** `voice_instruct` strings to story characters. "
        "Follow the OmniVoice attribute rules below exactly: only allowed tokens, comma + space, one per category. "
        "Honor **voice_gender**, **voice_age**, **voice_characteristics**, and **voice_accent** from each row. "
        "**voice_gender** is only male or female; if missing or unclear, treat as **male**. "
        "Output JSON only.\n\n"
        + _INSTRUCT_RULES
    )
    payload = {"characters": rows, "narrator_default": AudiobookPlan().narrator_instruct}
    user = f"""{json.dumps(payload, ensure_ascii=False)}

Return JSON:
{{"voices":[{{"name":string,"voice_instruct":string}}]}}

Rules:
- For **every** object in input ``characters``, output exactly one ``voices`` entry with the **same** ``name`` string.
- ``voice_instruct`` MUST use **only** tokens from the OmniVoice lists above (comma + space); **one token per category**; map ``voice_accent`` prose to one **accent** token (English book → English accent).
- Do **not** emit mood words, hyphenated invented tags, or comma-separated prose traits as tags.
- If there is no "Narrator" in the cast but you add one for omniscient prose, use an extra entry; otherwise do not invent names."""

    raw = chat_complete(system, user, max_new_tokens=1200)
    data = extract_json_object(raw)
    voices = data.get("voices")
    if not isinstance(voices, list):
        raise ValueError("Expected voices[]")

    by_name: dict[str, dict[str, Any]] = {}
    for v in voices:
        if isinstance(v, dict):
            n = str(v.get("name", "")).strip()
            if n:
                by_name[n.lower()] = v

    default_ins = "moderate pitch, american accent"
    out: list[CharacterCard] = []
    row_names_lower: set[str] = set()

    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        row_names_lower.add(name.lower())
        role = str(row.get("role", "")).strip()
        summary = str(row.get("summary", "")).strip()
        v = by_name.get(name.lower())
        llm_ins = str(v.get("voice_instruct", "")).strip() if v else ""
        if _instruct_from_llm_usable(llm_ins, default=default_ins):
            instruct = llm_ins
        else:
            instruct = _omnivoice_instruct_from_row(row)
            if llm_ins:
                logger.info(
                    "Using editor voice_* fields for %r (LLM instruct missing or generic)",
                    name,
                )
        out.append(
            CharacterCard(
                name=name,
                role=role,
                summary=summary,
                voice_instruct=instruct,
            )
        )

    for v in voices:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name", "")).strip()
        if not name or name.lower() in row_names_lower:
            continue
        llm_ins = str(v.get("voice_instruct", "")).strip() or default_ins
        role = str(v.get("role", "")).strip()
        summary = str(v.get("summary", "")).strip()
        out.append(
            CharacterCard(
                name=name,
                role=role,
                summary=summary,
                voice_instruct=llm_ins,
            )
        )

    return out


def _sample_line_for_character(card: CharacterCard) -> str:
    """Spoken line for clone preview when the user leaves the global sample line empty."""
    if card.summary.strip():
        t = card.summary.strip()
        if len(t) > 280:
            t = t[:277] + "..."
        return f"I'm {card.name}. {t}"
    if card.role.strip():
        return (
            f"I'm {card.name}, {card.role.strip()} — this is my voice for this audiobook."
        )
    return f"Hello — this is {card.name}'s voice for this story."


def build_clone_prompts_for_cast(
    cards: list[CharacterCard],
    *,
    sample_steps: int = 10,
    sample_line: str | None = None,
) -> tuple[dict[str, VoiceClonePrompt], dict[str, tuple[int, np.ndarray]]]:
    """Synthesize a short clip per character, build clone prompts, return prompts + preview audio."""
    global_sample = (sample_line or "").strip()
    prompts: dict[str, VoiceClonePrompt] = {}
    samples: dict[str, tuple[int, np.ndarray]] = {}
    for c in cards:
        key = c.name.strip()
        if not key:
            continue
        phrase = global_sample if global_sample else _sample_line_for_character(c)
        logger.info("Voice sample + clone prompt for %s", key)
        vcp, wav = build_voice_clone_from_instruct(
            c.voice_instruct,
            sample_text=phrase,
            num_step=sample_steps,
            language="English",
        )
        prompts[key] = vcp
        samples[key] = (SAMPLE_RATE, np.asarray(wav, dtype=np.float32).reshape(-1))
    return prompts, samples


def apply_external_reference_voice_to_character(
    plan: AudiobookPlan,
    character_name: str,
    audio: tuple[int, np.ndarray] | None,
    ref_transcript: str | None = None,
) -> AudiobookPlan:
    """Replace one cast member’s clone prompt with an uploaded reference clip."""
    from book2audio.synthesis import build_voice_clone_from_reference_audio

    name = (character_name or "").strip()
    if not name:
        raise ValueError("Select a character name")
    if not plan.characters:
        raise ValueError("No cast in session — run voice setup first")
    card = next((c for c in plan.characters if c.name.lower() == name.lower()), None)
    if card is None:
        raise ValueError(f"No character named {name!r} in the current cast")
    canonical = card.name.strip()
    vcp, preview, prate = build_voice_clone_from_reference_audio(
        audio,
        ref_text=(ref_transcript or "").strip() or None,
    )
    plan.clone_prompts[canonical] = vcp
    plan.voice_samples[canonical] = (prate, preview)
    return plan


def regenerate_single_voice_sample(
    plan: AudiobookPlan,
    character_name: str,
    *,
    sample_steps: int,
    sample_line: str | None,
) -> AudiobookPlan:
    """Re-run OmniVoice sample + clone prompt for one cast member; updates ``plan`` in place."""
    name = (character_name or "").strip()
    if not name:
        raise ValueError("Select a character name")
    if not plan.characters:
        raise ValueError("No cast in session — run voice setup first")
    card = next((c for c in plan.characters if c.name.lower() == name.lower()), None)
    if card is None:
        raise ValueError(f"No character named {name!r} in the current cast")
    name = card.name.strip()

    steps = max(4, int(sample_steps))
    line = (sample_line or "").strip() or None
    prompts, samples = build_clone_prompts_for_cast(
        [card],
        sample_steps=steps,
        sample_line=line,
    )
    plan.clone_prompts[name] = prompts[name]
    plan.voice_samples[name] = samples[name]
    return plan


def prepare_chapter_segments(
    chapter_body: str,
    character_names: list[str],
) -> list[TTSSegment]:
    """Qwen rewrites one chapter into labeled segments for TTS (IPA, emotion hints)."""
    names = ", ".join(character_names)
    system = (
        "You prepare novel text for expressive text-to-speech. JSON only. "
        "Each segment `speaker` MUST be exactly one of the allowed JSON strings (verbatim); "
        "otherwise the audiobook engine cannot select the correct voice clone."
    )
    body = (chapter_body or "").strip()[:14_000]
    cast_verbatim = "\n".join(f"  {json.dumps(n, ensure_ascii=False)}" for n in character_names)
    user = f"""Cast overview: {names}

Each segment's `speaker` must be EXACTLY one of these strings (copy spelling, spaces, and casing):
{cast_verbatim}
  "Narrator"

Chapter text:
{body}

Return JSON:
{{"segments":[
  {{"speaker":string,"text":string,"speed":number|null,"extra_instruct":string|null}}
]}}

Rules:
- Use `speaker` "Narrator" only for unattributed narration or inner monologue; use a cast name when a named character speaks (including lines like “they said” if the speaker is known from context).
- Never invent new `speaker` values or use pronouns (“he”, “she”, “they”) as `speaker` — always the character’s listed name or "Narrator".
- For tricky names, add IPA once inside `text`, e.g. Hermione (hɜːˈmaɪəni).
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
        sp = _normalize_segment_speaker_label(str(s.get("speaker", "Narrator"))) or "Narrator"
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
        vcp, canon_key = resolve_clone_prompt_for_speaker(
            clone_prompts,
            seg.speaker,
            fallback_key=fallback_key,
        )
        if vcp is None:
            logger.warning("No clone for speaker %r — skipping line", seg.speaker)
            continue
        norm_sp = _normalize_segment_speaker_label(seg.speaker)
        if norm_sp.lower() != canon_key.lower():
            logger.info(
                "Segment speaker %r → using voice clone for %r",
                seg.speaker,
                canon_key,
            )
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
