"""LLM + search orchestration for multi-character audiobook prep (Qwen + OmniVoice)."""

from __future__ import annotations

import io
import json
import logging
import re
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

from book2audio.llm_qwen import chat_complete, extract_json_object
from book2audio.search_web import format_snippets_for_llm, search_snippets
from book2audio.synthesis import (
    SAMPLE_RATE,
    build_voice_clone_from_instruct,
    build_voice_clone_from_reference_audio,
    omnivoice_age_should_omit_pitch,
    synthesize_voice_clone_to_numpy,
)
from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.voice_design import _INSTRUCT_CATEGORIES

# OmniVoice ``instruct`` keyword for hushed/breathy delivery (Style category; spelled in omnivoice pkg).
OMNIVOICE_HUSHED_STYLE_TOKEN: str = next(iter(_INSTRUCT_CATEGORIES[3]))

VOICE_ARCHIVE_JSON = "book2audio_session.json"
VOICE_ARCHIVE_VERSION = 1

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

**Style:** optional hushed / breathy speech — use the **Style** English token from
`voice-design.md` (exactly the short keyword listed in that row for this category).

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
    "**one per category** (gender, age, pitch, optional hushed-speech style token, accent for English). "
    "Never output raw mood words or hyphenated non-tokens (e.g. use `high pitch` not `high-pitched`).\n\n"
    + OMNIVOICE_VOICE_DESIGN_LLM_DOC
)

# Shared across character research, voice-tag assignment, and chapter segmentation prompts.
LLM_INSTRUCTIONS_TTS_SPEAKER_AND_CLONES = """
**Audiobook TTS linkage (read carefully):**
- Each JSON **`name`** is the **canonical speaker ID** for the rest of the pipeline. The app builds **one OmniVoice
  voice clone per `name`** (from `voice_*` → `voice_instruct`, then a short sample). Later, chapter segmentation
  emits a `speaker` string per line; TTS **selects the clone by that `speaker`**. It must be **exactly** one of the
  cast `name` values or **`"Narrator"`** — never pronouns (**he**, **she**, **they**), nicknames you did not list,
  or invented labels, or the line may get the wrong voice or be skipped.
- **OmniVoice** is driven by **`voice_instruct`** (voice design tokens) and the spoken **text**; it does **not**
  take the character’s **name** as a separate “role” control. Names matter because they **key** which saved clone
  runs for each segment.
- Include a **`Narrator`** row when you want a dedicated narration voice in the cast list; chapter prep still uses
  `speaker` **`"Narrator"`** for unattributed narration / inner monologue when no listed character is speaking.
""".strip()

CHARACTER_RESEARCH_SYSTEM_PROMPT = (
    "You extract literary casts with **how each character should sound** for audiobook casting with **OmniVoice** "
    "(downstream TTS uses fixed voice-design tokens from the user message — align voice_* fields with those lists). "
    "Each character **`name`** must be a **stable, canonical ID**: the same string is reused verbatim as the "
    "chapter-segment **`speaker`** field (or use **`Narrator`** for narration rows). "
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
    # Some LLM payloads use "age" instead of voice_age — keep voice_* canonical output.
    if "voice_age" not in out:
        raw_age = item.get("voice_age") or item.get("age")
        if raw_age is not None:
            s = str(raw_age).strip()
            if s:
                out["voice_age"] = s
    raw_g = out.get("voice_gender", "")
    gl = str(raw_g).strip().lower()
    if not gl or gl == "unknown":
        out["voice_gender"] = "male"
    elif gl == "female":
        out["voice_gender"] = "female"
    elif gl == "male":
        out["voice_gender"] = "male"
    return out


def _voice_age_fragment_for_row(row: dict[str, Any]) -> str:
    """Raw age string from editor/LLM row (supports common key aliases)."""
    for key in ("voice_age", "age", "voice_age_group", "life_stage"):
        raw = row.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return ""


def _normalize_age_text(s: str) -> str:
    t = (s or "").strip().lower()
    for ch in ("\u2013", "\u2014", "\u2212"):  # en, em, minus
        t = t.replace(ch, "-")
    t = t.replace("_", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def resolve_voice_age_token(age_raw: str) -> str:
    """Map ``voice_age`` / ``age`` free text to an OmniVoice **age** token.

    Longer phrases are matched first so e.g. *young adult* wins over *adult*.
    """
    s = _normalize_age_text(age_raw)
    if not s:
        return "middle-aged"

    exact_ok = {"child", "teenager", "young adult", "middle-aged", "elderly"}
    if s in exact_ok:
        return s
    # Hyphen/space variants already normalized for contains checks; allow compact tokens.
    compact = s.replace("-", " ").replace("  ", " ").strip()
    if compact in exact_ok:
        return compact

    aliases: list[tuple[str, str]] = [
        ("young adult", "young adult"),
        ("middle aged", "middle-aged"),
        ("middle-aged", "middle-aged"),
        ("young-adult", "young adult"),
        ("middle-age", "middle-aged"),
        ("octogenarian", "elderly"),
        ("nonagenarian", "elderly"),
        ("centenarian", "elderly"),
        ("adolescent", "teenager"),
        ("teenager", "teenager"),
        ("teenage", "teenager"),
        ("children", "child"),
        ("geriatric", "elderly"),
        ("elderly", "elderly"),
        ("seniors", "elderly"),
        ("senior", "elderly"),
        ("infant", "child"),
        ("toddler", "child"),
        ("teen", "teenager"),
        ("child", "child"),
        ("kids", "child"),
        ("kid", "child"),
        ("elder", "elderly"),
        ("old age", "elderly"),
        ("late middle", "middle-aged"),
        ("prime adult", "young adult"),
        ("twenties", "young adult"),
        ("thirties", "young adult"),
        ("forties", "middle-aged"),
        ("fifties", "middle-aged"),
        ("sixties", "elderly"),
        ("seventies", "elderly"),
        ("eighties", "elderly"),
        ("20s", "young adult"),
        ("30s", "young adult"),
        ("40s", "middle-aged"),
        ("50s", "middle-aged"),
        ("60s", "elderly"),
        ("70s", "elderly"),
        ("80s", "elderly"),
        ("90s", "elderly"),
        ("youth", "young adult"),
        ("young", "young adult"),
        ("ya", "young adult"),
        ("mature", "middle-aged"),
        ("middle", "middle-aged"),
    ]
    by_len = sorted(aliases, key=lambda x: len(x[0]), reverse=True)
    for needle, tag in by_len:
        if needle in s or s == needle.replace(" ", "-"):
            return tag
    if s == "adult":
        return "young adult"
    return "middle-aged"


def _omnivoice_instruct_from_row(row: dict[str, Any]) -> str:
    """Build OmniVoice tag string from editor voice_* fields when the LLM returns a weak instruct."""
    g = str(row.get("voice_gender", "male")).strip().lower()
    gender = "female" if g == "female" else "male"
    age = resolve_voice_age_token(_voice_age_fragment_for_row(row))
    ch = str(row.get("voice_characteristics", "")).lower()
    tok = OMNIVOICE_HUSHED_STYLE_TOKEN  # Style: ``whisper`` (see voice-design.md)
    wants_whisper = (
        tok in ch
        or "breathy" in ch
        or "hushed" in ch
        or "under breath" in ch
    )
    omit_pitch = omnivoice_age_should_omit_pitch(age)
    if not omit_pitch:
        if any(x in ch for x in ("very low", "deep", "gravel")):
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

    parts = [gender, age] if omit_pitch else [gender, age, pitch]
    if wants_whisper:
        parts.append(tok)
    parts.append(accent)
    return ", ".join(parts)


def _instruct_from_llm_usable(instruct: str, *, default: str = "moderate pitch, american accent") -> bool:
    s = (instruct or "").strip()
    if len(s) < 12:
        return False
    if s.lower() == default.lower():
        return False
    low = s.lower()
    tok = OMNIVOICE_HUSHED_STYLE_TOKEN
    young_age_word = any(
        x in low for x in ("child", "teenager", "儿童", "少年")
    )
    return ("male" in low or "female" in low) and (
        "accent" in low
        or "pitch" in low
        or tok in low
        or young_age_word
    )


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
    #: Web-snippet blob from the last **local** extract (search + Qwen); empty for external-LLM-only workflows.
    character_research_blob: str = ""
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

{LLM_INSTRUCTIONS_TTS_SPEAKER_AND_CLONES}

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
) -> tuple[dict[str, Any], str]:
    """Web search + Qwen: return ``(parsed JSON with characters[], research_blob)``."""

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
    return data, blob


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
        "Each returned `name` must match an input row’s `name` exactly — that string is the **speaker ID** "
        "for later chapter JSON. "
        "Output JSON only.\n\n"
        + _INSTRUCT_RULES
    )
    payload = {"characters": rows, "narrator_default": AudiobookPlan().narrator_instruct}
    user = f"""{json.dumps(payload, ensure_ascii=False)}

Return JSON:
{{"voices":[{{"name":string,"voice_instruct":string}}]}}

Rules:
- For **every** object in input ``characters``, output exactly one ``voices`` entry with the **same** ``name`` string (identical spelling and casing): that string is the **speaker ID** for chapter segmentation and voice-clone lookup.
- ``voice_instruct`` MUST use **only** tokens from the OmniVoice lists above (comma + space); **one token per category**; map ``voice_accent`` prose to one **accent** token (English book → English accent).
- Do **not** emit mood words, hyphenated invented tags, or comma-separated prose traits as tags.
- If there is no "Narrator" in the cast but you add one for omniscient prose, use an extra entry; otherwise do not invent names.

{LLM_INSTRUCTIONS_TTS_SPEAKER_AND_CLONES}"""

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


def safe_voice_archive_basename(name: str) -> str:
    """Filesystem-safe stem for WAV files in the voice archive (matches export naming)."""
    base = re.sub(r'[<>:"/\\|?*\n\r]+', "_", (name or "").strip()) or "voice"
    return base[:120]


def read_wav_bytes_mono_float32(data: bytes) -> tuple[int, np.ndarray]:
    """Load PCM WAV bytes → ``(sample_rate, mono float32)``. Prefer 16-bit mono/stereo."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
    if sample_width == 1:
        x = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sample_width == 2:
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(
            f"Unsupported WAV sample width {sample_width} (use 16-bit PCM for best results)"
        )
    if n_channels <= 1:
        mono = x.reshape(-1)
    else:
        mono = np.mean(x.reshape(-1, n_channels), axis=1)
    return framerate, mono.astype(np.float32)


def voice_archive_dict_from_plan(plan: AudiobookPlan) -> dict[str, Any]:
    """JSON sidecar written next to WAVs inside the downloadable archive."""
    voice_files = {
        n: f"{safe_voice_archive_basename(n)}.wav"
        for n in plan.voice_samples.keys()
    }
    return {
        "book2audio_archive_version": VOICE_ARCHIVE_VERSION,
        "raw_character_rows": plan.raw_character_rows,
        "character_cards": [
            {
                "name": c.name,
                "role": c.role,
                "summary": c.summary,
                "voice_instruct": c.voice_instruct,
            }
            for c in plan.characters
        ],
        "narrator_instruct": plan.narrator_instruct,
        "character_research_blob": plan.character_research_blob,
        "voice_files": voice_files,
    }


def _find_zip_member_ci(zf: zipfile.ZipFile, basename: str) -> str | None:
    want = Path(basename).name.lower()
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if Path(name).name.lower() == want:
            return name
    return None


def character_cards_from_voice_archive_manifest(manifest: dict[str, Any]) -> list[CharacterCard]:
    """Rebuild :class:`CharacterCard` instances from archive JSON (no Qwen)."""
    cards_raw = manifest.get("character_cards")
    rows_raw = manifest.get("raw_character_rows") or []
    rows: list = list(rows_raw) if isinstance(rows_raw, list) else []

    if isinstance(cards_raw, list) and cards_raw:
        out: list[CharacterCard] = []
        for d in cards_raw:
            if not isinstance(d, dict):
                continue
            n = str(d.get("name", "")).strip()
            if not n:
                continue
            vi = str(d.get("voice_instruct", "")).strip()
            if not vi:
                row = next(
                    (
                        r
                        for r in rows
                        if isinstance(r, dict)
                        and str(r.get("name", "")).strip().lower() == n.lower()
                    ),
                    None,
                )
                vi = _omnivoice_instruct_from_row(
                    row if row else {"name": n, "voice_gender": "male"}
                )
            out.append(
                CharacterCard(
                    name=n,
                    role=str(d.get("role", "")).strip(),
                    summary=str(d.get("summary", "")).strip(),
                    voice_instruct=vi,
                )
            )
        return out

    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        n = str(row.get("name", "")).strip()
        if not n:
            continue
        out.append(
            CharacterCard(
                name=n,
                role=str(row.get("role", "")).strip(),
                summary=str(row.get("summary", "")).strip(),
                voice_instruct=_omnivoice_instruct_from_row(row),
            )
        )
    return out


def _minimal_row_for_wav_import(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "role": "Imported from WAV",
        "summary": "Voice reference imported from filename; edit JSON for story context.",
        "voice_gender": "male",
        "voice_age": "middle-aged",
        "voice_characteristics": "",
        "voice_accent": "unknown",
    }


def import_voice_archive_from_zip_bytes(
    data: bytes,
    *,
    on_progress: Callable[[float, str], None] | None = None,
) -> AudiobookPlan:
    """Load ``book2audio_session.json`` + WAVs from a ZIP, or WAV-only ZIP / loose WAV repack."""
    plan = AudiobookPlan()
    plan.last_segments = []

    def _p(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, frac)), msg)

    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        manifest: dict[str, Any] | None = None
        try:
            raw_m = zf.read(VOICE_ARCHIVE_JSON).decode("utf-8")
            manifest = json.loads(raw_m)
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
            manifest = None

        voice_files: dict[str, Any] = {}
        use_wav_only = manifest is None

        if isinstance(manifest, dict):
            ver = manifest.get("book2audio_archive_version", 0)
            if ver > VOICE_ARCHIVE_VERSION:
                logger.warning(
                    "Archive format newer than this app (version %s)",
                    ver,
                )
            rows = manifest.get("raw_character_rows") or []
            plan.raw_character_rows = list(rows) if isinstance(rows, list) else []
            plan.character_research_blob = str(manifest.get("character_research_blob") or "")
            ni = manifest.get("narrator_instruct")
            if isinstance(ni, str) and ni.strip():
                plan.narrator_instruct = ni.strip()
            plan.characters = character_cards_from_voice_archive_manifest(manifest)
            vf = manifest.get("voice_files")
            voice_files = vf if isinstance(vf, dict) else {}
            if not plan.characters:
                use_wav_only = True

        if use_wav_only:
            wav_members = sorted(
                (
                    n
                    for n in zf.namelist()
                    if n.lower().endswith(".wav") and not n.endswith("/")
                ),
                key=lambda x: Path(x).name.lower(),
            )
            seen_base: set[str] = set()
            ordered: list[str] = []
            for m in wav_members:
                b = Path(m).name.lower()
                if b in seen_base:
                    continue
                seen_base.add(b)
                ordered.append(m)

            if not ordered:
                raise ValueError(
                    "No book2audio_session.json (or empty cast) and no .wav files in ZIP."
                )

            rows = []
            cards = []
            for i, member in enumerate(ordered):
                stem = Path(member).stem.strip()
                disp = stem or f"voice_{i + 1}"
                row = _minimal_row_for_wav_import(disp)
                rows.append(row)
                cards.append(
                    CharacterCard(
                        name=disp,
                        role=row["role"],
                        summary=row["summary"],
                        voice_instruct=_omnivoice_instruct_from_row(row),
                    )
                )
            plan.raw_character_rows = rows
            plan.characters = cards
            nload = len(cards)
            plan.clone_prompts = {}
            plan.voice_samples = {}
            for i, card in enumerate(plan.characters):
                member = ordered[i]
                _p((i + 1) / max(nload, 1), f"Import WAV {i + 1}/{nload} ({card.name})…")
                pcm = read_wav_bytes_mono_float32(zf.read(member))
                vcp, preview, prate = build_voice_clone_from_reference_audio(
                    pcm,
                    ref_text=None,
                )
                plan.clone_prompts[card.name] = vcp
                plan.voice_samples[card.name] = (prate, preview)
            _p(1.0, "Import finished.")
            plan.diagnostics = (
                f"Imported {nload} WAV-only voice(s); edit character JSON as needed."
            )
            return plan

        nload = len(plan.characters)
        plan.clone_prompts = {}
        plan.voice_samples = {}
        for i, card in enumerate(plan.characters):
            _p((i + 1) / max(nload, 1), f"Import {card.name} ({i + 1}/{nload})…")
            fn = str(voice_files.get(card.name) or "").strip()
            if not fn:
                fn = f"{safe_voice_archive_basename(card.name)}.wav"
            member = _find_zip_member_ci(zf, fn)
            if member is None:
                member = _find_zip_member_ci(
                    zf,
                    f"{safe_voice_archive_basename(card.name)}.wav",
                )
            if member is None:
                logger.warning("Missing WAV for %r (expected %r)", card.name, fn)
                continue
            pcm = read_wav_bytes_mono_float32(zf.read(member))
            vcp, preview, prate = build_voice_clone_from_reference_audio(
                pcm,
                ref_text=None,
            )
            plan.clone_prompts[card.name] = vcp
            plan.voice_samples[card.name] = (prate, preview)

        _p(1.0, "Import finished.")
        plan.diagnostics = f"Imported archive ({len(plan.voice_samples)} voice(s))."
        return plan


def import_voice_wav_files_from_paths(
    paths: list[str | Path],
    *,
    on_progress: Callable[[float, str], None] | None = None,
) -> AudiobookPlan:
    """Import one folder’s worth of WAV files (names from stems) via a temporary ZIP path."""
    uniq = sorted(
        {Path(p).resolve() for p in paths if str(p).strip()},
        key=lambda p: p.name.lower(),
    )
    wavs = [p for p in uniq if p.suffix.lower() == ".wav"]
    if not wavs:
        raise ValueError("Need at least one .wav file.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in wavs:
            zf.write(p, arcname=p.name)
    return import_voice_archive_from_zip_bytes(
        buf.getvalue(),
        on_progress=on_progress,
    )


def regenerate_single_voice_sample(
    plan: AudiobookPlan,
    character_name: str,
    *,
    sample_steps: int,
    sample_line: str | None,
    editor_text: str | None = None,
) -> AudiobookPlan:
    """Re-run OmniVoice sample + clone prompt for one cast member; updates ``plan`` in place.

    If *editor_text* is provided (current **Characters** JSON), it is parsed first so **role**,
    **summary**, and **voice_\\*** fields edited by hand are applied before regenerating.
    The selected character’s ``voice_instruct`` is rebuilt with
    :func:`_omnivoice_instruct_from_row` so tags always match the editor (regenerate does not
    call Qwen per character — use **Assign voice tags & generate samples** for LLM-assisted
    tagging of the full cast).
    """
    name = (character_name or "").strip()
    if not name:
        raise ValueError("Select a character name")
    if not plan.characters:
        raise ValueError("No cast in session — run voice setup first")

    if editor_text is not None and str(editor_text).strip():
        rows = parse_character_json(editor_text)
        if not rows:
            raise ValueError(
                "Character JSON is empty or invalid — fix the editor or run voice setup again."
            )
        plan.raw_character_rows = rows
        by_lower: dict[str, dict[str, Any]] = {}
        for r in rows:
            n = str(r.get("name", "")).strip()
            if n:
                by_lower[n.lower()] = r
        if name.lower() not in by_lower:
            raise ValueError(
                f"Character {name!r} not found in the JSON editor — check spelling or add this name."
            )
        new_chars: list[CharacterCard] = []
        for c in plan.characters:
            r = by_lower.get(c.name.lower())
            if r is not None:
                new_chars.append(
                    CharacterCard(
                        name=c.name,
                        role=str(r.get("role", "")).strip(),
                        summary=str(r.get("summary", "")).strip(),
                        voice_instruct=c.voice_instruct,
                    )
                )
            else:
                new_chars.append(c)
        plan.characters = new_chars
        row = by_lower[name.lower()]
        nm = str(row.get("name", "")).strip()
        fresh_card = CharacterCard(
            name=nm,
            role=str(row.get("role", "")).strip(),
            summary=str(row.get("summary", "")).strip(),
            voice_instruct=_omnivoice_instruct_from_row(row),
        )
        logger.info(
            "regenerate_single_voice_sample: editor sync for %r → voice_instruct=%r "
            "(voice_gender=%r voice_age=%r voice_accent=%r)",
            nm,
            fresh_card.voice_instruct,
            row.get("voice_gender"),
            row.get("voice_age"),
            row.get("voice_accent"),
        )
        plan.characters = [
            fresh_card if c.name.lower() == fresh_card.name.lower() else c
            for c in plan.characters
        ]
        name = fresh_card.name.strip()

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
        "In the host app, each cast name already has **one OmniVoice voice clone**; each segment’s "
        "`speaker` field **selects which clone** synthesizes that line (`voice_instruct` was fixed "
        "at casting time — it is not re-specified here). "
        "`speaker` MUST be exactly one of the allowed JSON strings (verbatim); "
        "otherwise the engine cannot select the correct voice."
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
- `extra_instruct` rarely: only one extra OmniVoice Style token when needed (otherwise null). See voice-design.md Style list.
- Split into natural beats (dialogue vs narration). Keep each text under ~400 characters when possible.

{LLM_INSTRUCTIONS_TTS_SPEAKER_AND_CLONES}"""
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
