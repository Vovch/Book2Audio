"""Gradio web UI for phrase → speech and audiobook pipeline (Qwen + OmniVoice)."""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import wave
import zipfile
from typing import Any

import gradio as gr
import numpy as np

from book2audio.audiobook_pipeline import (
    AudiobookPlan,
    EXAMPLE_CHARACTERS_EDITOR_JSON,
    VOICE_ARCHIVE_JSON,
    apply_external_reference_voice_to_character,
    assign_omnivoice_profiles,
    build_clone_prompts_for_cast,
    character_rows_to_editor_json,
    format_character_research_messages_for_external_llm,
    import_voice_archive_from_zip_bytes,
    import_voice_wav_files_from_paths,
    parse_character_json,
    prepare_chapter_segments,
    regenerate_single_voice_sample,
    research_characters,
    safe_voice_archive_basename,
    split_into_chapters,
    synthesize_segments,
    voice_archive_dict_from_plan,
)
from book2audio.device import pick_device_and_dtype
from book2audio.llm_qwen import default_llm_model_id
from book2audio.synthesis import synthesize_to_numpy

logger = logging.getLogger(__name__)


def _summarize_for_log(label: str, value: Any, *, max_str: int = 600) -> str:
    """Short, log-safe rendering of Gradio callback arguments (no huge blobs)."""
    if value is None:
        return f"{label}=None"
    if isinstance(value, AudiobookPlan):
        p = value
        names = [c.name for c in p.characters]
        vn = list(p.voice_samples.keys())
        instr = [c.voice_instruct[:80] + "…" if len(c.voice_instruct) > 80 else c.voice_instruct for c in p.characters[:8]]
        extra = "…" if len(p.characters) > 8 else ""
        return (
            f"{label}=AudiobookPlan(id={id(p)}, n_char={len(p.characters)}, names={names!r}, "
            f"voice_keys={vn!r}, n_prompts={len(p.clone_prompts)}, "
            f"instruct_preview[{len(instr)}]={instr!r}{extra})"
        )
    if isinstance(value, str):
        if len(value) > max_str:
            return f"{label}=str(len={len(value)}, head={value[:max_str]!r}...)"
        return f"{label}={value!r}"
    if isinstance(value, (int, float, bool)):
        return f"{label}={value!r}"
    if isinstance(value, np.ndarray):
        return f"{label}=ndarray(shape={value.shape}, dtype={value.dtype})"
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], np.ndarray)
    ):
        sr, arr = value
        return f"{label}=(sr={sr!r}, audio_ndarray(shape={arr.shape}, dtype={arr.dtype}))"
    if isinstance(value, gr.Progress):
        return f"{label}=<gr.Progress>"
    if isinstance(value, (list, tuple)):
        n = len(value)
        if n == 0:
            return f"{label}=empty_{type(value).__name__}"
        head = value[:5]
        parts = [_summarize_for_log(f"{label}[{i}]", x, max_str=120) for i, x in enumerate(head)]
        tail = f", …(+{n - len(head)} more)" if n > len(head) else ""
        return f"{label}={type(value).__name__}(n={n}, head=[{'; '.join(parts)}]{tail})"
    path = getattr(value, "path", None)
    if isinstance(path, str) and path:
        return f"{label}={type(value).__name__}(path={path!r})"
    return f"{label}={type(value).__name__}@{id(value)!r}"


def _log_ui_event(handler: str, level: int = logging.INFO, **params: Any) -> None:
    if not logger.isEnabledFor(level):
        return
    parts = [_summarize_for_log(k, v) for k, v in params.items()]
    logger.log(level, "UI event %s → %s", handler, " | ".join(parts))


def _wav_bytes_mono_float32(sample_rate: int, audio: np.ndarray) -> bytes:
    s = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = np.clip(s * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _zip_voice_samples_to_tempfile(plan: AudiobookPlan) -> str:
    manifest = voice_archive_dict_from_plan(plan)
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    fd, path = tempfile.mkstemp(prefix="book2audio_voice_samples_", suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(VOICE_ARCHIVE_JSON, manifest_json)
            for display_name, (rate, wav) in plan.voice_samples.items():
                fn = manifest["voice_files"][display_name]
                zf.writestr(fn, _wav_bytes_mono_float32(rate, wav))
        return path
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise


def _regenerate_selected_voice_sample(
    character_name: str,
    sample_steps: float,
    sample_line: str,
    characters_editor: str,
    session: AudiobookPlan | None,
) -> tuple[Any, str, AudiobookPlan]:
    _log_ui_event(
        "_regenerate_selected_voice_sample / Regenerate selected voice sample",
        character_name=character_name,
        sample_steps=sample_steps,
        sample_line=sample_line,
        characters_editor=characters_editor,
        session=session,
    )
    plan = session or AudiobookPlan()
    if not character_name:
        gr.Warning("Choose a character in the dropdown first.")
        return None, "No character selected.", plan
    pick = str(character_name).strip()
    canonical = next(
        (c.name for c in plan.characters if c.name.lower() == pick.lower()),
        pick,
    )
    try:
        regenerate_single_voice_sample(
            plan,
            pick,
            sample_steps=int(sample_steps),
            sample_line=sample_line,
            editor_text=characters_editor,
        )
    except ValueError as exc:
        gr.Warning(str(exc))
        return (
            plan.voice_samples.get(canonical),
            str(exc),
            plan,
        )
    except Exception as exc:
        gr.Error(f"Regenerate failed: {exc}")
        return (
            plan.voice_samples.get(canonical),
            str(exc),
            plan,
        )
    audio = plan.voice_samples.get(canonical)
    return (
        audio,
        f"Regenerated sample for “{canonical}”.",
        plan,
    )


def _apply_external_voice_sample(
    character_name: str,
    ref_audio: tuple[int, np.ndarray] | None,
    ref_transcript: str,
    session: AudiobookPlan | None,
) -> tuple[Any, str, AudiobookPlan]:
    _log_ui_event(
        "_apply_external_voice_sample / Use uploaded clip for selected character",
        character_name=character_name,
        ref_audio=ref_audio,
        ref_transcript=ref_transcript,
        session=session,
    )
    plan = session or AudiobookPlan()
    if not character_name:
        gr.Warning("Choose a character in the dropdown first.")
        return None, "No character selected.", plan
    pick = str(character_name).strip()
    canonical = next(
        (c.name for c in plan.characters if c.name.lower() == pick.lower()),
        pick,
    )
    try:
        apply_external_reference_voice_to_character(
            plan,
            pick,
            ref_audio,
            ref_transcript=ref_transcript,
        )
    except ValueError as exc:
        gr.Warning(str(exc))
        return (
            plan.voice_samples.get(canonical),
            str(exc),
            plan,
        )
    except Exception as exc:
        gr.Error(f"Could not use uploaded clip: {exc}")
        return (
            plan.voice_samples.get(canonical),
            str(exc),
            plan,
        )
    audio = plan.voice_samples.get(canonical)
    return (
        audio,
        f"Applied external reference clip to “{canonical}”.",
        plan,
    )


def _download_all_voice_samples_zip(session: AudiobookPlan | None) -> str | None:
    _log_ui_event("_download_all_voice_samples_zip / Build ZIP of all voice samples", session=session)
    plan = session or AudiobookPlan()
    if not plan.voice_samples:
        gr.Warning("No voice samples yet — run “Assign voice tags & generate samples” first.")
        return None
    try:
        return _zip_voice_samples_to_tempfile(plan)
    except Exception as exc:
        gr.Error(f"Could not build ZIP: {exc}")
        return None


def _resolve_upload_path(upload: Any) -> str | None:
    """Turn a Gradio File/FileData value into a local filesystem path."""
    if upload is None:
        return None
    # Gradio 6+: FileData (and similar) expose server temp path here.
    path = getattr(upload, "path", None)
    if isinstance(path, str) and path.strip():
        return path.strip()
    # Serialized FileData from the browser can arrive as a plain dict.
    if isinstance(upload, dict):
        p = upload.get("path")
        if isinstance(p, str) and p.strip():
            return p.strip()
    # Older tempfile wrapper / pathlib / str path
    name = getattr(upload, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    if isinstance(upload, str) and upload.strip():
        return upload.strip()
    return None


def _import_voice_archive_zip(
    upload: Any,
    session: AudiobookPlan | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, AudiobookPlan, gr.update, Any]:
    _log_ui_event(
        "_import_voice_archive_zip / Load voice archive ZIP",
        upload=upload,
        session=session,
        progress=progress,
    )
    path = _resolve_upload_path(upload)
    if not path:
        gr.Warning("Select a .zip (full archive or WAV-only).")
        plan = session or AudiobookPlan()
        names = list(plan.voice_samples.keys())
        pick = names[0] if names else None
        return (
            character_rows_to_editor_json(plan.raw_character_rows),
            "No archive selected.",
            plan,
            gr.update(choices=names, value=pick),
            plan.voice_samples.get(pick) if pick else None,
        )
    try:
        with open(path, "rb") as f:
            blob = f.read()

        def report(frac: float, msg: str) -> None:
            progress(frac, desc=msg)

        plan = import_voice_archive_from_zip_bytes(blob, on_progress=report)
    except Exception as exc:
        gr.Error(f"Voice archive import failed: {exc}")
        plan = session or AudiobookPlan()
        names = list(plan.voice_samples.keys())
        pick = names[0] if names else None
        return (
            character_rows_to_editor_json(plan.raw_character_rows),
            str(exc),
            plan,
            gr.update(choices=names, value=pick),
            plan.voice_samples.get(pick) if pick else None,
        )
    editor = character_rows_to_editor_json(plan.raw_character_rows)
    names = list(plan.voice_samples.keys()) or [c.name for c in plan.characters]
    first = names[0] if names else None
    audio = plan.voice_samples.get(first) if first else None
    return (
        editor,
        plan.diagnostics,
        plan,
        gr.update(choices=names, value=first),
        audio,
    )


def _import_voice_wavs_only(
    upload_list: Any,
    session: AudiobookPlan | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, AudiobookPlan, gr.update, Any]:
    _log_ui_event(
        "_import_voice_wavs_only / Load WAV files only",
        upload_list=upload_list,
        session=session,
        progress=progress,
    )
    raw = upload_list
    if raw is None:
        files: list[Any] = []
    elif isinstance(raw, list):
        files = raw
    else:
        files = [raw]
    paths = [_resolve_upload_path(f) for f in files]
    paths = [p for p in paths if p]
    if not paths:
        gr.Warning("Upload one or more .wav files.")
        plan = session or AudiobookPlan()
        names = list(plan.voice_samples.keys())
        pick = names[0] if names else None
        return (
            character_rows_to_editor_json(plan.raw_character_rows),
            "No WAVs selected.",
            plan,
            gr.update(choices=names, value=pick),
            plan.voice_samples.get(pick) if pick else None,
        )
    try:

        def report(frac: float, msg: str) -> None:
            progress(frac, desc=msg)

        plan = import_voice_wav_files_from_paths(paths, on_progress=report)
    except Exception as exc:
        gr.Error(f"WAV import failed: {exc}")
        plan = session or AudiobookPlan()
        names = list(plan.voice_samples.keys())
        pick = names[0] if names else None
        return (
            character_rows_to_editor_json(plan.raw_character_rows),
            str(exc),
            plan,
            gr.update(choices=names, value=pick),
            plan.voice_samples.get(pick) if pick else None,
        )
    editor = character_rows_to_editor_json(plan.raw_character_rows)
    names = list(plan.voice_samples.keys())
    first = names[0] if names else None
    audio = plan.voice_samples.get(first) if first else None
    return (
        editor,
        plan.diagnostics,
        plan,
        gr.update(choices=names, value=first),
        audio,
    )


def _synthesize_for_ui(text: str, instruct: str):
    _log_ui_event("_synthesize_for_ui / Generate speech (Phrase tab)", text=text, instruct=instruct)
    result = synthesize_to_numpy(text, instruct)
    if result is None:
        gr.Warning("Enter some text to synthesize.")
        return None
    rate, audio = result
    return rate, audio


def _segments_preview(segments) -> str:
    rows = [
        {
            "speaker": s.speaker,
            "text": s.text,
            "speed": s.speed,
            "extra_instruct": s.extra_instruct,
        }
        for s in segments
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _extract_characters_step(
    title: str,
    author: str,
    session: AudiobookPlan | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, AudiobookPlan, gr.update, Any]:
    """Web search + Qwen → fill the editable character list (user can edit afterward)."""
    _log_ui_event(
        "_extract_characters_step / Extract characters (web + Qwen)",
        title=title,
        author=author,
        session=session,
        progress=progress,
    )
    plan = session or AudiobookPlan()
    plan.voice_samples = {}
    plan.clone_prompts = {}
    plan.character_research_blob = ""
    plan.characters = []
    plan.raw_character_rows = []

    def report(frac: float, msg: str) -> None:
        progress(frac, desc=msg)

    try:
        data, blob = research_characters(title, author, on_progress=report)
        plan.character_research_blob = blob
    except Exception as exc:
        gr.Error(f"Character extraction failed: {exc}")
        return "", str(exc), plan, gr.update(choices=[], value=None), None
    chars = data.get("characters")
    if isinstance(chars, list):
        plan.raw_character_rows = [c for c in chars if isinstance(c, dict)]
    else:
        plan.raw_character_rows = []
    editor = character_rows_to_editor_json(plan.raw_character_rows)
    plan.diagnostics = f"Extracted {len(plan.raw_character_rows)} characters. Review/edit JSON, then run voice setup."
    return (
        editor,
        plan.diagnostics,
        plan,
        gr.update(choices=[], value=None),
        None,
    )


def _build_external_character_prompt(title: str, author: str) -> str:
    """Format system + user block for an external chat model (no local web search)."""
    _log_ui_event(
        "_build_external_character_prompt / Build research prompt for external LLM",
        title=title,
        author=author,
    )
    try:
        _sys, _user, copy_paste = format_character_research_messages_for_external_llm(
            title,
            author,
        )
    except Exception as exc:
        gr.Error(f"Could not build external prompt: {exc}")
        return ""
    return copy_paste


def _assign_voices_step(
    editor_text: str,
    sample_steps: float,
    sample_line: str,
    session: AudiobookPlan | None,
) -> tuple[str, AudiobookPlan, gr.update, Any]:
    """Parse editor JSON → Qwen voice tags → OmniVoice sample + clone prompt per role."""
    _log_ui_event(
        "_assign_voices_step / Assign voice tags & generate samples",
        editor_text=editor_text,
        sample_steps=sample_steps,
        sample_line=sample_line,
        session=session,
    )
    plan = session or AudiobookPlan()
    try:
        rows = parse_character_json(editor_text)
    except ValueError as exc:
        gr.Error(str(exc))
        return str(exc), plan, gr.update(choices=[], value=None), None
    if not rows:
        gr.Warning("Add at least one character with a non-empty \"name\" in the JSON editor.")
        return "No characters in editor.", plan, gr.update(choices=[], value=None), None

    steps = max(4, int(sample_steps))
    line = (sample_line or "").strip()
    try:
        plan.raw_character_rows = rows
        cards = assign_omnivoice_profiles(rows)
        plan.characters = cards
        prompts, samples = build_clone_prompts_for_cast(
            cards,
            sample_steps=steps,
            sample_line=line or None,
        )
        plan.clone_prompts = prompts
        plan.voice_samples = samples
    except Exception as exc:
        gr.Error(f"Voice setup failed: {exc}")
        return str(exc), plan, gr.update(choices=[], value=None), None

    names = list(samples.keys())
    first = names[0] if names else None
    audio = samples[first] if first else None
    plan.diagnostics = (
        f"Voice tags + samples for {len(names)} roles. Pick a name below to listen."
    )
    return (
        plan.diagnostics,
        plan,
        gr.update(choices=names, value=first),
        audio,
    )


def _voice_preview_select(character_name: str, session: AudiobookPlan | None) -> Any:
    _log_ui_event(
        "_voice_preview_select / Listen · character dropdown",
        level=logging.DEBUG,
        character_name=character_name,
        session=session,
    )
    plan = session or AudiobookPlan()
    if not character_name:
        return None
    return plan.voice_samples.get(character_name)


def _prepare_chapter(
    book: str,
    chapter_index: float,
    session: AudiobookPlan | None,
) -> tuple[str, str, AudiobookPlan]:
    _log_ui_event(
        "_prepare_chapter / Prepare chapter for TTS (Qwen)",
        book=book,
        chapter_index=chapter_index,
        session=session,
    )
    plan = session or AudiobookPlan()
    if not plan.characters or not plan.clone_prompts:
        gr.Warning('Run "Assign voice tags & generate samples" first.')
        return "", "Need cast + voice prompts.", plan
    chapters = split_into_chapters(book)
    if not chapters:
        return "", "Book text is empty.", plan
    idx = max(0, min(int(chapter_index), len(chapters) - 1))
    heading, body = chapters[idx]
    names = [c.name for c in plan.characters]
    try:
        segs = prepare_chapter_segments(body, names)
    except Exception as exc:
        gr.Error(f"Chapter prep failed: {exc}")
        return "", str(exc), plan
    plan.last_segments = segs
    plan.diagnostics = f"Chapter {idx}: {heading!r} → {len(segs)} segments"
    return _segments_preview(segs), plan.diagnostics, plan


def _synthesize_chapter(
    session: AudiobookPlan | None,
    tts_steps: float,
) -> tuple[Any, str, AudiobookPlan]:
    _log_ui_event(
        "_synthesize_chapter / Synthesize chapter audio",
        session=session,
        tts_steps=tts_steps,
    )
    plan = session or AudiobookPlan()
    if not plan.last_segments:
        gr.Warning("Prepare a chapter first.")
        return None, "No segments.", plan
    if not plan.clone_prompts:
        gr.Warning("Voice prompts missing.")
        return None, "Run voice setup.", plan
    steps = max(4, int(tts_steps))
    try:
        rate, wav = synthesize_segments(
            plan.last_segments,
            plan.clone_prompts,
            num_step=steps,
        )
    except Exception as exc:
        gr.Error(f"Synthesis failed: {exc}")
        return None, str(exc), plan
    plan.diagnostics = f"Synthesized chapter audio ({len(wav) / rate:.1f}s)."
    return (rate, wav), plan.diagnostics, plan


def main() -> None:
    # INFO: button / primary UI actions. DEBUG: also character dropdown changes.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _dm, dtype, label = pick_device_and_dtype()
    qwen_id = default_llm_model_id()
    with gr.Blocks(title="Book2Audio · OmniVoice") as demo:
        gr.Markdown(
            "# Book2Audio\n"
            "Quick **phrase TTS** or a guided **audiobook** flow: build a cast with an **external LLM** (recommended) or "
            f"local **Qwen** (`{qwen_id}`), then **OmniVoice** for speech.\n\n"
            f"**OmniVoice device:** `{label}` · **dtype:** `{dtype}`\n\n"
            "Set `EXA_API_KEY` if you use **local** character extraction (web search); otherwise DuckDuckGo is used. "
            "`BOOK2AUDIO_LLM_DEVICE=cpu` keeps the LLM on CPU if VRAM is tight."
        )
        with gr.Tabs():
            with gr.Tab("Phrase (quick)"):
                gr.Markdown(
                    "Uses the **omnivoice** package from PyPI ([OmniVoice](https://github.com/k2-fsa/OmniVoice)). "
                    "First run downloads weights from Hugging Face."
                )
                text = gr.Textbox(
                    label="Phrase or passage",
                    lines=5,
                    placeholder="Type what you want spoken…",
                )
                instruct = gr.Textbox(
                    label="Voice design (optional)",
                    lines=1,
                    placeholder='e.g. "female, low pitch, british accent"',
                )
                out = gr.Audio(label="Generated audio", type="numpy")
                btn = gr.Button("Generate speech", variant="primary")
                btn.click(_synthesize_for_ui, inputs=[text, instruct], outputs=out)

            with gr.Tab("Audiobook (step by step)"):
                session = gr.State()

                gr.Markdown("### Step 1 · Characters")
                gr.Markdown(
                    "**Recommended:** use **Build research prompt** with title + author, paste the block into ChatGPT / Claude / "
                    "etc., then paste the model’s JSON into **Characters (editable JSON)** below. "
                    "You can also **edit the JSON** directly or use **local extraction** (web search + Qwen) at the bottom.\n\n"
                    "Each object needs **`name`**; add **`role`**, **`summary`**, and voice fields **`voice_gender`** (**male** or **female**), "
                    "**`voice_age`**, **`voice_characteristics`** (how they sound), and **`voice_accent`** so Step 2 can cast them."
                )
                ab_title = gr.Textbox(
                    label="Book title",
                    placeholder="e.g. Pride and Prejudice",
                )
                ab_author = gr.Textbox(
                    label="Author (optional)",
                    placeholder="e.g. Jane Austen",
                )
                gr.Markdown("#### External LLM (main path)")
                gr.Markdown(
                    "Builds **one** copy-paste prompt from **title and author only** (no web search in this app). "
                    "Paste it into ChatGPT, Claude, etc.; put the reply in **Characters** below."
                )
                btn_external_prompt = gr.Button(
                    "Build research prompt for external LLM",
                    variant="primary",
                )
                external_prompt_block = gr.Textbox(
                    label="Copy-paste prompt for your LLM (single message)",
                    lines=22,
                    interactive=False,
                    buttons=["copy"],
                )

                characters_editor = gr.Textbox(
                    label="Characters (editable JSON · paste external LLM reply here)",
                    lines=16,
                    value=EXAMPLE_CHARACTERS_EDITOR_JSON,
                    placeholder='[\n  {"name": "Alice", "role": "…", "summary": "…",\n   "voice_gender": "female", "voice_age": "young adult",\n   "voice_characteristics": "soft, thoughtful", "voice_accent": "British"}\n]',
                )

                log_extract = gr.Textbox(label="Status · Step 1", lines=2, interactive=False)

                with gr.Accordion("Local extraction: web search + Qwen (optional)", open=False):
                    gr.Markdown(
                        "Runs **three web searches** and **one local Qwen call** to fill the editor—same schema as the external path, "
                        "but research snippets are gathered from the web here.\n\n"
                        "**Timing:** On CPU, the **first** Qwen load plus generation often takes **several minutes**. "
                        "Watch the **progress bar** (top-right) and **status** above; wait until this step finishes before clicking again."
                    )
                    btn_extract = gr.Button("Extract characters (web + Qwen)", variant="secondary")

                gr.Markdown("### Step 2 · Voice samples")
                gr.Markdown(
                    "Qwen assigns OmniVoice voice tags from the character list; OmniVoice then synthesizes **one short clip per character** "
                    "and builds voice-clone prompts. Edit the JSON above if needed, then run the button below.\n\n"
                    "**Transcription for reference clips** (WAV import / **Use uploaded clip** without typed text): by default Book2Audio "
                    "uses **local Parakeet ONNX** via `onnx-asr` "
                    "(`nemo-parakeet-tdt-0.6b-v3` from Hugging Face; **first run downloads the model**). "
                    "Install with `pip install \"book2audio[parakeet-stt]\"`. "
                    "Optional: set `BOOK2AUDIO_STT_BACKEND=http` and run a compatible OpenAI-style server "
                    "(e.g. `BOOK2AUDIO_STT_API_URL`), or **paste the spoken words** to skip STT.\n\n"
                    "You can also **upload your own short reference recording** for a character (after voice setup): pick them in the "
                    "dropdown, add optional transcript text, and click **Use uploaded clip**."
                )
                sample_line = gr.Textbox(
                    label="Line spoken in each voice sample (same line for every character if filled)",
                    value="Hello — this is my voice for this story.",
                    lines=1,
                    placeholder="Leave empty to use each character’s name + role/summary as the sample line.",
                )
                sample_steps = gr.Slider(
                    minimum=4,
                    maximum=24,
                    value=10,
                    step=1,
                    label="OmniVoice steps per voice sample",
                )
                btn_voices = gr.Button(
                    "Assign voice tags & generate samples",
                    variant="secondary",
                )
                log_voices = gr.Textbox(label="Status · Step 2", lines=2, interactive=False)

                voice_pick = gr.Dropdown(
                    label="Listen · character",
                    choices=[],
                    value=None,
                )
                voice_preview = gr.Audio(label="Voice sample", type="numpy")
                btn_regen_voice = gr.Button(
                    "Regenerate selected voice sample",
                    variant="secondary",
                )
                gr.Markdown(
                    "**Regenerate** reads the **current Characters JSON**, updates **role** / **summary** / "
                    "**voice_gender**, **voice_age**, **voice_characteristics**, **voice_accent** from the editor, then "
                    "rebuilds **voice_instruct** for the selected role with the same deterministic rules as the app "
                    "(no extra Qwen call). Run **Assign voice tags & generate samples** again if you want the LLM "
                    "to retag the whole cast."
                )
                external_ref_audio = gr.Audio(
                    label="External reference clip (optional · for selected character)",
                    type="numpy",
                )
                external_ref_transcript = gr.Textbox(
                    label="Words spoken in that clip (recommended; else local ONNX Parakeet / optional HTTP STT)",
                    lines=2,
                    placeholder="Exact transcript, or leave empty to transcribe with pip install \"book2audio[parakeet-stt]\" (HF download on first run) or BOOK2AUDIO_STT_BACKEND=http.",
                )
                btn_apply_external_voice = gr.Button(
                    "Use uploaded clip for selected character",
                    variant="secondary",
                )
                samples_zip = gr.File(
                    label="Download all voice samples (.zip)",
                    interactive=False,
                )
                btn_download_samples = gr.Button(
                    "Build ZIP of all voice samples",
                    variant="secondary",
                )
                gr.Markdown(
                    "The ZIP includes **`book2audio_session.json`** (character rows/cards, optional web-research blob, "
                    "and voice filenames) plus one **`.wav`** per character — suitable for backup or moving to another machine."
                )
                import_voice_zip = gr.File(
                    label="Import voice archive (.zip)",
                    file_types=[".zip"],
                )
                btn_import_voice_zip = gr.Button(
                    "Load voice archive ZIP",
                    variant="secondary",
                )
                import_voice_many = gr.File(
                    label="Import WAVs only (multiple · character name = filename without .wav)",
                    file_count="multiple",
                    file_types=[".wav"],
                )
                btn_import_voice_wavs = gr.Button(
                    "Load WAV files only",
                    variant="secondary",
                )

                with gr.Accordion("Later · chapter pipeline", open=False):
                    gr.Markdown(
                        "Uses the **same session** after voice setup. If you change the character JSON, run Step 2 again."
                    )
                    book_full = gr.Textbox(
                        label="Full book text (for chapter split)",
                        lines=10,
                        placeholder="Paste full book text; use Chapter headings for splits…",
                    )
                    chapter_index = gr.Slider(
                        minimum=0,
                        maximum=99,
                        value=0,
                        step=1,
                        label="Chapter index",
                    )
                    btn_prep = gr.Button("Prepare chapter for TTS (Qwen)", variant="secondary")
                    prepared = gr.Textbox(label="Segment preview (JSON)", lines=8)
                    log_prep = gr.Textbox(label="Status", lines=1, interactive=False)
                    tts_steps = gr.Slider(
                        minimum=6,
                        maximum=32,
                        value=16,
                        step=1,
                        label="OmniVoice steps per line",
                    )
                    btn_synth = gr.Button("Synthesize chapter audio", variant="primary")
                    chapter_audio = gr.Audio(label="Chapter audio", type="numpy")
                    log_synth = gr.Textbox(label="Status", lines=1, interactive=False)

                btn_extract.click(
                    _extract_characters_step,
                    inputs=[ab_title, ab_author, session],
                    outputs=[
                        characters_editor,
                        log_extract,
                        session,
                        voice_pick,
                        voice_preview,
                    ],
                    show_progress="full",
                    show_progress_on=log_extract,
                )
                btn_external_prompt.click(
                    _build_external_character_prompt,
                    inputs=[ab_title, ab_author],
                    outputs=[external_prompt_block],
                )
                btn_voices.click(
                    _assign_voices_step,
                    inputs=[characters_editor, sample_steps, sample_line, session],
                    outputs=[log_voices, session, voice_pick, voice_preview],
                )
                voice_pick.change(
                    _voice_preview_select,
                    inputs=[voice_pick, session],
                    outputs=voice_preview,
                )
                btn_regen_voice.click(
                    _regenerate_selected_voice_sample,
                    inputs=[voice_pick, sample_steps, sample_line, characters_editor, session],
                    outputs=[voice_preview, log_voices, session],
                    show_progress="full",
                )
                btn_apply_external_voice.click(
                    _apply_external_voice_sample,
                    inputs=[voice_pick, external_ref_audio, external_ref_transcript, session],
                    outputs=[voice_preview, log_voices, session],
                    show_progress="full",
                )
                btn_download_samples.click(
                    _download_all_voice_samples_zip,
                    inputs=[session],
                    outputs=[samples_zip],
                )
                btn_import_voice_zip.click(
                    _import_voice_archive_zip,
                    inputs=[import_voice_zip, session],
                    outputs=[
                        characters_editor,
                        log_voices,
                        session,
                        voice_pick,
                        voice_preview,
                    ],
                    show_progress="full",
                )
                btn_import_voice_wavs.click(
                    _import_voice_wavs_only,
                    inputs=[import_voice_many, session],
                    outputs=[
                        characters_editor,
                        log_voices,
                        session,
                        voice_pick,
                        voice_preview,
                    ],
                    show_progress="full",
                )
                btn_prep.click(
                    _prepare_chapter,
                    inputs=[book_full, chapter_index, session],
                    outputs=[prepared, log_prep, session],
                )
                btn_synth.click(
                    _synthesize_chapter,
                    inputs=[session, tts_steps],
                    outputs=[chapter_audio, log_synth, session],
                )

    demo.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
