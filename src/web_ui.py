"""Gradio web UI for phrase → speech and audiobook pipeline (Qwen + OmniVoice)."""

from __future__ import annotations

import json
import logging
from typing import Any

import gradio as gr

from book2audio.audiobook_pipeline import (
    AudiobookPlan,
    EXAMPLE_CHARACTERS_EDITOR_JSON,
    assign_omnivoice_profiles,
    build_clone_prompts_for_cast,
    character_rows_to_editor_json,
    format_character_research_messages_for_external_llm,
    parse_character_json,
    prepare_chapter_segments,
    research_characters,
    split_into_chapters,
    synthesize_segments,
)
from book2audio.device import pick_device_and_dtype
from book2audio.llm_qwen import default_llm_model_id
from book2audio.synthesis import synthesize_to_numpy

def _synthesize_for_ui(text: str, instruct: str):
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
    plan = session or AudiobookPlan()
    plan.voice_samples = {}
    plan.clone_prompts = {}
    plan.characters = []

    def report(frac: float, msg: str) -> None:
        progress(frac, desc=msg)

    try:
        data = research_characters(title, author, on_progress=report)
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
    plan = session or AudiobookPlan()
    if not character_name:
        return None
    return plan.voice_samples.get(character_name)


def _prepare_chapter(
    book: str,
    chapter_index: float,
    session: AudiobookPlan | None,
) -> tuple[str, str, AudiobookPlan]:
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
    logging.basicConfig(level=logging.INFO)
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
                    "and builds voice-clone prompts. Edit the JSON above if needed, then run the button below."
                )
                sample_line = gr.Textbox(
                    label="Line spoken in each voice sample",
                    value="Hello — this is my voice for this story.",
                    lines=1,
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
