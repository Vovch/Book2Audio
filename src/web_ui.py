"""Gradio web UI for phrase → speech (depends on :mod:`book2audio.synthesis`)."""

from __future__ import annotations

import logging

import gradio as gr

from book2audio.device import pick_device_and_dtype
from book2audio.synthesis import synthesize_to_numpy


def _synthesize_for_ui(text: str, instruct: str):
    result = synthesize_to_numpy(text, instruct)
    if result is None:
        gr.Warning("Enter some text to synthesize.")
        return None
    rate, audio = result
    return rate, audio


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _dm, dtype, label = pick_device_and_dtype()
    with gr.Blocks(title="Book2Audio · OmniVoice") as demo:
        gr.Markdown(
            "# Text to speech\n"
            "Uses the **omnivoice** package from PyPI ([OmniVoice](https://github.com/k2-fsa/OmniVoice)). "
            "First generation downloads weights from Hugging Face and can take a while.\n\n"
            f"**Device:** `{label}` · **dtype:** `{dtype}`\n\n"
            "AMD systems without NVIDIA: **auto** uses **CPU** for OmniVoice (DirectML is not "
            "selected—it currently crashes during synthesis with torch-directml). "
            "Override with env `BOOK2AUDIO_DEVICE=cpu|cuda|…`."
        )
        text = gr.Textbox(
            label="Phrase or passage",
            lines=5,
            placeholder="Type what you want spoken…",
        )
        instruct = gr.Textbox(
            label="Voice design (optional)",
            lines=1,
            placeholder='e.g. "female, low pitch, british accent" — leave empty for auto voice',
        )
        out = gr.Audio(label="Generated audio", type="numpy")
        btn = gr.Button("Generate speech", variant="primary")
        btn.click(_synthesize_for_ui, inputs=[text, instruct], outputs=out)

    demo.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
