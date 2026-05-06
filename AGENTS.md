# Book2Audio — Agent and contributor guide

This document orients anyone (human or automated agent) working in this repository. The project’s purpose is to build an application that turns **books into audiobooks**: from source text (and optionally structure) to listenable, high-quality spoken output.

## Vision

**Book2Audio** exists to make long-form reading accessible in audio form without sacrificing clarity or narrative flow. The core promise is a pipeline from book-like inputs (e.g. EPUB, PDF with extractable text, plain text, or pasted chapters) to finished or chaptered audio that users can play offline or stream.

The word “translate” here means **transform narrative text into speech** (text-to-speech and prosody), not necessarily machine translation between human languages—unless the product later adds that as an explicit feature.

## Product intent (high level)

- **Ingest**: Accept common book formats and normalize them into a clean, ordered text model (chapters, sections, front matter).
- **Prepare**: Handle typography artifacts (hyphenation, line breaks), dialogue attribution, and optional pronunciation hints (names, places, foreign terms).
- **Synthesize**: Generate speech with consistent voice(s), sensible pauses, and chapter boundaries suitable for bookmarking and resume.
- **Deliver**: Export audio files (e.g. M4A/MP3), optionally with metadata (title, author, chapter markers), and a simple way to preview or queue jobs.

Exact UX, stack, and deployment are intentionally unspecified here; they will evolve as the repo grows.

## Principles

1. **User responsibility for rights**: The app should assume users supply content they are allowed to process. Documentation and UI copy should remind users to respect copyright, licensing, and regional law. Do not design features whose primary purpose is to circumvent access controls or redistribute protected works without permission.
2. **Quality over raw speed**: Prefer understandable pacing, correct emphasis, and stable voices over the fastest possible first draft—while keeping batch jobs efficient where possible.
3. **Transparency**: Where the system guesses (pronunciation, speaker boundaries), make that observable or editable when feasible.
4. **Accessibility**: Consider listeners who rely on clear audio, adjustable speed, and simple navigation between chapters.

## Technical directions (non-binding)

Future implementation might touch:

- Document parsing and text cleanup
- Queues or jobs for long-running synthesis
- Integration with on-device or cloud TTS APIs
- Caching, resumability, and storage of intermediate artifacts
- Packaging for desktop, web, or mobile—TBD

Agents should follow existing patterns in the repo once code exists; until then, prefer small, reviewable changes and clear boundaries between parsing, business logic, and synthesis.

## Setup and run (local TTS web UI)

End-user-oriented copy also lives in [README.md](README.md); keep both in sync when install steps change.

**Code layout:** Python package **`book2audio`**, files in **`src/`** (`device.py`, `synthesis.py`, `web_ui.py`). Setuptools maps `book2audio` → `src/` so there is no `src/book2audio/` directory. Root [`tts_web.py`](tts_web.py) is a shim so `python tts_web.py` still works; the console script `book2audio-tts` calls `book2audio.web_ui:main`.

### Requirements

- Python **3.10+**
- Network access to **Hugging Face** on first run (weights for `k2-fsa/OmniVoice`)

### Helper scripts (recommended)

Repository root:

**Windows + AMD Radeon:** use **`install.bat`** / `pip install .` — Book2Audio runs OmniVoice on **CPU** (supported). **`install-amd.bat`** is optional (DirectML venv for other experiments).

|  | Default (includes **AMD via CPU** on Windows) | Optional DirectML venv (Windows only) |
| --- | --- | --- |
| Windows | `install.bat` | `install-amd.bat` or `install.bat amd` |
| macOS / Linux | `chmod +x install.sh && ./install.sh` | Do not use the `amd` profile on non-Windows; `install.sh amd` and `install-amd.sh` refuse to run there. |

Scripts create `.venv`, upgrade `pip`, and install `.` or `.[amd]`. When switching stacks, delete `.venv` first.

### Virtual environment (manual)

**Windows (PowerShell):**

```powershell
cd <Book2Audio>
python -m venv .venv
.\.venv\Scripts\pip install -U pip
```

**macOS / Linux:**

```bash
cd <Book2Audio>
python -m venv .venv
./.venv/bin/pip install -U pip
```

Activate the venv when working (Windows: `.\.venv\Scripts\Activate.ps1`; Unix: `source .venv/bin/activate`).

### Install: default (NVIDIA CUDA, Apple Silicon, CPU, or **Windows + AMD on CPU**)

```bash
pip install .
```

For **NVIDIA**, install a **CUDA-enabled** `torch` / `torchaudio` pair from [pytorch.org](https://pytorch.org/get-started/locally/) consistent with the user’s driver. Prefer a **separate venv** from any DirectML stack below.

For **AMD on Windows**, **CPU** inference is the supported path (`auto` or `BOOK2AUDIO_DEVICE=cpu`); do not expect stable OmniVoice on DirectML.

### Install: optional DirectML stack on Windows (advanced)

**Not** needed for Book2Audio on AMD. Use a **clean venv** only if you want `torch-directml` for other work. Consumer AMD on **native Windows** would use **torch-directml**, not ROCm, for DirectML-backed PyTorch.

```bash
pip install ".[amd]"
```

Pins: `torch-directml` (currently `torch` 2.4.x), `torchaudio==2.4.1`, `transformers>=5.3,<5.8`. See [Enable PyTorch with DirectML on Windows](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows).

### Run

```bash
book2audio-tts
# or: python tts_web.py  (shim → book2audio.web_ui)
```

UI: **http://127.0.0.1:7860** (binds `0.0.0.0`). First generation downloads large checkpoints.

### Environment variables

- `BOOK2AUDIO_DEVICE`: `auto` (default), `cuda`, `dml`, `mps`, `cpu`
- `BOOK2AUDIO_SKIP_DML`: `1` to skip DirectML

Device env vars: [src/device.py](src/device.py). UI: [src/web_ui.py](src/web_ui.py). Core TTS: [src/synthesis.py](src/synthesis.py).

### Hugging Face

If Hub downloads fail, users can set `HF_ENDPOINT` (e.g. `https://hf-mirror.com`) per OmniVoice README.

## GPU and inference devices

- **NVIDIA on Windows/Linux**: Default install uses PyTorch CUDA when available; no extra package flag is required (install the appropriate `torch` build from [pytorch.org](https://pytorch.org/get-started/locally/) if needed).
- **Apple Silicon**: Uses MPS when available.
- **AMD on native Windows** (e.g. Radeon): **CPU** is the supported Book2Audio path (`auto` resolves to CPU when CUDA/MPS are absent). **`BOOK2AUDIO_DEVICE=dml`** is unreliable for OmniVoice. GPU acceleration for this model means **NVIDIA/CUDA**, **MPS**, or moving to **Linux ROCm** — not native Windows AMD + DirectML for OmniVoice today. The optional **`[amd]`** extra is for a DirectML PyTorch venv, not a recommended OmniVoice backend.
- **AMD on Linux**: ROCm may be an option for some GPUs; this repo does not pin a ROCm environment—follow PyTorch + ROCm docs for your distro.

### Tests

- `pip install ".[dev]"` then `pytest`. Integration tests (`@pytest.mark.integration`) run real inference; `tests/conftest.py` sets CPU + `BOOK2AUDIO_SKIP_DML` by default.
- Faster local/CI runs: `pytest -m "not integration"`.
- Test WAVs use pytest `tmp_path` (OS temp); removed after the run. To keep artifacts: `pytest --basetemp=.\_pytest_artifacts` (example).

## What to avoid

- Shipping or encouraging use of the tool as a way to pirate books or strip DRM.
- Hard-coding provider keys, license-bypass tricks, or non-portable assumptions without documentation.
- Large refactors unrelated to a stated task (see project contribution norms as they appear in `README` or issue templates, when added).

## How agents should work in this repo

- Read nearby code and match style, naming, and error handling before adding features.
- Prefer incremental delivery: one vertical slice (e.g. “EPUB → single MP3 for one chapter”) before polishing every format edge case.
- When requirements are ambiguous, encode reasonable defaults in code but document assumptions in commit messages or brief in-code comments where it prevents repeated confusion.

## Glossary

- **Book**: A structured long-form work with ordered sections (chapters, optionally subsections).
- **Audiobook output**: Time-aligned audio representating that structure, with metadata that supports navigation.

---

*This file is the canonical high-level description of Book2Audio for the repository. Update it when the product scope or ethical guardrails materially change.*
