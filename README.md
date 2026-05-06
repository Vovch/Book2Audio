# Book2Audio

Turn books into audiobooks over time. Today this repo includes a small **Text-to-speech web demo** built on **[OmniVoice](https://github.com/k2-fsa/OmniVoice)** from PyPI (`omnivoice`). The library is a dependency only—no fork of OmniVoice is required.

Application code is the **`book2audio`** Python package, rooted in **`src/`** (`device.py`, `synthesis.py`, `web_ui.py`, `search_web.py`, `llm_qwen.py`, `audiobook_pipeline.py`, plus `__init__.py`). Root **`tts_web.py`** is only a shim so `python tts_web.py` still works after `pip install .`.

## Requirements

- **Python 3.10+**
- A **Hugging Face**–reachable network on first run (model weights download for `k2-fsa/OmniVoice` and the default `Qwen/Qwen3.5-0.8B` planner checkpoint)

For AI assistants and contributors working in this repo, see **[AGENTS.md](AGENTS.md)** for product intent, ethics, and the same setup steps in agent-oriented form.

---

## Quick install (scripts)

From the repository root:

**Windows with an AMD Radeon GPU:** use the **left column** — **`install.bat`** / **`pip install .`**. OmniVoice runs on **CPU** in this setup (stable). You do **not** need **`install-amd.bat`** for Book2Audio.

|  | Default (NVIDIA / Apple / CPU / **AMD via CPU**) | Optional: DirectML venv (Windows only) |
| --- | --- | --- |
| **Windows** | Run **`install.bat`** (Command Prompt or PowerShell) | **`install-amd.bat`** or **`install.bat amd`** |
| **macOS / Linux** | **`chmod +x install.sh`** then **`./install.sh`** | Not used — DirectML is Windows-only; **`./install.sh amd`** exits with an error on these platforms |

The scripts create **`.venv`**, upgrade **pip**, and run **`pip install .`** or **`pip install ".[amd]"`**. If you change between the default and DirectML stacks, remove **`.venv`** first, then run the matching script.

---

## Install (manual)

Create and activate a virtual environment, then upgrade `pip`.

**Windows (PowerShell):**

```powershell
cd path\to\Book2Audio
python -m venv .venv
.\.venv\Scripts\pip install -U pip
```

**macOS / Linux:**

```bash
cd path/to/Book2Audio
python -m venv .venv
./.venv/bin/pip install -U pip
```

### Option A — Default (NVIDIA CUDA, Apple Silicon, CPU, or **Windows + AMD on CPU**)

Installs `omnivoice` and pulls a recent **PyTorch** stack from PyPI.

```bash
pip install .
```

**NVIDIA:** For GPU inference, install a **CUDA-enabled** `torch` and `torchaudio` build that matches your driver using the official wizard: [PyTorch — Get Started](https://pytorch.org/get-started/locally/). Install that build before or after `pip install .`; resolve any version prompts in favor of a consistent `torch` / `torchaudio` pair.

**AMD on Windows:** This is the **supported** Book2Audio path — leave **`BOOK2AUDIO_DEVICE`** unset or use **`auto`** / **`cpu`**. Inference uses **CPU** (OmniVoice is not stable on **DirectML** today).

### Option B — Optional: DirectML stack on Windows (advanced)

**Not** required for Book2Audio on AMD — stay on **Option A** for normal use. This profile exists if you want a `torch-directml` environment for other experiments.

On **native Windows**, consumer AMD cards can use **DirectML** via `torch-directml`, not ROCm. Use a **fresh venv** dedicated to this stack.

```bash
pip install ".[amd]"
```

This extra pins a DirectML-compatible stack: `torch-directml` (currently `torch` 2.4.x), `torchaudio==2.4.1`, and `transformers<5.8` (required because newer `transformers` expects a newer PyTorch than DirectML ships).

**OmniVoice note:** Book2Audio’s **`auto` device never picks DirectML**; **`BOOK2AUDIO_DEVICE=dml`** is unsupported for reliable synthesis (often crashes during generation).

Background: [Enable PyTorch with DirectML on Windows](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows).

**Switching between DirectML and NVIDIA (CUDA):** use **separate virtual environments** so `torch`, `torchaudio`, and `transformers` stay consistent.

---

## Run the TTS web UI

From the project root, with the venv activated:

```bash
book2audio-tts
```

Or:

```bash
python tts_web.py
```

(`tts_web.py` is a thin shim; implementation lives in `src/` as import package `book2audio`.)

The app listens on **http://127.0.0.1:7860** (server binds to `0.0.0.0`). The first synthesis triggers a large model download; be patient.

### Windows PowerShell: “running scripts is disabled”

You do **not** have to run `Activate.ps1`. From the project root, call the venv directly:

```powershell
cd "c:\Users\vladi\Desktop\Book2Audio"
.\.venv\Scripts\book2audio-tts
```

Or:

```powershell
.\.venv\Scripts\python.exe tts_web.py
```

If you prefer activating the venv, allow scripts **for this session only**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

To allow local scripts permanently for your user (common dev setup):  
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`  
([Microsoft docs](https://learn.microsoft.com/powershell/module/microsoft.powershell.security/set-executionpolicy)).

---

## Environment variables (device selection)

| Variable | Purpose |
| -------- | ------- |
| `BOOK2AUDIO_DEVICE` | `auto` (default), `cuda`, `dml`, `mps`, or `cpu` |
| `BOOK2AUDIO_SKIP_DML` | Set to `1` to ignore DirectML even if `torch-directml` is installed |

With `auto`, order is: **CUDA → MPS → CPU**. **DirectML is not used in `auto`** (OmniVoice + torch-directml commonly fails with `RuntimeError: Cannot set version_counter for inference tensor`). On **AMD + Windows**, **`cpu`** (explicit or via `auto`) is the supported path; **`BOOK2AUDIO_DEVICE=dml`** is for experimentation only.

### Audiobook tab (Qwen + web search)

| Variable | Purpose |
| -------- | ------- |
| `BOOK2AUDIO_LLM_MODEL` | Optional: swap the planner checkpoint (default [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)). Non–Qwen-3.5 ids are loaded with `AutoModelForCausalLM`—use only if you know the model matches. |
| `BOOK2AUDIO_LLM_DEVICE` | Force LLM device: `cpu`, `cuda`, `mps` (otherwise follows OmniVoice device selection; DirectML backends default the LLM to **CPU**) |
| `EXA_API_KEY` | Optional; [Exa](https://exa.ai) search API. Without it, DuckDuckGo HTML search is used (no key, lower quality). |

The **Audiobook** Gradio tab runs: web snippets → Qwen character list → Qwen OmniVoice voice tags → a short **OmniVoice sample per role** (builds **voice-clone** prompts) → Qwen chapter segmentation (IPA/emotion hints in text) → **OmniVoice** line-by-line synthesis.

Character research uses **public web results**; for obscure works, results may be thin—use the **editable JSON** to fix or paste the cast yourself.

---

## Hugging Face connectivity

If downloads fail, OmniVoice’s README suggests using an HF mirror, for example:

```bash
# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

---

## Tests

Install dev extras and run **pytest** (integration tests perform **real** TTS using the OmniVoice cache; `tests/conftest.py` defaults to CPU + skips DirectML for stability):

```bash
pip install ".[dev]"
pytest
```

Omit slow integration tests: `pytest -m "not integration"`.

**Where test WAVs go:** integration tests write under pytest’s [`tmp_path`](https://docs.pytest.org/en/stable/how-to/tmp_path.html) (a per-test folder under your OS temp directory). That folder is **deleted when the pytest run ends**, so you will not see `pytest_synthesis_real.wav` in the repo unless you change the tests or run with e.g. `pytest --basetemp=.\_pytest_artifacts` to keep artifacts under the project.

---

## License

Follow the licenses of bundled dependencies (notably **OmniVoice** — Apache-2.0). Add a top-level `LICENSE` for **Book2Audio** when you finalize the project’s distribution terms.
