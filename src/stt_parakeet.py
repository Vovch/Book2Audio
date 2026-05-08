"""Reference-clip transcription: local Parakeet ONNX (``onnx-asr``) or optional HTTP STT API.

By default, **auto** uses **`onnx-asr`** when installed so the Parakeet model can
**download from Hugging Face** on first use. Optional HTTP mode posts multipart WAV to an
OpenAI-style ``/v1/audio/transcriptions`` endpoint.

Environment (``BOOK2AUDIO_*`` only):

**Backend**

- ``BOOK2AUDIO_STT_BACKEND`` — ``auto`` (default), ``onnx``, or ``http``.
  ``auto`` uses ONNX when ``onnx_asr`` is importable, otherwise HTTP.

**Local ONNX**

- ``BOOK2AUDIO_ONNX_ASR_MODEL`` — default ``nemo-parakeet-tdt-0.6b-v3``
- ``BOOK2AUDIO_ONNX_ASR_PROVIDERS`` — comma-separated ORT providers; on Windows the default
  is ``DmlExecutionProvider,CPUExecutionProvider`` if unset, else ``CPUExecutionProvider``
- ``BOOK2AUDIO_STT_CPU_ONLY`` — if ``1``, force CPU only for ONNX

**HTTP**

- ``BOOK2AUDIO_STT_API_URL`` (default ``http://127.0.0.1:5092/v1/audio/transcriptions``)
- ``BOOK2AUDIO_STT_TIMEOUT_SECONDS`` (default ``120``)
- ``BOOK2AUDIO_STT_MODEL`` (default ``parakeet``)

Install local STT: ``pip install "book2audio[parakeet-stt]"`` (or ``onnx-asr[hub]`` + ``onnxruntime``).
On Windows GPU you may use ``onnxruntime-directml`` instead of ``onnxruntime``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_API = "http://127.0.0.1:5092/v1/audio/transcriptions"

# DirectML / GPU OOM hints for ORT fallback to CPU.
_GPU_OOM_HINTS = (
    "not enough memory",
    "out of memory",
    "8007000e",
    "e_outofmemory",
    "resource exhausted",
    "failed to allocate",
)

_onnx_lock = threading.Lock()
_onnx_model: object | None = None
_onnx_providers: list[str] | None = None
# After GPU OOM during inference, next load pins CPU.
_onnx_runtime_cpu_override: bool = False


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    if v is not None and str(v).strip():
        return str(v).strip()
    return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _stt_backend() -> Literal["auto", "onnx", "http"]:
    raw = _env_str("BOOK2AUDIO_STT_BACKEND", "auto").lower()
    if raw in ("onnx", "onnx_asr", "local"):
        return "onnx"
    if raw in ("http", "remote"):
        return "http"
    return "auto"


def _cpu_only() -> bool:
    return _env_int("BOOK2AUDIO_STT_CPU_ONLY", 0) == 1


def _onnx_model_id() -> str:
    return _env_str("BOOK2AUDIO_ONNX_ASR_MODEL", "nemo-parakeet-tdt-0.6b-v3")


def _default_onnx_providers_env() -> str:
    if _cpu_only():
        return "CPUExecutionProvider"
    if sys.platform == "win32":
        return "DmlExecutionProvider,CPUExecutionProvider"
    return "CPUExecutionProvider"


def _parse_providers_csv(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _onnx_providers_list() -> list[str]:
    if _onnx_runtime_cpu_override:
        return ["CPUExecutionProvider"]
    raw = (os.environ.get("BOOK2AUDIO_ONNX_ASR_PROVIDERS") or "").strip()
    if raw:
        return _parse_providers_csv(raw)
    return _parse_providers_csv(_default_onnx_providers_env())


def _onnx_importable() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("onnx_asr") is not None
    except Exception:
        return False


def _looks_like_gpu_oom(exc: BaseException) -> bool:
    return any(h in f"{type(exc).__name__}: {exc}".lower() for h in _GPU_OOM_HINTS)


def _filter_providers(requested: list[str]) -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = [p for p in requested if p in available]
    if not preferred and "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return preferred


def _ensure_onnx_model(*, on_status: Any = None) -> Any:
    global _onnx_model, _onnx_providers
    import onnx_asr

    with _onnx_lock:
        if _onnx_model is not None:
            return _onnx_model

        requested = _onnx_providers_list()
        last_error: BaseException | None = None

        for attempt in range(2):
            to_try = requested if attempt == 0 else ["CPUExecutionProvider"]
            providers = _filter_providers(to_try)
            if not providers:
                providers = ["CPUExecutionProvider"]
            if on_status:
                on_status(
                    f"Loading Parakeet ONNX model {_onnx_model_id()!r} "
                    f"(providers={','.join(providers)}) — first run may download from Hugging Face…"
                )
            else:
                logger.info(
                    "Loading ONNX ASR %r with providers %s",
                    _onnx_model_id(),
                    providers,
                )
            try:
                model = onnx_asr.load_model(
                    _onnx_model_id(),
                    providers=providers,
                )
            except Exception as e:
                last_error = e
                if (
                    attempt == 0
                    and _looks_like_gpu_oom(e)
                    and to_try != ["CPUExecutionProvider"]
                ):
                    logger.warning(
                        "ONNX ASR GPU OOM while loading; retrying on CPU. "
                        "Set BOOK2AUDIO_STT_CPU_ONLY=1 to skip GPU providers."
                    )
                    continue
                raise

            _onnx_model = model
            _onnx_providers = providers
            if on_status:
                on_status(f"ONNX ASR ready ({providers[0]}).")
            return model

        assert last_error is not None
        raise last_error


def _transcribe_onnx_wav_bytes(wav_bytes: bytes) -> str:
    global _onnx_model, _onnx_providers, _onnx_runtime_cpu_override
    if not _onnx_importable():
        raise RuntimeError(
            "Local STT requires the onnx-asr package. Install with: "
            'pip install "book2audio[parakeet-stt]"'
        )
    fd, path_str = tempfile.mkstemp(prefix="book2audio-stt-", suffix=".wav")
    try:
        os.close(fd)
        path = Path(path_str)
        path.write_bytes(wav_bytes)
        try:
            model = _ensure_onnx_model()
            text = model.recognize(str(path))
        except Exception as e:
            if not _looks_like_gpu_oom(e):
                raise
            logger.warning(
                "ONNX ASR GPU OOM during inference; reloading on CPU. "
                "Set BOOK2AUDIO_STT_CPU_ONLY=1 to avoid this."
            )
            with _onnx_lock:
                _onnx_model = None
                _onnx_providers = None
                _onnx_runtime_cpu_override = True
            model = _ensure_onnx_model()
            text = model.recognize(str(path))
    finally:
        try:
            Path(path_str).unlink(missing_ok=True)
        except OSError:
            pass
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def _api_url() -> str:
    v = (os.environ.get("BOOK2AUDIO_STT_API_URL") or "").strip()
    return v or _DEFAULT_API


def _timeout_seconds() -> float:
    raw = (os.environ.get("BOOK2AUDIO_STT_TIMEOUT_SECONDS") or "").strip()
    if raw:
        try:
            return max(5.0, float(raw))
        except ValueError:
            pass
    return 120.0


def _http_model_name() -> str:
    v = (os.environ.get("BOOK2AUDIO_STT_MODEL") or "").strip()
    return v or "parakeet"


def coerce_stt_response_to_text(payload: Any) -> str | None:
    """Normalize STT JSON (OpenAI-style) to plain text."""
    if payload is None:
        return None
    if isinstance(payload, str):
        t = payload.strip()
        return t or None
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    segments = payload.get("segments")
    if isinstance(segments, list):
        parts: list[str] = []
        for seg in segments:
            if isinstance(seg, dict):
                seg_text = seg.get("text")
                if isinstance(seg_text, str) and seg_text.strip():
                    parts.append(seg_text.strip())
        if parts:
            return " ".join(parts).strip()
    return None


def mono_float32_pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """16-bit mono WAV in memory (PCM) for STT file input."""
    s = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = np.clip(s * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _transcribe_http_bytes(wav_bytes: bytes, *, wav_filename: str) -> str:
    url = _api_url()
    timeout = _timeout_seconds()
    model = _http_model_name()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                files={"file": (wav_filename, wav_bytes, "audio/wav")},
                data={"model": model},
            )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"STT server not reachable at {url!r} ({exc}). "
            "Nothing is listening there (common on Windows: WinError 10061 = connection refused). "
            'For local transcription without a server, install: pip install "book2audio[parakeet-stt]" '
            "(ONNX Parakeet downloads from Hugging Face on first use) or paste the spoken text as the transcript."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"STT request to {url!r} timed out after {timeout:g}s. "
            "Check that the transcription service is running and not overloaded."
        ) from exc
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"STT HTTP {resp.status_code} at {url!r}: {resp.text[:500]!r}. "
            "Fix the HTTP STT service, switch to local ONNX (see book2audio[parakeet-stt]), "
            "or paste the spoken text as the reference transcript."
        ) from exc
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        t = (resp.text or "").strip()
        if t:
            return t
        raise ValueError("STT service returned non-JSON empty body") from None
    text = coerce_stt_response_to_text(payload)
    if text:
        return text
    logger.warning("STT response had no usable text: %s", payload)
    raise ValueError(
        "STT service returned no transcript text. "
        "Ensure the STT service is running and try again, or paste the clip transcript."
    )


def transcribe_reference_float32(
    mono_float32: np.ndarray,
    sample_rate: int,
    *,
    wav_filename: str = "reference.wav",
) -> str:
    """Transcribe mono float32 reference audio using ONNX Parakeet and/or HTTP (see module doc)."""
    wav_bytes = mono_float32_pcm_to_wav_bytes(mono_float32, sample_rate)
    mode = _stt_backend()

    if mode == "http":
        return _transcribe_http_bytes(wav_bytes, wav_filename=wav_filename)
    if mode == "onnx":
        return _transcribe_onnx_wav_bytes(wav_bytes)

    # auto
    if _onnx_importable():
        try:
            return _transcribe_onnx_wav_bytes(wav_bytes)
        except Exception as e:
            logger.warning("Local ONNX STT failed (%s); falling back to HTTP.", e)
            return _transcribe_http_bytes(wav_bytes, wav_filename=wav_filename)

    return _transcribe_http_bytes(wav_bytes, wav_filename=wav_filename)
