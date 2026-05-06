"""Inference device selection for OmniVoice.

Environment variables:

- ``BOOK2AUDIO_DEVICE``: ``auto`` (default), ``cuda``, ``dml``, ``mps``, ``cpu``.
- ``BOOK2AUDIO_SKIP_DML``: set to ``1`` to ignore DirectML even if ``torch-directml`` is installed.

``auto`` does **not** pick DirectML: PyTorch **torch-directml** + OmniVoice currently raises
``RuntimeError: Cannot set version_counter for inference tensor`` during generation on many
setups (AMD ``privateuseone``). Use **CPU** (default on AMD when CUDA is absent), **CUDA**, or
**MPS**. You may still set ``BOOK2AUDIO_DEVICE=dml`` to experiment.

AMD Radeon on native Windows often uses DirectML via **torch-directml** (``pip install ".[amd]"``);
Book2Audio avoids selecting it automatically for OmniVoice stability.
See https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _try_torch_directml() -> Any | None:
    if os.environ.get("BOOK2AUDIO_SKIP_DML", "").strip().lower() in ("1", "true", "yes"):
        return None
    try:
        import torch_directml

        return torch_directml
    except ImportError:
        return None


def pick_device_and_dtype() -> tuple[Any, torch.dtype, str]:
    """Return ``(device_map, dtype, label)`` for :meth:`omnivoice.OmniVoice.from_pretrained`."""
    mode = (os.environ.get("BOOK2AUDIO_DEVICE") or "auto").strip().lower()
    tdml = _try_torch_directml()

    def cuda() -> tuple[str, torch.dtype, str]:
        return "cuda:0", torch.float16, "cuda:0"

    def mps() -> tuple[str, torch.dtype, str]:
        return "mps", torch.float16, "mps"

    def cpu() -> tuple[str, torch.dtype, str]:
        return "cpu", torch.float32, "cpu"

    def dml() -> tuple[torch.device, torch.dtype, str]:
        assert tdml is not None
        dev = tdml.device()
        return dev, torch.float32, str(dev)

    if mode == "cpu":
        return cpu()
    if mode == "cuda":
        if torch.cuda.is_available():
            return cuda()
        logger.warning("BOOK2AUDIO_DEVICE=cuda but CUDA is not available; falling back.")
    if mode == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return mps()
        logger.warning("BOOK2AUDIO_DEVICE=mps but MPS is not available; falling back.")
    if mode == "dml":
        if tdml is not None:
            logger.warning(
                "BOOK2AUDIO_DEVICE=dml: OmniVoice often fails on DirectML "
                "(e.g. RuntimeError: Cannot set version_counter for inference tensor). "
                "Prefer cpu or cuda."
            )
            return dml()
        logger.warning("BOOK2AUDIO_DEVICE=dml but torch-directml is not installed; falling back.")

    if mode not in ("auto", ""):
        logger.warning("Unknown BOOK2AUDIO_DEVICE=%r; using auto.", mode)

    if torch.cuda.is_available():
        return cuda()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return mps()
    # Skip DirectML in auto: incompatible with OmniVoice generation today (see module docstring).
    return cpu()
