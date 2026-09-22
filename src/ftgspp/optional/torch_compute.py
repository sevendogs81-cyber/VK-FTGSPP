"""Lazy Torch device resolution for optional initialization accelerators."""

from __future__ import annotations

from typing import Any


def resolve_torch_device(
    requested: str,
    *,
    purpose: str,
) -> tuple[str, Any | None]:
    """Resolve an optional compute device without making Torch a core import."""

    value = str(requested).lower()
    if value == "cpu":
        return "cpu", None
    if value != "auto" and not value.startswith("cuda"):
        raise ValueError(f"{purpose} device must be auto, cpu, cuda, or cuda:<index>")
    try:
        import torch
    except ImportError as exc:
        if value == "auto":
            return "cpu", None
        raise RuntimeError(
            f"CUDA {purpose} requires Torch; install an initialization extra"
        ) from exc
    if value == "auto":
        return ("cuda", torch) if torch.cuda.is_available() else ("cpu", None)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA {purpose} was requested but is unavailable")
    try:
        torch.empty(0, device=value)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"CUDA {purpose} device {value!r} is unavailable") from exc
    return value, torch


def available_torch_device(requested: str) -> tuple[str, Any | None]:
    """Resolve a model-selected device, falling back when CUDA is only nominal.

    Test doubles and CPU-only installations can report a CUDA-shaped model
    device. Postprocessing should remain usable there instead of trying to
    allocate on a device that does not exist.
    """

    value = str(requested).lower()
    if not value.startswith("cuda"):
        return "cpu", None
    try:
        return resolve_torch_device(value, purpose="initialization postprocessing")
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return "cpu", None
