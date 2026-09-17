"""FP64 reference math for versioned foreground image objectives."""

from __future__ import annotations

import numpy as np


def composite_v1(
    foreground: np.ndarray,
    transmittance: np.ndarray,
    reference_rgb: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    foreground = np.asarray(foreground)
    transmittance = np.asarray(transmittance)
    reference_rgb = np.asarray(reference_rgb)
    mask = np.asarray(mask)
    background = np.asarray(background)
    if foreground.shape != reference_rgb.shape or foreground.ndim != 3 or foreground.shape[-1] != 3:
        raise ValueError("foreground and reference_rgb must have matching HxWx3 shapes")
    if transmittance.shape not in {foreground.shape[:2], foreground.shape[:2] + (1,)}:
        raise ValueError("transmittance must be HxW or HxWx1")
    if mask.shape not in {foreground.shape[:2], foreground.shape[:2] + (1,)}:
        raise ValueError("mask must be HxW or HxWx1")
    if background.shape != (3,):
        raise ValueError("background must contain RGB")
    t = transmittance[..., None] if transmittance.ndim == 2 else transmittance
    m = mask[..., None] if mask.ndim == 2 else mask
    prediction_unclamped = foreground + background * t
    target = reference_rgb * m + background * (1.0 - m)
    return np.clip(prediction_unclamped, 0.0, 1.0), target


def composite_v1_pixel_vjp(
    upstream: np.ndarray, background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map ``dL/dS`` to raster state ``(F,T)``."""

    h = np.asarray(upstream)
    c = np.asarray(background)
    if h.ndim != 3 or h.shape[-1] != 3 or c.shape != (3,):
        raise ValueError("upstream must be HxWx3 and background must be RGB")
    return h, np.sum(h * c, axis=-1)


def color_correct_composite_v1(
    foreground: np.ndarray,
    transmittance: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    t = np.asarray(transmittance)
    if t.ndim == 2:
        t = t[..., None]
    return (
        np.asarray(foreground) @ np.asarray(matrix)
        + (1.0 - t) * np.asarray(bias)
        + t * np.asarray(background)
    )


def color_correct_composite_vjp(
    foreground: np.ndarray,
    transmittance: np.ndarray,
    upstream: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    background: np.ndarray,
) -> dict[str, np.ndarray]:
    f = np.asarray(foreground)
    t = np.asarray(transmittance)
    h = np.asarray(upstream)
    if t.ndim == 3:
        t = t[..., 0]
    axes = tuple(range(h.ndim - 1))
    return {
        "foreground": h @ np.asarray(matrix).T,
        "transmittance": np.sum(
            h * (np.asarray(background) - np.asarray(bias)), axis=-1,
        ),
        "matrix": f.reshape(-1, 3).T @ h.reshape(-1, 3),
        "bias": np.sum((1.0 - t)[..., None] * h, axis=axes),
    }
