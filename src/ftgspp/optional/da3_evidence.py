"""Depth-evidence helpers for DA3 candidate filtering."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def boundary_distance_map(foreground: np.ndarray) -> np.ndarray:
    """Euclidean distance, in processed-image pixels, to the mask boundary."""

    inside = np.asarray(foreground, bool)
    if inside.size == 0:
        return np.zeros(inside.shape, np.float32)
    distance = ndimage.distance_transform_edt(inside)
    outside = ndimage.distance_transform_edt(~inside)
    signed = np.where(inside, distance, -outside)
    return signed.astype(np.float32, copy=False)


def target_depth_is_trusted(
    *,
    observed_depth: np.ndarray,
    validity: np.ndarray,
    foreground: np.ndarray,
) -> np.ndarray:
    """Return the fixed v1 target-depth validity gate."""

    return (
        validity
        & foreground
        & np.isfinite(observed_depth)
        & (observed_depth > 1e-8)
    )
