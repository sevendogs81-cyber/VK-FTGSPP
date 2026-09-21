"""Foreground-aware evaluation metrics with explicit empty-region handling."""

from __future__ import annotations

import cv2
import numpy as np

from ftgspp.training.objective import composite_v1


def _safe_mean(values: np.ndarray) -> tuple[float | None, int]:
    count = int(values.size)
    return (float(np.mean(values)) if count else None), count


def _psnr_from_sum(squared_error_sum: float, samples: int) -> float | None:
    if samples <= 0:
        return None
    mse = float(squared_error_sum) / float(samples)
    return float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))


def _binary_morph(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    code = cv2.MORPH_DILATE if operation == "dilate" else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), code, kernel).astype(bool)


def boundary_fscore(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: int,
    validity: np.ndarray | None = None,
) -> float | None:
    """Tolerance-based boundary F-score without validity-edge contamination.

    Boundaries are extracted from the original silhouettes.  Only afterwards
    are boundary samples restricted to an eroded valid support.  Masking the
    silhouettes first would turn every valid/invalid transition into a false
    object boundary.
    """

    prediction = np.asarray(prediction, bool)
    target = np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError("boundary masks must have the same shape")
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = prediction & ~cv2.erode(prediction.astype(np.uint8), kernel).astype(bool)
    target_boundary = target & ~cv2.erode(target.astype(np.uint8), kernel).astype(bool)
    if validity is not None:
        valid = np.asarray(validity, bool)
        if valid.shape != prediction.shape:
            raise ValueError("boundary validity must match the masks")
        # A match may use samples up to ``tolerance`` pixels away.  Eroding by
        # that radius ensures neither side of the comparison is inferred from
        # an ignored neighborhood.
        support = _binary_morph(valid, max(1, int(tolerance)), "erode")
        pred_boundary &= support
        target_boundary &= support
    if not pred_boundary.any() and not target_boundary.any():
        return None
    pred_near = _binary_morph(pred_boundary, tolerance, "dilate")
    target_near = _binary_morph(target_boundary, tolerance, "dilate")
    precision = float(np.mean(target_near[pred_boundary])) if pred_boundary.any() else 0.0
    recall = float(np.mean(pred_near[target_boundary])) if target_boundary.any() else 0.0
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def foreground_reference_roi(
    mask: np.ndarray,
    validity: np.ndarray | None = None,
    *,
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    """Return a deterministic reference-mask bounding ROI as y0,y1,x0,x1.

    The ROI is defined only by the reference foreground and validity, never by
    the prediction. This makes per-checkpoint ROI metrics directly comparable.
    """

    foreground = np.asarray(mask).squeeze() >= 0.5
    valid = (
        np.ones_like(foreground, bool)
        if validity is None else np.asarray(validity).squeeze() >= 0.5
    )
    if foreground.shape != valid.shape:
        raise ValueError("foreground ROI mask and validity shapes differ")
    ys, xs = np.nonzero(foreground & valid)
    if not len(xs):
        return None
    radius = max(0, int(padding))
    height, width = foreground.shape
    return (
        max(0, int(ys.min()) - radius),
        min(height, int(ys.max()) + 1 + radius),
        max(0, int(xs.min()) - radius),
        min(width, int(xs.max()) + 1 + radius),
    )


def foreground_metrics(
    foreground: np.ndarray,
    transmittance: np.ndarray,
    reference_rgb: np.ndarray,
    mask: np.ndarray,
    validity: np.ndarray | None = None,
    *,
    background_colors=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    opacity_threshold: float = 0.5,
) -> dict[str, object]:
    f = np.asarray(foreground, np.float32)
    t = np.asarray(transmittance, np.float32).squeeze()
    rgb = np.asarray(reference_rgb, np.float32)
    m = np.asarray(mask, np.float32).squeeze()
    v = np.ones_like(m, bool) if validity is None else np.asarray(validity).squeeze() >= 0.5
    if f.shape != rgb.shape or f.shape[:2] != t.shape or t.shape != m.shape or m.shape != v.shape:
        raise ValueError("foreground metric inputs have incompatible shapes")
    if not v.any():
        raise ValueError("foreground metrics require at least one valid pixel")
    alpha = np.clip(1.0 - t, 0.0, 1.0)
    hard_target = m >= 0.5
    hard_prediction = alpha >= float(opacity_threshold)
    intersection = int(np.count_nonzero(v & hard_target & hard_prediction))
    union = int(np.count_nonzero(v & (hard_target | hard_prediction)))
    soft_intersection = float(np.sum(alpha[v]*m[v], dtype=np.float64))
    soft_union = float(np.sum(
        alpha[v]+m[v]-alpha[v]*m[v], dtype=np.float64,
    ))
    soft_mass = float(np.sum(alpha[v]+m[v], dtype=np.float64))
    tolerance = max(1, round(2 * max(m.shape) / 1024))
    foreground_roi = foreground_reference_roi(m, v, padding=tolerance)
    foreground_region = v & hard_target
    foreground_error = (np.clip(f, 0.0, 1.0) - rgb * m[..., None]) ** 2
    foreground_count = int(np.count_nonzero(foreground_region))
    foreground_squared_error_sum = (
        float(np.sum(foreground_error[foreground_region], dtype=np.float64))
        if foreground_count else 0.0
    )
    foreground_mse = (
        foreground_squared_error_sum / (3 * foreground_count)
        if foreground_count else None
    )
    dilated = _binary_morph(hard_target, tolerance, "dilate")
    eroded = _binary_morph(hard_target, tolerance, "erode")
    outside = v & ~dilated
    inside = v & eroded
    leakage, leakage_count = _safe_mean(alpha[outside])
    holes, holes_count = _safe_mean((1.0 - alpha)[inside])
    boundary_region = v & (dilated ^ eroded)
    absolute_mask_error = np.abs(alpha - m)
    boundary_mae, boundary_count = _safe_mean(
        absolute_mask_error[boundary_region],
    )
    full_mae, full_count = _safe_mean(absolute_mask_error[v])
    fixed: list[dict[str, object]] = []
    for raw_color in background_colors:
        color = np.asarray(raw_color, np.float32)
        pred, target = composite_v1(f, t, rgb, m, color)
        difference = pred[v].astype(np.float64)
        difference -= target[v]
        np.square(difference, out=difference)
        squared_error_sum = float(np.sum(difference, dtype=np.float64))
        samples = int(np.count_nonzero(v)) * 3
        fixed.append({
            "background": [float(x) for x in color],
            "psnr": _psnr_from_sum(squared_error_sum, samples),
            "valid_pixels": int(np.count_nonzero(v)),
            "squared_error_sum": squared_error_sum,
            "samples": samples,
        })
    return {
        "fixed_backgrounds": fixed,
        "foreground_psnr": (
            None if foreground_mse is None else (
                float("inf") if foreground_mse == 0.0
                else float(-10.0 * np.log10(foreground_mse))
            )
        ),
        "foreground_squared_error_sum": foreground_squared_error_sum,
        "foreground_samples": foreground_count,
        "silhouette_iou": None if union == 0 else intersection / union,
        "silhouette_intersection": intersection,
        "silhouette_union": union,
        "soft_silhouette_iou": (
            None if soft_union <= 0.0 else soft_intersection/soft_union
        ),
        "soft_silhouette_dice": (
            None if soft_mass <= 0.0 else 2.0*soft_intersection/soft_mass
        ),
        "soft_silhouette_intersection": soft_intersection,
        "soft_silhouette_union": soft_union,
        "soft_silhouette_mass": soft_mass,
        "boundary_fscore": boundary_fscore(
            hard_prediction, hard_target, tolerance, validity=v,
        ),
        "boundary_tolerance_pixels": tolerance,
        "foreground_roi": (
            None if foreground_roi is None else {
                "y0": foreground_roi[0], "y1": foreground_roi[1],
                "x0": foreground_roi[2], "x1": foreground_roi[3],
                "pixels": (
                    (foreground_roi[1] - foreground_roi[0])
                    * (foreground_roi[3] - foreground_roi[2])
                ),
                "padding_pixels": tolerance,
                "defined_by": "reference_mask_and_validity",
            }
        ),
        "background_opacity_leakage": leakage,
        "background_opacity_sum": (
            float(np.sum(alpha[outside], dtype=np.float64))
            if leakage_count else 0.0
        ),
        "background_samples": leakage_count,
        "foreground_holes": holes,
        "foreground_holes_sum": (
            float(np.sum((1.0-alpha)[inside], dtype=np.float64))
            if holes_count else 0.0
        ),
        "foreground_interior_samples": holes_count,
        "boundary_mask_mae": boundary_mae,
        "boundary_mask_abs_error_sum": (
            float(np.sum(absolute_mask_error[boundary_region], dtype=np.float64))
            if boundary_count else 0.0
        ),
        "boundary_samples": boundary_count,
        "mask_mae": full_mae,
        "mask_abs_error_sum": float(
            np.sum(absolute_mask_error[v], dtype=np.float64)
        ),
        "valid_samples": full_count,
    }
