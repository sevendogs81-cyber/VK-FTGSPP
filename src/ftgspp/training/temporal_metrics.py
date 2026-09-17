"""Torch-free diagnostics for FTGSPP temporal support and cutoff sensitivity."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, np.float64)
    result = np.empty_like(x)
    positive = x >= 0.0
    result[positive] = 1.0/(1.0+np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x/(1.0+exp_x)
    return result


def temporal_support_diagnostics(
    model,
    timestamps: Sequence[float],
    cutoffs: Sequence[float],
    *,
    max_time_samples: int = 32,
    chunk_size: int = 262_144,
) -> dict[str, object]:
    """Summarize materialized activity without rendering or changing cutoff."""

    time_values = np.asarray(sorted(set(float(value) for value in timestamps)), np.float64)
    if not len(time_values) or not np.isfinite(time_values).all():
        raise ValueError("temporal diagnostics require finite timestamps")
    if len(time_values) > max_time_samples:
        indices = np.linspace(
            0, len(time_values)-1, max_time_samples, dtype=np.int64,
        )
        time_values = time_values[np.unique(indices)]
    thresholds = np.asarray(list(cutoffs), np.float64)
    if (
        thresholds.ndim != 1 or not len(thresholds)
        or not np.isfinite(thresholds).all()
        or np.any(thresholds < 0.0) or np.any(thresholds >= 1.0)
    ):
        raise ValueError("temporal cutoffs must be finite and in [0, 1)")
    centers = np.asarray(model.times, np.float64).reshape(-1)
    duration_raw = np.asarray(model.duration_raw, np.float64).reshape(-1)
    gates = _sigmoid(20.0*np.asarray(model.gate_raw, np.float64).reshape(-1))
    base_opacity = _sigmoid(
        np.asarray(model.opacity_logits, np.float64).reshape(-1)
    )
    if not (
        len(centers) == len(duration_raw) == len(gates) == len(base_opacity)
    ) or not len(centers):
        raise ValueError("model temporal arrays are empty or inconsistent")
    max_duration = float(model.max_duration)
    durations = (
        2.0*max_duration*_sigmoid(duration_raw)
        if np.isfinite(max_duration) else np.exp(duration_raw)
    )
    durations = np.maximum(durations, 1e-8)
    active = np.zeros((len(time_values), len(thresholds)), np.int64)
    opacity_sum = np.zeros(len(time_values), np.float64)
    phi_sum = np.zeros(len(time_values), np.float64)
    count = len(centers)
    for start in range(0, count, max(1, int(chunk_size))):
        stop = min(count, start+max(1, int(chunk_size)))
        center = centers[start:stop]
        duration = durations[start:stop]
        gate = gates[start:stop]
        opacity = base_opacity[start:stop]
        for time_index, timestamp in enumerate(time_values):
            z = (timestamp-center)/duration
            marginal = np.exp(-0.5*z*z)
            phi = gate+(1.0-gate)*marginal
            active[time_index] += np.count_nonzero(
                phi[:, None] > thresholds[None, :], axis=0,
            )
            phi_sum[time_index] += float(np.sum(phi, dtype=np.float64))
            opacity_sum[time_index] += float(
                np.sum(opacity*phi, dtype=np.float64)
            )
    per_time = [{
        "timestamp": float(timestamp),
        "mean_temporal_opacity": float(phi_sum[index]/count),
        "materialized_opacity_sum": float(opacity_sum[index]),
        "active_fraction": {
            f"{threshold:.9g}": float(active[index, cutoff_index]/count)
            for cutoff_index, threshold in enumerate(thresholds)
        },
    } for index, timestamp in enumerate(time_values)]
    sensitivity = {
        f"{threshold:.9g}": {
            "mean_active_fraction": float(np.mean(active[:, index]/count)),
            "minimum_active_fraction": float(np.min(active[:, index]/count)),
            "maximum_active_fraction": float(np.max(active[:, index]/count)),
        }
        for index, threshold in enumerate(thresholds)
    }
    return {
        "schema": "ftgspp-temporal-support-v1",
        "gaussians": count,
        "sampled_timestamps": len(time_values),
        "cutoff_sensitivity": sensitivity,
        "per_time_bin": per_time,
        "duration_seconds": {
            "p01": float(np.percentile(durations, 1)),
            "median": float(np.median(durations)),
            "p99": float(np.percentile(durations, 99)),
        },
        "gate": {
            "p01": float(np.percentile(gates, 1)),
            "median": float(np.median(gates)),
            "p99": float(np.percentile(gates, 99)),
        },
    }
