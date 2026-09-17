"""Contiguous NumPy representation of FTGSPP's explicit 4D Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

TEMPORAL_SCALE_MULTIPLIER = 2.0


def _f32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


@dataclass
class NativeGaussians:
    means: np.ndarray
    log_scales: np.ndarray
    quaternions: np.ndarray
    opacity_logits: np.ndarray
    sh_coefficients: np.ndarray
    times: np.ndarray
    duration_raw: np.ndarray
    velocities: np.ndarray
    gate_raw: np.ndarray
    max_duration: float

    def __post_init__(self) -> None:
        self.means = _f32(self.means)
        self.log_scales = _f32(self.log_scales)
        self.quaternions = _f32(self.quaternions)
        self.opacity_logits = _f32(self.opacity_logits)
        self.sh_coefficients = _f32(self.sh_coefficients)
        self.times = _f32(self.times)
        self.duration_raw = _f32(self.duration_raw)
        self.velocities = _f32(self.velocities)
        self.gate_raw = _f32(self.gate_raw)
        self.validate()

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def validate(self) -> None:
        n = len(self)
        expected = {
            "means": (n, 3), "log_scales": (n, 3), "quaternions": (n, 4),
            "opacity_logits": (n, 1), "sh_coefficients": (n, 16, 3),
            "times": (n, 1), "duration_raw": (n, 1), "velocities": (n, 3),
            "gate_raw": (n, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "means": self.means, "log_scales": self.log_scales,
            "quaternions": self.quaternions, "opacity_logits": self.opacity_logits,
            "sh_coefficients": self.sh_coefficients, "times": self.times,
            "duration_raw": self.duration_raw, "velocities": self.velocities,
            "gate_raw": self.gate_raw,
            "max_duration": np.asarray(self.max_duration, dtype=np.float32),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, np.ndarray]) -> "NativeGaussians":
        return cls(
            means=values["means"], log_scales=values["log_scales"],
            quaternions=values["quaternions"], opacity_logits=values["opacity_logits"],
            sh_coefficients=values["sh_coefficients"], times=values["times"],
            duration_raw=values["duration_raw"], velocities=values["velocities"],
            gate_raw=values["gate_raw"], max_duration=float(np.asarray(values["max_duration"])),
        )

    @classmethod
    def from_native(cls, values: Mapping[str, np.ndarray], max_duration: float) -> "NativeGaussians":
        scale_opacity = np.asarray(values["log_scales_opacity"])
        time_duration = np.asarray(values["times_durations"])
        velocity_gate = np.asarray(values["velocities_gates"])
        return cls(
            means=np.asarray(values["means"]), log_scales=scale_opacity[:, :3],
            quaternions=np.asarray(values["rotations"]), opacity_logits=scale_opacity[:, 3:4],
            sh_coefficients=np.asarray(values["sh_coeffs"]), times=time_duration[:, :1],
            duration_raw=time_duration[:, 1:2], velocities=velocity_gate[:, :3],
            gate_raw=velocity_gate[:, 3:4], max_duration=max_duration,
        )

    def temporal_log_scale(self) -> np.ndarray:
        if np.isinf(self.max_duration):
            return self.duration_raw
        sigmoid = 1.0 / (1.0 + np.exp(-self.duration_raw))
        return np.log(np.maximum(self.max_duration * TEMPORAL_SCALE_MULTIPLIER * sigmoid, np.finfo(np.float32).tiny))
