"""NumPy reference for native per-camera affine color correction.

Matches ``ftgspp.compat.torch_models.colorcorrect.ColorCorrector``:
``c' = c @ (mat + I) + vec`` with L2 regularization on mat/vec.
Forward/backward image transforms and Adam for the 12 floats/camera run as
Vulkan compute shaders. This module is intentionally excluded from production
training and exists for unit tests and optional offline inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NativeColorCorrectors:
    params: np.ndarray  # (N, 12) float32: mat 3x3 row-major + vec 3
    m: np.ndarray
    v: np.ndarray
    lr: float = 1e-3
    reg: float = 1e-3

    @classmethod
    def create(cls, num_cameras: int, lr: float = 1e-3, reg: float = 1e-3) -> "NativeColorCorrectors":
        n = int(num_cameras)
        zeros = np.zeros((n, 12), dtype=np.float32)
        return cls(params=zeros.copy(), m=zeros.copy(), v=zeros.copy(), lr=lr, reg=reg)

    def matrix_bias(self, camera_index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.params[int(camera_index)]
        mat = row[:9].reshape(3, 3) + np.eye(3, dtype=np.float32)
        vec = row[9:12]
        return mat, vec

    def apply_numpy(self, rgb: np.ndarray, camera_index: int) -> np.ndarray:
        """Host reference used by unit tests and LPIPS pre-correct."""
        mat, vec = self.matrix_bias(camera_index)
        flat = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
        # Torch ColorCorrector: rgb @ (mat + I) + vec
        out = flat @ mat + vec
        return out.reshape(rgb.shape)

    def adam_step(self, grad: np.ndarray, step: int) -> None:
        """Adam update matching ``reg * sum(params.square())`` in FTGSPP-main."""
        g = np.asarray(grad, dtype=np.float32) + 2.0 * self.reg * self.params
        beta1, beta2, eps = 0.9, 0.999, 1e-15
        self.m = beta1 * self.m + (1.0 - beta1) * g
        self.v = beta2 * self.v + (1.0 - beta2) * (g * g)
        t = max(int(step), 1)
        mhat = self.m / (1.0 - beta1**t)
        vhat = self.v / (1.0 - beta2**t)
        self.params = self.params - self.lr * mhat / (np.sqrt(vhat) + eps)
