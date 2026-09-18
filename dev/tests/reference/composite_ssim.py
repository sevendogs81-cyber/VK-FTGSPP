"""Independent FP64 reference for the native composite_v1 L1/SSIM graph."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


COEFFICIENTS = np.asarray([
    0.001028380123898387,
    0.0075987582094967365,
    0.036000773310661316,
    0.10936068743467331,
    0.21300552785396576,
    0.26601171493530273,
    0.21300552785396576,
    0.10936068743467331,
    0.036000773310661316,
    0.0075987582094967365,
    0.001028380123898387,
], np.float64)
KERNEL = np.outer(COEFFICIENTS, COEFFICIENTS)


def valid_window_centers(validity: np.ndarray) -> np.ndarray:
    valid = np.asarray(validity, bool)
    centers = cv2.erode(
        valid.astype(np.uint8), np.ones((11, 11), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    centers[:5] = False
    centers[-5:] = False
    centers[:, :5] = False
    centers[:, -5:] = False
    return centers


def _convolve(value: np.ndarray) -> np.ndarray:
    return ndimage.convolve(
        np.asarray(value, np.float64), KERNEL, mode="constant", cval=0.0,
    )


def composite_l1_ssim_reference(
    state: np.ndarray,
    reference_rgba: np.ndarray,
    validity: np.ndarray,
    background: np.ndarray,
    *,
    ssim_lambda: float = 0.2,
) -> tuple[dict[str, float], np.ndarray]:
    """Return scalar terms and dL/d(F,T) using the shader's exact reduction."""

    state64 = np.asarray(state, np.float64)
    reference = np.asarray(reference_rgba, np.uint8)
    valid = np.asarray(validity, bool)
    bg = np.asarray(background, np.float64)
    rgb = reference[..., :3].astype(np.float64) / 255.0
    mask = reference[..., 3].astype(np.float64) / 255.0
    raw_prediction = state64[..., :3] + state64[..., 3, None] * bg
    prediction = np.clip(raw_prediction, 0.0, 1.0)
    target = rgb * mask[..., None] + bg * (1.0 - mask[..., None])
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0:
        raise ValueError("reference requires at least one valid pixel")
    centers = valid_window_centers(valid)
    valid_windows = int(np.count_nonzero(centers))
    l1_only = valid_windows == 0
    l1_mix = 1.0 if l1_only else 1.0 - float(ssim_lambda)
    l1 = l1_mix * float(np.mean(np.abs(prediction[valid] - target[valid])))
    gradient = np.zeros_like(prediction)
    gradient[valid] = (
        l1_mix * np.sign(prediction[valid] - target[valid])
        / (3.0 * valid_pixels)
    )
    ssim_loss = 0.0
    if not l1_only:
        values: list[np.ndarray] = []
        for channel in range(3):
            x = prediction[..., channel]
            y = target[..., channel]
            mu_x = _convolve(x)
            mu_y = _convolve(y)
            second_x = _convolve(x * x)
            second_y = _convolve(y * y)
            cross = _convolve(x * y)
            mu_x_sq = mu_x * mu_x
            mu_y_sq = mu_y * mu_y
            variance_x = second_x - mu_x_sq
            variance_y = second_y - mu_y_sq
            covariance = cross - mu_x * mu_y
            a = mu_x_sq + mu_y_sq + 0.01**2
            b = variance_x + variance_y + 0.03**2
            c = 2.0 * mu_x * mu_y + 0.01**2
            d = 2.0 * covariance + 0.03**2
            ssim = c * d / (a * b)
            values.append(ssim[centers])

            d_mu = (
                (2.0 * mu_y * d) / (a * b)
                - (2.0 * mu_y * c) / (a * b)
                - (2.0 * mu_x * c * d) / (a * a * b)
                + (2.0 * mu_x * c * d) / (a * b * b)
            )
            d_second = -(c * d) / (a * b * b)
            d_cross = 2.0 * c / (a * b)
            d_mu = np.where(centers, d_mu, 0.0)
            d_second = np.where(centers, d_second, 0.0)
            d_cross = np.where(centers, d_cross, 0.0)
            gradient[..., channel] += (
                -float(ssim_lambda) / (3.0 * valid_windows)
            ) * (
                _convolve(d_mu)
                + 2.0 * x * _convolve(d_second)
                + y * _convolve(d_cross)
            )
        mean_ssim = float(np.mean(np.concatenate(values)))
        ssim_loss = float(ssim_lambda) * (1.0 - mean_ssim)

    clamp_derivative = (
        (raw_prediction >= 0.0) & (raw_prediction <= 1.0)
    )
    gradient *= clamp_derivative
    state_gradient = np.empty_like(state64)
    state_gradient[..., :3] = gradient
    state_gradient[..., 3] = np.sum(gradient * bg, axis=-1)
    return {
        "l1": l1,
        "ssim": ssim_loss,
        "image_loss": l1 + ssim_loss,
    }, state_gradient
