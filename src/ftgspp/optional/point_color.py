"""Visibility-aware multi-view image-based coloring for prepared points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ftgspp.data.scene import View


@dataclass(frozen=True)
class PointColorResult:
    rgb: np.ndarray
    keep: np.ndarray
    source_count: np.ndarray
    disagreement: np.ndarray


IBR_COLOR_VERSION = "ftgspp-ibr-color-v2"


def _nearest_legal_pixel(
    legal: np.ndarray, x: int, y: int, radius: int,
) -> tuple[int, int] | None:
    height, width = legal.shape
    for distance in range(max(0, int(radius)) + 1):
        candidates: list[tuple[int, int, int]] = []
        for dy in range(-distance, distance + 1):
            for dx in range(-distance, distance + 1):
                if max(abs(dx), abs(dy)) != distance:
                    continue
                xx, yy = x + dx, y + dy
                if 0 <= xx < width and 0 <= yy < height and legal[yy, xx]:
                    candidates.append((dx * dx + dy * dy, yy, xx))
        if candidates:
            _distance, yy, xx = min(candidates)
            return xx, yy
    return None


def project_points(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(points, np.float64)
    homogeneous = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    camera = homogeneous @ np.asarray(view.w2c, np.float64).T
    z = camera[:, 2]
    k = np.asarray(view.intrinsic, np.float64)
    u = k[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + k[0, 2]
    v = k[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + k[1, 2]
    return u, v, z


def color_points_ibr(
    points: np.ndarray,
    views: Sequence[View],
    *,
    mask_threshold: float = 0.8,
    visibility_relative_tolerance: float = 0.01,
    min_sources: int = 1,
    mode: str = "visibility_robust",
    depth_rank_views: int = 3,
    z_buffer_radius: int = 1,
    source_search_radius: int = 2,
    normals: np.ndarray | None = None,
    angle_weight_power: float = 0.0,
) -> PointColorResult:
    """Color points from strict foreground pixels with a per-view z-buffer."""

    xyz = np.asarray(points, np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points must be a finite Nx3 array")
    if not views:
        raise ValueError("IBR coloring requires at least one view")
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("IBR mask_threshold must be in [0, 1]")
    if not np.isfinite(visibility_relative_tolerance) or visibility_relative_tolerance < 0.0:
        raise ValueError("IBR visibility_relative_tolerance must be finite and nonnegative")
    if int(min_sources) <= 0 or int(min_sources) > len(views):
        raise ValueError("IBR min_sources must be between one and the view count")
    if mode not in {"visibility_robust", "depth_rank_top3"}:
        raise ValueError("IBR mode must be visibility_robust or depth_rank_top3")
    if int(depth_rank_views) <= 0:
        raise ValueError("IBR depth_rank_views must be positive")
    if int(z_buffer_radius) < 0 or int(source_search_radius) < 0:
        raise ValueError("IBR pixel radii must be nonnegative")
    if not np.isfinite(angle_weight_power) or float(angle_weight_power) < 0.0:
        raise ValueError("IBR angle_weight_power must be finite and nonnegative")
    normal_values: np.ndarray | None = None
    if normals is not None:
        normal_values = np.asarray(normals, np.float64)
        if normal_values.shape != xyz.shape or not np.isfinite(normal_values).all():
            raise ValueError("IBR normals must be a finite Nx3 array")
        lengths = np.linalg.norm(normal_values, axis=1, keepdims=True)
        normal_values = normal_values / np.maximum(lengths, 1e-12)
    samples: list[list[tuple[float, np.ndarray, float]]] = [
        [] for _ in range(len(xyz))
    ]
    for view in views:
        image = view.load_rgb()[..., :3]
        mask = view.load_mask()
        validity = view.load_validity() >= 0.5
        height, width = view.size
        u, v, z = project_points(xyz, view)
        x = np.rint(u - 0.5).astype(np.int64)
        y = np.rint(v - 0.5).astype(np.int64)
        inside = (
            (z > 1e-8) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        )
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        legal = validity & (mask >= mask_threshold)
        selected_x = x[ids].copy()
        selected_y = y[ids].copy()
        strict = legal[selected_y, selected_x]
        if int(source_search_radius) > 0 and not np.all(strict):
            for local_index in np.flatnonzero(~strict):
                found = _nearest_legal_pixel(
                    legal, int(selected_x[local_index]), int(selected_y[local_index]),
                    int(source_search_radius),
                )
                if found is not None:
                    selected_x[local_index], selected_y[local_index] = found
                    strict[local_index] = True
        ids = ids[strict]
        selected_x = selected_x[strict]
        selected_y = selected_y[strict]
        if not len(ids):
            continue
        depth_buffer = np.full((height, width), np.inf, np.float64)
        # Build coverage at the original projected pixels. A minimum filter
        # then makes sparse point samples conservative over neighboring pixels.
        np.minimum.at(depth_buffer, (y[ids], x[ids]), z[ids])
        if int(z_buffer_radius) > 0:
            from scipy import ndimage
            depth_buffer = ndimage.minimum_filter(
                depth_buffer, size=2 * int(z_buffer_radius) + 1,
                mode="constant", cval=np.inf,
            )
        camera_center = np.linalg.inv(view.w2c)[:3, 3]
        for point_id, sample_x, sample_y in zip(
            ids.tolist(), selected_x.tolist(), selected_y.tolist(), strict=True,
        ):
            if mode == "visibility_robust":
                nearest = depth_buffer[y[point_id], x[point_id]]
                tolerance = max(1e-5, visibility_relative_tolerance * nearest)
                if z[point_id] > nearest + tolerance:
                    continue
            to_camera = camera_center - xyz[point_id]
            ray_norm = float(np.linalg.norm(to_camera))
            angle_weight = 1.0
            if normal_values is not None and float(angle_weight_power) > 0.0:
                view_direction = to_camera / max(ray_norm, 1e-12)
                cosine = max(0.0, float(np.dot(normal_values[point_id], view_direction)))
                if cosine <= 0.0:
                    continue
                angle_weight = cosine ** float(angle_weight_power)
            weight = (
                1.0
                if mode == "depth_rank_top3"
                else float(mask[sample_y, sample_x]) * angle_weight
                / max(ray_norm * ray_norm, 1e-6)
            )
            samples[point_id].append((
                float(z[point_id]),
                image[sample_y, sample_x].astype(np.float64),
                weight,
            ))
    rgb = np.zeros((len(xyz), 3), np.uint8)
    count = np.asarray([len(values) for values in samples], np.int32)
    disagreement = np.full(len(xyz), np.nan, np.float32)
    for index, point_samples in enumerate(samples):
        if not point_samples:
            continue
        if mode == "depth_rank_top3":
            point_samples = sorted(point_samples, key=lambda item: item[0])[
                :int(depth_rank_views)
            ]
        values = np.stack([item[1] for item in point_samples])
        sample_weights = np.asarray([item[2] for item in point_samples], np.float64)
        # Robustly reject samples whose RGB residual is far above the median.
        median = np.median(values, axis=0)
        residual = np.linalg.norm(values - median, axis=1)
        cutoff = np.median(residual) + 2.5 * max(np.median(np.abs(residual - np.median(residual))), 1.0)
        accepted = residual <= cutoff
        values = values[accepted]
        sample_weights = sample_weights[accepted]
        rgb[index] = np.clip(
            np.rint(np.average(values, axis=0, weights=sample_weights)), 0, 255,
        ).astype(np.uint8)
        disagreement[index] = float(np.mean(residual[accepted]))
        count[index] = int(np.count_nonzero(accepted))
    return PointColorResult(
        rgb=rgb,
        keep=count >= int(min_sources),
        source_count=count,
        disagreement=disagreement,
    )
