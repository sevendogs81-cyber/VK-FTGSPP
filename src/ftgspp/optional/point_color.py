"""Visibility-aware multi-view image-based coloring for prepared points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ftgspp.data.scene import View


@dataclass(frozen=True)
class PointColorResult:
    rgb: np.ndarray
    keep: np.ndarray
    source_count: np.ndarray
    disagreement: np.ndarray


IBR_COLOR_VERSION = "ftgspp-ibr-color-v3"


def _nearest_legal_pixels(
    legal: np.ndarray, x: np.ndarray, y: np.ndarray, radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve nearest legal source pixels for a projected point batch."""

    height, width = legal.shape
    selected_x = np.asarray(x, np.int64).copy()
    selected_y = np.asarray(y, np.int64).copy()
    found = legal[selected_y, selected_x].copy()
    for distance in range(1, max(0, int(radius)) + 1):
        offsets: list[tuple[int, int, int]] = []
        for dy in range(-distance, distance + 1):
            for dx in range(-distance, distance + 1):
                if max(abs(dx), abs(dy)) != distance:
                    continue
                offsets.append((dx * dx + dy * dy, dy, dx))
        # This order matches min((distance_squared, yy, xx)) from the scalar
        # implementation because every candidate starts from the same x/y.
        for _distance_squared, dy, dx in sorted(offsets):
            pending = np.flatnonzero(~found)
            if not len(pending):
                break
            xx = x[pending] + dx
            yy = y[pending] + dy
            inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            legal_pending = np.zeros(len(pending), bool)
            legal_pending[inside] = legal[yy[inside], xx[inside]]
            accepted = pending[legal_pending]
            selected_x[accepted] = x[accepted] + dx
            selected_y[accepted] = y[accepted] + dy
            found[accepted] = True
    return selected_x, selected_y, found


def project_points(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(points, np.float64)
    w2c = np.asarray(view.w2c, np.float64)
    camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
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
    view_count = len(views)
    point_count = len(xyz)
    colors = np.zeros((view_count, point_count, 3), np.uint8)
    weights = np.zeros((view_count, point_count), np.float64)
    depths = np.full((view_count, point_count), np.inf, np.float64)
    available = np.zeros((view_count, point_count), bool)
    for view_index, view in enumerate(views):
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
        selected_x, selected_y, legal_source = _nearest_legal_pixels(
            legal, x[ids], y[ids], int(source_search_radius),
        )
        ids = ids[legal_source]
        selected_x = selected_x[legal_source]
        selected_y = selected_y[legal_source]
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
        if mode == "visibility_robust":
            nearest = depth_buffer[y[ids], x[ids]]
            tolerance = np.maximum(
                1e-5, float(visibility_relative_tolerance) * nearest,
            )
            visible = z[ids] <= nearest + tolerance
            ids = ids[visible]
            selected_x = selected_x[visible]
            selected_y = selected_y[visible]
        if not len(ids):
            continue
        camera_center = np.linalg.inv(view.w2c)[:3, 3]
        to_camera = camera_center - xyz[ids]
        ray_norm = np.linalg.norm(to_camera, axis=1)
        squared_distance = np.maximum(ray_norm * ray_norm, 1e-6)
        angle_weight = np.ones(len(ids), np.float64)
        if normal_values is not None and float(angle_weight_power) > 0.0:
            direction = to_camera / np.maximum(ray_norm, 1e-12)[:, None]
            cosine = np.maximum(
                0.0, np.einsum("ij,ij->i", normal_values[ids], direction),
            )
            front = cosine > 0.0
            ids = ids[front]
            selected_x = selected_x[front]
            selected_y = selected_y[front]
            squared_distance = squared_distance[front]
            angle_weight = cosine[front] ** float(angle_weight_power)
        colors[view_index, ids] = image[selected_y, selected_x, :3]
        depths[view_index, ids] = z[ids]
        weights[view_index, ids] = (
            1.0 if mode == "depth_rank_top3" else
            mask[selected_y, selected_x] * angle_weight / squared_distance
        )
        available[view_index, ids] = True

    rgb = np.zeros((point_count, 3), np.uint8)
    count = np.zeros(point_count, np.int32)
    disagreement = np.full(point_count, np.nan, np.float32)
    aggregation_chunk = 262_144
    for start in range(0, point_count, aggregation_chunk):
        end = min(point_count, start + aggregation_chunk)
        selected = available[:, start:end].copy()
        if mode == "depth_rank_top3":
            order = np.argsort(
                np.where(selected, depths[:, start:end], np.inf),
                axis=0, kind="stable",
            )
            ranked = np.zeros_like(selected)
            columns = np.arange(end - start)
            for rank in range(min(int(depth_rank_views), view_count)):
                row = order[rank]
                valid = selected[row, columns]
                ranked[row[valid], columns[valid]] = True
            selected = ranked
        active = np.flatnonzero(np.any(selected, axis=0))
        if not len(active):
            continue
        chosen = selected[:, active].T
        values = colors[:, start:end][:, active].transpose(1, 0, 2).astype(np.float64)
        sample_weights = weights[:, start:end][:, active].T
        masked_values = np.where(chosen[:, :, None], values, np.nan)
        median = np.nanmedian(masked_values, axis=1)
        residual = np.linalg.norm(values - median[:, None, :], axis=2)
        masked_residual = np.where(chosen, residual, np.nan)
        residual_median = np.nanmedian(masked_residual, axis=1)
        residual_mad = np.nanmedian(
            np.abs(masked_residual - residual_median[:, None]), axis=1,
        )
        cutoff = residual_median + 2.5 * np.maximum(residual_mad, 1.0)
        accepted = chosen & (residual <= cutoff[:, None])
        accepted_weights = np.where(accepted, sample_weights, 0.0)
        weight_sum = accepted_weights.sum(axis=1)
        weighted_rgb = np.einsum("nv,nvc->nc", accepted_weights, values)
        zero_weight = weight_sum <= 0.0
        if np.any(zero_weight):
            accepted_count = np.maximum(accepted.sum(axis=1), 1)
            fallback = np.einsum("nv,nvc->nc", accepted, values) / accepted_count[:, None]
            weighted_rgb[zero_weight] = fallback[zero_weight]
            weight_sum[zero_weight] = 1.0
        output_ids = start + active
        rgb[output_ids] = np.clip(
            np.rint(weighted_rgb / weight_sum[:, None]), 0, 255,
        ).astype(np.uint8)
        accepted_count = accepted.sum(axis=1)
        count[output_ids] = accepted_count
        disagreement[output_ids] = (
            np.where(accepted, residual, 0.0).sum(axis=1)
            / np.maximum(accepted_count, 1)
        ).astype(np.float32)
    return PointColorResult(
        rgb=rgb,
        keep=count >= int(min_sources),
        source_count=count,
        disagreement=disagreement,
    )
