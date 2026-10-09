"""Visibility-aware multi-view image-based coloring for prepared points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ftgspp.data.scene import View
from ftgspp.optional.torch_compute import resolve_torch_device
from ftgspp.optional.preparation_views import prefetch_rgb


@dataclass(frozen=True)
class PointColorResult:
    rgb: np.ndarray
    keep: np.ndarray
    source_count: np.ndarray
    disagreement: np.ndarray
    device: str = "cpu"


IBR_COLOR_VERSION = "ftgspp-ibr-color-v5-query-subset"


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


def _torch_masked_median(values, selected, *, torch):
    """Match NumPy's even-sample median while keeping reductions on CUDA."""

    expanded = selected
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    ordered = torch.sort(
        torch.where(expanded, values, torch.full_like(values, float("inf"))),
        dim=1,
    ).values
    count = selected.sum(dim=1).long()
    lower = ((count - 1) // 2).clamp_min(0)
    upper = (count // 2).clamp_min(0)
    if values.ndim == 3:
        channels = values.shape[2]
        lower_value = torch.gather(
            ordered, 1, lower[:, None, None].expand(-1, 1, channels),
        ).squeeze(1)
        upper_value = torch.gather(
            ordered, 1, upper[:, None, None].expand(-1, 1, channels),
        ).squeeze(1)
    else:
        lower_value = torch.gather(ordered, 1, lower[:, None]).squeeze(1)
        upper_value = torch.gather(ordered, 1, upper[:, None]).squeeze(1)
    return (lower_value + upper_value) * 0.5


def _color_points_ibr_cuda(
    xyz: np.ndarray,
    views: Sequence[View],
    *,
    torch,
    device: str,
    mask_threshold: float,
    visibility_relative_tolerance: float,
    min_sources: int,
    mode: str,
    depth_rank_views: int,
    z_buffer_radius: int,
    source_search_radius: int,
    normals: np.ndarray | None,
    angle_weight_power: float,
    query_indices: np.ndarray,
) -> PointColorResult:
    """CUDA implementation of projection, z-buffering and robust IBR."""

    with torch.inference_mode():
        points = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        normal_values = None
        if normals is not None:
            normal_values = torch.as_tensor(
                normals, dtype=torch.float32, device=device,
            )
            normal_values = normal_values / torch.linalg.vector_norm(
                normal_values, dim=1, keepdim=True,
            ).clamp_min(1e-12)
        view_count = len(views)
        point_count = len(query_indices)
        query_map = torch.full((len(points),), -1, dtype=torch.int64, device=device)
        query_map[torch.as_tensor(query_indices, device=device)] = torch.arange(point_count, device=device)
        colors = torch.zeros(
            (view_count, point_count, 3), dtype=torch.uint8, device=device,
        )
        weights = torch.zeros(
            (view_count, point_count), dtype=torch.float32, device=device,
        )
        depths = torch.full(
            (view_count, point_count), float("inf"),
            dtype=torch.float32, device=device,
        )
        available = torch.zeros(
            (view_count, point_count), dtype=torch.bool, device=device,
        )
        for view_index, view, decoded in prefetch_rgb(views):
            image = torch.as_tensor(
                np.ascontiguousarray(decoded[..., :3]), device=device,
            )
            mask = torch.as_tensor(
                np.ascontiguousarray(view.load_mask()),
                dtype=torch.float32, device=device,
            )
            validity = torch.as_tensor(
                np.ascontiguousarray(view.load_validity() >= 0.5), device=device,
            )
            height, width = view.size
            w2c = torch.as_tensor(view.w2c, dtype=torch.float32, device=device)
            intrinsic = torch.as_tensor(
                view.intrinsic, dtype=torch.float32, device=device,
            )
            camera = points @ w2c[:3, :3].T + w2c[:3, 3]
            z = camera[:, 2]
            u = intrinsic[0, 0] * camera[:, 0] / z.clamp_min(1e-12) + intrinsic[0, 2]
            v = intrinsic[1, 1] * camera[:, 1] / z.clamp_min(1e-12) + intrinsic[1, 2]
            x = torch.round(u - 0.5).long()
            y = torch.round(v - 0.5).long()
            inside = (
                (z > 1e-8) & (x >= 0) & (x < width)
                & (y >= 0) & (y < height)
            )
            ids = torch.nonzero(inside, as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            legal = validity & (mask >= float(mask_threshold))
            source_x = x[ids].clone()
            source_y = y[ids].clone()
            found = legal[source_y, source_x].clone()
            for distance in range(1, int(source_search_radius) + 1):
                if bool(torch.all(found)):
                    break
                offsets = []
                for dy in range(-distance, distance + 1):
                    for dx in range(-distance, distance + 1):
                        if max(abs(dx), abs(dy)) == distance:
                            offsets.append((dx * dx + dy * dy, dy, dx))
                for _distance_squared, dy, dx in sorted(offsets):
                    pending = ~found
                    xx = x[ids] + dx
                    yy = y[ids] + dy
                    candidate_inside = (
                        pending & (xx >= 0) & (xx < width)
                        & (yy >= 0) & (yy < height)
                    )
                    safe_x = xx.clamp(0, width - 1)
                    safe_y = yy.clamp(0, height - 1)
                    accepted = candidate_inside & legal[safe_y, safe_x]
                    source_x[accepted] = xx[accepted]
                    source_y[accepted] = yy[accepted]
                    found |= accepted
            ids = ids[found]
            source_x = source_x[found]
            source_y = source_y[found]
            if ids.numel() == 0:
                continue
            depth_buffer = torch.full(
                (height * width,), float("inf"),
                dtype=torch.float32, device=device,
            )
            pixel = y[ids] * width + x[ids]
            depth_buffer.scatter_reduce_(
                0, pixel, z[ids], reduce="amin", include_self=True,
            )
            depth_buffer = depth_buffer.reshape(height, width)
            # Full geometry remains the occlusion reference. Only queries
            # enter the expensive multi-view color aggregation.
            queried = query_map[ids] >= 0
            ids, source_x, source_y = ids[queried], source_x[queried], source_y[queried]
            if ids.numel() == 0:
                continue
            if int(z_buffer_radius) > 0:
                radius = int(z_buffer_radius)
                nearest = torch.full(
                    (ids.numel(),), float("inf"),
                    dtype=torch.float32, device=device,
                )
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        xx = x[ids] + dx
                        yy = y[ids] + dy
                        query_inside = (
                            (xx >= 0) & (xx < width)
                            & (yy >= 0) & (yy < height)
                        )
                        candidate = depth_buffer[
                            yy.clamp(0, height - 1), xx.clamp(0, width - 1)
                        ]
                        nearest = torch.minimum(
                            nearest,
                            torch.where(
                                query_inside, candidate,
                                torch.full_like(candidate, float("inf")),
                            ),
                        )
            else:
                nearest = depth_buffer[y[ids], x[ids]]
            if mode == "visibility_robust":
                tolerance = torch.maximum(
                    torch.full_like(nearest, 1e-5),
                    float(visibility_relative_tolerance) * nearest,
                )
                visible = z[ids] <= nearest + tolerance
                ids = ids[visible]
                source_x = source_x[visible]
                source_y = source_y[visible]
            if ids.numel() == 0:
                continue
            c2w = torch.linalg.inv(w2c)
            to_camera = c2w[:3, 3] - points[ids]
            ray_norm = torch.linalg.vector_norm(to_camera, dim=1)
            squared_distance = (ray_norm * ray_norm).clamp_min(1e-6)
            angle_weight = torch.ones_like(squared_distance)
            if normal_values is not None and float(angle_weight_power) > 0.0:
                direction = to_camera / ray_norm.clamp_min(1e-12)[:, None]
                cosine = torch.clamp(
                    torch.sum(normal_values[ids] * direction, dim=1), min=0.0,
                )
                front = cosine > 0.0
                ids = ids[front]
                source_x = source_x[front]
                source_y = source_y[front]
                squared_distance = squared_distance[front]
                angle_weight = cosine[front] ** float(angle_weight_power)
            output_ids = query_map[ids]
            colors[view_index, output_ids] = image[source_y, source_x, :3]
            depths[view_index, output_ids] = z[ids]
            weights[view_index, output_ids] = (
                torch.ones_like(squared_distance)
                if mode == "depth_rank_top3" else
                mask[source_y, source_x] * angle_weight / squared_distance
            )
            available[view_index, output_ids] = True

        rgb = torch.zeros((point_count, 3), dtype=torch.uint8, device=device)
        count = torch.zeros(point_count, dtype=torch.int32, device=device)
        disagreement = torch.full(
            (point_count,), float("nan"), dtype=torch.float32, device=device,
        )
        aggregation_chunk = 262_144
        for start in range(0, point_count, aggregation_chunk):
            end = min(point_count, start + aggregation_chunk)
            selected = available[:, start:end].clone()
            if mode == "depth_rank_top3":
                ranked = torch.zeros_like(selected)
                ordered = torch.argsort(
                    torch.where(
                        selected, depths[:, start:end],
                        torch.full_like(depths[:, start:end], float("inf")),
                    ),
                    dim=0, stable=True,
                )
                columns = torch.arange(end - start, device=device)
                for rank in range(min(int(depth_rank_views), view_count)):
                    row = ordered[rank]
                    valid = selected[row, columns]
                    ranked[row[valid], columns[valid]] = True
                selected = ranked
            active = torch.nonzero(
                torch.any(selected, dim=0), as_tuple=False,
            ).flatten()
            if active.numel() == 0:
                continue
            chosen = selected[:, active].T
            values = colors[:, start:end][:, active].permute(1, 0, 2).float()
            sample_weights = weights[:, start:end][:, active].T
            median = _torch_masked_median(values, chosen, torch=torch)
            residual = torch.linalg.vector_norm(values - median[:, None, :], dim=2)
            residual_median = _torch_masked_median(residual, chosen, torch=torch)
            residual_mad = _torch_masked_median(
                torch.abs(residual - residual_median[:, None]), chosen, torch=torch,
            )
            cutoff = residual_median + 2.5 * torch.clamp(residual_mad, min=1.0)
            accepted = chosen & (residual <= cutoff[:, None])
            accepted_weights = torch.where(
                accepted, sample_weights, torch.zeros_like(sample_weights),
            )
            weight_sum = accepted_weights.sum(dim=1)
            weighted_rgb = torch.einsum("nv,nvc->nc", accepted_weights, values)
            zero_weight = weight_sum <= 0.0
            if bool(torch.any(zero_weight)):
                accepted_count = accepted.sum(dim=1).clamp_min(1)
                fallback = torch.einsum(
                    "nv,nvc->nc", accepted.float(), values,
                ) / accepted_count[:, None]
                weighted_rgb[zero_weight] = fallback[zero_weight]
                weight_sum[zero_weight] = 1.0
            output_ids = start + active
            rgb[output_ids] = torch.clamp(
                torch.round(weighted_rgb / weight_sum[:, None]), 0, 255,
            ).to(torch.uint8)
            accepted_count = accepted.sum(dim=1)
            count[output_ids] = accepted_count.to(torch.int32)
            disagreement[output_ids] = (
                torch.where(accepted, residual, torch.zeros_like(residual)).sum(dim=1)
                / accepted_count.clamp_min(1)
            )
        return PointColorResult(
            rgb=rgb.cpu().numpy(),
            keep=(count >= int(min_sources)).cpu().numpy(),
            source_count=count.cpu().numpy(),
            disagreement=disagreement.cpu().numpy(),
            device=device,
        )


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
    device: str = "auto",
    query_indices: np.ndarray | None = None,
) -> PointColorResult:
    """Color queries using all points as the per-view occlusion reference.

    Results follow ``query_indices`` order, or input order when omitted.
    Sampling queries never removes occluders from the depth buffer.
    """

    xyz = np.asarray(points, np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points must be a finite Nx3 array")
    if query_indices is None:
        query_indices = np.arange(len(xyz), dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices)
        if (query_indices.ndim != 1 or query_indices.dtype.kind not in "iu"
                or np.any(query_indices < 0) or np.any(query_indices >= len(xyz))
                or len(np.unique(query_indices)) != len(query_indices)):
            raise ValueError("IBR query_indices must be unique valid integer indices")
        query_indices = query_indices.astype(np.int64, copy=False)
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
    resolved_device, torch = resolve_torch_device(
        device, purpose="IBR point coloring",
    )
    if resolved_device.startswith("cuda"):
        return _color_points_ibr_cuda(
            xyz, views, torch=torch, device=resolved_device,
            mask_threshold=float(mask_threshold),
            visibility_relative_tolerance=float(visibility_relative_tolerance),
            min_sources=int(min_sources), mode=mode,
            depth_rank_views=int(depth_rank_views),
            z_buffer_radius=int(z_buffer_radius),
            source_search_radius=int(source_search_radius),
            normals=normal_values,
            angle_weight_power=float(angle_weight_power),
            query_indices=query_indices,
        )
    view_count = len(views)
    point_count = len(query_indices)
    query_map = np.full(len(xyz), -1, np.int64)
    query_map[query_indices] = np.arange(point_count)
    colors = np.zeros((view_count, point_count, 3), np.uint8)
    weights = np.zeros((view_count, point_count), np.float64)
    depths = np.full((view_count, point_count), np.inf, np.float64)
    available = np.zeros((view_count, point_count), bool)
    for view_index, view, decoded in prefetch_rgb(views):
        image = decoded[..., :3]
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
        queried = query_map[ids] >= 0
        ids, selected_x, selected_y = ids[queried], selected_x[queried], selected_y[queried]
        if not len(ids):
            continue
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
        output_ids = query_map[ids]
        colors[view_index, output_ids] = image[selected_y, selected_x, :3]
        depths[view_index, output_ids] = z[ids]
        weights[view_index, output_ids] = (
            1.0 if mode == "depth_rank_top3" else
            mask[selected_y, selected_x] * angle_weight / squared_distance
        )
        available[view_index, output_ids] = True

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
        device="cpu",
    )
