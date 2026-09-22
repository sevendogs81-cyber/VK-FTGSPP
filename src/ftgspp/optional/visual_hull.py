"""Calibrated mask carving and visibility-aware IBR point preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from ftgspp.config.compose import load_config
from ftgspp.data.scene import Scene, View
from ftgspp.io.ply import read_point_ply, write_point_ply_atomic
from ftgspp.io.provenance import source_identity
from ftgspp.optional.point_color import (
    IBR_COLOR_VERSION,
    color_points_ibr,
    project_points,
)
from ftgspp.optional.torch_compute import resolve_torch_device
from ftgspp.training.init_plan import initialization_seed
from ftgspp.training.point_sampling import spatial_sample_indices

VISUAL_HULL_SCHEMA = "ftgspp-visual-hull-ibr-v3"
VISUAL_HULL_METHODS = {"carving", "point_filtering"}


def _resolve_carving_device(requested: str):
    """Compatibility wrapper for the shared optional device resolver."""

    return resolve_torch_device(requested, purpose="visual-hull preparation")


def _cuda_support_counts(
    xyz: np.ndarray,
    prepared_views,
    *,
    torch,
    device: str,
    min_views: int,
    support_fraction: float,
    return_counts: bool = False,
):
    """Return kept local ids and a small support histogram from CUDA.

    Visibility and support stay on the device.  Only selected indices and the
    per-chunk diagnostic histogram cross the PCIe boundary.
    """

    with torch.inference_mode():
        points = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        visible = torch.zeros(len(points), dtype=torch.int16, device=device)
        support = torch.zeros(len(points), dtype=torch.int16, device=device)
        for w2c, intrinsic, support_mask, validity, height, width in prepared_views:
            camera = points @ w2c[:3, :3].T + w2c[:3, 3]
            depth = camera[:, 2]
            u = intrinsic[0, 0] * camera[:, 0] / depth.clamp_min(1e-12) + intrinsic[0, 2]
            v = intrinsic[1, 1] * camera[:, 1] / depth.clamp_min(1e-12) + intrinsic[1, 2]
            x = torch.round(u - 0.5).long()
            y = torch.round(v - 0.5).long()
            inside = (
                (depth > 1e-8) & (x >= 0) & (x < width)
                & (y >= 0) & (y < height)
            )
            safe_x = x.clamp(0, width - 1)
            safe_y = y.clamp(0, height - 1)
            legal = inside & validity[safe_y, safe_x]
            visible += legal.to(torch.int16)
            support += (
                legal & support_mask[safe_y, safe_x]
            ).to(torch.int16)
        required = torch.ceil(
            torch.clamp(visible, min=1).to(torch.float32)
            * float(support_fraction)
        ).to(torch.int16)
        keep = (visible >= int(min_views)) & (support >= required)
        kept = torch.nonzero(keep, as_tuple=False).flatten().cpu().numpy()
        histogram = torch.bincount(
            torch.clamp(support.to(torch.int64), max=len(prepared_views)),
            minlength=len(prepared_views) + 1,
        ).cpu().numpy()
        visible_histogram = torch.bincount(
            torch.clamp(visible.to(torch.int64), max=len(prepared_views)),
            minlength=len(prepared_views) + 1,
        ).cpu().numpy()
        counts = (
            (support.cpu().numpy(), visible.cpu().numpy())
            if return_counts else None
        )
    return kept, histogram, visible_histogram, counts


def carve_visual_hull(
    views: Sequence[View],
    bounds: np.ndarray,
    *,
    resolution: int | tuple[int, int, int] = 128,
    mask_threshold: float = 0.5,
    min_views: int | None = None,
    support_fraction: float = 0.9,
    surface_band: bool = True,
    surface_band_layers: int = 2,
    coarse_resolution: int | tuple[int, int, int] | None = None,
    mask_dilation_pixels: int = 0,
    chunk_points: int = 1_000_000,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, object]]:
    """Carve voxel centers, using chunked CUDA projection when available."""

    if not views:
        raise ValueError("visual hull requires at least one calibrated view")
    bounds = np.asarray(bounds, np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("bounds must be finite [[xmin,ymin,zmin],[xmax,ymax,zmax]]")
    shape = (resolution,) * 3 if isinstance(resolution, int) else tuple(int(v) for v in resolution)
    if len(shape) != 3 or min(shape) <= 1:
        raise ValueError("visual-hull resolution must have three values greater than one")
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("visual-hull mask_threshold must be in [0, 1]")
    resolved_min_views = (
        min(3, len(views)) if min_views is None else int(min_views)
    )
    if len(views) < 2:
        raise ValueError("visual-hull carving requires at least two views")
    if resolved_min_views <= 0 or resolved_min_views > len(views):
        raise ValueError("visual-hull min_views must be between one and the view count")
    if not 0.0 <= float(support_fraction) <= 1.0:
        raise ValueError("visual-hull support_fraction must be in [0, 1]")
    if int(chunk_points) <= 0:
        raise ValueError("visual-hull chunk_points must be positive")
    if int(mask_dilation_pixels) < 0:
        raise ValueError("mask_dilation_pixels cannot be negative")
    if int(surface_band_layers) <= 0:
        raise ValueError("surface_band_layers must be positive")
    resolved_device, torch = _resolve_carving_device(device)
    masks = [view.load_mask() for view in views]
    support_masks = [
        ndimage.binary_dilation(
            mask >= float(mask_threshold), iterations=int(mask_dilation_pixels),
        ) if int(mask_dilation_pixels) > 0 else mask >= float(mask_threshold)
        for mask in masks
    ]
    validities = [view.load_validity() >= 0.5 for view in views]
    cuda_views = None
    if resolved_device.startswith("cuda"):
        cuda_views = [(
            torch.as_tensor(view.w2c, dtype=torch.float32, device=resolved_device),
            torch.as_tensor(
                view.intrinsic, dtype=torch.float32, device=resolved_device,
            ),
            torch.as_tensor(support_mask, device=resolved_device),
            torch.as_tensor(validity, device=resolved_device),
            int(view.size[0]),
            int(view.size[1]),
        ) for view, support_mask, validity in zip(
            views, support_masks, validities, strict=True,
        )]
    axes = [
        np.linspace(bounds[0, axis], bounds[1, axis], shape[axis], endpoint=False)
        + (bounds[1, axis] - bounds[0, axis]) / (2.0 * shape[axis])
        for axis in range(3)
    ]
    total = int(np.prod(shape))
    occupied = np.zeros(total, bool)
    support_hist = np.zeros(len(views) + 1, np.int64)
    coarse_volume: np.ndarray | None = None
    fine_candidates: np.ndarray | None = None
    coarse_diagnostics: dict[str, object] | None = None
    if coarse_resolution is not None:
        coarse_shape = (
            (int(coarse_resolution),) * 3
            if isinstance(coarse_resolution, int)
            else tuple(int(value) for value in coarse_resolution)
        )
        if len(coarse_shape) != 3 or min(coarse_shape) <= 1:
            raise ValueError("coarse visual-hull resolution must exceed one")
        if any(coarse >= fine for coarse, fine in zip(coarse_shape, shape, strict=True)):
            raise ValueError("coarse visual-hull resolution must be below fine resolution")
        coarse_points, coarse_diagnostics = carve_visual_hull(
            views, bounds, resolution=coarse_shape,
            mask_threshold=mask_threshold, min_views=resolved_min_views,
            support_fraction=support_fraction, surface_band=False,
            surface_band_layers=surface_band_layers,
            coarse_resolution=None,
            mask_dilation_pixels=mask_dilation_pixels,
            chunk_points=chunk_points,
            device=resolved_device,
        )
        extent = bounds[1] - bounds[0]
        coarse_indices = np.floor(
            (coarse_points.astype(np.float64) - bounds[0])
            / extent * np.asarray(coarse_shape, np.float64)
        ).astype(np.int64)
        coarse_indices = np.clip(
            coarse_indices, 0, np.asarray(coarse_shape) - 1,
        )
        coarse_volume = np.zeros(coarse_shape, bool)
        coarse_volume[tuple(coarse_indices.T)] = True
        # One coarse-cell guard band prevents a thin surface close to a cell
        # boundary from being discarded before fine evaluation.
        coarse_volume = ndimage.binary_dilation(coarse_volume, iterations=1)
        coarse_axis = [
            np.minimum(
                np.arange(shape[axis], dtype=np.int64)
                * coarse_shape[axis] // shape[axis],
                coarse_shape[axis] - 1,
            )
            for axis in range(3)
        ]
        fine_candidates = np.flatnonzero(
            coarse_volume[np.ix_(*coarse_axis)].reshape(-1)
        )
    evaluated_candidates = 0
    active_chunk_points = int(chunk_points)
    minimum_chunk_points = active_chunk_points
    cuda_oom_reductions = 0
    candidate_total = total if fine_candidates is None else int(len(fine_candidates))
    start = 0
    while start < candidate_total:
        end = min(candidate_total, start + active_chunk_points)
        flat = (
            np.arange(start, end, dtype=np.int64)
            if fine_candidates is None else fine_candidates[start:end]
        )
        ix, iy, iz = np.unravel_index(flat, shape)
        xyz = np.stack((axes[0][ix], axes[1][iy], axes[2][iz]), axis=1)
        if resolved_device.startswith("cuda"):
            try:
                kept_local, chunk_histogram, _visible_histogram, _counts = _cuda_support_counts(
                    xyz, cuda_views,
                    torch=torch, device=resolved_device,
                    min_views=resolved_min_views,
                    support_fraction=float(support_fraction),
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or active_chunk_points <= 1:
                    raise
                active_chunk_points = max(1, active_chunk_points // 2)
                minimum_chunk_points = min(
                    minimum_chunk_points, active_chunk_points,
                )
                cuda_oom_reductions += 1
                torch.cuda.empty_cache()
                continue
            occupied[flat[kept_local]] = True
            support_hist += chunk_histogram
        else:
            visible = np.zeros(len(xyz), np.int16)
            support = np.zeros(len(xyz), np.int16)
            for view, support_mask, validity in zip(
                views, support_masks, validities, strict=True,
            ):
                u, v, depth = project_points(xyz, view)
                x = np.rint(u - 0.5).astype(np.int64)
                y = np.rint(v - 0.5).astype(np.int64)
                height, width = view.size
                inside = (
                    (depth > 1e-8) & (x >= 0) & (x < width)
                    & (y >= 0) & (y < height)
                )
                ids = np.flatnonzero(inside)
                if not len(ids):
                    continue
                ids = ids[validity[y[ids], x[ids]]]
                visible[ids] += 1
                support[ids] += support_mask[y[ids], x[ids]]
            keep = (visible >= resolved_min_views) & (
                support >= np.ceil(np.maximum(visible, 1) * float(support_fraction))
            )
            occupied[flat] = keep
            support_hist += np.bincount(
                np.minimum(support, len(views)), minlength=len(views) + 1,
            )
        evaluated_candidates += len(flat)
        minimum_chunk_points = min(minimum_chunk_points, end - start)
        start = end
    volume_count = int(np.count_nonzero(occupied))
    if volume_count == 0:
        raise RuntimeError("visual-hull carving produced an empty volume")
    volume = occupied.reshape(shape)
    selected = (
        volume & ~ndimage.binary_erosion(
            volume, iterations=int(surface_band_layers), border_value=0,
        )
        if surface_band else volume
    )
    ix, iy, iz = np.nonzero(selected)
    points = np.stack((axes[0][ix], axes[1][iy], axes[2][iz]), axis=1).astype(np.float32)
    extent = bounds[1] - bounds[0]
    diagnostics: dict[str, object] = {
        "schema": VISUAL_HULL_SCHEMA,
        "resolution": list(shape),
        "voxel_size": (extent / np.asarray(shape)).tolist(),
        "candidate_count": total,
        "coarse_candidate_count": int(candidate_total),
        "evaluated_candidate_count": int(evaluated_candidates),
        "volume_count": volume_count,
        "output_count": int(len(points)),
        "surface_band": bool(surface_band),
        "surface_band_layers": int(surface_band_layers),
        "mask_threshold": float(mask_threshold),
        "min_views": resolved_min_views,
        "support_fraction": float(support_fraction),
        "mask_dilation_pixels": int(mask_dilation_pixels),
        "device": resolved_device,
        "requested_chunk_points": int(chunk_points),
        "minimum_chunk_points": int(minimum_chunk_points),
        "cuda_oom_reductions": int(cuda_oom_reductions),
        "coarse": coarse_diagnostics,
        "support_histogram": support_hist.tolist(),
    }
    return points, diagnostics


def filter_points_by_visual_hull(
    points: np.ndarray,
    views: Sequence[View],
    *,
    mask_threshold: float = 0.5,
    min_views: int | None = None,
    support_fraction: float = 0.9,
    mask_dilation_pixels: int = 0,
    device: str = "auto",
    chunk_points: int = 1_000_000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conservatively retain an existing cloud inside calibrated silhouettes."""

    xyz = np.asarray(points, np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("visual-hull input points must be a finite Nx3 array")
    if len(views) < 2:
        raise ValueError("visual-hull point filtering requires at least two views")
    resolved_min = min(3, len(views)) if min_views is None else int(min_views)
    if resolved_min < 1 or resolved_min > len(views):
        raise ValueError("visual-hull min_views is outside the available view count")
    if not 0.0 <= float(support_fraction) <= 1.0:
        raise ValueError("visual-hull support_fraction must be in [0, 1]")
    if int(mask_dilation_pixels) < 0:
        raise ValueError("mask_dilation_pixels cannot be negative")
    if int(chunk_points) <= 0:
        raise ValueError("visual-hull chunk_points must be positive")
    resolved_device, torch = _resolve_carving_device(device)
    masks = []
    validities = []
    for view in views:
        mask = view.load_mask() >= float(mask_threshold)
        if int(mask_dilation_pixels) > 0:
            mask = ndimage.binary_dilation(
                mask, iterations=int(mask_dilation_pixels),
            )
        masks.append(mask)
        validities.append(view.load_validity() >= 0.5)
    visible = np.zeros(len(xyz), np.int16)
    support = np.zeros(len(xyz), np.int16)
    keep = np.zeros(len(xyz), bool)
    if resolved_device.startswith("cuda"):
        prepared = [(
            torch.as_tensor(view.w2c, dtype=torch.float32, device=resolved_device),
            torch.as_tensor(view.intrinsic, dtype=torch.float32, device=resolved_device),
            torch.as_tensor(mask, device=resolved_device),
            torch.as_tensor(validity, device=resolved_device),
            int(view.size[0]), int(view.size[1]),
        ) for view, mask, validity in zip(views, masks, validities, strict=True)]
        for start in range(0, len(xyz), int(chunk_points)):
            end = min(len(xyz), start + int(chunk_points))
            kept, _support_hist, _visible_hist, counts = _cuda_support_counts(
                xyz[start:end], prepared, torch=torch, device=resolved_device,
                min_views=resolved_min, support_fraction=float(support_fraction),
                return_counts=True,
            )
            keep[start + kept] = True
            assert counts is not None
            support[start:end], visible[start:end] = counts
    else:
        for view, mask, validity in zip(views, masks, validities, strict=True):
            u, v, depth = project_points(xyz, view)
            x = np.rint(u - 0.5).astype(np.int64)
            y = np.rint(v - 0.5).astype(np.int64)
            height, width = view.size
            inside = (
                (depth > 1e-8) & (x >= 0) & (x < width)
                & (y >= 0) & (y < height)
            )
            ids = np.flatnonzero(inside)
            if not len(ids):
                continue
            ids = ids[validity[y[ids], x[ids]]]
            visible[ids] += 1
            support[ids] += mask[y[ids], x[ids]]
        keep = (visible >= resolved_min) & (
            support >= np.ceil(np.maximum(visible, 1) * float(support_fraction))
        )
    return keep, {
        "input_count": int(len(xyz)),
        "kept_count": int(np.count_nonzero(keep)),
        "min_views": resolved_min,
        "support_fraction": float(support_fraction),
        "device": resolved_device,
        "visible_histogram": np.bincount(
            visible, minlength=len(views) + 1,
        ).tolist(),
        "support_histogram": np.bincount(
            support, minlength=len(views) + 1,
        ).tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounds_from_point_clouds(
    clouds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    quantile: float = 0.005,
    padding_fraction: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate stable carving bounds from DA3/user keyframe clouds."""

    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    point_count = 0
    for xyz, _rgb in clouds:
        points = np.asarray(xyz, np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point-cloud bounds require Nx3 coordinates")
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        point_count += len(points)
        trim = float(quantile) if len(points) >= 200 else 0.0
        lowers.append(np.quantile(points, trim, axis=0))
        uppers.append(np.quantile(points, 1.0 - trim, axis=0))
    if not lowers:
        raise ValueError("cannot derive visual-hull bounds from empty point clouds")
    raw_lower = np.min(np.stack(lowers), axis=0)
    raw_upper = np.max(np.stack(uppers), axis=0)
    extent = raw_upper - raw_lower
    scale = max(float(np.max(extent)), 1e-4)
    padding = np.maximum(extent * float(padding_fraction), scale * 0.005)
    bounds = np.stack((raw_lower - padding, raw_upper + padding))
    return bounds, {
        "kind": "point_cloud_quantiles",
        "quantile": float(quantile),
        "padding_fraction": float(padding_fraction),
        "source_points": int(point_count),
        "raw_bounds": np.stack((raw_lower, raw_upper)).tolist(),
        "resolved_bounds": bounds.tolist(),
    }


def prepare_visual_hull_points(
    config_path: Path,
    output: Path,
    bounds: np.ndarray | None,
    *,
    method: str = "carving",
    resolution: int = 128,
    mask_threshold: float = 0.5,
    color_mask_threshold: float = 0.8,
    min_views: int | None = None,
    support_fraction: float = 0.9,
    mask_dilation_pixels: int = 2,
    sampling: str = "surface_band",
    surface_band_layers: int = 2,
    coarse_resolution: int | None = None,
    max_points: int | None = None,
    input_points: Path | None = None,
    color_mode: str = "visibility_robust",
    visibility_relative_tolerance: float = 0.01,
    color_min_sources: int = 1,
    z_buffer_radius: int = 1,
    source_search_radius: int = 2,
    device: str = "auto",
    chunk_points: int = 1_000_000,
) -> Path:
    if method not in VISUAL_HULL_METHODS:
        raise ValueError(
            f"visual-hull method must be one of {sorted(VISUAL_HULL_METHODS)}"
        )
    if method == "carving" and bounds is not None and input_points is not None:
        raise ValueError("carving accepts --bounds or --input-points, not both")
    if method == "point_filtering" and bounds is not None:
        raise ValueError("--bounds is only valid with method=carving")
    if method == "carving" and sampling not in {"surface_band", "volume"}:
        raise ValueError("visual-hull sampling must be surface_band or volume")
    if int(chunk_points) <= 0:
        raise ValueError("visual-hull chunk_points must be positive")
    if max_points is not None and int(max_points) <= 0:
        raise ValueError("visual-hull max_points must be positive when set")
    input_root = (
        None if input_points is None else input_points.expanduser().resolve()
    )
    if input_root is not None and not input_root.is_dir():
        raise FileNotFoundError(input_root)
    config = load_config(config_path)
    scene = Scene.from_config(config)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_coarse = None if method == "point_filtering" else (
        max(2, int(resolution) // 4)
        if coarse_resolution is None else int(coarse_resolution)
    )
    frames = list(range(0, len(scene.frames), int(config.init.keyframe_stride)))
    if frames[-1] != len(scene.frames) - 1:
        frames.append(len(scene.frames) - 1)
    source_clouds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None
    source_hashes: dict[str, str] = {}
    point_source: dict[str, object] | None = None
    if input_root is None and (method == "point_filtering" or bounds is None):
        from ftgspp.optional.da3 import generate_da3_points

        generated = generate_da3_points(config, scene, frames)
        source_clouds = generated
        input_root = Path(generated.cache_root).resolve()
        try:
            source_manifest = json.loads(
                (input_root / "manifest.json").read_text(encoding="utf-8")
            )
            source_hashes = {
                str(row["ply"]): str(row["ply_sha256"])
                for row in source_manifest.get("keyframes", [])
                if isinstance(row, dict)
                and isinstance(row.get("ply"), str)
                and isinstance(row.get("ply_sha256"), str)
            }
        except (OSError, json.JSONDecodeError):
            source_hashes = {}
        point_source = {
            "kind": "da3_generated" if generated.generated else "da3_cache",
            "path": str(input_root),
        }
    elif input_root is not None:
        if method == "carving" and bounds is None:
            source_clouds = [
                read_point_ply(
                    input_root / f"f{int(scene.frames[position]):06d}.ply"
                )
                for position in frames
            ]
        point_source = {"kind": "prepared_cloud", "path": str(input_root)}
    if not scene.calibration_available:
        raise ValueError(
            "visual-hull preparation requires calibrated cameras or DA3 camera estimation"
        )
    bounds_source: dict[str, object]
    if method == "carving" and bounds is None:
        assert source_clouds is not None
        bounds, bounds_source = _bounds_from_point_clouds(source_clouds)
        bounds_source["point_source"] = point_source
    else:
        bounds_source = {"kind": "explicit"} if bounds is not None else {
            "kind": "not_applicable",
        }
    rows: list[dict[str, object]] = []
    resolved_preparation_device: str | None = None
    resolved_coloring_device: str | None = None
    tolerance = float(config.init.da3.sync_tolerance_fraction) / float(scene.fps)
    seed = initialization_seed(config)
    rng = np.random.default_rng(seed)
    for cloud_index, position in enumerate(
        tqdm(frames, desc=f"Preparing {method} points")
    ):
        frame = int(scene.frames[position])
        # Use an actual capture timestamp as the keyframe coordinate. Camera
        # offsets make logical ``frame/fps`` an unsuitable synchronization
        # anchor for real multi-camera rigs.
        reference = float(scene.view(frame, scene.train_cameras[0]).timestamp)
        views = scene.views_nearest_timestamp(
            reference, scene.train_cameras,
            tolerance_seconds=tolerance, train_only=True,
        )
        if method == "carving":
            points, diagnostics = carve_visual_hull(
                views, np.asarray(bounds),
                resolution=resolution, mask_threshold=mask_threshold,
                min_views=min_views, support_fraction=support_fraction,
                surface_band=sampling == "surface_band",
                surface_band_layers=surface_band_layers,
                coarse_resolution=resolved_coarse,
                mask_dilation_pixels=mask_dilation_pixels,
                chunk_points=chunk_points,
                device=device,
            )
            resolved_preparation_device = str(diagnostics["device"])
            source_record: dict[str, object] = {
                "kind": "carved_grid",
                "bounds_source": bounds_source["kind"],
            }
        else:
            source_path = input_root / f"f{frame:06d}.ply"
            if source_clouds is not None:
                source_xyz, _source_rgb = source_clouds[cloud_index]
                source_digest = source_hashes.get(source_path.name)
                if source_digest is None and source_path.is_file():
                    source_digest = _sha256(source_path)
            else:
                source_xyz, _source_rgb, source_digest = read_point_ply(
                    source_path, return_sha256=True,
                )
            keep, diagnostics = filter_points_by_visual_hull(
                source_xyz, views, mask_threshold=mask_threshold,
                min_views=min_views, support_fraction=support_fraction,
                mask_dilation_pixels=mask_dilation_pixels,
                device=device, chunk_points=chunk_points,
            )
            resolved_preparation_device = str(diagnostics["device"])
            points = source_xyz[keep]
            source_record = {
                "kind": "input_cloud_hull_filter",
                "path": str(source_path),
                "sha256": source_digest,
            }
            if not len(points):
                raise RuntimeError(
                    f"visual-hull filtering removed every point for f{frame:06d}"
                )
        colored = color_points_ibr(
            points, views, mask_threshold=color_mask_threshold,
            visibility_relative_tolerance=visibility_relative_tolerance,
            min_sources=color_min_sources, mode=color_mode,
            z_buffer_radius=z_buffer_radius,
            source_search_radius=source_search_radius,
            device=device,
        )
        resolved_coloring_device = str(
            getattr(colored, "device", resolved_preparation_device or "cpu")
        )
        points = points[colored.keep]
        rgb = colored.rgb[colored.keep]
        if not len(points):
            raise RuntimeError(f"visual hull f{frame:06d} has no trusted color source")
        if max_points is not None and len(points) > int(max_points):
            keep = spatial_sample_indices(rng, points, int(max_points))
            points, rgb = points[keep], rgb[keep]
        ply, ply_digest = write_point_ply_atomic(
            output / f"f{frame:06d}.ply", points, rgb,
            return_sha256=True,
        )
        rows.append({
            "frame": frame,
            "reference_timestamp": reference,
            "views": [{
                "camera_id": int(view.camera_id),
                "frame": int(view.frame),
                "timestamp": float(view.timestamp),
                "residual_seconds": abs(float(view.timestamp) - reference),
            } for view in views],
            "ply": ply.name,
            "ply_sha256": ply_digest,
            "colored_points": int(len(points)),
            "color_sources_histogram": np.bincount(
                colored.source_count, minlength=len(views) + 1,
            ).tolist(),
            "geometry_diagnostics": diagnostics,
            "geometry_source": source_record,
        })
    manifest = {
        "schema": VISUAL_HULL_SCHEMA,
        "method": method,
        "dataset_fingerprint": scene.checkpoint_metadata()["fingerprint"],
        "bounds": None if bounds is None else np.asarray(bounds, float).tolist(),
        "bounds_source": bounds_source,
        "seed": seed,
        "sampling": sampling if method == "carving" else None,
        "coarse_resolution": resolved_coarse,
        "fine_resolution": int(resolution) if method == "carving" else None,
        "mask_threshold": float(mask_threshold),
        "color_mask_threshold": float(color_mask_threshold),
        "min_views": (
            None if min_views is None else int(min_views)
        ),
        "support_fraction": float(support_fraction),
        "mask_dilation_pixels": int(mask_dilation_pixels),
        "surface_band_layers": (
            int(surface_band_layers) if method == "carving" else None
        ),
        "geometry_source": (
            "carved_grid" if method == "carving" else "input_cloud_hull_filter"
        ),
        "requested_device": device,
        "device": resolved_preparation_device,
        "chunk_points": int(chunk_points),
        "input_points": None if input_root is None else str(input_root),
        "point_source": point_source,
        "coloring": {
            "version": IBR_COLOR_VERSION,
            "mode": color_mode,
            "visibility_relative_tolerance": float(visibility_relative_tolerance),
            "min_sources": int(color_min_sources),
            "z_buffer_radius": int(z_buffer_radius),
            "source_search_radius": int(source_search_radius),
            "robust_color": "weighted_median_mad_v1",
            "device": resolved_coloring_device,
        },
        "max_points": None if max_points is None else int(max_points),
        "rows": rows,
    }
    source = source_identity()
    manifest["source_identity"] = {
        "schema": source["schema"],
        "set_sha256": source["set_sha256"],
    }
    manifest_path = output / "manifest.json"
    temporary = output / f".manifest.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    completed = output / "completed.json"
    completed_tmp = output / f".completed.{os.getpid()}.tmp"
    completed_tmp.write_text(json.dumps({
        "schema": VISUAL_HULL_SCHEMA,
        "manifest_sha256": _sha256(manifest_path),
        "keyframes": len(rows),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(completed_tmp, completed)
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ftgspp prepare points")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method", choices=tuple(sorted(VISUAL_HULL_METHODS)),
        default="carving",
        help=(
            "carving builds points from masks inside DA3-derived bounds "
            "(default); point_filtering filters the DA3 cloud"
        ),
    )
    parser.add_argument(
        "--bounds", type=float, nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="optional carving bounds override; default derives robust bounds from DA3",
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--color-mask-threshold", type=float, default=0.8)
    parser.add_argument("--min-views", type=int)
    parser.add_argument("--support-fraction", type=float, default=0.9)
    parser.add_argument("--mask-dilation-pixels", type=int, default=2)
    parser.add_argument(
        "--sampling", choices=("surface_band", "volume"),
        default="surface_band",
    )
    parser.add_argument("--coarse-resolution", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--surface-band-layers", type=int, default=2)
    parser.add_argument(
        "--input-points", type=Path,
        help=(
            "optional DA3/user PLY directory used for filtering or automatic "
            "carving bounds; default generates/reuses the configured DA3 cache"
        ),
    )
    parser.add_argument(
        "--device", default="auto",
        help="geometry projection and IBR device: auto, cpu, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--chunk-points", type=int, default=1_000_000,
        help="candidate voxels per projection chunk; CUDA OOM halves it automatically",
    )
    parser.add_argument(
        "--color-mode", choices=("visibility_robust", "depth_rank_top3"),
        default="visibility_robust",
    )
    parser.add_argument("--visibility-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--color-min-sources", type=int, default=1)
    parser.add_argument("--z-buffer-radius", type=int, default=1)
    parser.add_argument("--source-search-radius", type=int, default=2)
    args = parser.parse_args(argv)
    if args.method == "carving" and args.bounds is not None and args.input_points is not None:
        parser.error("carving accepts --bounds or --input-points, not both")
    if args.method == "point_filtering" and args.bounds is not None:
        parser.error("--bounds is only valid with --method carving")
    bounds = (
        None if args.bounds is None
        else np.asarray(args.bounds, np.float64).reshape(2, 3)
    )
    print(prepare_visual_hull_points(
        args.config, args.output, bounds,
        method=args.method,
        resolution=args.resolution,
        mask_threshold=args.mask_threshold,
        color_mask_threshold=args.color_mask_threshold,
        min_views=args.min_views,
        support_fraction=args.support_fraction,
        mask_dilation_pixels=args.mask_dilation_pixels,
        sampling=args.sampling,
        surface_band_layers=args.surface_band_layers,
        coarse_resolution=args.coarse_resolution,
        max_points=args.max_points,
        input_points=args.input_points,
        color_mode=args.color_mode,
        visibility_relative_tolerance=args.visibility_relative_tolerance,
        color_min_sources=args.color_min_sources,
        z_buffer_radius=args.z_buffer_radius,
        source_search_radius=args.source_search_radius,
        device=args.device,
        chunk_points=args.chunk_points,
    ))
