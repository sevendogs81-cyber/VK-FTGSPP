"""Calibrated mask carving and visibility-aware IBR point preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

from ftgspp.config.compose import load_config
from ftgspp.data.scene import Scene, View
from ftgspp.io.ply import read_point_ply, write_point_ply_atomic
from ftgspp.io.provenance import source_identity
from ftgspp.training.point_sampling import spatial_sample_indices
from ftgspp.optional.point_color import (
    IBR_COLOR_VERSION, color_points_ibr, project_points,
)


VISUAL_HULL_SCHEMA = "ftgspp-visual-hull-ibr-v1"


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
) -> tuple[np.ndarray, dict[str, object]]:
    """Carve voxel centers; projections outside a view provide no evidence."""

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
    masks = [view.load_mask() for view in views]
    support_masks = [
        ndimage.binary_dilation(
            mask >= float(mask_threshold), iterations=int(mask_dilation_pixels),
        ) if int(mask_dilation_pixels) > 0 else mask >= float(mask_threshold)
        for mask in masks
    ]
    validities = [view.load_validity() >= 0.5 for view in views]
    axes = [
        np.linspace(bounds[0, axis], bounds[1, axis], shape[axis], endpoint=False)
        + (bounds[1, axis] - bounds[0, axis]) / (2.0 * shape[axis])
        for axis in range(3)
    ]
    total = int(np.prod(shape))
    occupied = np.zeros(total, bool)
    support_hist = np.zeros(len(views) + 1, np.int64)
    coarse_volume: np.ndarray | None = None
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
    evaluated_candidates = 0
    for start in range(0, total, int(chunk_points)):
        flat = np.arange(start, min(total, start + int(chunk_points)), dtype=np.int64)
        ix, iy, iz = np.unravel_index(flat, shape)
        xyz = np.stack((axes[0][ix], axes[1][iy], axes[2][iz]), axis=1)
        if coarse_volume is not None:
            coarse_shape = coarse_volume.shape
            coarse_index = np.stack((
                np.minimum(ix * coarse_shape[0] // shape[0], coarse_shape[0] - 1),
                np.minimum(iy * coarse_shape[1] // shape[1], coarse_shape[1] - 1),
                np.minimum(iz * coarse_shape[2] // shape[2], coarse_shape[2] - 1),
            ), axis=1)
            candidate = coarse_volume[tuple(coarse_index.T)]
            flat = flat[candidate]
            xyz = xyz[candidate]
            if not len(flat):
                continue
        evaluated_candidates += len(flat)
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
                (depth > 1e-8) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            )
            ids = np.flatnonzero(inside)
            if not len(ids):
                continue
            valid = validity[y[ids], x[ids]]
            ids = ids[valid]
            visible[ids] += 1
            support[ids] += support_mask[y[ids], x[ids]]
        keep = (visible >= resolved_min_views) & (
            support >= np.ceil(np.maximum(visible, 1) * float(support_fraction))
        )
        occupied[flat] = keep
        support_hist += np.bincount(
            np.minimum(support, len(views)), minlength=len(views) + 1,
        )
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
        "evaluated_candidate_count": int(evaluated_candidates),
        "volume_count": volume_count,
        "output_count": int(len(points)),
        "surface_band": bool(surface_band),
        "surface_band_layers": int(surface_band_layers),
        "mask_threshold": float(mask_threshold),
        "min_views": resolved_min_views,
        "support_fraction": float(support_fraction),
        "mask_dilation_pixels": int(mask_dilation_pixels),
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
    visible = np.zeros(len(xyz), np.int16)
    support = np.zeros(len(xyz), np.int16)
    for view in views:
        mask = view.load_mask() >= float(mask_threshold)
        if int(mask_dilation_pixels) > 0:
            mask = ndimage.binary_dilation(
                mask, iterations=int(mask_dilation_pixels),
            )
        validity = view.load_validity() >= 0.5
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


def prepare_visual_hull_points(
    config_path: Path,
    output: Path,
    bounds: np.ndarray | None,
    *,
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
) -> Path:
    config = load_config(config_path)
    scene = Scene.from_config(config)
    if not scene.calibration_available:
        raise ValueError("visual-hull preparation requires calibrated cameras")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sampling not in {"surface_band", "volume"}:
        raise ValueError("visual-hull sampling must be surface_band or volume")
    if max_points is not None and int(max_points) <= 0:
        raise ValueError("visual-hull max_points must be positive when set")
    if input_points is None and bounds is None:
        raise ValueError("visual-hull carving requires explicit world bounds")
    input_root = (
        None if input_points is None else input_points.expanduser().resolve()
    )
    resolved_coarse = (
        max(2, int(resolution) // 4)
        if coarse_resolution is None else int(coarse_resolution)
    )
    frames = list(range(0, len(scene.frames), int(config.init.keyframe_stride)))
    if frames[-1] != len(scene.frames) - 1:
        frames.append(len(scene.frames) - 1)
    rows: list[dict[str, object]] = []
    tolerance = float(config.init.da3.sync_tolerance_fraction) / float(scene.fps)
    rng = np.random.default_rng(int(config.train.seed))
    for position in frames:
        frame = int(scene.frames[position])
        # Use an actual capture timestamp as the keyframe coordinate. Camera
        # offsets make logical ``frame/fps`` an unsuitable synchronization
        # anchor for real multi-camera rigs.
        reference = float(scene.view(frame, scene.train_cameras[0]).timestamp)
        views = scene.views_nearest_timestamp(
            reference, scene.train_cameras,
            tolerance_seconds=tolerance, train_only=True,
        )
        if input_root is None:
            points, diagnostics = carve_visual_hull(
                views, np.asarray(bounds),
                resolution=resolution, mask_threshold=mask_threshold,
                min_views=min_views, support_fraction=support_fraction,
                surface_band=sampling == "surface_band",
                surface_band_layers=surface_band_layers,
                coarse_resolution=resolved_coarse,
                mask_dilation_pixels=mask_dilation_pixels,
            )
            source_record: dict[str, object] = {"kind": "carved_grid"}
        else:
            source_path = input_root / f"f{frame:06d}.ply"
            source_xyz, _source_rgb = read_point_ply(source_path)
            keep, diagnostics = filter_points_by_visual_hull(
                source_xyz, views, mask_threshold=mask_threshold,
                min_views=min_views, support_fraction=support_fraction,
                mask_dilation_pixels=mask_dilation_pixels,
            )
            points = source_xyz[keep]
            source_record = {
                "kind": "input_cloud_hull_filter",
                "path": str(source_path),
                "sha256": _sha256(source_path),
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
        )
        points = points[colored.keep]
        rgb = colored.rgb[colored.keep]
        if not len(points):
            raise RuntimeError(f"visual hull f{frame:06d} has no trusted color source")
        if max_points is not None and len(points) > int(max_points):
            keep = spatial_sample_indices(rng, points, int(max_points))
            points, rgb = points[keep], rgb[keep]
        ply = write_point_ply_atomic(
            output / f"f{frame:06d}.ply", points, rgb,
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
            "ply_sha256": _sha256(ply),
            "colored_points": int(len(points)),
            "color_sources_histogram": np.bincount(
                colored.source_count, minlength=len(views) + 1,
            ).tolist(),
            "carving": diagnostics,
            "geometry_source": source_record,
        })
    manifest = {
        "schema": VISUAL_HULL_SCHEMA,
        "foreground_compatible": True,
        "dataset_fingerprint": scene.checkpoint_metadata()["fingerprint"],
        "bounds": None if bounds is None else np.asarray(bounds, float).tolist(),
        "seed": int(config.train.seed),
        "sampling": sampling,
        "coarse_resolution": resolved_coarse,
        "fine_resolution": int(resolution),
        "mask_threshold": float(mask_threshold),
        "color_mask_threshold": float(color_mask_threshold),
        "min_views": (
            None if min_views is None else int(min_views)
        ),
        "support_fraction": float(support_fraction),
        "mask_dilation_pixels": int(mask_dilation_pixels),
        "surface_band_layers": int(surface_band_layers),
        "geometry_source": (
            "carved_grid" if input_root is None else "input_cloud_hull_filter"
        ),
        "input_points": None if input_root is None else str(input_root),
        "coloring": {
            "version": IBR_COLOR_VERSION,
            "mode": color_mode,
            "visibility_relative_tolerance": float(visibility_relative_tolerance),
            "min_sources": int(color_min_sources),
            "z_buffer_radius": int(z_buffer_radius),
            "source_search_radius": int(source_search_radius),
            "robust_color": "weighted_median_mad_v1",
        },
        "max_points": None if max_points is None else int(max_points),
        "rows": rows,
    }
    source = source_identity()
    manifest["source_identity"] = {
        "schema": source["schema"],
        "set_sha256": source["set_sha256"],
        "git_patch_sha256": source["git_patch_sha256"],
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
    parser.add_argument("--bounds", type=float, nargs=6,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
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
        help="optional DA3/user PLY directory to hull-filter before IBR recoloring",
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
    if args.bounds is None and args.input_points is None:
        parser.error("--bounds is required unless --input-points is supplied")
    bounds = (
        None if args.bounds is None
        else np.asarray(args.bounds, np.float64).reshape(2, 3)
    )
    print(prepare_visual_hull_points(
        args.config, args.output, bounds,
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
    ))
