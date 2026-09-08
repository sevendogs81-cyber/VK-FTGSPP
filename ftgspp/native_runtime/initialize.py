"""NumPy/SciPy port of FTGSPP's explicit Gaussian initialization."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ftgspp.native_runtime.dataset import NativeDataset
from ftgspp.native_runtime.model import NativeGaussians, TEMPORAL_SCALE_MULTIPLIER
from ftgspp.native_runtime.ply import read_point_ply

SH_C0 = 0.28209479177387814
TEMPORAL_OUTLIER_DISP_ALPHA = 3.0
DEFAULT_FLOW_WORKERS = 8
MAX_FLOW_WORKERS = 16
logger = logging.getLogger("ftgspp.native")


def _logit(value: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).eps
    value = np.clip(value, eps, 1.0-eps)
    return np.log(value / (1.0-value))


def _spacing(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.ones(1, dtype=np.float32) * 1e-3
    k = min(3, len(points))
    distances, _ = cKDTree(points).query(points, k=k)
    if distances.ndim == 1:
        distances = distances[:, None]
    return np.sqrt(np.maximum(np.mean(distances**2, axis=1), 1e-7)).astype(np.float32)


def _velocity(query: np.ndarray, reference: np.ndarray, delta_time: float, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(query)
    k = min(k, len(reference))
    _distances, indices = cKDTree(reference).query(query, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]
    return ((reference[indices] - query[:, None]) / delta_time).mean(axis=1).astype(np.float32)


def _select_point_indices(
    rng: np.random.Generator,
    population_size: int,
    budget: int,
) -> np.ndarray:
    """Select source points without replacement, preserving all points that fit."""
    population_size = int(population_size)
    budget = int(budget)
    if population_size <= 0:
        raise ValueError("point cloud must contain at least one point")
    if budget <= 0:
        raise ValueError("point budget must be greater than zero")
    if population_size <= budget:
        return np.arange(population_size, dtype=np.int64)
    return rng.choice(population_size, size=budget, replace=False)


def _sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    if not len(uv):
        return np.empty((0, image.shape[-1]), np.float32)
    sampled = cv2.remap(
        np.asarray(image, np.float32), uv[:, 0].astype(np.float32).reshape(-1, 1),
        uv[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(len(uv), -1)


def _project(xyz: np.ndarray, w2c: np.ndarray, intrinsic: np.ndarray, size: tuple[int, int]):
    xyz_camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    depth = xyz_camera[:, 2]
    homogeneous = xyz_camera @ intrinsic.T
    uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-7)
    height, width = size
    valid = (depth > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, depth, valid


def _backproject(uv: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    uv_h = np.concatenate([uv, np.ones((len(uv), 1), np.float32)], axis=1)
    xyz_camera = (uv_h @ np.linalg.inv(intrinsic).T) * depth[:, None]
    c2w = np.linalg.inv(w2c)
    return xyz_camera @ c2w[:3, :3].T + c2w[:3, 3]


def _flow_worker_count(num_cameras: int) -> int:
    raw = os.environ.get("FTGSPP_INIT_FLOW_WORKERS", str(DEFAULT_FLOW_WORKERS))
    try:
        requested = int(raw)
    except ValueError:
        requested = DEFAULT_FLOW_WORKERS
    return max(1, min(requested, int(num_cameras), MAX_FLOW_WORKERS))


def _load_flow_npz(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache, allow_pickle=False) as archive:
        flow = np.asarray(archive["flow"], np.float32)
        covis = np.asarray(archive["covis"], np.float32).squeeze()
    if flow.ndim != 3 or flow.shape[-1] != 2 or covis.ndim != 2:
        raise ValueError(f"invalid UFM cache shapes in {cache}: flow={flow.shape}, covis={covis.shape}")
    return flow, covis


def _resize_flow(
    flow: np.ndarray, covis: np.ndarray, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    if flow.shape[:2] == (height, width):
        return flow, covis
    old_h, old_w = flow.shape[:2]
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= width / max(old_w, 1)
    flow[..., 1] *= height / max(old_h, 1)
    covis = cv2.resize(covis, (width, height), interpolation=cv2.INTER_LINEAR)
    return flow, covis


def _camera_ufm_contribution(
    query: np.ndarray,
    dataset: NativeDataset,
    frame_0: int,
    frame_1: int,
    delta_time: float,
    cache: Path,
    camera_id: int,
    covis_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not cache.is_file():
        raise FileNotFoundError(cache)
    view_0, view_1 = dataset.view(frame_0, camera_id), dataset.view(frame_1, camera_id)
    height, width = view_0.size
    uv, depth, valid = _project(query, view_0.w2c, view_0.intrinsic, view_0.size)
    valid_ids = np.flatnonzero(valid)
    if not len(valid_ids):
        return None
    flow, covis = _resize_flow(*_load_flow_npz(cache), width, height)
    source = uv[valid_ids]
    source_depth = depth[valid_ids]
    destination = source + _sample(flow, source)
    covis_source = _sample(covis, source)[:, 0]
    keep = (
        (covis_source >= covis_threshold)
        & (destination[:, 0] >= 0) & (destination[:, 0] < width)
        & (destination[:, 1] >= 0) & (destination[:, 1] < height)
    )
    if not np.any(keep):
        return None
    ids = valid_ids[keep]
    source, destination, source_depth = source[keep], destination[keep], source_depth[keep]
    xyz_source = _backproject(source, source_depth, view_0.intrinsic, view_0.w2c)
    xyz_destination = _backproject(destination, source_depth, view_1.intrinsic, view_1.w2c)
    velocity = (xyz_destination-xyz_source) / delta_time
    weight = np.maximum(_sample(covis, source)[:, 0], 1e-6)
    return ids, velocity, weight


def _reduce_ufm_contributions(
    query: np.ndarray,
    contributions: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None],
) -> tuple[np.ndarray, np.ndarray]:
    velocity_sum = np.zeros_like(query, np.float32)
    weight_sum = np.zeros(len(query), np.float32)
    for item in contributions:
        if item is None:
            continue
        ids, velocity, weight = item
        velocity_sum[ids] += velocity * weight[:, None]
        weight_sum[ids] += weight
    valid = weight_sum > 0
    result = np.zeros_like(query, np.float32)
    result[valid] = velocity_sum[valid] / weight_sum[valid, None]
    return result, valid


def _temporal_ufm_velocity(
    query: np.ndarray,
    dataset: NativeDataset,
    frame_0: int,
    frame_1: int,
    delta_time: float,
    cache_root: Path,
    covis_threshold: float,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pair = cache_root / f"f{frame_0:06d}_f{frame_1:06d}"
    if not pair.is_dir():
        raise FileNotFoundError(f"flow pair directory not found: {pair}")
    jobs = [
        (pair / f"c{camera_index:03d}.npz", camera_id)
        for camera_index, camera_id in enumerate(dataset.cameras)
    ]
    if executor is None or len(jobs) <= 1:
        contributions = [
            _camera_ufm_contribution(
                query, dataset, frame_0, frame_1, delta_time,
                cache, camera_id, covis_threshold,
            )
            for cache, camera_id in jobs
        ]
    else:
        futures = [
            executor.submit(
                _camera_ufm_contribution,
                query, dataset, frame_0, frame_1, delta_time,
                cache, camera_id, covis_threshold,
            )
            for cache, camera_id in jobs
        ]
        contributions = [future.result() for future in futures]
    return _reduce_ufm_contributions(query, contributions)


def initialize_explicit(config, dataset: NativeDataset) -> NativeGaussians:
    if config.init.temporal_motion_adapted and not Path(config.init.temporal_flow_path).is_dir():
        raise FileNotFoundError(config.init.temporal_flow_path)
    frame_positions = list(range(0, len(dataset.frames), config.init.keyframe_stride))
    if frame_positions[-1] != len(dataset.frames)-1:
        frame_positions.append(len(dataset.frames)-1)
    point_paths = [Path(config.init.points_path) / f"f{dataset.frames[index]:06d}.ply" for index in frame_positions]
    points = [read_point_ply(path) for path in point_paths]
    per_frame = max(1, int(config.init.num_gaussians) // len(points))
    selected_counts = [min(per_frame, len(xyz)) for xyz, _rgb in points]
    logger.info(
        "Gaussian init: keyframes=%d target=%d per_keyframe_budget=%d "
        "available=%d selected=%d full_clouds=%d subsampled_clouds=%d "
        "sampling=without_replacement",
        len(points), int(config.init.num_gaussians), per_frame,
        sum(len(xyz) for xyz, _rgb in points), sum(selected_counts),
        sum(len(xyz) <= per_frame for xyz, _rgb in points),
        sum(len(xyz) > per_frame for xyz, _rgb in points),
    )
    rng = np.random.default_rng(config.train.seed)
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in (
        "means", "log_scales", "quaternions", "opacity_logits", "sh_coefficients",
        "times", "duration_raw", "velocities", "gate_raw",
    )}
    flow_workers = (
        _flow_worker_count(len(dataset.cameras))
        if config.init.temporal_motion_adapted else 1
    )
    previous_cv_threads = cv2.getNumThreads()
    executor: ThreadPoolExecutor | None = None
    if config.init.temporal_motion_adapted:
        logger.info(
            "UFM velocity init: %d keyframes, %d cameras, flow_workers=%d "
            "(FTGSPP_INIT_FLOW_WORKERS)",
            len(points), len(dataset.cameras), flow_workers,
        )
        cv2.setNumThreads(1)
        if flow_workers > 1:
            executor = ThreadPoolExecutor(max_workers=flow_workers)
    try:
        _initialize_explicit_frames(
            config, dataset, points, frame_positions, per_frame, rng, chunks,
            executor,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if config.init.temporal_motion_adapted:
            cv2.setNumThreads(previous_cv_threads)
    values = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    return NativeGaussians(**values, max_duration=float(config.model.max_duration))


def _initialize_explicit_frames(
    config, dataset: NativeDataset, points, frame_positions, per_frame, rng, chunks,
    executor: ThreadPoolExecutor | None,
) -> None:
    flow_path = Path(config.init.temporal_flow_path)
    covis_threshold = float(config.init.temporal_covis_thresh)
    for point_index, (xyz, rgb) in enumerate(points):
        other_index = point_index+1 if point_index+1 < len(points) else point_index-1
        ref_xyz = points[other_index][0]
        frame = dataset.frames[frame_positions[point_index]]
        other_frame = dataset.frames[frame_positions[other_index]]
        time = dataset.view(frame, dataset.cameras[0]).timestamp
        other_time = dataset.view(other_frame, dataset.cameras[0]).timestamp
        delta_time = other_time-time
        selection = _select_point_indices(rng, len(xyz), per_frame)
        n = len(selection)
        means = xyz[selection].astype(np.float32, copy=True)
        times = (rng.random((n, 1), dtype=np.float32)-0.5) * delta_time + time
        velocity = _velocity(means, ref_xyz, delta_time, config.init.num_velocity_nns)
        spacing = _spacing(means)
        if config.init.temporal_motion_adapted:
            if point_index == 0 or (point_index + 1) % 10 == 0 or point_index + 1 == len(points):
                logger.info(
                    "UFM velocity init keyframe %d/%d (f%06d -> f%06d)",
                    point_index + 1, len(points), frame, other_frame,
                )
            temporal, valid = _temporal_ufm_velocity(
                means, dataset, frame, other_frame, delta_time,
                flow_path, covis_threshold, executor=executor,
            )
            displacement = np.linalg.norm(temporal, axis=1) * abs(delta_time)
            override = valid & (displacement <= TEMPORAL_OUTLIER_DISP_ALPHA*spacing)
            velocity[override] = temporal[override]
        chunks["means"].append(means)
        chunks["log_scales"].append(np.log(np.maximum(config.init.scale * spacing[:, None], 1e-12)).repeat(3, axis=1))
        quaternion = np.zeros((n, 4), np.float32)
        quaternion[:, 0] = 1.0
        chunks["quaternions"].append(quaternion)
        chunks["opacity_logits"].append(_logit(np.full((n, 1), config.init.opacity, np.float32)))
        sh = np.zeros((n, 16, 3), np.float32)
        sh[:, 0] = (rgb[selection].astype(np.float32) / 255.0 - 0.5) / SH_C0
        chunks["sh_coefficients"].append(sh)
        chunks["times"].append(times.astype(np.float32))
        duration = np.full((n, 1), config.init.duration, np.float32)
        if np.isinf(config.model.max_duration):
            duration = np.log(np.maximum(duration, np.finfo(np.float32).tiny))
        else:
            duration = _logit(duration / (config.model.max_duration * TEMPORAL_SCALE_MULTIPLIER))
        chunks["duration_raw"].append(duration)
        chunks["velocities"].append(velocity)
        gate = -1.0 if config.model.marginal_gating else -100.0
        chunks["gate_raw"].append(np.full((n, 1), gate, np.float32))
