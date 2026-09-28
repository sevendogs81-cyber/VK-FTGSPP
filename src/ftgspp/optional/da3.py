"""Optional Depth Anything 3 point-cloud initialization.

The core FTGSPP package intentionally remains Torch-free.  This module is
imported only when ``init.backend`` selects DA3 (or ``auto`` cannot find a
complete user point cloud), and imports the upstream DA3 package lazily.

DA3 is used as a geometry initializer, not as a second training renderer: the
known FTGSPP camera poses/intrinsics are supplied to DA3, its depth maps are
unprojected into the existing world coordinate system, and RGB is sampled from
the same processed images.  This keeps the native model and checkpoint ABI
unchanged while making the generated cloud interchangeable with a user PLY.

Per-keyframe views are fused in world space (voxel occupancy) before any cap.
``init.backend: auto`` uses a complete user RGB PLY set when present. Otherwise
DA3 sizes itself from ``num_gaussians``, keyframe stride, the frame span, and
native resolution (``quality: auto``). Explicit ``process_res`` / ``max_views`` /
``max_points`` still override. Gaussian means are sampled later from the dense
cache; 3-NN velocity queries that cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ftgspp.data.scene import Scene, View
from ftgspp.io.init_cache import (
    DA3_SCHEMA,
    namespace_lock,
    publish_cache_pointer,
    published_cache_candidates,
    resolve_published_cache_root,
    write_json_atomic,
)
from ftgspp.io.ply import read_point_ply, write_point_ply_atomic
from ftgspp.optional.da3_evidence import target_depth_is_trusted
from ftgspp.optional.torch_compute import available_torch_device, resolve_torch_device
from ftgspp.optional.da3_schedule import (
    foreground_schedule_metadata,
    inference_pixel_cost,
    input_camera_ids,
    pre_fuse_per_view_cap,
    resolve_da3_schedule_for_scene,
    select_representative_camera_ids,
)
from ftgspp.training.init_plan import (
    bind_schedule_to_options,
    effective_da3_plan,
    initialization_seed,
    l1_identity_payload,
    l2_identity_payload,
    namespace_hash,
)
from ftgspp.training.point_sampling import (
    auto_voxel_size,
    select_point_indices,
    voxel_fuse_points,
)
from ftgspp.training.stage_timer import StageTimer

logger = logging.getLogger("ftgspp.native")


@dataclass
class _PredictionBundle:
    clouds: list[tuple[np.ndarray, np.ndarray]]
    depths: list[np.ndarray]
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    masks: list[np.ndarray]
    validities: list[np.ndarray]
    transforms: list[np.ndarray]
    camera_ids: list[int] | None = None


class DA3UnavailableError(RuntimeError):
    """Raised when DA3 initialization was requested but its optional stack is absent."""


class DA3PointClouds(list[tuple[np.ndarray, np.ndarray]]):
    """List-compatible point output bound to the exact published cache."""

    def __init__(
        self, points: Iterable[tuple[np.ndarray, np.ndarray]],
        *, cache_root: Path, generated: bool,
    ) -> None:
        super().__init__(points)
        self.cache_root = Path(cache_root)
        self.generated = bool(generated)


def model_identity(options: Any) -> dict[str, object]:
    """Return a cache identity for the selected DA3 model source."""

    model_name = str(getattr(options, "model_name", ""))
    model_path = getattr(options, "model_path", None)
    if model_path is None:
        return {"name": model_name, "path": None, "sha256": None}
    path = Path(model_path).expanduser()
    if not path.exists():
        return {"name": model_name, "path": str(path), "sha256": "missing"}
    if path.is_file():
        return {"name": model_name, "path": str(path.resolve()), "sha256": _sha256(path)}
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {
            ".safetensors", ".bin", ".pt", ".pth", ".json",
        }
    )
    entries = {
        item.relative_to(path).as_posix(): _sha256(item) for item in files
    }
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "name": model_name,
        "path": str(path.resolve()),
        "files": entries,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _loaded_model_identity(
    model: Any, options: Any, *, hash_state_dict: bool = True,
) -> dict[str, object]:
    """Bind a prepared cache to the actual tensors loaded for inference.

    A local model path is already content-hashed by :func:`model_identity`, and
    a Hugging Face model with a resolved commit is immutable for cache
    purposes.  The expensive tensor-to-CPU hash is retained as a fallback for
    ad-hoc model objects whose source cannot be identified.
    """

    identity = model_identity(options)
    config = getattr(model, "config", None)
    identity.update({
        "resolved_name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "loaded_state_dict_sha256": None,
    })
    source_hash = identity.get("sha256")
    source_verified = (
        isinstance(source_hash, str)
        and source_hash not in {"", "missing"}
    ) or bool(identity.get("commit_hash"))
    if not hash_state_dict:
        identity["loaded_state_dict_status"] = (
            "source_verified" if source_verified else "skipped"
        )
        return identity
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        identity["loaded_state_dict_status"] = "unavailable"
        return identity
    digest = hashlib.sha256()
    state = state_dict()
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach"):
            raise RuntimeError(f"DA3 state_dict entry {name!r} is not a tensor")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(name).encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
    identity["loaded_state_dict_sha256"] = digest.hexdigest()
    identity["loaded_state_dict_status"] = "hashed"
    return identity


def _target_extent(options: Any) -> float | None:
    raw = getattr(options, "target_bounds", None)
    if raw is None:
        return None
    bounds = np.asarray(raw, np.float64)
    if bounds.shape == (6,):
        bounds = bounds.reshape(2, 3)
    if (
        bounds.shape != (2, 3) or not np.isfinite(bounds).all()
        or np.any(bounds[1] <= bounds[0])
    ):
        raise ValueError(
            "init.da3.target_bounds must be finite "
            "[xmin,ymin,zmin,xmax,ymax,zmax]"
        )
    return float(np.max(bounds[1] - bounds[0]))


def _validate_prediction_depth_type(prediction: Any, options: Any) -> str:
    configured = str(getattr(options, "depth_type", "camera_z"))
    if configured != "camera_z":
        raise ValueError("DA3 backprojection supports only depth_type=camera_z")
    declared = getattr(prediction, "depth_type", None)
    metadata = getattr(prediction, "metadata", None)
    if declared is None and isinstance(metadata, dict):
        declared = metadata.get("depth_type")
    if declared is not None and str(declared).lower() not in {
        "camera_z", "camera-z", "z", "z_depth",
    }:
        raise RuntimeError(
            "DA3 prediction depth_type is incompatible with camera-z "
            f"backprojection: {declared!r}"
        )
    return "camera_z"


def _sha256(path: Path) -> str:
    from ftgspp.io.init_cache import file_sha256
    return file_sha256(path)


def _load_model(options: Any, *, fallback_gpu: int):
    """Import and construct DA3 without making Torch a core dependency."""

    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise DA3UnavailableError(
            "DA3 initialization requires the upstream depth-anything-3 package "
            "and its Torch dependencies. Install it in the active environment "
            "(the upstream package currently has its own dependency constraints) "
            "or use init.backend=points with a user point cloud."
        ) from exc

    device_name = str(getattr(options, "device", "auto"))
    if device_name == "auto":
        if torch.cuda.is_available():
            # Vulkan and CUDA device ordinals are not guaranteed to match.
            # Leave CUDA selection to the runtime (or let users set an
            # explicit ``init.da3.device``) instead of guessing from run.gpu.
            device_name = "cuda"
        else:
            device_name = "cpu"
    model_source = getattr(options, "model_path", None)
    if model_source:
        model_source = str(Path(model_source).expanduser())
    else:
        model_source = str(getattr(options, "model_name", "depth-anything/DA3-BASE"))
    logger.info(
        "DA3 initialization: loading %s on %s (process_res=%d, max_views=%d)",
        model_source,
        device_name,
        int(getattr(options, "process_res", 504)),
        int(getattr(options, "max_views", 8)),
    )
    if str(device_name).startswith("cuda") and torch.cuda.is_available():
        try:
            # Peak-memory telemetry is diagnostic only. Some pre-release Torch
            # builds reject an explicit device after DA3 imports even though
            # allocator statistics for the current CUDA device are available.
            with torch.cuda.device(device_name):
                torch.cuda.reset_peak_memory_stats()
        except (RuntimeError, ValueError) as exc:
            logger.warning("unable to reset CUDA peak-memory telemetry: %s", exc)
    try:
        model = DepthAnything3.from_pretrained(model_source)
        model = model.to(device_name)
        model.eval()
    except Exception as exc:  # pragma: no cover - model/network dependent
        raise RuntimeError(
            f"failed to load DA3 model {model_source!r} on {device_name!r}; "
            "set init.da3.model_path to a local checkpoint for offline runs"
        ) from exc
    return model, device_name


def _camera_chunks(
    camera_ids: Sequence[int],
    max_views: int,
    *,
    anchor_count: int = 1,
    minimum_chunk_size: int = 1,
) -> list[list[int]]:
    """Split cameras while retaining anchors and avoiding undersized tails.

    Pose-conditioned DA3 inference uses Umeyama alignment internally and needs
    at least three camera centers.  When requested, move cameras out of earlier
    chunks to make the tail large enough without increasing the forward batch.
    Only configurations with too few unique cameras to rebalance use repeated
    anchors; callers must then discard their duplicate predictions.
    """

    cameras = [int(value) for value in camera_ids]
    if not cameras:
        raise ValueError("DA3 initialization requires at least one camera")
    max_views = max(1, int(max_views))
    minimum_chunk_size = max(1, int(minimum_chunk_size))
    if len(cameras) <= max_views:
        return [cameras]
    anchor_count = max(0, min(int(anchor_count), len(cameras)))
    if anchor_count >= max_views:
        raise ValueError(
            "init.da3.max_views must exceed the number of alignment anchors "
            f"({anchor_count}) when the camera rig is split"
        )
    anchors = cameras[:anchor_count]
    remainder = cameras[anchor_count:]
    width = max_views - anchor_count
    chunks = [
        [*anchors, *remainder[offset:offset + width]]
        for offset in range(0, len(remainder), width)
    ]
    if len(chunks[-1]) >= minimum_chunk_size:
        return chunks
    if anchor_count:
        raise ValueError(
            "alignment anchors do not provide the requested minimum DA3 "
            f"chunk size ({minimum_chunk_size})"
        )
    if max_views < minimum_chunk_size:
        raise ValueError(
            "init.da3.max_views must be at least "
            f"{minimum_chunk_size} for pose-conditioned DA3 inference"
        )

    deficit = minimum_chunk_size - len(chunks[-1])
    moved: list[int] = []
    for chunk in reversed(chunks[:-1]):
        transfer = min(deficit, max(0, len(chunk) - minimum_chunk_size))
        if transfer:
            moved.extend(chunk[-transfer:])
            del chunk[-transfer:]
            deficit -= transfer
        if deficit == 0:
            break
    chunks[-1][:0] = sorted(moved, key=cameras.index)

    if deficit:
        tail = set(chunks[-1])
        padding = [camera for camera in cameras if camera not in tail][:deficit]
        if len(padding) != deficit:
            raise ValueError(
                "not enough distinct cameras to construct a valid DA3 chunk"
            )
        chunks[-1][:0] = padding
    return chunks


def _as_homogeneous(ext: np.ndarray) -> np.ndarray:
    ext = np.asarray(ext, dtype=np.float32)
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3] = ext
        return result
    raise ValueError(f"expected a 3x4 or 4x4 world-to-camera matrix, got {ext.shape}")


def _stratified_valid_pixels(
    valid: np.ndarray,
    *,
    max_points: int | None,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select spatially distributed valid pixels before 3D unprojection."""

    ys, xs = np.nonzero(valid)
    if max_points is None or len(xs) <= int(max_points):
        return ys, xs
    budget = max(1, int(max_points))
    generator = rng if rng is not None else np.random.default_rng(0)
    height, width = valid.shape
    rows = max(1, min(height, int(np.sqrt(budget * height / max(width, 1)))))
    cols = max(1, min(width, budget // rows))
    cells = (
        (ys.astype(np.int64) * rows // height) * cols
        + xs.astype(np.int64) * cols // width
    )
    order = generator.permutation(len(xs))
    _unique, first = np.unique(cells[order], return_index=True)
    chosen = order[first]
    if len(chosen) > budget:
        chosen = generator.choice(chosen, size=budget, replace=False)
    elif len(chosen) < budget:
        used = np.zeros(len(xs), dtype=bool)
        used[chosen] = True
        remainder = np.flatnonzero(~used)
        extra = generator.choice(
            remainder, size=budget - len(chosen), replace=False,
        )
        chosen = np.concatenate((chosen, extra))
    return ys[chosen], xs[chosen]


def depth_to_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    w2c: np.ndarray,
    rgb: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    sky: np.ndarray | None = None,
    confidence_percentile: float = 20.0,
    foreground_mask: np.ndarray | None = None,
    validity: np.ndarray | None = None,
    mask_threshold: float = 0.5,
    max_points: int | None = None,
    rng: np.random.Generator | None = None,
    return_selection: bool = False,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Convert one DA3 depth/RGB view into a world-space cloud.

    Valid pixels are confidence/mask filtered, then optionally stratified in
    image space before the matrix-heavy unprojection. Multi-view fusion and
    the final dense-cloud cap happen later. Pixel centers use DA3's processed
    intrinsic matrix. A confidence percentile of 20 keeps the top 80 percent.
    """

    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    rgb = np.asarray(rgb)
    if depth.ndim != 2:
        raise ValueError(f"DA3 depth must be HxW, got {depth.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"DA3 intrinsic must be 3x3, got {intrinsic.shape}")
    if rgb.ndim != 3 or rgb.shape[:2] != depth.shape or rgb.shape[2] < 3:
        raise ValueError(
            f"DA3 RGB must be HxWx3/4 matching depth, got {rgb.shape} vs {depth.shape}"
        )
    if confidence is not None and np.asarray(confidence).shape != depth.shape:
        raise ValueError("DA3 confidence map shape does not match depth")
    if sky is not None and np.asarray(sky).shape != depth.shape:
        raise ValueError("DA3 sky mask shape does not match depth")
    if foreground_mask is not None and np.asarray(foreground_mask).shape != depth.shape:
        raise ValueError("DA3 foreground mask shape does not match depth")
    if validity is not None and np.asarray(validity).shape != depth.shape:
        raise ValueError("DA3 validity mask shape does not match depth")
    if not 0.0 <= float(confidence_percentile) <= 100.0:
        raise ValueError("DA3 confidence_percentile must be in [0, 100]")
    if max_points is not None and int(max_points) <= 0:
        raise ValueError("max_points must be positive when set")

    valid = np.isfinite(depth) & (depth > 1e-6)
    if foreground_mask is not None:
        foreground = np.asarray(foreground_mask, dtype=np.float32)
        valid &= np.isfinite(foreground) & (foreground >= float(mask_threshold))
    if validity is not None:
        valid &= np.asarray(validity) >= 0.5
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        finite_conf = np.isfinite(confidence)
        valid &= finite_conf
        if np.any(valid) and confidence_percentile > 0.0:
            threshold = np.percentile(confidence[valid], confidence_percentile)
            valid &= confidence >= threshold
    if sky is not None:
        valid &= ~np.asarray(sky, dtype=bool)
    ys, xs = _stratified_valid_pixels(
        valid, max_points=max_points, rng=rng,
    )
    if not len(xs):
        empty_xyz = np.empty((0, 3), np.float32)
        empty_rgb = np.empty((0, 3), np.uint8)
        if return_selection:
            return empty_xyz, empty_rgb, {
                "ys": np.empty(0, np.int64),
                "xs": np.empty(0, np.int64),
            }
        return empty_xyz, empty_rgb
    resolved_device, torch = (
        available_torch_device(device)
        if str(device).startswith("cuda") else
        resolve_torch_device(device, purpose="DA3 unprojection")
    )
    uv = np.stack(
        (xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5), axis=1,
    )
    if resolved_device.startswith("cuda"):
        with torch.inference_mode():
            uv_gpu = torch.as_tensor(uv, dtype=torch.float32, device=resolved_device)
            uv_h_gpu = torch.cat((
                uv_gpu,
                torch.ones((len(uv_gpu), 1), dtype=torch.float32, device=resolved_device),
            ), dim=1)
            intrinsic_gpu = torch.as_tensor(
                intrinsic, dtype=torch.float32, device=resolved_device,
            )
            selected_depth = torch.as_tensor(
                np.ascontiguousarray(depth[ys, xs]),
                dtype=torch.float32, device=resolved_device,
            )
            camera_xyz_gpu = (
                uv_h_gpu @ torch.linalg.inv(intrinsic_gpu).T
            ) * selected_depth[:, None]
            c2w_gpu = torch.linalg.inv(torch.as_tensor(
                _as_homogeneous(w2c), dtype=torch.float32, device=resolved_device,
            ))
            world_xyz = (
                camera_xyz_gpu @ c2w_gpu[:3, :3].T + c2w_gpu[:3, 3]
            ).cpu().numpy()
    else:
        uv_h = np.concatenate((uv, np.ones((len(uv), 1), np.float32)), axis=1)
        camera_xyz = (uv_h @ np.linalg.inv(intrinsic).T) * depth[ys, xs, None]
        c2w = np.linalg.inv(_as_homogeneous(w2c))
        world_xyz = camera_xyz @ c2w[:3, :3].T + c2w[:3, 3]
    colors = np.asarray(rgb[ys, xs, :3], dtype=np.uint8)
    finite = np.isfinite(world_xyz).all(axis=1)
    if return_selection:
        return (
            world_xyz[finite].astype(np.float32),
            colors[finite],
            {
                "ys": ys[finite],
                "xs": xs[finite],
                "device": resolved_device,
            },
        )
    return world_xyz[finite].astype(np.float32), colors[finite]


def _processed_scalar_map(
    value: np.ndarray,
    target_size: tuple[int, int],
    method: str,
    *,
    interpolation: int,
    source_to_processed: np.ndarray | None = None,
) -> np.ndarray:
    """Replay DA3's declared resize/crop geometry for a scalar side input."""

    import cv2

    source = np.asarray(value, np.float32)
    target_h, target_w = (int(item) for item in target_size)
    if source.shape == (target_h, target_w) and source_to_processed is None:
        return np.ascontiguousarray(source)
    if source_to_processed is not None:
        matrix = np.asarray(source_to_processed, np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("DA3 source_to_processed transform must be a finite 3x3 matrix")
        result = cv2.warpPerspective(
            source, matrix, (target_w, target_h), flags=int(interpolation),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    elif str(method).endswith("_crop"):
        raise RuntimeError(
            "DA3 crop preprocessing requires the model prediction to expose an "
            "exact source_to_processed 3x3 transform; refusing to guess crop offsets"
        )
    else:
        # The *_resize APIs have no crop/pad translation. Their exact scale is
        # fully determined by the returned processed raster dimensions.
        result = cv2.resize(source, (target_w, target_h), interpolation=interpolation)
    if result.shape != (target_h, target_w):
        raise RuntimeError(
            f"DA3 scalar transform produced {result.shape}, expected {(target_h, target_w)}"
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def _prediction_transform_matrix(
    prediction: Any,
    index: int,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    method: str,
    *,
    convention: str = "source_index_scale",
    metadata_present: list[bool] | None = None,
) -> np.ndarray:
    """Return the exact source-pixel to processed-pixel transform.

    Newer DA3 adapters can expose ``source_to_processed`` (or one of the
    compatibility attribute names below) as N×3×3 matrices or per-view dicts.
    Crop/pad methods must provide it. Pure resize remains exactly derivable from
    the source and returned target raster sizes.

    ``convention`` records how that matrix is interpreted. ``pixel_center`` is
    only applied when requested; it is not inferred from output sizes alone.
    """

    raw: Any = None
    for attribute in (
        "source_to_processed", "image_transforms", "processed_transforms",
        "preprocess_transforms",
    ):
        candidate = getattr(prediction, attribute, None)
        if candidate is not None:
            raw = candidate
            break
    metadata = getattr(prediction, "metadata", None)
    if raw is None and isinstance(metadata, dict):
        raw = next((metadata.get(name) for name in (
            "source_to_processed", "image_transforms", "processed_transforms",
        ) if metadata.get(name) is not None), None)
    if raw is not None:
        if isinstance(raw, dict):
            item = raw.get(index, raw.get(str(index), raw))
        else:
            array = np.asarray(raw) if not isinstance(raw, (list, tuple)) else None
            if array is not None and array.shape == (3, 3):
                item = array
            else:
                item = raw[index]
        if isinstance(item, dict):
            declared_source = item.get("source_size")
            declared_target = item.get("target_size", item.get("processed_size"))
            if declared_source is not None and tuple(map(int, declared_source)) != tuple(source_size):
                raise RuntimeError("DA3 transform source_size does not match its input view")
            if declared_target is not None and tuple(map(int, declared_target)) != tuple(target_size):
                raise RuntimeError("DA3 transform target_size does not match its depth raster")
            item = next((item.get(name) for name in (
                "source_to_processed", "matrix", "affine",
            ) if item.get(name) is not None), None)
        matrix = np.asarray(item, np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack((matrix, np.asarray([0.0, 0.0, 1.0])))
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise RuntimeError("DA3 prediction contains an invalid processed-image transform")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise RuntimeError("DA3 processed-image transform is singular")
        if metadata_present is not None:
            metadata_present.append(True)
        if str(convention) == "pixel_center":
            return pixel_center_to_index_matrix(matrix)
        return matrix
    if metadata_present is not None:
        metadata_present.append(False)
    if str(method).endswith("_crop"):
        raise RuntimeError(
            "DA3 crop preprocessing did not return source_to_processed metadata; "
            "use a resize method or update the DA3 adapter"
        )
    source_h, source_w = map(int, source_size)
    target_h, target_w = map(int, target_size)
    if str(convention) == "pixel_center":
        return np.asarray([
            [target_w / source_w, 0.0, 0.5 * (target_w / source_w - 1.0)],
            [0.0, target_h / source_h, 0.5 * (target_h / source_h - 1.0)],
            [0.0, 0.0, 1.0],
        ], np.float64)
    return np.asarray([
        [target_w / source_w, 0.0, 0.0],
        [0.0, target_h / source_h, 0.0],
        [0.0, 0.0, 1.0],
    ], np.float64)


def _intrinsic_to_source(
    processed_intrinsic: np.ndarray, source_to_processed: np.ndarray,
) -> np.ndarray:
    result = np.linalg.inv(np.asarray(source_to_processed, np.float64)) @ np.asarray(
        processed_intrinsic, np.float64,
    )
    result /= result[2, 2]
    if not np.isfinite(result).all() or result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise RuntimeError("DA3 transform produced an invalid source-space intrinsic")
    return np.ascontiguousarray(result, dtype=np.float32)


def _geometric_consistency_filter_cuda(
    xyz: np.ndarray,
    *,
    source_index: int,
    depths: Sequence[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    foreground_masks: Sequence[np.ndarray],
    validities: Sequence[np.ndarray],
    min_consistent_views: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    return_diagnostics: bool,
    mask_threshold: float,
    torch,
    device: str,
    prepared: dict[str, object] | None = None,
):
    """CUDA projection and depth-evidence accumulation for DA3 clouds."""

    with torch.inference_mode():
        points = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        supports = torch.ones(len(points), dtype=torch.int16, device=device)
        conflicts = torch.zeros(len(points), dtype=torch.int16, device=device)
        residuals = []
        evidence_count = 0
        occluded_count = 0
        unobservable_count = 0
        prepared = {} if prepared is None else prepared
        extrinsics_gpu = prepared.get("extrinsics")
        if extrinsics_gpu is None:
            extrinsics_gpu = torch.as_tensor(
                extrinsics, dtype=torch.float32, device=device,
            )
        intrinsics_gpu = prepared.get("intrinsics")
        if intrinsics_gpu is None:
            intrinsics_gpu = torch.as_tensor(
                intrinsics, dtype=torch.float32, device=device,
            )
        depth_tensors = prepared.get("depths")
        validity_tensors = prepared.get("validities")
        foreground_tensors = prepared.get("foreground_masks")
        for target in range(len(depths)):
            if target == int(source_index):
                continue
            transform = extrinsics_gpu[target]
            camera = points @ transform[:3, :3].T + transform[:3, 3]
            z = camera[:, 2]
            k = intrinsics_gpu[target]
            x = torch.round(
                k[0, 0] * camera[:, 0] / z.clamp_min(1e-12)
                + k[0, 2] - 0.5
            ).long()
            y = torch.round(
                k[1, 1] * camera[:, 1] / z.clamp_min(1e-12)
                + k[1, 2] - 0.5
            ).long()
            height, width = np.asarray(depths[target]).shape
            inside = (
                (z > 1e-8) & (x >= 0) & (x < width)
                & (y >= 0) & (y < height)
            )
            ids = torch.nonzero(inside, as_tuple=False).flatten()
            unobservable_count += int(len(points) - ids.numel())
            if ids.numel() == 0:
                continue
            depth_gpu = (
                depth_tensors[target] if depth_tensors is not None else
                torch.as_tensor(
                    np.ascontiguousarray(depths[target]),
                    dtype=torch.float32, device=device,
                )
            )
            validity_gpu = (
                validity_tensors[target] if validity_tensors is not None else
                torch.as_tensor(
                    np.ascontiguousarray(validities[target]), device=device,
                )
            )
            foreground_gpu = (
                foreground_tensors[target] if foreground_tensors is not None else
                torch.as_tensor(
                    np.ascontiguousarray(foreground_masks[target]),
                    dtype=torch.float32, device=device,
                )
            )
            observed = depth_gpu[y[ids], x[ids]]
            trusted = (
                (validity_gpu[y[ids], x[ids]] >= 0.5)
                & (foreground_gpu[y[ids], x[ids]] >= float(mask_threshold))
                & torch.isfinite(observed) & (observed > 1e-8)
            )
            if not bool(torch.any(trusted)):
                continue
            ids = ids[trusted]
            observed = observed[trusted]
            tolerance = torch.maximum(
                float(relative_tolerance) * observed,
                torch.full_like(observed, float(absolute_tolerance)),
            )
            delta = z[ids] - observed
            residuals.append(torch.abs(delta) / observed.clamp_min(1e-12))
            evidence_count += ids.numel()
            agree = torch.abs(delta) <= tolerance
            behind = delta > tolerance
            ahead = delta < -tolerance
            supports[ids[agree]] += 1
            conflicts[ids[ahead]] += 1
            occluded_count += int(torch.count_nonzero(behind).item())
        keep = (supports >= int(min_consistent_views)) & (conflicts == 0)
        if not return_diagnostics:
            return (
                keep.cpu().numpy(), supports.cpu().numpy(), conflicts.cpu().numpy(),
            )
        if residuals:
            all_rel = torch.cat(residuals)
            quantiles = torch.quantile(
                all_rel, torch.tensor([0.5, 0.9, 0.99], device=device),
            )
            residual_report = {
                "count": int(all_rel.numel()),
                "p50": float(quantiles[0].item()),
                "p90": float(quantiles[1].item()),
                "p99": float(quantiles[2].item()),
                "max": float(torch.max(all_rel).item()),
            }
        else:
            residual_report = {
                "count": 0, "p50": None, "p90": None, "p99": None, "max": None,
            }
        return (
            keep.cpu().numpy(), supports.cpu().numpy(), conflicts.cpu().numpy(), {
                "evidence_samples": int(evidence_count),
                "occluded_samples": int(occluded_count),
                "unobservable_samples": int(unobservable_count),
                "relative_depth_error": residual_report,
                "relative_depth_error_all": residual_report,
                "mask_threshold": float(mask_threshold),
                "device": device,
            },
        )


def geometric_consistency_filter(
    xyz: np.ndarray,
    *,
    source_index: int,
    depths: Sequence[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    foreground_masks: Sequence[np.ndarray],
    validities: Sequence[np.ndarray],
    min_consistent_views: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    return_diagnostics: bool = False,
    mask_threshold: float = 0.5,
    device: str = "auto",
    _prepared_cuda: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, object]
]:
    """Return keep/support/conflict arrays using conservative depth evidence."""

    resolved_device, torch = (
        available_torch_device(device)
        if str(device).startswith("cuda") else
        resolve_torch_device(device, purpose="DA3 geometric consistency")
    )
    if resolved_device.startswith("cuda"):
        return _geometric_consistency_filter_cuda(
            xyz, source_index=source_index, depths=depths,
            extrinsics=extrinsics, intrinsics=intrinsics,
            foreground_masks=foreground_masks, validities=validities,
            min_consistent_views=min_consistent_views,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            return_diagnostics=return_diagnostics,
            mask_threshold=mask_threshold, torch=torch, device=resolved_device,
            prepared=_prepared_cuda,
        )

    points = np.asarray(xyz, np.float64)
    supports = np.ones(len(points), np.int16)
    conflicts = np.zeros(len(points), np.int16)
    residuals: list[np.ndarray] = []
    evidence_count = 0
    occluded_count = 0
    unobservable_count = 0
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    for target in range(len(depths)):
        if target == int(source_index):
            continue
        camera = homogeneous @ np.asarray(extrinsics[target], np.float64).T
        z = camera[:, 2]
        k = np.asarray(intrinsics[target], np.float64)
        x = np.rint(k[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + k[0, 2] - 0.5).astype(np.int64)
        y = np.rint(k[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + k[1, 2] - 0.5).astype(np.int64)
        height, width = np.asarray(depths[target]).shape
        inside = (z > 1e-8) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        ids = np.flatnonzero(inside)
        unobservable_count += int(np.count_nonzero(~inside))
        if not len(ids):
            continue
        observed = np.asarray(depths[target], np.float64)[y[ids], x[ids]]
        valid = np.asarray(validities[target])[y[ids], x[ids]] >= 0.5
        foreground = (
            np.asarray(foreground_masks[target], np.float32)[y[ids], x[ids]]
            >= float(mask_threshold)
        )
        trusted = target_depth_is_trusted(
            observed_depth=observed,
            validity=valid,
            foreground=foreground,
        )
        if not np.any(trusted):
            continue
        ids = ids[trusted]
        observed = observed[trusted]
        tolerance = np.maximum(
            float(relative_tolerance) * observed,
            float(absolute_tolerance),
        )
        delta = z[ids] - observed
        relative = np.abs(delta) / np.maximum(observed, 1e-12)
        residuals.append(relative)
        evidence_count += len(ids)
        agree = np.abs(delta) <= tolerance
        behind = delta > tolerance
        ahead = delta < -tolerance
        supports[ids[agree]] += 1
        conflicts[ids[ahead]] += 1
        occluded_count += int(np.count_nonzero(behind))
    keep = (supports >= int(min_consistent_views)) & (conflicts == 0)
    if residuals:
        all_rel = np.concatenate(residuals)
        kept_mask = keep
        # residuals are per evidence sample, not per point; report separately.
        residual_report = {
            "count": int(len(all_rel)),
            "p50": float(np.percentile(all_rel, 50)),
            "p90": float(np.percentile(all_rel, 90)),
            "p99": float(np.percentile(all_rel, 99)),
            "max": float(np.max(all_rel)),
        }
    else:
        all_rel = np.empty(0, np.float64)
        residual_report = {
            "count": 0, "p50": None, "p90": None, "p99": None, "max": None,
        }
    if not return_diagnostics:
        return keep, supports, conflicts
    return keep, supports, conflicts, {
        "evidence_samples": int(evidence_count),
        "occluded_samples": int(occluded_count),
        "unobservable_samples": int(unobservable_count),
        "relative_depth_error": residual_report,
        "relative_depth_error_all": residual_report,
        "mask_threshold": float(mask_threshold),
        "device": "cpu",
    }


def _fuse_view_clouds(
    clouds: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    voxel_size: float | None,
    voxel_ratio: float | None = None,
    target_extent: float | None = None,
    max_points: int,
    rng: np.random.Generator,
    report: dict[str, object] | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Voxel-fuse multi-view unprojections, then spatially cap the dense cache.

    Returns ``(xyz, rgb, raw_count, voxel_size)``.  The fused cloud is the
    3-NN reference; Gaussian-budget sampling happens in ``initialize_explicit``.
    """

    points = [(xyz, rgb) for xyz, rgb in clouds if len(xyz)]
    if not points:
        raise RuntimeError(
            "DA3 produced no valid points after foreground, validity, positive-depth, "
            "confidence and sky filters; inspect empty masks and processed-coordinate "
            "alignment before changing thresholds"
        )
    resolved_device, _torch = (
        available_torch_device(device)
        if str(device).startswith("cuda") else
        resolve_torch_device(device, purpose="DA3 voxel fusion")
    )
    per_view = pre_fuse_per_view_cap(int(max_points), len(points))
    trimmed: list[tuple[np.ndarray, np.ndarray]] = []
    for xyz, rgb in points:
        if len(xyz) > per_view:
            keep = select_point_indices(
                rng, len(xyz), per_view, xyz=xyz,
            )
            trimmed.append((xyz[keep], rgb[keep]))
        else:
            trimmed.append((xyz, rgb))
    xyz = np.concatenate([item[0] for item in trimmed], axis=0)
    rgb = np.concatenate([item[1] for item in trimmed], axis=0)
    raw_count = int(len(xyz))
    voxel_diag: dict[str, object] = {}
    if voxel_size is None and voxel_ratio is not None:
        ratio = float(voxel_ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("init.da3.fusion_voxel_ratio must be positive and finite")
        scene_extent = (
            float(target_extent)
            if target_extent is not None
            else float(np.max(np.ptp(xyz, axis=0)))
        )
        if not np.isfinite(scene_extent) or scene_extent <= 0.0:
            raise ValueError("DA3 target extent must be positive and finite")
        size = max(ratio * scene_extent, 1e-6)
        voxel_diag = {"mode": "extent", "voxel_size": float(size)}
    elif voxel_size is None:
        size = auto_voxel_size(
            xyz, rng=rng, diagnostics=voxel_diag,
        )
        voxel_diag["mode"] = "spacing"
    else:
        size = float(voxel_size)
        if not np.isfinite(size) or size <= 0.0:
            raise ValueError("init.da3.voxel_size must be a positive finite length")
        voxel_diag = {"mode": "explicit", "voxel_size": size}
    fused_xyz, fused_rgb, _winner = voxel_fuse_points(
        xyz, rgb, voxel_size=size, device=resolved_device,
    )
    if len(fused_xyz) > int(max_points):
        sample_diag: dict[str, object] = {}
        keep = select_point_indices(
            rng, len(fused_xyz), int(max_points), xyz=fused_xyz,
            diagnostics=sample_diag,
        )
        fused_xyz, fused_rgb = fused_xyz[keep], fused_rgb[keep]
        voxel_diag["cap_sampling"] = sample_diag
    else:
        voxel_diag["cap_sampling"] = {"strategy": "all", "coverage_count": int(len(fused_xyz))}
    voxel_diag.update({
        "raw_count": raw_count,
        "fused_count": int(len(fused_xyz)),
        "final_voxel_size": float(size),
        "pre_unproject_cap_per_view": int(per_view),
        "device": resolved_device,
    })
    if report is not None:
        report.update(voxel_diag)
    return (
        fused_xyz.astype(np.float32, copy=False),
        fused_rgb.astype(np.uint8, copy=False),
        raw_count,
        size,
    )


def _scaled_intrinsic(
    intrinsic: np.ndarray,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    """Scale a pixel-space intrinsic matrix between image resolutions."""

    from_h, from_w = (int(value) for value in from_size)
    to_h, to_w = (int(value) for value in to_size)
    result = np.asarray(intrinsic, np.float32).copy()
    result[0] *= np.float32(to_w / from_w)
    result[1] *= np.float32(to_h / from_h)
    result[2, 2] = 1.0
    return np.ascontiguousarray(result)


def _rig_alignment(
    local_extrinsics: np.ndarray,
    global_extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find a similarity mapping from a local DA3 chunk to the shared rig."""

    local_poses = np.linalg.inv(local_extrinsics)
    global_poses = np.linalg.inv(global_extrinsics)
    rotation_sum = np.zeros((3, 3), dtype=np.float64)
    for local, global_ in zip(local_poses, global_poses):
        rotation_sum += global_[:3, :3] @ local[:3, :3].T
    u, _singular, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    local_centers = local_poses[:, :3, 3].astype(np.float64)
    global_centers = global_poses[:, :3, 3].astype(np.float64)
    local_centered = local_centers - local_centers.mean(axis=0)
    global_centered = global_centers - global_centers.mean(axis=0)
    denominator = float(np.sum(local_centered * local_centered))
    scale = (
        float(np.sqrt(np.sum(global_centered * global_centered) / denominator))
        if denominator > 1e-12 else 1.0
    )
    translation = (
        global_centers.mean(axis=0)
        - scale * (rotation @ local_centers.mean(axis=0))
    )
    return rotation.astype(np.float32), translation.astype(np.float32), scale


def _apply_rig_alignment(
    extrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> np.ndarray:
    poses = np.linalg.inv(np.asarray(extrinsics, np.float32))
    aligned = np.zeros_like(poses)
    aligned[:, 3, 3] = 1.0
    aligned[:, :3, :3] = rotation[None] @ poses[:, :3, :3]
    aligned[:, :3, 3] = (
        scale * (rotation @ poses[:, :3, 3, None])[..., 0]
        + translation[None]
    )
    return np.linalg.inv(aligned).astype(np.float32)


def _prediction_cameras(prediction: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 did not return camera poses and intrinsics")
    extrinsics = np.stack([
        _as_homogeneous(value) for value in np.asarray(prediction.extrinsics)
    ])
    intrinsics = np.asarray(prediction.intrinsics, np.float32)
    if extrinsics.shape != (count, 4, 4) or intrinsics.shape != (count, 3, 3):
        raise RuntimeError(
            "DA3 camera output shape mismatch: "
            f"extrinsics={extrinsics.shape}, intrinsics={intrinsics.shape}"
        )
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise RuntimeError("DA3 returned non-finite camera parameters")
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise RuntimeError("DA3 returned a non-positive focal length")
    return extrinsics, intrinsics


def _conditioned_camera_diagnostics(
    predicted: np.ndarray,
    conditioned: np.ndarray,
) -> dict[str, float]:
    """Compare DA3's returned rig with the calibrated rig supplied to it.

    ``align_to_input_ext_scale=True`` is expected to leave conditioned camera
    poses in the input world frame.  We still validate that assumption before
    attaching predicted depths to the locked FTGSPP calibration; otherwise a
    model/API change could silently create a second coordinate system.
    """

    predicted_pose = np.linalg.inv(np.asarray(predicted, np.float64))
    conditioned_pose = np.linalg.inv(np.asarray(conditioned, np.float64))
    center_error = np.linalg.norm(
        predicted_pose[:, :3, 3] - conditioned_pose[:, :3, 3], axis=1,
    )
    centers = conditioned_pose[:, :3, 3]
    baseline = float(np.max(np.linalg.norm(
        centers[:, None] - centers[None, :], axis=-1,
    ))) if len(centers) > 1 else 1.0
    baseline = max(baseline, 1e-6)
    angles: list[float] = []
    for predicted_rotation, conditioned_rotation in zip(
        predicted_pose[:, :3, :3], conditioned_pose[:, :3, :3], strict=True,
    ):
        relative = predicted_rotation @ conditioned_rotation.T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosine))))
    result = {
        "max_center_error": float(center_error.max(initial=0.0)),
        "max_center_error_over_baseline": float(
            center_error.max(initial=0.0) / baseline
        ),
        "max_rotation_error_degrees": max(angles, default=0.0),
    }
    # The threshold is deliberately a compatibility guard, not a geometry
    # tuning knob. Small numerical round-off is accepted; a materially changed
    # pose must be diagnosed instead of being mixed with the input rig.
    if (
        result["max_center_error_over_baseline"] > 0.02
        or result["max_rotation_error_degrees"] > 2.0
    ):
        raise RuntimeError(
            "DA3 returned camera poses inconsistent with the conditioned "
            "calibration: "
            f"center/baseline={result['max_center_error_over_baseline']:.6g}, "
            f"rotation={result['max_rotation_error_degrees']:.6g} degrees"
        )
    return result


def _resolved_sky_mask(options: Any) -> bool:
    value = getattr(options, "use_sky_mask", True)
    if value is None:
        raise RuntimeError(
            "init.da3.use_sky_mask must be resolved before unprojection; "
            "bind the schedule onto a copy of the options first"
        )
    return bool(value)


def _run_inference(
    model: Any,
    options: Any,
    images: list[np.ndarray],
    *,
    frame: int,
    cameras: Sequence[int],
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
):
    try:
        return model.inference(
            images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=True,
            infer_gs=False,
            use_ray_pose=bool(getattr(options, "use_ray_pose", False)),
            ref_view_strategy=str(
                getattr(options, "ref_view_strategy", "saddle_balanced")
            ),
            process_res=int(getattr(options, "process_res", 504)),
            process_res_method=str(
                getattr(options, "process_res_method", "upper_bound_resize")
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"DA3 inference failed for frame {frame} cameras {list(cameras)}"
        ) from exc


def _cloud_from_prediction(
    prediction: Any,
    index: int,
    w2c: np.ndarray,
    options: Any,
    *,
    depth_scale: float,
    foreground_mask: np.ndarray | None = None,
    validity: np.ndarray | None = None,
    max_points: int | None = None,
    rng: np.random.Generator | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.processed_images is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 did not return processed images/intrinsics")
    conf = None if prediction.conf is None else prediction.conf[index]
    return depth_to_points(
        np.asarray(prediction.depth[index], np.float32) * np.float32(depth_scale),
        prediction.intrinsics[index],
        w2c,
        prediction.processed_images[index],
        confidence=conf,
        sky=(
            prediction.sky[index]
            if _resolved_sky_mask(options)
            and prediction.sky is not None
            else None
        ),
        confidence_percentile=float(
            getattr(options, "confidence_percentile", 20.0)
        ),
        foreground_mask=foreground_mask,
        validity=validity,
        mask_threshold=float(getattr(options, "mask_threshold", 0.5)),
        max_points=max_points,
        rng=rng,
        return_selection=False,
        device=device,
    )


def _prediction_side_inputs(
    views: Sequence[View], prediction: Any, options: Any,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    foreground_only = bool(getattr(options, "foreground_only", False))
    method = str(getattr(options, "process_res_method", "upper_bound_resize"))
    convention = str(getattr(options, "pixel_transform_convention", "source_index_scale"))
    masks: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    transforms: list[np.ndarray] = []
    for index, view in enumerate(views):
        target_size = tuple(int(value) for value in np.asarray(prediction.depth[index]).shape)
        present: list[bool] = []
        transform = _prediction_transform_matrix(
            prediction, index, tuple(view.size), target_size, method,
            convention=convention, metadata_present=present,
        )
        transforms.append(transform)
        warp = bool(present and present[-1]) or str(convention) == "pixel_center"
        mask = (
            view.load_mask() if foreground_only
            else np.ones(view.size, dtype=np.float32)
        )
        processed_mask = _processed_scalar_map(
            mask, target_size, method, interpolation=1,
            source_to_processed=transform if warp else None,
        )
        masks.append(processed_mask)
        validities.append(_processed_scalar_map(
            view.load_validity(), target_size, method, interpolation=0,
            source_to_processed=transform if warp else None,
        ))
    return masks, validities, transforms


def _prediction_bundle(
    prediction: Any,
    views: Sequence[View],
    extrinsics: np.ndarray,
    options: Any,
    *,
    depth_scale: float,
    max_points_per_view: int | None = None,
    rng: np.random.Generator | None = None,
    device: str = "auto",
) -> _PredictionBundle:
    _validate_prediction_depth_type(prediction, options)
    masks, validities, transforms = _prediction_side_inputs(
        views, prediction, options,
    )
    depths = [
        np.asarray(value, np.float32) * np.float32(depth_scale)
        for value in prediction.depth
    ]
    raw_clouds: list[tuple[np.ndarray, np.ndarray]] = []
    camera_ids = [int(view.camera_id) for view in views]
    for index in range(len(views)):
        xyz, rgb = _cloud_from_prediction(
            prediction, index, extrinsics[index], options,
            depth_scale=depth_scale,
            foreground_mask=masks[index], validity=validities[index],
            max_points=max_points_per_view, rng=rng,
            device=device,
        )
        raw_clouds.append((xyz, rgb))
    return _PredictionBundle(
        clouds=raw_clouds,
        depths=depths,
        extrinsics=np.asarray(extrinsics, np.float32),
        intrinsics=np.asarray(prediction.intrinsics, np.float32),
        masks=masks,
        validities=validities,
        transforms=transforms,
        camera_ids=camera_ids,
    )


def _filter_prediction_bundles(
    bundles: Sequence[_PredictionBundle], options: Any, *, device: str = "auto",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Run consistency over all synchronized cameras, independent of chunks."""

    if not bundles:
        raise ValueError("DA3 global consistency requires at least one prediction chunk")
    raw_clouds = [cloud for bundle in bundles for cloud in bundle.clouds]
    depths = [value for bundle in bundles for value in bundle.depths]
    masks = [value for bundle in bundles for value in bundle.masks]
    validities = [value for bundle in bundles for value in bundle.validities]
    extrinsics = np.concatenate([bundle.extrinsics for bundle in bundles], axis=0)
    intrinsics = np.concatenate([bundle.intrinsics for bundle in bundles], axis=0)
    mask_threshold = float(getattr(options, "mask_threshold", 0.5))
    clouds: list[tuple[np.ndarray, np.ndarray]] = []
    support_hist = np.zeros(len(raw_clouds) + 1, np.int64)
    conflicts = 0
    low_support = 0
    evidence_samples = 0
    occluded_samples = 0
    residual_values: list[tuple[int, float | None]] = []
    primary_reject = {"low_support": 0, "depth_conflict": 0, "kept": 0}
    scene_extent = 0.0
    nonempty = [value[0] for value in raw_clouds if len(value[0])]
    if nonempty:
        combined = np.concatenate(nonempty)
        scene_extent = float(np.max(np.ptp(combined, axis=0)))
    locked_extent = _target_extent(options)
    scale_mode = str(getattr(
        options,
        "consistency_scale_mode",
        "target_extent" if locked_extent is not None else "predicted_extent",
    ))
    if scale_mode == "target_extent":
        if locked_extent is None:
            raise ValueError(
                "consistency_scale_mode=target_extent requires target_bounds"
            )
        tolerance_scale = float(locked_extent)
        tolerance_scale_source = "target_bounds"
    elif scale_mode == "predicted_extent":
        tolerance_scale = float(scene_extent)
        tolerance_scale_source = "predicted_cloud"
    elif scale_mode == "median_depth":
        positive_depths = [
            depth[np.isfinite(depth) & (depth > 0.0) & np.asarray(validity, bool)]
            for depth, validity in zip(depths, validities, strict=True)
        ]
        positive_depths = [values for values in positive_depths if values.size]
        if positive_depths:
            tolerance_scale = float(
                np.median(np.concatenate(positive_depths))
            )
            tolerance_scale_source = "median_positive_camera_z"
        else:
            tolerance_scale = float(scene_extent)
            tolerance_scale_source = "predicted_cloud_fallback"
    else:
        raise ValueError(
            f"unsupported consistency_scale_mode: {scale_mode}"
        )
    absolute_override = getattr(options, "absolute_depth_tolerance", None)
    absolute = (
        float(absolute_override)
        if absolute_override is not None else (
            float(getattr(options, "absolute_depth_tolerance_ratio", 0.002))
            * max(tolerance_scale, 1e-6)
        )
    )
    minimum = int(getattr(options, "min_consistent_views", 1))
    if minimum > len(raw_clouds):
        raise RuntimeError(
            "DA3 consistency requires "
            f"{minimum} views but only {len(raw_clouds)} synchronized views are available"
        )
    resolved_device, torch = (
        available_torch_device(device)
        if str(device).startswith("cuda") else
        resolve_torch_device(device, purpose="DA3 geometric consistency")
    )
    prepared_cuda = None
    if resolved_device.startswith("cuda") and minimum > 1:
        prepared_cuda = {
            "extrinsics": torch.as_tensor(
                extrinsics, dtype=torch.float32, device=resolved_device,
            ),
            "intrinsics": torch.as_tensor(
                intrinsics, dtype=torch.float32, device=resolved_device,
            ),
            "depths": [
                torch.as_tensor(
                    np.ascontiguousarray(value),
                    dtype=torch.float32, device=resolved_device,
                ) for value in depths
            ],
            "foreground_masks": [
                torch.as_tensor(
                    np.ascontiguousarray(value),
                    dtype=torch.float32, device=resolved_device,
                ) for value in masks
            ],
            "validities": [
                torch.as_tensor(
                    np.ascontiguousarray(value), device=resolved_device,
                ) for value in validities
            ],
        }
    for index, (xyz, rgb) in enumerate(raw_clouds):
        if not len(xyz) or minimum <= 1:
            clouds.append((xyz, rgb))
            if len(xyz):
                support_hist[1] += len(xyz)
                primary_reject["kept"] += len(xyz)
            continue
        keep, support, conflict, evidence = geometric_consistency_filter(
            xyz,
            source_index=index,
            depths=depths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            foreground_masks=masks,
            validities=validities,
            min_consistent_views=minimum,
            relative_tolerance=float(getattr(options, "relative_depth_tolerance", 0.02)),
            absolute_tolerance=absolute,
            return_diagnostics=True,
            mask_threshold=mask_threshold,
            device=resolved_device,
            _prepared_cuda=prepared_cuda,
        )
        low = support < minimum
        depth_conflict = conflict > 0
        primary = np.full(len(xyz), "kept", dtype=object)
        primary[low] = "low_support"
        primary[depth_conflict & ~low] = "depth_conflict"
        primary_reject["low_support"] += int(np.count_nonzero(primary == "low_support"))
        primary_reject["depth_conflict"] += int(
            np.count_nonzero(primary == "depth_conflict")
        )
        primary_reject["kept"] += int(np.count_nonzero(keep))
        support_hist += np.bincount(
            np.minimum(support, len(raw_clouds)), minlength=len(raw_clouds) + 1,
        )
        conflicts += int(np.count_nonzero(conflict))
        low_support += int(np.count_nonzero(low))
        evidence_samples += int(evidence["evidence_samples"])
        occluded_samples += int(evidence["occluded_samples"])
        relative = evidence["relative_depth_error"]
        if isinstance(relative, dict):
            residual_values.append((
                int(relative.get("count", 0)),
                None if relative.get("p90") is None else float(relative["p90"]),
            ))
        clouds.append((xyz[keep], rgb[keep]))
    return clouds, {
        "raw_points": int(sum(len(value[0]) for value in raw_clouds)),
        "consistent_points": int(sum(len(value[0]) for value in clouds)),
        "support_histogram": support_hist.tolist(),
        "conflicting_points": conflicts,
        "low_support_points": low_support,
        "evidence_samples": evidence_samples,
        "occluded_samples": occluded_samples,
        "per_source_relative_depth_error_p90": [
            {"count": count, "p90": p90} for count, p90 in residual_values
        ],
        "scene_extent": scene_extent,
        "tolerance_scale": tolerance_scale,
        "tolerance_scale_source": tolerance_scale_source,
        # Compatibility aliases retained for existing diagnostics readers.
        "tolerance_extent": tolerance_scale,
        "tolerance_extent_source": tolerance_scale_source,
        "absolute_depth_tolerance": absolute,
        "pre_unproject_cap_per_view": (
            None
        ),
        "source_to_processed": [
            matrix.tolist() for bundle in bundles for matrix in bundle.transforms
        ],
        "global_consistency_views": len(raw_clouds),
        "mask_threshold": mask_threshold,
        "primary_reject_reason": primary_reject,
        "device": resolved_device,
    }


def _clouds_from_prediction(
    prediction: Any,
    views: Sequence[View],
    extrinsics: np.ndarray,
    options: Any,
    *,
    depth_scale: float,
    max_points_per_view: int | None = None,
    rng: np.random.Generator | None = None,
    device: str = "auto",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    bundle = _prediction_bundle(
        prediction, views, extrinsics, options,
        depth_scale=depth_scale,
        max_points_per_view=max_points_per_view,
        rng=rng,
        device=device,
    )
    clouds, diagnostics = _filter_prediction_bundles(
        [bundle], options, device=device,
    )
    diagnostics["pre_unproject_cap_per_view"] = (
        None if max_points_per_view is None else int(max_points_per_view)
    )
    return clouds, diagnostics


def _unique_camera_bundle(
    bundles: Sequence[_PredictionBundle], camera_chunks: Sequence[Sequence[int]],
) -> _PredictionBundle:
    """Drop repeated alignment anchors before global consistency accounting."""

    seen: set[int] = set()
    selected: list[tuple[_PredictionBundle, int, int]] = []
    for bundle, cameras in zip(bundles, camera_chunks, strict=True):
        if len(bundle.clouds) != len(cameras):
            raise RuntimeError("DA3 bundle/camera cardinality mismatch")
        for index, camera in enumerate(cameras):
            if int(camera) not in seen:
                selected.append((bundle, index, int(camera)))
                seen.add(int(camera))
    return _PredictionBundle(
        clouds=[bundle.clouds[index] for bundle, index, _camera in selected],
        depths=[bundle.depths[index] for bundle, index, _camera in selected],
        extrinsics=np.stack([bundle.extrinsics[index] for bundle, index, _camera in selected]),
        intrinsics=np.stack([bundle.intrinsics[index] for bundle, index, _camera in selected]),
        masks=[bundle.masks[index] for bundle, index, _camera in selected],
        validities=[bundle.validities[index] for bundle, index, _camera in selected],
        transforms=[bundle.transforms[index] for bundle, index, _camera in selected],
        camera_ids=[camera for _bundle, _index, camera in selected],
    )


def _synchronized_views(
    dataset: Scene,
    frame: int,
    camera_ids: Sequence[int],
    options: Any,
    *,
    reference_timestamp: float | None = None,
) -> tuple[list[View], float, list[float]]:
    reference = (
        float(dataset.view(int(frame), int(camera_ids[0])).timestamp)
        if reference_timestamp is None else float(reference_timestamp)
    )
    tolerance = float(getattr(options, "sync_tolerance_fraction", 0.25)) / float(dataset.fps)
    train_only = set(int(value) for value in camera_ids) <= set(dataset.train_cameras)
    views = dataset.views_nearest_timestamp(
        reference, camera_ids, tolerance_seconds=tolerance, train_only=train_only,
    )
    return views, reference, [abs(float(view.timestamp) - reference) for view in views]


def _cache_root(config: Any) -> Path:
    configured = getattr(config.init.da3, "cache_path", None)
    if configured is not None:
        return Path(configured)
    return Path(config.data.root) / (
        f"points_da3_stride{int(config.init.keyframe_stride)}"
    )


def pixel_center_to_index_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a pixel-center homography into OpenCV index coordinates."""

    shift = np.asarray(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], np.float64,
    )
    inverse = np.asarray(
        [[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]], np.float64,
    )
    return inverse @ np.asarray(matrix, np.float64) @ shift


def resolve_da3_cache_root(root: Path) -> Path:
    """Resolve an atomically published cache namespace, with legacy fallback."""

    return resolve_published_cache_root(root)


def _selected_fusion_cameras(
    dataset: Scene,
    options: Any,
    schedule: Any,
    candidate_camera_ids: Sequence[int],
    frame_positions: Sequence[int],
) -> list[int]:
    candidate = [int(value) for value in candidate_camera_ids]
    if not getattr(dataset, "calibration_available", True):
        return list(candidate)
    reference_frame = int(dataset.frames[int(frame_positions[0])])
    w2c = {
        camera: _as_homogeneous(dataset.view(reference_frame, camera).w2c)
        for camera in candidate
    }
    return select_representative_camera_ids(candidate, w2c, schedule.fusion_views)


def da3_l1_identity(
    config: Any,
    dataset: Scene,
    frame_positions: Sequence[int],
    *,
    schedule: Any,
    candidate_camera_ids: Sequence[int],
    selected_camera_ids: Sequence[int],
    loaded_model_identity: Mapping[str, object],
) -> dict[str, object]:
    options = config.init.da3
    plan = effective_da3_plan(options, schedule)
    masks = getattr(getattr(config, "data", None), "masks", None)
    mask_affects_rgb = str(getattr(masks, "rgb_encoding", "straight")) == "premultiplied"
    return l1_identity_payload(
        pipeline_mode=str(getattr(getattr(config, "pipeline", None), "mode", "scene")),
        dataset_fingerprint=dataset.checkpoint_metadata()["fingerprint"],
        model_identity=loaded_model_identity,
        plan=plan,
        candidate_camera_ids=candidate_camera_ids,
        selected_camera_ids=selected_camera_ids,
        process_res_method=str(getattr(options, "process_res_method", "upper_bound_resize")),
        depth_type=str(getattr(options, "depth_type", "camera_z")),
        use_ray_pose=bool(getattr(options, "use_ray_pose", False)),
        ref_view_strategy=str(getattr(options, "ref_view_strategy", "saddle_balanced")),
        input_cameras=str(getattr(options, "input_cameras", "train")),
        foreground_only=bool(getattr(options, "foreground_only", False)),
        mask_threshold=float(getattr(options, "mask_threshold", 0.5)),
        sync_tolerance_fraction=float(getattr(options, "sync_tolerance_fraction", 0.25)),
        seed=initialization_seed(config),
        keyframe_frames=[
            int(dataset.frames[int(position)]) for position in frame_positions
        ],
        mask_affects_rgb=mask_affects_rgb,
        foreground_metadata=foreground_schedule_metadata(schedule),
    )


def da3_l2_identity(
    config: Any,
    dataset: Scene,
    frame_positions: Sequence[int],
    *,
    schedule: Any,
    candidate_camera_ids: Sequence[int],
    selected_camera_ids: Sequence[int],
    loaded_model_identity: Mapping[str, object],
) -> dict[str, object]:
    options = config.init.da3
    plan = effective_da3_plan(options, schedule)
    pre_cap = pre_fuse_per_view_cap(int(schedule.max_points), len(selected_camera_ids))
    l1 = da3_l1_identity(
        config, dataset, frame_positions, schedule=schedule,
        candidate_camera_ids=candidate_camera_ids,
        selected_camera_ids=selected_camera_ids,
        loaded_model_identity=loaded_model_identity,
    )
    return l2_identity_payload(
        l1=l1, plan=plan, options=options, pre_unproject_cap_per_view=pre_cap,
    )


def _cache_namespace(
    config: Any,
    dataset: Scene,
    frame_positions: Sequence[int],
    *,
    schedule: Any,
    candidate_camera_ids: Sequence[int],
    loaded_model_identity: dict[str, object],
    selected_camera_ids: Sequence[int] | None = None,
) -> str:
    selected = (
        list(selected_camera_ids)
        if selected_camera_ids is not None
        else (
            list(candidate_camera_ids)
            if not hasattr(dataset, "view")
            else _selected_fusion_cameras(
                dataset, config.init.da3, schedule, candidate_camera_ids,
                frame_positions,
            )
        )
    )
    identity = da3_l2_identity(
        config, dataset, frame_positions, schedule=schedule,
        candidate_camera_ids=candidate_camera_ids,
        selected_camera_ids=selected,
        loaded_model_identity=loaded_model_identity,
    )
    digest = namespace_hash(identity)
    mode = str(getattr(getattr(config, "pipeline", None), "mode", "scene"))
    mode = "".join(character if character.isalnum() else "-" for character in mode)
    return (
        f"{mode}-init{initialization_seed(config)}"
        f"-kf{len(frame_positions)}"
        f"-v{len(selected)}"
        f"-r{int(schedule.process_res)}"
        f"-{digest[:12]}"
    )


def _save_camera_manifest(
    cache_root: Path,
    camera_ids: Sequence[int],
    extrinsics: dict[int, np.ndarray],
    intrinsics: dict[int, np.ndarray],
    sizes: dict[int, tuple[int, int]],
) -> dict[str, object]:
    ordered = [int(camera) for camera in camera_ids]
    path = cache_root / "camera_manifest.npz"
    temporary = cache_root / f".camera_manifest.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        camera_ids=np.asarray(ordered, np.int64),
        extrinsics=np.stack([extrinsics[camera] for camera in ordered]),
        intrinsics=np.stack([intrinsics[camera] for camera in ordered]),
        sizes=np.asarray([sizes[camera] for camera in ordered], np.int64),
    )
    os.replace(temporary, path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "camera_ids": ordered,
    }


def _fuse_keyframe_cloud(
    clouds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    options: Any,
    voxel_size: float | None,
    voxel_ratio: float | None,
    max_points: int,
    rng: np.random.Generator,
    allow_empty: bool,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, int, float, dict[str, object]]:
    report: dict[str, object] = {}
    try:
        xyz, rgb, raw_count, size = _fuse_view_clouds(
            clouds, voxel_size=voxel_size, voxel_ratio=voxel_ratio,
            target_extent=_target_extent(options), max_points=max_points, rng=rng,
            report=report,
            device=device,
        )
    except RuntimeError:
        if not allow_empty:
            raise
        xyz = np.empty((0, 3), np.float32)
        rgb = np.empty((0, 3), np.uint8)
        raw_count = 0
        size = float(voxel_size) if voxel_size is not None else 1e-3
        report["empty"] = True
    return xyz, rgb, raw_count, size, report


def _serializable_geometry(diagnostics: dict[str, object]) -> dict[str, object]:
    return dict(diagnostics)


def _run_chunk_inference(
    model: Any,
    options: Any,
    views: Sequence[View],
    *,
    frame: int,
    cameras: Sequence[int],
    timer: StageTimer,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
) -> Any:
    images = [view.load_rgb()[..., :3] for view in views]
    inference_intrinsics = (
        None if extrinsics is None
        else np.asarray(intrinsics, np.float32) if intrinsics is not None
        else np.stack([np.asarray(view.intrinsic, np.float32) for view in views])
    )
    with timer.stage("da3_forward"):
        return _run_inference(
            model, options, images,
            frame=frame, cameras=cameras,
            extrinsics=extrinsics, intrinsics=inference_intrinsics,
        )


def generate_da3_points(
    config: Any,
    dataset: Scene,
    frame_positions: Sequence[int],
) -> DA3PointClouds:
    """Reuse or atomically publish one complete, immutable point-cloud set.

    The lock belongs to the cache base because current.json is shared by its
    namespaces. Check again after acquiring it, before loading a CUDA model;
    concurrent runs then share the first completed generation. Different cache
    bases can still initialize independently.
    """

    if not frame_positions:
        raise ValueError("DA3 initialization requires at least one keyframe")
    from ftgspp.data.scene import _load_da3_camera_manifest
    from ftgspp.training.initialize import _valid_da3_cache

    base = _cache_root(config)
    frames = [int(dataset.frames[int(position)]) for position in frame_positions]
    with namespace_lock(base):
        configured_model = model_identity(config.init.da3)
        for root in published_cache_candidates(base):
            if not _valid_da3_cache(
                config, dataset, root, expected_frames=frames,
                frame_positions=frame_positions,
                configured_model_identity=configured_model,
                verify_ply_hashes=False,
            ):
                continue
            geometry = None
            try:
                if not getattr(dataset, "calibration_available", True):
                    geometry = _load_da3_camera_manifest(
                        config, dataset.cameras, cache_root=root,
                    )
                    if geometry is None:
                        continue
                manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_hashes = {
                    str(row["ply"]): str(row["ply_sha256"])
                    for row in manifest.get("keyframes", [])
                    if isinstance(row, dict)
                    and isinstance(row.get("ply"), str)
                    and isinstance(row.get("ply_sha256"), str)
                }
                points = []
                for frame in frames:
                    path = root / f"f{frame:06d}.ply"
                    xyz, rgb, digest = read_point_ply(path, return_sha256=True)
                    if expected_hashes.get(path.name) != digest:
                        raise ValueError(f"DA3 cache keyframe hash mismatch: {path}")
                    points.append((xyz, rgb))
            except (OSError, ValueError) as exc:
                logger.warning("ignoring unreadable DA3 generation %s: %s", root, exc)
                continue
            if geometry is not None:
                dataset.update_camera_geometry(
                    {camera: value[0] for camera, value in geometry.items()},
                    {camera: value[1] for camera, value in geometry.items()},
                    {camera: value[2] for camera, value in geometry.items()},
                )
            if root != base:
                publish_cache_pointer(base, root, manifest_path)
            return DA3PointClouds(points, cache_root=root, generated=False)

        with tempfile.TemporaryDirectory(prefix=".building-", dir=base) as staging:
            temporary = Path(staging)
            # Geometry recovery replaces view records. Keep those changes local
            # until publication succeeds so a failed attempt can be retried on
            # the same Scene without switching coordinate systems or requests.
            working_dataset = copy.copy(dataset)
            points = _generate_da3_points(config, working_dataset, frame_positions, temporary)
            manifest = json.loads((temporary / "manifest.json").read_text(encoding="utf-8"))
            root = base / manifest["cache_namespace"]
            if root.exists():
                # Never replace a published directory while another reader
                # holds its path, even when that generation is damaged.
                root = base / f"{root.name}-{uuid.uuid4().hex[:12]}"
            os.rename(temporary, root)
            publish_cache_pointer(base, root, root / "manifest.json")
            if not getattr(dataset, "calibration_available", True):
                cameras = {
                    camera: working_dataset.view(frames[0], camera)
                    for camera in dataset.cameras
                }
                dataset.update_camera_geometry(
                    {camera: view.w2c for camera, view in cameras.items()},
                    {camera: view.intrinsic for camera, view in cameras.items()},
                    {camera: view.size for camera, view in cameras.items()},
                )
            return DA3PointClouds(points, cache_root=root, generated=True)


def _generate_da3_points(
    config: Any,
    dataset: Scene,
    frame_positions: Sequence[int],
    cache_root: Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate RGB point clouds and, when needed, an estimated camera rig.

    Calibrated datasets use pose-conditioned DA3 directly.  When calibration
    files are absent, DA3 first estimates a static rig from the first keyframe.
    Later keyframes are conditioned on that rig, keeping every point cloud in
    one coordinate system.  Large rigs use two shared anchor cameras to align
    independently inferred chunks without forcing all views into GPU memory.
    """

    started = time.perf_counter()
    timer = StageTimer()
    if not frame_positions:
        raise ValueError("DA3 initialization requires at least one keyframe")
    options = config.init.da3
    uncalibrated = not bool(getattr(dataset, "calibration_available", True))
    candidate_camera_ids = input_camera_ids(dataset, options)
    if uncalibrated:
        logger.info(
            "camera calibration not found; DA3 will estimate a %d-camera rig",
            len(candidate_camera_ids),
        )
    if not candidate_camera_ids:
        raise ValueError("DA3 initialization requires at least one input camera")
    if len(candidate_camera_ids) < 2 and "metric" not in str(
        getattr(options, "model_name", "")
    ).lower() and getattr(options, "model_path", None) is None:
        logger.warning(
            "single-camera DA3 initialization has an arbitrary global scale; "
            "use multiple simultaneous views or a DA3Metric checkpoint"
        )

    schedule = resolve_da3_schedule_for_scene(config, dataset, frame_positions)
    options = bind_schedule_to_options(config.init.da3, schedule)
    plan = effective_da3_plan(config.init.da3, schedule)
    if uncalibrated:
        # All views are needed once to estimate the unknown rig. Later frames
        # use a representative subset selected from the recovered geometry.
        camera_ids = list(candidate_camera_ids)
    else:
        reference_frame = int(dataset.frames[int(frame_positions[0])])
        candidate_w2c = {
            camera: _as_homogeneous(dataset.view(reference_frame, camera).w2c)
            for camera in candidate_camera_ids
        }
        camera_ids = select_representative_camera_ids(
            candidate_camera_ids, candidate_w2c, schedule.fusion_views,
        )
    native_hw = schedule.native_hw
    logger.info(
        "DA3 schedule quality=%s native=%dx%d process_res=%d max_views=%d "
        "fusion_views=%d/%d max_points=%d per_keyframe_gaussians=%d "
        "min_consistent_views=%d use_sky_mask=%s%s",
        schedule.quality, native_hw[0], native_hw[1],
        schedule.process_res, schedule.max_views,
        schedule.fusion_views, len(candidate_camera_ids), schedule.max_points,
        schedule.per_keyframe_gaussians, schedule.min_consistent_views,
        schedule.use_sky_mask,
        " (clamped to native, no upscale)" if schedule.upscale_clamped else "",
    )
    if schedule.foreground_fraction is not None:
        logger.info(
            "DA3 foreground schedule mask_fraction=%.6f visibility=%.3f "
            "sampled_views=%d nonempty_views=%d",
            schedule.foreground_fraction,
            schedule.foreground_visibility_fraction,
            schedule.foreground_sampled_views,
            schedule.foreground_nonempty_views,
        )
    logger.info("DA3 selected input cameras: %s", camera_ids)
    logger.info("DA3 pixel cost %s", inference_pixel_cost(native_hw, int(schedule.process_res)))
    with timer.stage("model_load"):
        model, device = _load_model(options, fallback_gpu=int(config.run.gpu))
    postprocess_device, _postprocess_torch = available_torch_device(device)
    logger.info("DA3 postprocessing device: %s", postprocess_device)
    try:
        # The model source is content/commit identified above; avoid copying every
        # tensor back to CPU just to derive a redundant state-dict hash.
        with timer.stage("model_identity"):
            loaded_model_identity = _loaded_model_identity(
                model, options, hash_state_dict=False,
            )
            if loaded_model_identity["loaded_state_dict_status"] == "skipped":
                loaded_model_identity = _loaded_model_identity(model, options)

        max_views = int(schedule.max_views)
        max_points = int(schedule.max_points)
        voxel_size = getattr(options, "voxel_size", None)
        voxel_mode = str(getattr(options, "fusion_voxel_mode", "spacing"))
        if voxel_mode not in {"spacing", "extent"}:
            raise ValueError(f"unsupported fusion_voxel_mode: {voxel_mode}")
        voxel_ratio = (
            float(getattr(options, "fusion_voxel_ratio", 0.002))
            if voxel_mode == "extent" and voxel_size is None else None
        )
        namespace = _cache_namespace(
            config, dataset, frame_positions, schedule=schedule,
            candidate_camera_ids=candidate_camera_ids,
            loaded_model_identity=loaded_model_identity,
            selected_camera_ids=camera_ids,
        )
        identity_cameras = [int(value) for value in camera_ids]
        input_l2_identity = da3_l2_identity(
            config, dataset, frame_positions, schedule=schedule,
            candidate_camera_ids=candidate_camera_ids,
            selected_camera_ids=identity_cameras,
            loaded_model_identity=loaded_model_identity,
        )
        rng = np.random.default_rng(initialization_seed(config))
        result: list[tuple[np.ndarray, np.ndarray]] = []
        ply_hashes: dict[int, str] = {}
        diagnostics_rows: list[dict[str, object]] = []

        rig_extrinsics: dict[int, np.ndarray] = {}
        runtime_intrinsics: dict[int, np.ndarray] = {}
        runtime_sizes: dict[int, tuple[int, int]] = {}
        input_intrinsics: dict[int, np.ndarray] = {}
        camera_manifest_record: dict[str, object] | None = None

        if uncalibrated:
            reference_position = int(frame_positions[0])
            reference_frame = int(dataset.frames[reference_position])
            frame_reference = float(
                dataset.view(reference_frame, candidate_camera_ids[0]).timestamp
            )
            anchor_count = min(2, len(camera_ids))
            reference_per_view_cap = pre_fuse_per_view_cap(
                max_points, len(camera_ids),
            )
            rig_chunks = _camera_chunks(
                camera_ids, max_views, anchor_count=anchor_count,
            )
            reference_bundles: list[_PredictionBundle] = []
            reference_bundle_cameras: list[list[int]] = []
            for chunk_index, camera_chunk in enumerate(rig_chunks):
                views, reference_timestamp, residuals = _synchronized_views(
                    dataset, reference_frame, camera_chunk, options,
                    reference_timestamp=frame_reference,
                )
                prediction = _run_chunk_inference(
                    model, options, views,
                    frame=reference_frame, cameras=camera_chunk,
                    timer=timer,
                )
                local_ext, local_k = _prediction_cameras(
                    prediction, len(camera_chunk),
                )
                existing = [
                    index for index, camera in enumerate(camera_chunk)
                    if camera in rig_extrinsics
                ]
                if existing:
                    global_anchors = np.stack([
                        rig_extrinsics[camera_chunk[index]] for index in existing
                    ])
                    rotation, translation, chunk_scale = _rig_alignment(
                        local_ext[existing], global_anchors,
                    )
                    aligned_ext = _apply_rig_alignment(
                        local_ext, rotation, translation, chunk_scale,
                    )
                else:
                    aligned_ext = local_ext
                    chunk_scale = 1.0

                with timer.stage("unproject_filter"):
                    bundle = _prediction_bundle(
                        prediction, views, aligned_ext, options,
                        depth_scale=chunk_scale,
                        max_points_per_view=reference_per_view_cap,
                        rng=rng,
                        device=postprocess_device,
                    )
                reference_bundles.append(bundle)
                reference_bundle_cameras.append(list(camera_chunk))
                new_indices: list[int] = []
                for index, camera in enumerate(camera_chunk):
                    if camera in rig_extrinsics:
                        continue
                    new_indices.append(index)
                    depth_size = tuple(
                        int(value) for value in prediction.depth[index].shape
                    )
                    # ``depth_size`` is DA3's internal processing resolution,
                    # not the resolution used by the Vulkan renderer. Keep
                    # the configured Scene size for training/evaluation and
                    # only use depth_size for the unprojection below.
                    input_size = tuple(int(value) for value in views[index].size)
                    rig_extrinsics[camera] = aligned_ext[index]
                    transform = _prediction_transform_matrix(
                        prediction, index, input_size, depth_size,
                        str(getattr(options, "process_res_method", "upper_bound_resize")),
                    )
                    target_k = _intrinsic_to_source(local_k[index], transform)
                    runtime_intrinsics[camera] = target_k
                    runtime_sizes[camera] = input_size
                    input_intrinsics[camera] = target_k
                diagnostics_rows.append({
                    "frame": reference_frame,
                    "reference_timestamp": reference_timestamp,
                    "cameras": [int(value) for value in camera_chunk],
                    "actual_frames": [int(view.frame) for view in views],
                    "sync_residual_seconds": residuals,
                    "chunk": int(chunk_index),
                    "new_view_indices": new_indices,
                    "geometry": {
                        "raw_points": int(sum(len(value[0]) for value in bundle.clouds)),
                        "global_filter_applied_after_all_chunks": True,
                    },
                })
                logger.info(
                    "DA3 rig chunk %d/%d: cameras=%s scale=%.6g",
                    chunk_index + 1, len(rig_chunks), camera_chunk, chunk_scale,
                )
            if set(rig_extrinsics) != set(camera_ids):
                raise RuntimeError("DA3 camera estimation did not cover the full rig")
            unique_reference_bundle = _unique_camera_bundle(
                reference_bundles, reference_bundle_cameras,
            )
            with timer.stage("global_consistency"):
                reference_clouds, global_diagnostics = _filter_prediction_bundles(
                    [unique_reference_bundle], options, device=postprocess_device,
                )
            for row in diagnostics_rows:
                if int(row["frame"]) == reference_frame:
                    row["global_geometry"] = _serializable_geometry(global_diagnostics)
            with timer.stage("voxel_fusion"):
                xyz, rgb, raw_count, fuse_voxel, fuse_report = _fuse_keyframe_cloud(
                    reference_clouds, options=options, voxel_size=voxel_size,
                    voxel_ratio=voxel_ratio, max_points=max_points, rng=rng,
                    allow_empty=bool(getattr(options, "foreground_only", False)),
                    device=postprocess_device,
                )
            result.append((xyz, rgb))
            with timer.stage("cache_write"):
                ply_path, ply_digest = write_point_ply_atomic(
                    cache_root / f"f{reference_frame:06d}.ply", xyz, rgb,
                    allow_empty=len(xyz) == 0,
                    return_sha256=True,
                )
                ply_hashes[reference_frame] = ply_digest
            logger.info(
                "DA3 initialization keyframe %d/%d (f%06d): %d fused points "
                "(raw=%d voxel=%.5g, dense 3-NN reference)",
                1, len(frame_positions), reference_frame, len(xyz), raw_count, fuse_voxel,
            )
            start_ordinal = 1
            camera_ids = select_representative_camera_ids(
                candidate_camera_ids, rig_extrinsics, schedule.fusion_views,
            )
            logger.info("DA3 selected conditioned cameras: %s", camera_ids)
            conditioned_chunks = _camera_chunks(
                camera_ids, max_views, anchor_count=0, minimum_chunk_size=3,
            )
        else:
            for camera in dataset.cameras:
                view = dataset.view(dataset.frames[int(frame_positions[0])], camera)
                rig_extrinsics[int(camera)] = _as_homogeneous(view.w2c)
                runtime_intrinsics[int(camera)] = np.asarray(view.intrinsic, np.float32)
                runtime_sizes[int(camera)] = tuple(int(value) for value in view.size)
                input_intrinsics[int(camera)] = np.asarray(view.intrinsic, np.float32)
            start_ordinal = 0
            conditioned_chunks = _camera_chunks(
                camera_ids, max_views, anchor_count=0, minimum_chunk_size=3,
            )

        per_view_cap = pre_fuse_per_view_cap(max_points, len(camera_ids))
        for ordinal in range(start_ordinal, len(frame_positions)):
            frame = int(dataset.frames[int(frame_positions[ordinal])])
            frame_reference = float(
                dataset.view(frame, candidate_camera_ids[0]).timestamp
            )
            frame_bundles: list[_PredictionBundle] = []
            frame_bundle_cameras: list[list[int]] = []
            frame_diagnostic_start = len(diagnostics_rows)
            for chunk_index, camera_chunk in enumerate(conditioned_chunks):
                views, reference_timestamp, residuals = _synchronized_views(
                    dataset, frame, camera_chunk, options,
                    reference_timestamp=frame_reference,
                )
                extrinsics = np.stack([
                    rig_extrinsics[camera] for camera in camera_chunk
                ])
                prediction = _run_chunk_inference(
                    model, options, views,
                    frame=frame, cameras=camera_chunk,
                    timer=timer,
                    extrinsics=extrinsics,
                    intrinsics=np.stack([input_intrinsics[camera] for camera in camera_chunk]),
                )
                predicted_ext, _predicted_k = _prediction_cameras(
                    prediction, len(camera_chunk),
                )
                camera_diagnostics = _conditioned_camera_diagnostics(
                    predicted_ext, extrinsics,
                )
                with timer.stage("unproject_filter"):
                    bundle = _prediction_bundle(
                        # The input rig is authoritative. Prediction intrinsics
                        # remain authoritative in processed-image coordinates.
                        prediction, views, extrinsics, options,
                        depth_scale=1.0,
                        max_points_per_view=per_view_cap,
                        rng=rng,
                        device=postprocess_device,
                    )
                frame_bundles.append(bundle)
                frame_bundle_cameras.append(list(camera_chunk))
                diagnostics_rows.append({
                    "frame": frame,
                    "reference_timestamp": reference_timestamp,
                    "cameras": [int(value) for value in camera_chunk],
                    "actual_frames": [int(view.frame) for view in views],
                    "sync_residual_seconds": residuals,
                    "chunk": int(chunk_index),
                    "conditioned_camera_check": camera_diagnostics,
                    "geometry": {
                        "raw_points": int(sum(len(value[0]) for value in bundle.clouds)),
                        "global_filter_applied_after_all_chunks": True,
                    },
                })
            with timer.stage("global_consistency"):
                unique_frame_bundle = _unique_camera_bundle(
                    frame_bundles, frame_bundle_cameras,
                )
                frame_clouds, global_diagnostics = _filter_prediction_bundles(
                    [unique_frame_bundle], options, device=postprocess_device,
                )
            for row in diagnostics_rows[frame_diagnostic_start:]:
                row["global_geometry"] = _serializable_geometry(global_diagnostics)
            with timer.stage("voxel_fusion"):
                xyz, rgb, raw_count, fuse_voxel, fuse_report = _fuse_keyframe_cloud(
                    frame_clouds, options=options, voxel_size=voxel_size,
                    voxel_ratio=voxel_ratio, max_points=max_points, rng=rng,
                    allow_empty=bool(getattr(options, "foreground_only", False)),
                    device=postprocess_device,
                )
            result.append((xyz, rgb))
            with timer.stage("cache_write"):
                ply_path, ply_digest = write_point_ply_atomic(
                    cache_root / f"f{frame:06d}.ply", xyz, rgb,
                    allow_empty=len(xyz) == 0,
                    return_sha256=True,
                )
                ply_hashes[frame] = ply_digest
            logger.info(
                "DA3 initialization keyframe %d/%d (f%06d): %d fused points "
                "(raw=%d voxel=%.5g, dense 3-NN reference)",
                ordinal + 1, len(frame_positions), frame, len(xyz), raw_count, fuse_voxel,
            )

        if not any(len(xyz) for xyz, _rgb in result):
            raise RuntimeError(
                "DA3 produced no valid points for any keyframe after filtering "
                "and fusion; refusing to synthesize geometry"
            )
        if uncalibrated:
            dataset.update_camera_geometry(
                rig_extrinsics, runtime_intrinsics, runtime_sizes,
            )
            camera_manifest_record = _save_camera_manifest(
                cache_root, dataset.cameras,
                rig_extrinsics, runtime_intrinsics, runtime_sizes,
            )
            logger.info("saved DA3 camera rig: %s", cache_root / "camera_manifest.npz")
        dataset_identity = dataset.checkpoint_metadata()
        reference_times: dict[int, float] = {}
        for row in diagnostics_rows:
            row_frame = int(row["frame"])
            reference_times.setdefault(
                row_frame, float(row["reference_timestamp"]),
            )
        keyframes = []
        for position in frame_positions:
            frame = int(dataset.frames[int(position)])
            ply_path = cache_root / f"f{frame:06d}.ply"
            keyframes.append({
                "frame": frame,
                "reference_timestamp": reference_times[frame],
                "ply": ply_path.name,
                "ply_sha256": ply_hashes[frame],
            })
        cuda_peak_bytes: int | None = None
        try:
            import torch
            if str(device).startswith("cuda") and torch.cuda.is_available():
                with torch.cuda.device(device):
                    cuda_peak_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        manifest = {
            "schema": DA3_SCHEMA,
            "pipeline_mode": str(getattr(
                getattr(config, "pipeline", None), "mode", "scene",
            )),
            "cache_namespace": namespace,
            "dataset_fingerprint": dataset_identity["fingerprint"],
            "model_name": str(getattr(options, "model_name", "")),
            "model_path": (
                str(getattr(options, "model_path"))
                if getattr(options, "model_path", None) is not None else None
            ),
            "model_identity": loaded_model_identity,
            "camera_manifest": camera_manifest_record,
            "input_l2_identity": input_l2_identity,
            "quality": str(getattr(options, "quality", "balanced")),
            "schedule": {
                "native_hw": [int(native_hw[0]), int(native_hw[1])],
                "per_keyframe_gaussians": int(schedule.per_keyframe_gaussians),
                "upscale_clamped": bool(schedule.upscale_clamped),
                "candidate_input_views": int(len(candidate_camera_ids)),
                "fusion_views": int(len(camera_ids)),
                "selected_camera_ids": [int(value) for value in camera_ids],
                "pre_unproject_cap_per_view": int(per_view_cap),
                "min_consistent_views": int(schedule.min_consistent_views),
                "use_sky_mask": bool(schedule.use_sky_mask),
                "foreground": foreground_schedule_metadata(schedule),
            },
            "preprocess": {
                "resolution": int(schedule.process_res),
                "method": str(getattr(options, "process_res_method", "upper_bound_resize")),
                "depth_type": str(getattr(options, "depth_type", "camera_z")),
                "postprocess_device": postprocess_device,
            },
            "max_views": max_views,
            "use_sky_mask": bool(schedule.use_sky_mask),
            "requested": {
                "use_sky_mask": plan.requested_use_sky_mask,
                "quality": plan.requested_quality,
                "process_res": plan.requested_process_res,
            },
            "l1_identity": da3_l1_identity(
                config, dataset, frame_positions, schedule=schedule,
                candidate_camera_ids=candidate_camera_ids,
                selected_camera_ids=identity_cameras,
                loaded_model_identity=loaded_model_identity,
            ),
            "l2_identity": da3_l2_identity(
                config, dataset, frame_positions, schedule=schedule,
                candidate_camera_ids=candidate_camera_ids,
                selected_camera_ids=identity_cameras,
                loaded_model_identity=loaded_model_identity,
            ),
            "use_ray_pose": bool(getattr(options, "use_ray_pose", False)),
            "ref_view_strategy": str(getattr(options, "ref_view_strategy", "saddle_balanced")),
            "input_cameras": str(getattr(options, "input_cameras", "train")),
            "foreground_only": bool(getattr(options, "foreground_only", False)),
            "mask_threshold": float(getattr(options, "mask_threshold", 0.5)),
            "confidence_percentile": float(getattr(options, "confidence_percentile", 20.0)),
            "min_consistent_views": int(schedule.min_consistent_views),
            "relative_depth_tolerance": float(getattr(options, "relative_depth_tolerance", 0.02)),
            "absolute_depth_tolerance": (
                None if getattr(options, "absolute_depth_tolerance", None) is None
                else float(options.absolute_depth_tolerance)
            ),
            "absolute_depth_tolerance_ratio": float(getattr(options, "absolute_depth_tolerance_ratio", 0.002)),
            "consistency_scale_mode": str(getattr(
                options, "consistency_scale_mode", "median_depth",
            )),
            "sync_tolerance_fraction": float(getattr(options, "sync_tolerance_fraction", 0.25)),
            "fusion_voxel_ratio": float(getattr(options, "fusion_voxel_ratio", 0.002)),
            "fusion_voxel_mode": str(getattr(
                options, "fusion_voxel_mode", "spacing",
            )),
            "target_bounds": (
                None if getattr(options, "target_bounds", None) is None
                else [float(value) for value in options.target_bounds]
            ),
            "voxel_size": None if voxel_size is None else float(voxel_size),
            "max_points": max_points,
            "seed": initialization_seed(config),
            "keyframe_frames": [
                int(dataset.frames[int(position)]) for position in frame_positions
            ],
            "keyframes": keyframes,
            "diagnostics": diagnostics_rows,
            "performance": {
                "wall_seconds": float(time.perf_counter() - started),
                "cuda_peak_allocated_bytes": cuda_peak_bytes,
                "cuda_device": str(device) if cuda_peak_bytes is not None else None,
                **timer.report(),
            },
        }
        from ftgspp.io.provenance import source_identity
        source = source_identity()
        manifest["source_identity"] = {
            "schema": source["schema"],
            "set_sha256": source["set_sha256"],
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        manifest["fingerprint"] = fingerprint
        write_json_atomic(cache_root / "manifest.json", manifest)
        write_json_atomic(cache_root / "completed.json", {
            "schema": DA3_SCHEMA,
            "manifest_fingerprint": fingerprint,
            "keyframes": len(result),
        })
        return result
    finally:
        # CUDA and Vulkan allocations compete for VRAM even though their device
        # ordinals are independent.  Release DA3 before native upload.
        try:
            import torch
            del model
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - cleanup must not mask inference errors
            pass
