"""NumPy/SciPy port of FTGSPP's explicit Gaussian initialization."""

from __future__ import annotations

import copy
import json
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ftgspp.data.scene import Scene
from ftgspp.io.model import NativeGaussians, TEMPORAL_SCALE_MULTIPLIER
from ftgspp.io.ply import read_point_ply
from ftgspp.training.init_plan import (
    L3_ALGORITHM,
    initialization_seed,
    l3_identity_payload,
)
from ftgspp.training.point_sampling import select_point_indices
from ftgspp.training.stage_timer import StageTimer

SH_C0 = 0.28209479177387814
SPARSE_JITTER_SCALE = 0.25
logger = logging.getLogger("ftgspp.native")


@dataclass(frozen=True)
class KeyframeCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    reference_timestamp: float
    frame: int
    provenance: dict[str, object]


def _cloud_arrays(cloud: KeyframeCloud | tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return (cloud.xyz, cloud.rgb) if isinstance(cloud, KeyframeCloud) else cloud


def _keyframe_clouds(
    points: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    root: Path | None,
    frames: Sequence[int],
    dataset: Scene,
    source: str,
) -> list[KeyframeCloud]:
    manifest_times = _manifest_reference_timestamps(root, frames)
    manifest_path = Path(root) / "manifest.json" if root is not None else None
    manifest_sha256 = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path is not None and manifest_path.is_file() else None
    )
    anchor_camera = int(dataset.train_cameras[0])
    result: list[KeyframeCloud] = []
    for index, ((xyz, rgb), frame) in enumerate(zip(points, frames, strict=True)):
        timestamp = (
            float(manifest_times[index])
            if manifest_times is not None
            else float(dataset.view(int(frame), anchor_camera).timestamp)
        )
        result.append(KeyframeCloud(
            xyz=np.asarray(xyz, np.float32),
            rgb=np.asarray(rgb, np.uint8),
            reference_timestamp=timestamp,
            frame=int(frame),
            provenance={
                "source": source,
                "root": None if root is None else str(Path(root).resolve()),
                "manifest_sha256": manifest_sha256,
                "timestamp_source": (
                    "point_manifest" if manifest_times is not None
                    else "selected_source_training_camera"
                ),
            },
        ))
    return result


def _complete_point_clouds(
    root: Path | None, frames: Sequence[int], *, label: str,
    allow_empty: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Read a complete ``f<frame>.ply`` set, returning ``None`` if absent.

    A complete-set check is important for ``auto``: silently mixing a stale
    user cloud with newly inferred DA3 frames can produce discontinuous motion
    initialization.  Partial sets are therefore treated as a cache miss and
    regenerated as one coherent set.
    """

    if root is None:
        return None
    root = Path(root)
    paths = [root / f"f{int(frame):06d}.ply" for frame in frames]
    if not all(path.is_file() for path in paths):
        return None
    try:
        points = [read_point_ply(path) for path in paths]
    except (OSError, ValueError) as exc:
        logger.warning("ignoring invalid %s point-cloud cache: %s", label, exc)
        return None
    if any(len(xyz) == 0 for xyz, _rgb in points) and not allow_empty:
        logger.warning("ignoring empty %s point-cloud cache in %s", label, root)
        return None
    return points


def _valid_prepared_point_manifest(
    root: Path,
    dataset: Scene,
    frames: Sequence[int],
    *,
    require_foreground_compatibility: bool = False,
) -> bool:
    """Validate the optional visual-hull manifest beside explicit PLYs.

    User-authored PLY directories without a manifest remain supported.  When a
    library-generated manifest is present, however, stale masks/images or a
    replaced PLY must not be silently accepted by ``init.backend=points``.
    """

    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.is_file():
        return not require_foreground_compatibility
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = json.loads(
            (Path(root) / "completed.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or not isinstance(completed, dict):
        return False
    # A manifest means this directory is managed prepared output, rather than
    # an unversioned user PLY directory.  Unknown schemas must fail closed:
    # otherwise a newer/incompatible or partially replaced cache silently
    # bypasses every integrity check below.
    if manifest.get("schema") != "ftgspp-visual-hull-ibr-v1":
        return False
    if completed.get("schema") != "ftgspp-visual-hull-ibr-v1":
        return False
    if require_foreground_compatibility and not bool(
        manifest.get("foreground_compatible", True)
    ):
        return False
    if manifest.get("dataset_fingerprint") != dataset.checkpoint_metadata()["fingerprint"]:
        return False
    source_record = manifest.get("source_identity")
    if not isinstance(source_record, dict):
        return False
    from ftgspp.io.provenance import source_identity
    if source_record.get("set_sha256") != source_identity()["set_sha256"]:
        return False
    if completed.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        return False
    if completed.get("keyframes") != len(frames):
        return False
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return False
    by_frame: dict[int, dict[str, object]] = {}
    try:
        for row in rows:
            if not isinstance(row, dict) or row.get("frame") is None:
                return False
            frame = int(row["frame"])
            if frame in by_frame:
                return False
            by_frame[frame] = row
    except (TypeError, ValueError):
        return False
    if set(by_frame) != {int(frame) for frame in frames}:
        return False
    for frame in frames:
        row = by_frame[int(frame)]
        relative = row.get("ply")
        if not isinstance(relative, str):
            return False
        path = Path(root) / relative
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        if digest != row.get("ply_sha256"):
            return False
    return True


def _valid_da3_cache(
    config, dataset: Scene, root: Path | None, expected_frames: Sequence[int] | None = None,
    frame_positions: Sequence[int] | None = None,
    *, configured_model_identity: dict[str, object] | None = None,
) -> bool:
    if root is None:
        return False
    from ftgspp.optional.da3 import resolve_da3_cache_root
    root = resolve_da3_cache_root(Path(root))
    manifest_path = root / "manifest.json"
    completed_path = root / "completed.json"
    if not manifest_path.is_file() or not completed_path.is_file():
        logger.warning("DA3 cache lacks manifest/completed marker: %s", root)
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("DA3 cache metadata is invalid in %s: %s", root, exc)
        return False
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "ftgspp-da3-foreground-v1"
        or not isinstance(completed, dict)
        or completed.get("schema") != "ftgspp-da3-foreground-v1"
    ):
        logger.warning("DA3 cache schema is unsupported: %s", root)
        return False
    namespace = manifest.get("cache_namespace")
    namespace_dir = (
        isinstance(namespace, str)
        and len(namespace) == 64
        and all(character in "0123456789abcdef" for character in namespace)
    )
    # Published SelfCap trees flatten ``<sha256>/`` into ``points_stride1/``.
    # Only reject a hash-named directory that does not match its manifest.
    if namespace is not None and (
        not isinstance(namespace, str)
        or (namespace_dir and root.name != namespace and len(root.name) == 64)
    ):
        logger.warning("DA3 cache namespace does not match its directory: %s", root)
        return False
    options = config.init.da3
    from ftgspp.optional.da3 import (
        da3_l2_identity, model_identity as da3_model_identity,
        resolve_da3_cache_root,
    )
    from ftgspp.optional.da3_schedule import (
        input_camera_ids,
        resolve_da3_schedule_for_scene,
    )
    positions = list(frame_positions) if frame_positions is not None else None
    if positions is None and expected_frames is not None:
        try:
            positions = [
                int(dataset.frames.index(int(frame))) for frame in expected_frames
            ]
        except ValueError:
            logger.warning("DA3 cache keyframe is not in the loaded scene: %s", root)
            return False
    if not positions:
        logger.warning("DA3 cache validation needs keyframe positions: %s", root)
        return False
    # An estimated rig was inferred using all cameras. Loading that rig must
    # not change the request's camera set/schedule merely because the Scene is
    # now calibrated. Before restoration, compare the provisional input
    # identity; afterwards compare the identity with recovered geometry.
    schedule_dataset = dataset
    if manifest.get("camera_manifest") is not None:
        schedule_dataset = copy.copy(dataset)
        schedule_dataset.calibration_available = False
    schedule = resolve_da3_schedule_for_scene(config, schedule_dataset, positions)
    candidate_camera_ids = input_camera_ids(schedule_dataset, options)
    stored = manifest.get(
        "l2_identity" if getattr(dataset, "calibration_available", True)
        else "input_l2_identity"
    )
    if not isinstance(stored, dict):
        logger.warning("DA3 cache has no l2_identity; refusing silent reuse: %s", root)
        return False
    stored_selected = stored.get("selected_camera_ids")
    if not isinstance(stored_selected, list):
        stored_selected = (
            manifest.get("schedule", {}).get("selected_camera_ids", candidate_camera_ids)
            if isinstance(manifest.get("schedule"), dict) else candidate_camera_ids
        )
    configured_model = (
        da3_model_identity(options)
        if configured_model_identity is None else configured_model_identity
    )
    identity = da3_l2_identity(
        config, dataset, positions, schedule=schedule,
        candidate_camera_ids=candidate_camera_ids,
        selected_camera_ids=stored_selected,
        loaded_model_identity=configured_model,
    )
    mismatches = {
        key: (stored.get(key), value)
        for key, value in identity.items() if stored.get(key) != value
    }
    mismatches.pop("model_identity", None)
    cached_model = manifest.get("model_identity")
    if not isinstance(cached_model, dict) or any(
        cached_model.get(key) != value
        for key, value in configured_model.items()
        if key != "loaded_state_dict_sha256"
    ):
        mismatches["model_identity"] = (cached_model, configured_model)
    if mismatches:
        logger.warning("DA3 cache fingerprint inputs changed: %s", mismatches)
        return False
    if completed.get("manifest_fingerprint") != manifest.get("fingerprint"):
        logger.warning("DA3 completed marker does not match its manifest: %s", root)
        return False
    unsigned = dict(manifest)
    stored_fingerprint = unsigned.pop("fingerprint", None)
    try:
        encoded = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError):
        return False
    actual_fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if stored_fingerprint != actual_fingerprint:
        logger.warning("DA3 manifest fingerprint is invalid: %s", root)
        return False
    camera_record = manifest.get("camera_manifest")
    if camera_record is not None:
        from ftgspp.io.init_cache import file_sha256
        if not isinstance(camera_record, dict) or camera_record.get("path") != "camera_manifest.npz":
            return False
        try:
            if file_sha256(root / "camera_manifest.npz") != camera_record.get("sha256"):
                return False
        except OSError:
            return False
    keyframes = manifest.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        logger.warning("DA3 manifest has no keyframe integrity table: %s", root)
        return False
    for row in keyframes:
        if not isinstance(row, dict) or not isinstance(row.get("ply"), str):
            logger.warning("DA3 keyframe manifest row is invalid: %s", root)
            return False
        path = root / row["ply"]
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            logger.warning("DA3 cache keyframe is unreadable: %s", path)
            return False
        if digest != row.get("ply_sha256"):
            logger.warning("DA3 cache keyframe hash mismatch: %s", path)
            return False
    if completed.get("keyframes") != len(keyframes):
        logger.warning("DA3 completed marker keyframe count is invalid: %s", root)
        return False
    keyframe_frames = [
        int(row.get("frame")) for row in keyframes
        if isinstance(row, dict) and row.get("frame") is not None
    ]
    if expected_frames is not None and keyframe_frames != [
        int(frame) for frame in expected_frames
    ]:
        logger.warning("DA3 cache keyframe frame list changed: %s", root)
        return False
    return True


def _initial_points(
    config, dataset: Scene, frame_positions: Sequence[int],
) -> list[KeyframeCloud]:
    """Resolve user PLYs, a DA3 cache, or lazy DA3 inference in that order."""

    frames = [int(dataset.frames[index]) for index in frame_positions]
    backend = str(getattr(config.init, "backend", "auto"))
    points_path = getattr(config.init, "points_path", None)
    if backend in {"auto", "points"}:
        if not getattr(dataset, "calibration_available", True):
            if backend == "points":
                raise ValueError(
                    "init.backend=points cannot initialize an uncalibrated dataset; "
                    "use init.backend=da3/auto so DA3 can estimate camera poses"
                )
            logger.warning(
                "dataset has no camera calibration; ignoring user PLYs because "
                "their world coordinate system cannot be attached to the videos"
            )
        else:
            points = _complete_point_clouds(points_path, frames, label="user")
            if points is not None:
                foreground_auto = (
                    backend == "auto"
                    and str(getattr(config.pipeline, "mode", "scene")) == "foreground"
                )
                if _valid_prepared_point_manifest(
                    points_path,
                    dataset,
                    frames,
                    require_foreground_compatibility=foreground_auto,
                ):
                    logger.info("Gaussian init source: user point clouds (%s)", points_path)
                    return _keyframe_clouds(
                        points, root=points_path, frames=frames,
                        dataset=dataset, source="user_points",
                    )
                logger.warning(
                    "prepared point-cloud manifest is stale, incomplete, or not "
                    "foreground-compatible under %s",
                    points_path,
                )
                if backend == "points":
                    raise ValueError(
                        f"point-cloud manifest validation failed under {points_path}; "
                        "regenerate the prepared initialization"
                    )
        if backend == "points":
            expected = Path(points_path) if points_path is not None else "<unset>"
            raise FileNotFoundError(
                f"init.backend=points requires a complete point-cloud set in {expected}; "
                "provide PLYs or choose init.backend=da3/auto"
            )
        if points_path is not None:
            logger.warning(
                "user point-cloud set is incomplete under %s; trying DA3 initialization",
                points_path,
            )

    da3_options = getattr(config.init, "da3", None)
    cache_base = getattr(da3_options, "cache_path", None) if da3_options is not None else None
    if cache_base is None:
        cache_base = Path(config.data.root) / (
            f"points_da3_stride{int(config.init.keyframe_stride)}"
        )
    from ftgspp.optional.da3 import resolve_da3_cache_root
    cache_path = resolve_da3_cache_root(Path(cache_base))
    cached = _complete_point_clouds(
        cache_path, frames, label="DA3",
        allow_empty=bool(getattr(da3_options, "foreground_only", False)),
    )
    if cached is not None and not _valid_da3_cache(
        config, dataset, cache_path, expected_frames=frames,
        frame_positions=frame_positions,
    ):
        cached = None
    if cached is not None and getattr(dataset, "calibration_available", True):
        logger.info("Gaussian init source: cached DA3 point clouds (%s)", cache_path)
        return _keyframe_clouds(
            cached, root=cache_path, frames=frames,
            dataset=dataset, source="da3_cache",
        )
    if cached is not None:
        logger.warning(
            "DA3 point cache has no matching camera manifest; regenerating the "
            "uncalibrated initialization"
        )

    if backend not in {"auto", "da3"}:
        raise ValueError(f"unsupported init.backend: {backend}")
    try:
        from ftgspp.optional.da3 import generate_da3_points
    except Exception as exc:  # pragma: no cover - import itself is Torch-free
        raise RuntimeError("failed to import the optional DA3 initializer") from exc
    generated = generate_da3_points(config, dataset, frame_positions)
    return _keyframe_clouds(
        generated, root=generated.cache_root, frames=frames,
        dataset=dataset, source="da3_generated" if generated.generated else "da3_cache",
    )


def _l2_fingerprint_from_points(points: Sequence[KeyframeCloud]) -> str | None:
    for cloud in points:
        root = cloud.provenance.get("root")
        if not isinstance(root, str):
            continue
        path = Path(root) / "manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fingerprint = payload.get("fingerprint")
        if isinstance(fingerprint, str):
            return fingerprint
    return None


def _manifest_reference_timestamps(
    root: Path | None, frames: Sequence[int],
) -> list[float] | None:
    """Read keyframe times produced by DA3 or visual-hull preparation."""

    if root is None:
        return None
    path = Path(root) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_rows = manifest.get("keyframes", manifest.get("rows"))
    if not isinstance(raw_rows, list):
        # Early foreground-v1 DA3 manifests stored one diagnostics row per
        # camera chunk.  Accept those rows, taking the first time per frame.
        raw_rows = manifest.get("diagnostics")
    if not isinstance(raw_rows, list):
        return None
    times: dict[int, float] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        try:
            frame = int(row["frame"])
            timestamp = float(row["reference_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(timestamp):
            times.setdefault(frame, timestamp)
    if any(int(frame) not in times for frame in frames):
        return None
    return [times[int(frame)] for frame in frames]


def _initial_reference_timestamps(
    config, dataset: Scene, frame_positions: Sequence[int],
) -> list[float] | None:
    """Resolve the manifest belonging to the point source selected by init."""

    frames = [int(dataset.frames[index]) for index in frame_positions]
    backend = str(getattr(config.init, "backend", "auto"))
    points_path = getattr(config.init, "points_path", None)
    if backend in {"auto", "points"} and points_path is not None:
        paths = [Path(points_path) / f"f{frame:06d}.ply" for frame in frames]
        if all(path.is_file() for path in paths):
            timestamps = _manifest_reference_timestamps(points_path, frames)
            if timestamps is not None:
                return timestamps
    da3 = getattr(config.init, "da3", None)
    return _manifest_reference_timestamps(
        getattr(da3, "cache_path", None), frames,
    )


def _logit(value: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).eps
    value = np.clip(value, eps, 1.0-eps)
    return np.log(value / (1.0-value))


def _spacing(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.ones(1, dtype=np.float32) * 1e-3
    # ``cKDTree.query`` includes the query point itself as the first result.
    # Query one extra neighbor so the local scale is based on three *other*
    # points whenever the cloud is large enough.  Including the zero self
    # distance makes every scale systematically too small by mixing a value
    # that carries no geometric information into the RMS estimate.
    k = min(4, len(points))
    distances, _ = cKDTree(points).query(points, k=k, workers=-1)
    if distances.ndim == 1:
        distances = distances[:, None]
    nonself = distances[:, 1:] if distances.shape[1] > 1 else distances
    return np.sqrt(np.maximum(np.mean(nonself**2, axis=1), 1e-7)).astype(np.float32)


def _velocity(
    query: np.ndarray,
    reference: np.ndarray,
    delta_time: float,
    k: int,
) -> np.ndarray:
    if k <= 0 or len(reference) == 0 or len(query) == 0:
        return np.zeros_like(query)
    k = min(k, len(reference))
    _distances, indices = cKDTree(reference).query(query, k=k, workers=-1)
    if indices.ndim == 1:
        indices = indices[:, None]
    candidates = (reference[indices] - query[:, None]) / delta_time
    return np.median(candidates, axis=1).astype(np.float32)


def _select_point_indices(
    rng: np.random.Generator,
    population_size: int,
    budget: int,
    xyz: np.ndarray | None = None,
    *,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """Select source points without replacement; spatial when ``xyz`` is given."""
    return select_point_indices(
        rng, population_size, budget, xyz, diagnostics=diagnostics,
    )


def _largest_remainder(
    total: int,
    weights: np.ndarray,
    capacities: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate an integer total in O(keyframes) water-filling rounds."""

    count = len(weights)
    if total < 0 or count == 0:
        raise ValueError("invalid keyframe budget allocation")
    allocation = np.zeros(count, np.int64)
    remaining = int(total)
    active = np.ones(count, bool)
    while remaining and bool(np.any(active)):
        indices = np.flatnonzero(active)
        active_weights = np.maximum(weights[indices], 0.0)
        if not np.any(active_weights):
            active_weights = np.ones(len(indices), np.float64)
        shares = remaining * active_weights / float(active_weights.sum())
        if capacities is not None:
            rooms = capacities[indices] - allocation[indices]
            saturated = shares >= rooms
            if np.any(saturated):
                chosen = indices[saturated]
                grants = capacities[chosen] - allocation[chosen]
                allocation[chosen] += grants
                remaining -= int(grants.sum())
                active[chosen] = False
                continue
        floors = np.floor(shares).astype(np.int64)
        if capacities is not None:
            floors = np.minimum(
                floors, capacities[indices] - allocation[indices],
            )
        allocation[indices] += floors
        remaining -= int(floors.sum())
        if remaining:
            fractions = shares - np.floor(shares)
            order = indices[np.argsort(-fractions, kind="mergesort")]
            for index in order:
                if not remaining:
                    break
                if capacities is None or allocation[index] < capacities[index]:
                    allocation[index] += 1
                    remaining -= 1
        break
    if remaining:
        raise ValueError(
            "combined keyframe point population is below the requested budget"
        )
    return allocation


def allocate_keyframe_budgets(
    capacities: Sequence[int],
    target: int,
    *,
    adaptive_weights: Sequence[float] | None = None,
) -> list[int]:
    """Allocate the exact global Gaussian budget across keyframes.

    Adaptive weights grow with sqrt(point count), which responds to genuinely
    sparse views while bounding dense-cloud dominance. If the unique
    population is exhausted, all unique points are assigned first and the
    remaining budget is distributed as jittered replicas.
    """

    values = np.asarray(capacities, np.int64)
    if values.ndim != 1 or len(values) == 0 or np.any(values <= 0):
        raise ValueError("every keyframe point cloud must be non-empty")
    target = int(target)
    if target < len(values):
        raise ValueError(
            "init.num_gaussians must be at least the number of keyframes"
        )
    weights = (
        np.sqrt(values.astype(np.float64))
        if adaptive_weights is None
        else np.asarray(adaptive_weights, np.float64)
    )
    if (
        weights.shape != values.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("adaptive keyframe weights must be positive and finite")
    median = float(np.median(weights))
    if median > 0.0:
        weights = np.clip(weights, median / 4.0, median * 4.0)
    unique_total = int(values.sum())
    if target <= unique_total:
        # Reserve one Gaussian for every keyframe before weighted water
        # filling, so adaptation can never remove a temporal anchor.
        reserved = np.ones(len(values), np.int64)
        return (
            reserved + _largest_remainder(
                target-len(values), weights, values-reserved,
            )
        ).tolist()
    extra = _largest_remainder(target - unique_total, weights)
    return (values + extra).tolist()


def keyframe_budget_weights(
    clouds: Sequence[KeyframeCloud],
) -> list[float]:
    """Combine source capacity, surface complexity, and adjacent motion.

    Every factor is normalized by its keyframe median, so world units and
    capture scale cancel. Sampling is deterministic and bounded.
    """

    capacities = np.asarray([len(cloud.xyz) for cloud in clouds], np.float64)
    centroids: list[np.ndarray] = []
    extents: list[float] = []
    shape_complexity: list[float] = []
    for cloud in clouds:
        xyz = np.asarray(cloud.xyz, np.float64)
        if len(xyz) > 4096:
            indices = np.linspace(0, len(xyz)-1, 4096, dtype=np.int64)
            xyz = xyz[indices]
        center = np.median(xyz, axis=0)
        centered = xyz-center
        covariance = centered.T@centered/max(len(centered), 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        energy = float(eigenvalues.sum())
        effective_rank = (
            energy*energy/float(np.sum(eigenvalues*eigenvalues))
            if np.any(eigenvalues) else 1.0
        )
        centroids.append(center)
        extents.append(max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-8))
        shape_complexity.append(max(effective_rank, 1.0))
    motion = np.zeros(len(clouds), np.float64)
    for index in range(len(clouds)):
        distances = []
        if index:
            distances.append(float(np.linalg.norm(centroids[index]-centroids[index-1])))
        if index+1 < len(clouds):
            distances.append(float(np.linalg.norm(centroids[index+1]-centroids[index])))
        motion[index] = max(distances, default=0.0)

    def normalized(values: np.ndarray) -> np.ndarray:
        positive = values[values > 0.0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        return np.maximum(values/scale, 1e-3)

    capacity_factor = normalized(np.sqrt(capacities))
    complexity_factor = normalized(np.asarray(shape_complexity, np.float64))
    extent_factor = normalized(np.sqrt(np.asarray(extents, np.float64)))
    motion_factor = 1.0+normalized(motion)*0.25 if np.any(motion) else np.ones_like(motion)
    weights = (
        capacity_factor
        * np.sqrt(complexity_factor)
        * np.sqrt(extent_factor)
        * np.sqrt(motion_factor)
    )
    median = float(np.median(weights))
    weights = np.clip(weights, median/4.0, median*4.0)
    return [float(value) for value in weights]


def initialize_explicit(config, dataset: Scene) -> NativeGaussians:
    started = time.perf_counter()
    timer = StageTimer()
    frame_positions = list(range(0, len(dataset.frames), config.init.keyframe_stride))
    if frame_positions[-1] != len(dataset.frames)-1:
        frame_positions.append(len(dataset.frames)-1)
    with timer.stage("resolve_source"):
        points = _initial_points(config, dataset, frame_positions)
    target = int(config.init.num_gaussians)
    nonempty = [index for index, cloud in enumerate(points) if len(cloud.xyz) > 0]
    if not nonempty:
        raise ValueError(
            "no valid initialization candidates in any keyframe; refusing to "
            "synthesize random world points"
        )
    capacities = [len(cloud.xyz) for cloud in points]
    nonempty_clouds = [points[index] for index in nonempty]
    nonempty_capacities = [capacities[index] for index in nonempty]
    adaptive_weights = keyframe_budget_weights(nonempty_clouds)
    nonempty_budgets = allocate_keyframe_budgets(
        nonempty_capacities,
        target,
        adaptive_weights=adaptive_weights,
    )
    budgets = [0] * len(points)
    for index, budget in zip(nonempty, nonempty_budgets, strict=True):
        budgets[index] = int(budget)
    selected_counts = list(budgets)
    logger.info(
        "Gaussian init: keyframes=%d target=%d "
        "available=%d selected=%d resampled=%d full_clouds=%d "
        "subsampled_clouds=%d sampling=%s algorithm=%s",
        len(points), int(config.init.num_gaussians),
        sum(capacities), sum(selected_counts),
        sum(max(0, budget-capacity) for budget, capacity in zip(
            budgets, capacities, strict=True,
        )),
        sum(capacity <= budget for capacity, budget in zip(
            capacities, budgets, strict=True,
        ) if capacity > 0),
        sum(capacity > budget for capacity, budget in zip(
            capacities, budgets, strict=True,
        )),
        "adaptive_surface_grid",
        L3_ALGORITHM,
    )
    rng = np.random.default_rng(initialization_seed(config))
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in (
        "means", "log_scales", "quaternions", "opacity_logits", "sh_coefficients",
        "times", "duration_raw", "velocities", "gate_raw",
    )}
    with timer.stage("sample_velocity_scale"):
        frame_reports = _initialize_explicit_frames(
            config, dataset, points, frame_positions, budgets, rng, chunks,
        )
    nonempty_parts = [part for part in chunks["means"] if len(part)]
    if not nonempty_parts:
        raise ValueError("initialization produced no Gaussians")
    values = {
        name: np.concatenate([part for part in parts if len(part)], axis=0)
        for name, parts in chunks.items()
    }
    cuda_peak_bytes: int | None = None
    da3_performance: dict[str, object] | None = None
    for cloud in points:
        root = cloud.provenance.get("root")
        if not isinstance(root, str):
            continue
        try:
            payload = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") == "ftgspp-da3-foreground-v1" and isinstance(
            payload.get("performance"), dict,
        ):
            da3_performance = dict(payload["performance"])
            raw_peak = da3_performance.get("cuda_peak_allocated_bytes")
            cuda_peak_bytes = None if raw_peak is None else int(raw_peak)
            break
    report = {
        "schema": "ftgspp-keyframe-cloud-init-v1",
        "pipeline_mode": str(getattr(
            getattr(config, "pipeline", None), "mode", "scene",
        )),
        "backend": str(getattr(config.init, "backend", "auto")),
        "requested_gaussians": target,
        "actual_gaussians": int(len(values["means"])),
        "comparable_budget": True,
        "incomparable_reason": None,
        "keyframe_budget_weights": adaptive_weights,
        "algorithm": L3_ALGORITHM,
        "l3_identity": l3_identity_payload(
            config=config,
            l2_fingerprint=_l2_fingerprint_from_points(points),
            keyframe_frames=[cloud.frame for cloud in points],
        ),
        "replica_fraction": float(
            sum(max(0, budget - capacity) for budget, capacity in zip(
                budgets, capacities, strict=True,
            )) / max(int(len(values["means"])), 1)
        ),
        "temporal_initialization": {
            "duration_seconds": float(config.init.duration),
            "duration_multiplier": float(getattr(
                config.init, "duration_multiplier", 1.0,
            )),
            "gate_raw": (
                float(getattr(config.model, "gate_init_logit", -1.0))
                if config.model.marginal_gating else -100.0
            ),
            "marginal_gating": bool(config.model.marginal_gating),
            "marginal_threshold": float(getattr(
                config.model, "marginal_threshold", 0.05,
            )),
            "velocity_neighbors": int(config.init.num_velocity_nns),
        },
        "performance": {
            "wall_seconds": float(time.perf_counter() - started),
            "cuda_peak_allocated_bytes": cuda_peak_bytes,
            "da3": da3_performance,
            **timer.report(),
            "included_da3_generation": any(
                cloud.provenance.get("source") == "da3_generated" for cloud in points
            ),
        },
        "keyframes": [{
            "frame": cloud.frame,
            "reference_timestamp": cloud.reference_timestamp,
            "available_points": len(cloud.xyz),
            "unique_candidates": len(cloud.xyz),
            "target_budget": int(budget),
            "selected_points": int(budget),
            "unique_selected_points": int(min(budget, len(cloud.xyz))),
            "resampled_points": int(max(0, budget - len(cloud.xyz))),
            "replica_count": int(max(0, budget - len(cloud.xyz))),
            "provenance": cloud.provenance,
            **(frame_reports[index] if index < len(frame_reports) else {}),
        } for index, (cloud, budget) in enumerate(zip(points, budgets, strict=True))],
    }
    output = Path(config.run.output_path) / "initialization_manifest.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return NativeGaussians(**values, max_duration=float(config.model.max_duration))


def keyframe_neighbor_indices(capacities: Sequence[int]) -> list[int | None]:
    """Use the next nonempty frame, or the previous one at the sequence end."""

    result: list[int | None] = [None] * len(capacities)
    previous: int | None = None
    for index, capacity in enumerate(capacities):
        if capacity > 0:
            if previous is not None:
                result[previous] = index
                result[index] = previous
            previous = index
    return result


def sample_keyframe_points(
    rng: np.random.Generator, xyz: np.ndarray, budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Sample unique sources and deterministic replicas, preserving RNG order."""

    unique_budget = min(budget, len(xyz))
    sample_diag: dict[str, object] = {}
    selection_unique = _select_point_indices(
        rng, len(xyz), unique_budget, xyz=xyz,
        diagnostics=sample_diag,
    )
    resampled = budget - unique_budget
    if resampled:
        parent_choice = rng.choice(unique_budget, size=resampled, replace=True)
        parent_source = selection_unique[parent_choice]
        selection = np.concatenate([
            selection_unique, np.asarray(parent_source, np.int64),
        ])
    else:
        selection = selection_unique
        parent_choice = np.empty(0, np.int64)
    source_means = xyz[selection].astype(np.float32, copy=True)
    unique_source = xyz[selection_unique].astype(np.float32, copy=True)
    unique_spacing = _spacing(unique_source)
    if resampled:
        noise = rng.normal(size=(resampled, 3)).astype(np.float32)
        noise /= np.maximum(
            np.linalg.norm(noise, axis=1, keepdims=True), 1e-12,
        )
        radii = (
            rng.random((resampled, 1), dtype=np.float32)
            * unique_spacing[parent_choice][:, None]
            * np.float32(SPARSE_JITTER_SCALE)
        )
        source_means[-resampled:] = (
            xyz[parent_source].astype(np.float32) + noise * radii
        )
    spacing = np.concatenate([
        unique_spacing,
        unique_spacing[parent_choice] if resampled else np.empty(0, np.float32),
    ])
    return selection, source_means, spacing, sample_diag


def _initialize_explicit_frames(
    config, dataset: Scene, points, frame_positions, per_frame, rng, chunks,
    *, reference_timestamps: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    budgets = (
        [int(per_frame)] * len(points)
        if isinstance(per_frame, (int, np.integer)) else list(per_frame)
    )
    neighbors = keyframe_neighbor_indices([len(_cloud_arrays(cloud)[0]) for cloud in points])
    reports: list[dict[str, object]] = []
    for point_index, cloud in enumerate(points):
        xyz, rgb = _cloud_arrays(cloud)
        frame = dataset.frames[frame_positions[point_index]]
        t_ref = (
            float(cloud.reference_timestamp)
            if isinstance(cloud, KeyframeCloud)
            else float(reference_timestamps[point_index])
            if reference_timestamps is not None
            else float(dataset.view(frame, dataset.cameras[0]).timestamp)
        )
        budget = int(budgets[point_index])
        if budget <= 0 or len(xyz) == 0:
            for name, width in (
                ("means", 3), ("log_scales", 3), ("quaternions", 4),
                ("opacity_logits", 1), ("times", 1), ("duration_raw", 1),
                ("velocities", 3), ("gate_raw", 1),
            ):
                chunks[name].append(np.empty((0, width), np.float32))
            chunks["sh_coefficients"].append(np.empty((0, 16, 3), np.float32))
            reports.append({"skipped": True, "reason": "empty_or_zero_budget"})
            continue
        other_index = neighbors[point_index]
        if other_index is None:
            ref_xyz = xyz
            delta_time = 1.0 / max(float(dataset.fps), 1e-6)
        else:
            other_cloud = points[other_index]
            ref_xyz = _cloud_arrays(other_cloud)[0]
            other_frame = dataset.frames[frame_positions[other_index]]
            other_time = (
                float(other_cloud.reference_timestamp)
                if isinstance(other_cloud, KeyframeCloud)
                else float(reference_timestamps[other_index])
                if reference_timestamps is not None
                else float(dataset.view(other_frame, dataset.cameras[0]).timestamp)
            )
            delta_time = other_time - t_ref
            if not np.isfinite(delta_time) or abs(delta_time) < 1e-12:
                raise ValueError(
                    "point-cloud keyframe reference timestamps must be finite "
                    "and distinct"
                )
        unique_budget = min(budget, len(xyz))
        resampled = budget - unique_budget
        selection, source_means, spacing, sample_diag = sample_keyframe_points(rng, xyz, budget)
        n = len(selection)
        times = (
            np.full((n, 1), t_ref, np.float32)
            if other_index is None
            else (rng.random((n, 1), dtype=np.float32) - 0.5) * delta_time + t_ref
        )
        velocity = (
            np.zeros_like(source_means)
            if other_index is None else _velocity(
                source_means,
                ref_xyz,
                delta_time,
                config.init.num_velocity_nns,
            )
        )
        means = source_means
        if other_index is not None:
            shift = (times - np.float32(t_ref)) * velocity
            means = source_means + shift.astype(np.float32)
        opacity = np.full((n, 1), config.init.opacity, np.float32)
        chunks["means"].append(means)
        chunks["log_scales"].append(
            np.log(np.maximum(config.init.scale * spacing[:, None], 1e-12)).repeat(3, axis=1)
        )
        quaternion = np.zeros((n, 4), np.float32)
        quaternion[:, 0] = 1.0
        chunks["quaternions"].append(quaternion)
        chunks["opacity_logits"].append(_logit(opacity))
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
        gate = (
            float(getattr(config.model, "gate_init_logit", -1.0))
            if config.model.marginal_gating else -100.0
        )
        chunks["gate_raw"].append(np.full((n, 1), gate, np.float32))
        restored = means - (times - np.float32(t_ref)) * velocity
        reports.append({
            "sampling": sample_diag,
            "unique_selected_points": int(unique_budget),
            "replica_count": int(resampled),
            "source_restore_mae": float(np.mean(np.abs(restored - source_means))),
        })
    return reports
