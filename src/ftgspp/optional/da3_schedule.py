"""Resolution and dense-cloud schedule for DA3 initialization.

Users set ``num_gaussians``, ``keyframe_stride``, and the frame span. This
module derives the DA3 pass from those plus native capture size. The fused PLY
is only a 3-NN reference; Gaussian means are sampled later to
``num_gaussians // keyframes``.

``quality: auto`` (the default) balances speed and quality: it stays on the
balanced preset unless the per-keyframe Gaussian budget is so small that extra
DA3 detail would be thrown away. ``high`` is opt-in. Process resolution never
upscales and is capped well below 4K.

Geometric gates are not speed knobs. Multi-view captures always require a
second supporting view, and the sky mask stays on only for monocular fusion.
Small rigs fuse every input camera; dense studio arrays keep a geometrically
spread subset so initializer cost does not grow with redundant viewpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

DA3_PATCH = 14
DA3_PRESETS = ("fast", "balanced", "high")
DA3_QUALITIES = ("auto", *DA3_PRESETS)
# Below this many Gaussians per keyframe, auto uses ``fast`` (DA3 pixels would
# mostly be discarded at Gaussian sampling). Otherwise auto uses ``balanced``.
AUTO_FAST_BUDGET = 4_000
AUTO_BASE_BUDGET = 8_333
AUTO_MIN_RES = 392
AUTO_MAX_RES = 1008
# Fused 3-NN cache relative to per-keyframe Gaussians. Eight is enough for
# spatial sampling without materializing the paper's 700k-candidate pool.
AUTO_DENSE_MULT = 8
AUTO_MAX_POINTS = 120_000
# Consumer/prosumer rigs are cheap enough to fuse entirely. Larger arrays
# still pick a spread subset from the Gaussian budget.
AUTO_SMALL_RIG_CAMERAS = 8
FOREGROUND_STATS_MAX_VIEWS = 96
FOREGROUND_FRACTION_FLOOR = 1.0 / 64.0

# Caps are 14-multiples. max_points_cap is the fused 3-NN cache, not Gaussians.
# max_views is the DA3 forward batch (VRAM), not how many cameras are fused.
_QUALITY: dict[str, dict[str, int]] = {
    "fast": {
        "res_cap": 504,
        "dense_mult": 4,
        "max_views": 4,
        "max_points_cap": 80_000,
    },
    "balanced": {
        "res_cap": 1008,
        "dense_mult": 8,
        "max_views": 6,
        "max_points_cap": 160_000,
    },
    "high": {
        "res_cap": 1512,
        "dense_mult": 12,
        "max_views": 6,
        "max_points_cap": 250_000,
    },
}


@dataclass(frozen=True)
class ForegroundCoverage:
    """Deterministic mask statistics used by the shared auto scheduler."""

    median_visible_fraction: float
    visibility_fraction: float
    sampled_views: int
    nonempty_views: int
    foreground_pixels: int
    valid_pixels: int


@dataclass(frozen=True)
class DA3Schedule:
    quality: str
    process_res: int
    max_views: int
    fusion_views: int
    max_points: int
    per_keyframe_gaussians: int
    native_hw: tuple[int, int]
    upscale_clamped: bool
    min_consistent_views: int
    use_sky_mask: bool
    foreground_fraction: float | None
    foreground_visibility_fraction: float | None
    foreground_sampled_views: int
    foreground_nonempty_views: int
    foreground_pixels: int
    foreground_valid_pixels: int


def align_process_res(length: int) -> int:
    """Round down to a DINOv2 patch multiple; keep tiny test sizes as-is."""

    size = max(1, int(length))
    aligned = (size // DA3_PATCH) * DA3_PATCH
    return aligned if aligned >= DA3_PATCH else size


def native_long_side(height: int, width: int) -> int:
    return max(int(height), int(width))


def per_keyframe_gaussians(num_gaussians: int, n_keyframes: int) -> int:
    return max(1, int(num_gaussians) // max(1, int(n_keyframes)))


def _evenly_sample(values: Sequence[int], count: int) -> list[int]:
    items = [int(value) for value in values]
    if count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, max(1, int(count)), dtype=np.int64)
    return [items[int(index)] for index in np.unique(indices)]


def foreground_coverage_for_scene(
    dataset: Any,
    frame_positions: Sequence[int],
    camera_ids: Sequence[int],
    *,
    mask_threshold: float,
    max_sampled_views: int = FOREGROUND_STATS_MAX_VIEWS,
) -> ForegroundCoverage:
    """Measure usable target size without scanning an entire large capture.

    Every candidate camera is represented when the cap permits; keyframes are
    sampled uniformly over time. Resolution scaling uses the median non-empty
    view so a temporarily absent target does not request unbounded DA3 pixels.
    The visibility fraction remains recorded as a separate diagnostic.
    """

    cameras = [int(camera) for camera in camera_ids]
    positions = [int(position) for position in frame_positions]
    if not cameras or not positions:
        raise ValueError("foreground coverage requires cameras and keyframes")
    cap = max(1, int(max_sampled_views))
    sampled_cameras = _evenly_sample(cameras, min(len(cameras), cap))
    frame_count = max(1, cap // len(sampled_cameras))
    sampled_positions = _evenly_sample(positions, min(len(positions), frame_count))
    fractions: list[float] = []
    foreground_pixels = 0
    valid_pixels = 0
    sampled_views = 0
    nonempty_views = 0
    for position in sampled_positions:
        frame = int(dataset.frames[position])
        for camera in sampled_cameras:
            view = dataset.view(frame, camera)
            mask = np.asarray(view.load_mask(), np.float32)
            validity = np.asarray(view.load_validity()) >= 0.5
            if mask.shape != validity.shape:
                raise ValueError(
                    "foreground mask and validity must have matching shapes "
                    f"for camera={camera} frame={frame}"
                )
            finite_valid = validity & np.isfinite(mask)
            count = int(np.count_nonzero(finite_valid))
            if count == 0:
                continue
            foreground = finite_valid & (mask >= float(mask_threshold))
            selected = int(np.count_nonzero(foreground))
            sampled_views += 1
            valid_pixels += count
            foreground_pixels += selected
            fraction = float(selected / count)
            if selected:
                nonempty_views += 1
                fractions.append(fraction)
    if sampled_views == 0:
        raise ValueError(
            "foreground DA3 scheduling found no valid mask pixels in its "
            "deterministic camera/keyframe sample"
        )
    if not fractions:
        raise ValueError(
            "foreground DA3 scheduling found an empty target in every sampled "
            "training view; inspect the mask manifest and target selection"
        )
    return ForegroundCoverage(
        median_visible_fraction=float(np.median(np.asarray(fractions, np.float64))),
        visibility_fraction=float(nonempty_views / sampled_views),
        sampled_views=sampled_views,
        nonempty_views=nonempty_views,
        foreground_pixels=foreground_pixels,
        valid_pixels=valid_pixels,
    )


def _quality_spec(quality: str) -> dict[str, int]:
    name = str(quality or "balanced").strip().lower()
    if name not in _QUALITY:
        raise ValueError(
            f"unsupported init.da3.quality: {quality!r} "
            f"(expected one of {list(DA3_QUALITIES)})"
        )
    return _QUALITY[name]


def infer_quality(*, num_gaussians: int, n_keyframes: int) -> str:
    """Pick a speed/quality preset from the training Gaussian budget.

    Auto never selects ``high``: 1512-long DA3 is an explicit quality opt-in.
    Automatic sizing then continuously scales process resolution, dense cache,
    representative cameras, and GPU batch from the same per-keyframe budget.
    """

    budget = per_keyframe_gaussians(num_gaussians, n_keyframes)
    return "fast" if budget <= AUTO_FAST_BUDGET else "balanced"


def resolve_quality(
    configured: Any,
    *,
    num_gaussians: int,
    n_keyframes: int,
) -> str:
    name = str(configured or "auto").strip().lower()
    if name in {"auto", ""}:
        return infer_quality(num_gaussians=num_gaussians, n_keyframes=n_keyframes)
    _quality_spec(name)
    return name


def _positive_or_none(value: Any) -> int | None:
    if value is None:
        return None
    number = int(value)
    return number if number > 0 else None


def resolve_process_res(
    configured: Any,
    *,
    native_hw: tuple[int, int],
    quality: str,
) -> tuple[int, bool]:
    """Return (process_res, upscale_clamped). Never upscales the capture."""

    spec = _quality_spec(quality)
    long_side = native_long_side(*native_hw)
    native_aligned = align_process_res(long_side)
    requested = _positive_or_none(configured)
    target = native_aligned if requested is None else requested
    if requested is None:
        target = min(native_aligned, int(spec["res_cap"]))
        target = align_process_res(target)
    clamped = target > native_aligned
    if clamped:
        target = native_aligned
    return max(1, int(target)), clamped


def resolve_max_points(
    configured: Any,
    *,
    num_gaussians: int,
    n_keyframes: int,
    quality: str,
) -> int:
    requested = _positive_or_none(configured)
    if requested is not None:
        return requested
    spec = _quality_spec(quality)
    budget = per_keyframe_gaussians(num_gaussians, n_keyframes)
    target = int(budget) * int(spec["dense_mult"])
    return int(min(max(target, 2 * budget), int(spec["max_points_cap"])))


def resolve_max_views(
    configured: Any,
    *,
    n_cameras: int,
    quality: str,
) -> int:
    """DA3 GPU batch size. Fusion still uses every input camera via chunks."""

    cameras = max(1, int(n_cameras))
    requested = _positive_or_none(configured)
    if requested is not None:
        return min(requested, cameras)
    return min(int(_quality_spec(quality)["max_views"]), cameras)


def resolve_auto_process_res(
    configured: Any,
    *,
    native_hw: tuple[int, int],
    num_gaussians: int,
    n_keyframes: int,
    foreground_fraction: float | None = None,
) -> tuple[int, bool]:
    """Scale DA3 pixels with budget, capture size, and usable target area."""

    requested = _positive_or_none(configured)
    if requested is not None:
        return resolve_process_res(
            requested, native_hw=native_hw, quality="balanced",
        )
    budget = per_keyframe_gaussians(num_gaussians, n_keyframes)
    target = int(round(
        504.0 * math.sqrt(float(budget) / float(AUTO_BASE_BUDGET))
    ))
    target = align_process_res(min(max(target, AUTO_MIN_RES), AUTO_MAX_RES))
    if foreground_fraction is not None:
        fraction = float(foreground_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("foreground_fraction must be finite and in (0, 1]")
        # DA3 candidate pixels scale approximately with long_side^2 * mask
        # fraction. Compensate a small target while preserving both the native
        # image ceiling and the existing automatic compute ceiling.
        target = align_process_res(int(round(
            target / math.sqrt(max(fraction, FOREGROUND_FRACTION_FLOOR))
        )))
    native_aligned = align_process_res(native_long_side(*native_hw))
    return (
        max(1, min(target, native_aligned, AUTO_MAX_RES)),
        bool(target > native_aligned),
    )


def resolve_auto_max_points(
    configured: Any,
    *,
    num_gaussians: int,
    n_keyframes: int,
) -> int:
    requested = _positive_or_none(configured)
    if requested is not None:
        return requested
    budget = per_keyframe_gaussians(num_gaussians, n_keyframes)
    return int(min(AUTO_DENSE_MULT * budget, AUTO_MAX_POINTS))


def resolve_fusion_views(
    configured: Any,
    *,
    n_cameras: int,
    num_gaussians: int,
    n_keyframes: int,
    automatic: bool,
) -> int:
    """Number of geometrically representative cameras fused per keyframe.

    Rigs at or below ``AUTO_SMALL_RIG_CAMERAS`` always use every input
    camera: dropping a GoPro from an eight-camera capture to save one DA3
    forward is not worth the coverage hole. Dense arrays still subsample by
    Gaussian budget so a 24-camera studio does not pay RoMa-like cost.
    """

    cameras = max(1, int(n_cameras))
    requested = _positive_or_none(configured)
    if requested is not None:
        return min(requested, cameras)
    if not automatic:
        return cameras
    if cameras <= AUTO_SMALL_RIG_CAMERAS:
        return cameras
    budget = per_keyframe_gaussians(num_gaussians, n_keyframes)
    if budget <= AUTO_FAST_BUDGET:
        target = 6
    elif budget <= 50_000:
        target = 8
    else:
        target = 10
    return min(target, cameras)


def resolve_min_consistent_views(
    configured: Any,
    *,
    fusion_views: int,
) -> int:
    """Require a second view when one exists; stay at one for monocular."""

    views = max(1, int(fusion_views))
    if configured is None:
        return min(2, views)
    value = int(configured)
    if value <= 0 or value > views:
        raise ValueError(
            "init.da3.min_consistent_views must be between one and the "
            f"resolved fusion view count ({views})"
        )
    return value


def resolve_use_sky_mask(
    configured: Any,
    *,
    fusion_views: int,
) -> bool:
    """Keep the sky mask for monocular fusion; drop it once a second view exists.

    Indoor walls and windows are often labeled sky. A second supporting depth
    already rejects unbounded sky floaters, so the mask would only punch holes
    in background geometry.
    """

    if configured is not None:
        return bool(configured)
    return int(fusion_views) <= 1


def resolve_auto_max_views(
    configured: Any,
    *,
    fusion_views: int,
    process_res: int,
) -> int:
    """Choose a conservative DA3 forward batch; fusion may span batches."""

    requested = _positive_or_none(configured)
    if requested is not None:
        return min(requested, max(1, int(fusion_views)))
    cap = 8 if int(process_res) <= 560 else 6
    return min(cap, max(1, int(fusion_views)))


def input_camera_ids(dataset: Any, options: Any) -> list[int]:
    """Return cameras that may contribute geometry without hold-out leakage.

    Uncalibrated captures need every camera to estimate one rig.  Once a scene
    is calibrated, however, evaluation cameras are never DA3 inputs: using
    their RGB/depth would make the held-out split part of model construction.
    ``input_cameras=all`` is therefore rejected instead of being a misleading
    compatibility alias.
    """

    uncalibrated = not bool(getattr(dataset, "calibration_available", True))
    if uncalibrated:
        return [int(camera) for camera in dataset.cameras]
    if str(getattr(options, "input_cameras", "train")) == "all":
        raise ValueError(
            "init.da3.input_cameras=all is unsafe for a calibrated scene because "
            "it includes hold-out/evaluation cameras; use input_cameras=train"
        )
    return [int(camera) for camera in dataset.train_cameras]


def select_representative_camera_ids(
    camera_ids: Sequence[int],
    w2c_by_camera: Mapping[int, np.ndarray],
    count: int,
) -> list[int]:
    """Greedily cover camera positions and optical axes.

    The selector is deterministic and independent of numeric camera IDs. It
    reduces redundant dense rigs while preserving wide-baseline/elevation
    coverage. If calibration is degenerate, deterministic input order wins.
    """

    cameras = [int(camera) for camera in camera_ids]
    target = min(max(1, int(count)), len(cameras))
    if target >= len(cameras):
        return cameras
    centers: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for camera in cameras:
        c2w = np.linalg.inv(np.asarray(w2c_by_camera[camera], np.float64))
        centers.append(c2w[:3, 3])
        direction = c2w[:3, 2]
        norm = float(np.linalg.norm(direction))
        directions.append(direction / norm if norm > 1e-12 else np.zeros(3))
    center_array = np.stack(centers)
    centered = center_array - np.mean(center_array, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale > 1e-12:
        centered /= scale
    else:
        centered[:] = 0.0
    features = np.concatenate(
        (centered, 0.5 * np.stack(directions)), axis=1,
    )
    if not np.isfinite(features).all():
        return cameras[:target]

    selected: list[int] = [
        int(np.argmax(np.sum(
            (features - np.mean(features, axis=0, keepdims=True)) ** 2,
            axis=1,
        )))
    ]
    minimum_distance = np.full(len(cameras), np.inf, np.float64)
    while len(selected) < target:
        last = selected[-1]
        distance = np.sum((features - features[last]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
        selected.append(int(np.argmax(minimum_distance)))
    selected_set = set(selected)
    return [camera for index, camera in enumerate(cameras) if index in selected_set]


def resolve_da3_schedule_for_scene(
    config: Any,
    dataset: Any,
    frame_positions: Sequence[int],
) -> DA3Schedule:
    """Resolve auto DA3 knobs from the capture and Gaussian budget."""

    if not frame_positions:
        raise ValueError("DA3 initialization requires at least one keyframe")
    options = config.init.da3
    camera_ids = input_camera_ids(dataset, options)
    if not camera_ids:
        raise ValueError("DA3 initialization requires at least one input camera")
    reference_frame = int(dataset.frames[int(frame_positions[0])])
    native_hw = native_image_size(sizes=[
        tuple(int(value) for value in dataset.view(reference_frame, camera).size)
        for camera in camera_ids
    ])
    coverage = (
        foreground_coverage_for_scene(
            dataset, frame_positions, camera_ids,
            mask_threshold=float(getattr(options, "mask_threshold", 0.5)),
        )
        if bool(getattr(options, "foreground_only", False)) else None
    )
    return resolve_da3_schedule(
        options,
        native_hw=native_hw,
        num_gaussians=int(getattr(config.init, "num_gaussians", 0) or 0) or 200_000,
        n_keyframes=len(frame_positions),
        n_cameras=len(camera_ids),
        foreground_coverage=coverage,
    )


def resolve_da3_schedule(
    options: Any,
    *,
    native_hw: tuple[int, int],
    num_gaussians: int,
    n_keyframes: int,
    n_cameras: int,
    foreground_coverage: ForegroundCoverage | None = None,
) -> DA3Schedule:
    configured_quality = str(
        getattr(options, "quality", "auto") or "auto"
    ).strip().lower()
    automatic = configured_quality == "auto"
    quality = resolve_quality(
        configured_quality,
        num_gaussians=num_gaussians,
        n_keyframes=n_keyframes,
    )
    if automatic:
        process_res, upscale_clamped = resolve_auto_process_res(
            getattr(options, "process_res", None),
            native_hw=native_hw,
            num_gaussians=num_gaussians,
            n_keyframes=n_keyframes,
            foreground_fraction=(
                None if foreground_coverage is None
                else foreground_coverage.median_visible_fraction
            ),
        )
        max_points = resolve_auto_max_points(
            getattr(options, "max_points", None),
            num_gaussians=num_gaussians,
            n_keyframes=n_keyframes,
        )
    else:
        process_res, upscale_clamped = resolve_process_res(
            getattr(options, "process_res", None),
            native_hw=native_hw,
            quality=quality,
        )
        max_points = resolve_max_points(
            getattr(options, "max_points", None),
            num_gaussians=num_gaussians,
            n_keyframes=n_keyframes,
            quality=quality,
        )
    fusion_views = resolve_fusion_views(
        getattr(options, "fusion_views", None),
        n_cameras=n_cameras,
        num_gaussians=num_gaussians,
        n_keyframes=n_keyframes,
        automatic=automatic,
    )
    min_consistent_views = resolve_min_consistent_views(
        getattr(options, "min_consistent_views", None),
        fusion_views=fusion_views,
    )
    use_sky_mask = resolve_use_sky_mask(
        getattr(options, "use_sky_mask", None),
        fusion_views=fusion_views,
    )
    return DA3Schedule(
        quality=quality,
        process_res=process_res,
        max_views=(
            resolve_auto_max_views(
                getattr(options, "max_views", None),
                fusion_views=fusion_views,
                process_res=process_res,
            )
            if automatic else resolve_max_views(
                getattr(options, "max_views", None),
                n_cameras=fusion_views,
                quality=quality,
            )
        ),
        fusion_views=fusion_views,
        max_points=max_points,
        per_keyframe_gaussians=per_keyframe_gaussians(num_gaussians, n_keyframes),
        native_hw=(int(native_hw[0]), int(native_hw[1])),
        upscale_clamped=upscale_clamped,
        min_consistent_views=min_consistent_views,
        use_sky_mask=use_sky_mask,
        foreground_fraction=(
            None if foreground_coverage is None
            else float(foreground_coverage.median_visible_fraction)
        ),
        foreground_visibility_fraction=(
            None if foreground_coverage is None
            else float(foreground_coverage.visibility_fraction)
        ),
        foreground_sampled_views=(
            0 if foreground_coverage is None
            else int(foreground_coverage.sampled_views)
        ),
        foreground_nonempty_views=(
            0 if foreground_coverage is None
            else int(foreground_coverage.nonempty_views)
        ),
        foreground_pixels=(
            0 if foreground_coverage is None
            else int(foreground_coverage.foreground_pixels)
        ),
        foreground_valid_pixels=(
            0 if foreground_coverage is None
            else int(foreground_coverage.valid_pixels)
        ),
    )


def native_image_size(
    views: Sequence[Any] | None = None,
    *,
    sizes: Sequence[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Largest H/W among the first-keyframe input views."""

    collected: list[tuple[int, int]] = []
    if sizes is not None:
        collected.extend((int(h), int(w)) for h, w in sizes)
    if views is not None:
        for view in views:
            height, width = (int(value) for value in view.size)
            collected.append((height, width))
    if not collected:
        raise ValueError("DA3 schedule requires at least one native image size")
    return max(item[0] for item in collected), max(item[1] for item in collected)


def apply_schedule_to_options(options: Any, schedule: DA3Schedule) -> None:
    """Write resolved ints back so inference/manifest see concrete values."""

    options.quality = str(schedule.quality)
    options.process_res = int(schedule.process_res)
    options.max_views = int(schedule.max_views)
    options.fusion_views = int(schedule.fusion_views)
    options.max_points = int(schedule.max_points)
    options.min_consistent_views = int(schedule.min_consistent_views)
    options.use_sky_mask = bool(schedule.use_sky_mask)


def foreground_schedule_metadata(schedule: DA3Schedule) -> dict[str, object]:
    """Stable cache/run metadata for target-aware automatic sizing."""

    fraction = getattr(schedule, "foreground_fraction", None)
    return {
        "enabled": fraction is not None,
        "median_visible_fraction": fraction,
        "visibility_fraction": getattr(
            schedule, "foreground_visibility_fraction", None,
        ),
        "sampled_views": int(getattr(schedule, "foreground_sampled_views", 0)),
        "nonempty_views": int(getattr(schedule, "foreground_nonempty_views", 0)),
        "foreground_pixels": int(getattr(schedule, "foreground_pixels", 0)),
        "valid_pixels": int(getattr(schedule, "foreground_valid_pixels", 0)),
        "resolution_rule": (
            "budget_long_side/sqrt(max(mask_fraction,1/64));"
            "capped_by_native_and_auto_max"
            if fraction is not None else "not_applied"
        ),
    }


def pre_fuse_per_view_cap(max_points: int, n_views: int) -> int:
    """Per-view spatial cap so fusion never concatenates tens of millions of points."""

    views = max(1, int(n_views))
    raw_budget = 4 * max(1, int(max_points))
    return max(256, (raw_budget + views - 1) // views)


def inference_pixel_cost(
    native_hw: tuple[int, int], process_res: int,
) -> dict[str, int]:
    """Record 2D processed-pixel cost, not only the long-side length."""

    height, width = int(native_hw[0]), int(native_hw[1])
    long_side = max(height, width)
    scale = min(1.0, float(process_res) / float(max(long_side, 1)))
    processed_h = max(1, int(round(height * scale)))
    processed_w = max(1, int(round(width * scale)))
    return {
        "native_height": height,
        "native_width": width,
        "processed_height": processed_h,
        "processed_width": processed_w,
        "processed_pixels": int(processed_h * processed_w),
    }
