"""Immutable effective initialization plan and cache-stage identities."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from typing import Any

from ftgspp.io.provenance import stage_identity

L1_ALGORITHM = "ftgspp-da3-l1-v1"
L2_ALGORITHM = "ftgspp-da3-l2-v3"
L3_ALGORITHM = "ftgspp-init-l3-v4"

L1_SOURCE_FILES = (
    "src/ftgspp/optional/da3.py",
    "src/ftgspp/optional/da3_schedule.py",
    "src/ftgspp/training/init_plan.py",
    "src/ftgspp/data/scene.py",
    "src/ftgspp/io/provenance.py",
)
L2_SOURCE_FILES = L1_SOURCE_FILES + (
    "src/ftgspp/optional/da3_evidence.py",
    "src/ftgspp/training/point_sampling.py",
    "src/ftgspp/io/init_cache.py",
    "src/ftgspp/io/ply.py",
)
L3_SOURCE_FILES = (
    "src/ftgspp/training/initialize.py",
    "src/ftgspp/training/point_sampling.py",
    "src/ftgspp/training/init_plan.py",
)


def initialization_seed(config: Any) -> int:
    """Return the seed shared by DA3 geometry and L3 initialization.

    ``init.seed`` is optional for backward compatibility.  When omitted, the
    historical behavior derives initialization randomness from ``train.seed``.
    Setting it explicitly lets multiple training seeds reuse one identical L2
    geometry cache and one identical L3 initial model.
    """

    raw = getattr(getattr(config, "init", None), "seed", None)
    return int(getattr(config.train, "seed") if raw is None else raw)


@dataclass(frozen=True)
class EffectiveDA3Plan:
    """Resolved DA3 inputs. User config is not mutated to produce this."""

    quality: str
    process_res: int
    max_views: int
    fusion_views: int
    max_points: int
    min_consistent_views: int
    use_sky_mask: bool
    upscale_clamped: bool
    native_hw: tuple[int, int]
    per_keyframe_gaussians: int
    pixel_transform_convention: str
    requested_use_sky_mask: bool | None
    requested_quality: str
    requested_process_res: int | None


def bind_schedule_to_options(options: Any, schedule: Any) -> Any:
    """Copy ``options`` and write resolved schedule fields onto the copy."""

    bound = replace(options) if is_dataclass(options) and not isinstance(options, type) else copy.copy(options)
    from ftgspp.optional.da3_schedule import apply_schedule_to_options
    apply_schedule_to_options(bound, schedule)
    return bound


def effective_da3_plan(options: Any, schedule: Any) -> EffectiveDA3Plan:
    requested_sky = getattr(options, "use_sky_mask", None)
    return EffectiveDA3Plan(
        quality=str(schedule.quality),
        process_res=int(schedule.process_res),
        max_views=int(schedule.max_views),
        fusion_views=int(schedule.fusion_views),
        max_points=int(schedule.max_points),
        min_consistent_views=int(schedule.min_consistent_views),
        use_sky_mask=bool(getattr(schedule, "use_sky_mask", True)),
        upscale_clamped=bool(getattr(schedule, "upscale_clamped", False)),
        native_hw=tuple(int(value) for value in getattr(schedule, "native_hw", (1, 1))),
        per_keyframe_gaussians=int(getattr(schedule, "per_keyframe_gaussians", 1)),
        pixel_transform_convention=str(
            getattr(options, "pixel_transform_convention", "source_index_scale")
        ),
        requested_use_sky_mask=(
            None if requested_sky is None else bool(requested_sky)
        ),
        requested_quality=str(getattr(options, "quality", "auto") or "auto"),
        requested_process_res=(
            None if getattr(options, "process_res", None) is None
            else int(options.process_res)
        ),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_bounds(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value]


def l1_identity_payload(
    *,
    pipeline_mode: str,
    dataset_fingerprint: str,
    model_identity: Mapping[str, object],
    plan: EffectiveDA3Plan,
    candidate_camera_ids: Sequence[int],
    selected_camera_ids: Sequence[int],
    process_res_method: str,
    depth_type: str,
    use_ray_pose: bool,
    ref_view_strategy: str,
    input_cameras: str,
    foreground_only: bool,
    mask_threshold: float,
    sync_tolerance_fraction: float,
    seed: int,
    keyframe_frames: Sequence[int],
    mask_affects_rgb: bool,
    foreground_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "schema": "ftgspp-da3-l1-identity-v1",
        "algorithm": L1_ALGORITHM,
        "pipeline_mode": str(pipeline_mode),
        "dataset_fingerprint": str(dataset_fingerprint),
        "model_identity": dict(model_identity),
        "keyframe_frames": [int(value) for value in keyframe_frames],
        "candidate_camera_ids": [int(value) for value in candidate_camera_ids],
        "selected_camera_ids": [int(value) for value in selected_camera_ids],
        "effective": {
            "quality": plan.quality,
            "process_res": plan.process_res,
            "max_views": plan.max_views,
            "fusion_views": plan.fusion_views,
            "use_sky_mask": plan.use_sky_mask,
            "upscale_clamped": plan.upscale_clamped,
            "native_hw": [plan.native_hw[0], plan.native_hw[1]],
            "pixel_transform_convention": plan.pixel_transform_convention,
        },
        "preprocess": {
            "method": str(process_res_method),
            "depth_type": str(depth_type),
            "use_ray_pose": bool(use_ray_pose),
            "ref_view_strategy": str(ref_view_strategy),
            "input_cameras": str(input_cameras),
            "sync_tolerance_fraction": float(sync_tolerance_fraction),
        },
        "mask_in_network_input": bool(mask_affects_rgb),
        "foreground_only": bool(foreground_only),
        "mask_threshold": (
            float(mask_threshold) if mask_affects_rgb else None
        ),
        "foreground": None if foreground_metadata is None else dict(foreground_metadata),
        "seed": int(seed),
        "stage_identity": stage_identity(L1_SOURCE_FILES),
    }


def l2_identity_payload(
    *,
    l1: Mapping[str, object],
    plan: EffectiveDA3Plan,
    options: Any,
    pre_unproject_cap_per_view: int,
) -> dict[str, object]:
    payload = dict(l1)
    payload["schema"] = "ftgspp-da3-l2-identity-v1"
    payload["algorithm"] = L2_ALGORITHM
    payload["stage_identity"] = stage_identity(L2_SOURCE_FILES)
    payload["geometry"] = {
        "max_points": int(plan.max_points),
        "min_consistent_views": int(plan.min_consistent_views),
        "confidence_percentile": float(
            getattr(options, "confidence_percentile", 20.0)
        ),
        "mask_threshold": float(getattr(options, "mask_threshold", 0.5)),
        "foreground_only": bool(getattr(options, "foreground_only", False)),
        "relative_depth_tolerance": float(
            getattr(options, "relative_depth_tolerance", 0.02)
        ),
        "absolute_depth_tolerance": _optional_float(
            getattr(options, "absolute_depth_tolerance", None)
        ),
        "absolute_depth_tolerance_ratio": float(
            getattr(options, "absolute_depth_tolerance_ratio", 0.002)
        ),
        "consistency_scale_mode": str(
            getattr(options, "consistency_scale_mode", "median_depth")
        ),
        "fusion_voxel_ratio": float(getattr(options, "fusion_voxel_ratio", 0.002)),
        "fusion_voxel_mode": str(getattr(options, "fusion_voxel_mode", "spacing")),
        "voxel_size": _optional_float(getattr(options, "voxel_size", None)),
        "target_bounds": _optional_bounds(getattr(options, "target_bounds", None)),
        "pre_unproject_cap_per_view": int(pre_unproject_cap_per_view),
    }
    return payload


def l3_identity_payload(
    *,
    config: Any,
    l2_fingerprint: str | None,
    keyframe_frames: Sequence[int],
) -> dict[str, object]:
    init = config.init
    return {
        "schema": "ftgspp-init-l3-identity-v1",
        "algorithm": L3_ALGORITHM,
        "l2_fingerprint": l2_fingerprint,
        "keyframe_frames": [int(value) for value in keyframe_frames],
        "backend": str(getattr(init, "backend", "auto")),
        "num_gaussians": int(init.num_gaussians),
        "scale": float(init.scale),
        "opacity": float(init.opacity),
        "duration": float(init.duration),
        "duration_multiplier": float(getattr(init, "duration_multiplier", 1.0)),
        "num_velocity_nns": int(init.num_velocity_nns),
        "sh_degree": int(getattr(init, "sh_degree", 0)),
        "seed": initialization_seed(config),
        "gate_init_logit": float(getattr(config.model, "gate_init_logit", -1.0)),
        "marginal_gating": bool(config.model.marginal_gating),
        "stage_identity": stage_identity(L3_SOURCE_FILES),
    }


def namespace_hash(payload: Mapping[str, object]) -> str:
    from ftgspp.io.init_cache import fingerprint_payload
    return fingerprint_payload(dict(payload))
