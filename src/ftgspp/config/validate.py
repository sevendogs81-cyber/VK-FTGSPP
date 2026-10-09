import math
from pathlib import Path

from ftgspp.config.schema import AppConfig, DEFAULT_FPS
from ftgspp.config.training_schedule import TRAINING_PRESETS


RELOCATION_MODES = {"partial_copy", "exact_copy", "3d_mcmc"}
PIPELINE_MODES = {"scene", "foreground"}
SCORE_MODES = {"default", "gate_included"}
INIT_BACKENDS = {"auto", "carving", "point_filtering", "points", "da3"}
MASK_SOURCES = {"none", "external", "alpha"}
RGB_ENCODINGS = {"straight", "premultiplied"}
BACKGROUND_MODES = {"fixed", "random_solid"}
BACKGROUND_RANK_POLICIES = {"shared", "independent"}
BACKGROUND_HARD_PHASES = {"inherit", "fixed_black"}
EVAL_SELECTORS = {"psnr", "foreground_psnr"}
DA3_RESIZE_METHODS = {
    "upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop",
}
DA3_REF_VIEW_STRATEGIES = {"first", "middle", "saddle_balanced", "saddle_sim_range"}
DA3_QUALITIES = {"auto", "fast", "balanced", "high"}
DA3_DEPTH_TYPES = {"camera_z"}
DA3_CONSISTENCY_SCALE_MODES = {"median_depth", "predicted_extent", "target_extent"}
DA3_FUSION_VOXEL_MODES = {"spacing", "extent"}
DA3_PIXEL_CONVENTIONS = {"source_index_scale", "pixel_center"}
DISTRIBUTED_MODES = {"off", "sync"}
DISTRIBUTED_REDUCTIONS = {"mean"}
DISTRIBUTED_BACKENDS = {"auto", "pair_host"}
DISTRIBUTED_REDUCTION_DTYPES = {"fp32"}
MAX_SUPPORTED_WORLD_SIZE = 2
HOST_COLLECTIVE_ALGORITHMS = {
    "auto", "pair_exchange_gpu_merge",
}
HOST_COLLECTIVE_CODECS = {"auto_exact", "dense", "sorted_sparse"}
HOST_COLLECTIVE_REDUCTION_TREES = {
    "fixed_rank_order_v1",
}
HOST_COLLECTIVE_TOPOLOGY_POLICIES = {"device_uuid_pci_numa_v1"}
CLOCK_ABI = {
    "optimizer": "optimizer_steps",
    "samples": "views_seen",
    "wall": "wall_seconds",
}
DISTRIBUTED_SAMPLING_POLICIES = {
    "flat_epoch", "unique", "with_replacement", "co_frame",
    "torch_random_sampler",
}
DISTRIBUTED_LR_SCALING = {
    "none", "linear_lr", "sqrt_lr", "effective_batch_adam", "steps_div_p",
}


def compute_derived_fields(config: AppConfig) -> AppConfig:
    requested_fps = float(config.data.fps)
    if not math.isfinite(requested_fps):
        raise ValueError("data.fps must be finite")
    fps = requested_fps if requested_fps > 0 else DEFAULT_FPS
    config.data.fps = fps
    num_frames = max(config.data.frames.stop - config.data.frames.start, 1)
    config.model.max_duration = num_frames / fps
    if config.init.duration is None:
        config.init.duration = (
            max(config.init.keyframe_stride, 1)
            / fps
            * float(config.init.duration_multiplier)
        )
    world_size = (
        len(config.distributed.device_ids)
        if config.distributed.mode == "sync"
        else 1
    )
    return config


def apply_world_size_training_policy(
    config: AppConfig, *, world_size: int,
) -> None:
    """Apply explicit large-batch LR / step compensation in place."""

    policy = str(config.distributed.world_size_lr_scaling)
    if policy == "none" or world_size <= 1:
        return
    if policy in {"linear_lr", "sqrt_lr", "effective_batch_adam"}:
        batch = world_size
        scale_base = float(world_size)
        if policy == "effective_batch_adam":
            scale_base = float(batch)
        scale = (
            float(world_size)
            if policy == "linear_lr"
            else math.sqrt(scale_base)
        )
        config.train.lrs = {
            key: float(value) * scale for key, value in config.train.lrs.items()
        }
        return
    if policy == "steps_div_p":
        config.train.iterations = max(1, math.ceil(config.train.iterations / world_size))
        # Keep relocation window inside the shortened schedule.
        relocation = config.train.relocation
        if relocation.stop > config.train.iterations:
            relocation.stop = config.train.iterations
        if relocation.start >= relocation.stop:
            relocation.start = max(0, relocation.stop - max(relocation.every, 1))
        return
    raise ValueError(
        f"unsupported distributed.world_size_lr_scaling: {policy} "
        f"(expected one of {sorted(DISTRIBUTED_LR_SCALING)})"
    )


def validate_static_config(config: AppConfig) -> None:
    batch_size = config.train.batch_size
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size not in (1, 4):
        raise ValueError("train.batch_size must be the integer 1 or 4")
    if batch_size == 4:
        if (config.distributed.mode != "off" or config.pipeline.mode != "foreground"
                or config.train.color_correction or config.train.hard_separation
                or config.distributed.sampling_policy != "flat_epoch"
                or config.distributed.world_size_lr_scaling != "none"):
            raise ValueError(
                "batch_size=4 requires single-device foreground, flat_epoch, "
                "color_correction=false, hard_separation=false and no automatic LR scaling"
            )
    pipeline_mode = str(config.pipeline.mode)
    if config.train.preset is not None:
        if config.train.preset not in TRAINING_PRESETS.get(pipeline_mode, {}):
            raise ValueError("train.preset must be preview, balanced, or extended")
    if pipeline_mode not in PIPELINE_MODES:
        raise ValueError(
            f"unsupported pipeline.mode: {pipeline_mode} "
            f"(expected one of {sorted(PIPELINE_MODES)})"
        )
    if (
        not math.isfinite(float(config.model.marginal_threshold))
        or not 0.0 <= float(config.model.marginal_threshold) < 1.0
    ):
        raise ValueError("model.marginal_threshold must be finite and in [0, 1)")
    if not math.isfinite(float(config.model.gate_init_logit)):
        raise ValueError("model.gate_init_logit must be finite")
    if config.run.runtime != "native":
        raise ValueError(
            "run.runtime must be 'native'; it is the only training runtime"
        )
    if config.data.frames.stop <= config.data.frames.start:
        raise ValueError("data.frames.stop must be greater than data.frames.start")
    masks = config.data.masks
    if str(masks.source) not in MASK_SOURCES:
        raise ValueError(
            f"unsupported data.masks.source: {masks.source} "
            f"(expected one of {sorted(MASK_SOURCES)})"
        )
    if str(masks.rgb_encoding) not in RGB_ENCODINGS:
        raise ValueError(
            f"unsupported data.masks.rgb_encoding: {masks.rgb_encoding}"
        )
    if masks.source == "external" and masks.root is None and masks.manifest_path is None:
        raise ValueError(
            "data.masks.root or data.masks.manifest_path is required for external masks"
        )
    if masks.rgb_encoding == "premultiplied" and masks.source == "none":
        raise ValueError(
            "data.masks.rgb_encoding=premultiplied requires source=external or alpha"
        )
    if masks.required and masks.source == "none":
        raise ValueError("data.masks.required=true requires source=external or alpha")
    if pipeline_mode == "scene":
        if masks.source != "none":
            raise ValueError(
                "pipeline.mode=scene reconstructs full-frame RGB and does not "
                "accept a foreground mask; use pipeline.mode=foreground"
            )
    else:
        if masks.source == "none" or not bool(masks.required):
            raise ValueError(
                "pipeline.mode=foreground requires mandatory external or alpha masks"
            )
        if config.init.backend == "auto":
            raise ValueError(
                "pipeline.mode=foreground requires an explicit init.backend: "
                "carving (default), point_filtering, points, or da3"
            )
        if config.init.backend == "da3" and not bool(config.init.da3.foreground_only):
            raise ValueError(
                "pipeline.mode=foreground requires foreground-aware DA3 initialization"
            )
    if bool(config.init.da3.foreground_only) and masks.source == "none":
        raise ValueError(
            "init.da3.foreground_only=true requires data.masks.source=external "
            "or alpha; an all-image fallback is not a foreground initializer"
        )
    if pipeline_mode == "foreground":
        if masks.rgb_encoding != "straight":
            raise ValueError(
                f"{config.pipeline.image_objective} requires straight RGB; "
                "premultiplied inputs must be unpremultiplied during preparation"
            )
    background = config.train.background
    if str(background.mode) not in BACKGROUND_MODES:
        raise ValueError(f"unsupported train.background.mode: {background.mode}")
    color = list(background.color)
    if len(color) != 3 or not all(math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0 for v in color):
        raise ValueError("train.background.color must contain three finite values in [0, 1]")
    if str(background.rank_policy) not in BACKGROUND_RANK_POLICIES:
        raise ValueError(
            f"unsupported train.background.rank_policy: {background.rank_policy}"
        )
    if isinstance(background.seed, bool) or int(background.seed) < 0:
        raise ValueError("train.background.seed must be a non-negative integer")
    if str(background.hard_phase) not in BACKGROUND_HARD_PHASES:
        raise ValueError(
            f"unsupported train.background.hard_phase: {background.hard_phase}"
        )
    if str(config.eval.selector) not in EVAL_SELECTORS:
        raise ValueError(f"unsupported eval.selector: {config.eval.selector}")
    if not 0.0 < float(config.eval.opacity_threshold) < 1.0:
        raise ValueError("eval.opacity_threshold must be in (0, 1)")
    for name in (
        "selector_min_silhouette_iou",
        "selector_max_background_opacity_leakage",
        "selector_max_foreground_holes",
        "selector_min_boundary_fscore",
    ):
        raw = getattr(config.eval, name)
        if raw is not None and (
            not math.isfinite(float(raw)) or not 0.0 <= float(raw) <= 1.0
        ):
            raise ValueError(f"eval.{name} must be null or in [0, 1]")
    if int(config.eval.candidate_max_views) < 0:
        raise ValueError("eval.candidate_max_views must not be negative")
    cutoffs = [float(value) for value in config.eval.temporal_cutoffs]
    if (
        not cutoffs
        or any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in cutoffs)
        or cutoffs != sorted(set(cutoffs))
    ):
        raise ValueError(
            "eval.temporal_cutoffs must be sorted unique finite values in [0, 1)"
        )
    if (
        config.eval.selector == "foreground_psnr"
        and pipeline_mode != "foreground"
    ):
        raise ValueError(
            "eval.selector=foreground_psnr requires pipeline.mode=foreground"
        )
    for eval_color in config.eval.background_colors:
        if len(eval_color) != 3 or not all(
            math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0
            for v in eval_color
        ):
            raise ValueError(
                "eval.background_colors entries must contain three finite values in [0, 1]"
            )
    if not config.eval.background_colors:
        raise ValueError("eval.background_colors must contain at least one color")
    if pipeline_mode == "foreground" and any(
        abs(float(channel)) > 1e-6
        for channel in config.eval.background_colors[0]
    ):
        raise ValueError(
            "foreground eval requires eval.background_colors[0] to be black; "
            "foreground PSNR and foreground SSIM share that target"
        )
    if config.init.backend not in INIT_BACKENDS:
        raise ValueError(
            f"unsupported init.backend: {config.init.backend} "
            f"(expected one of {sorted(INIT_BACKENDS)})"
        )
    if config.init.backend in {"points", "point_filtering"} and config.init.points_path is None:
        raise ValueError(
            f"init.points_path is required when init.backend={config.init.backend}"
        )
    if config.init.backend in {"carving", "point_filtering"}:
        if pipeline_mode != "foreground":
            raise ValueError(
                f"init.backend={config.init.backend} requires pipeline.mode=foreground"
            )
        if config.init.points_path is None:
            raise ValueError(
                f"init.points_path is required when init.backend={config.init.backend}"
            )
    da3 = config.init.da3
    if not str(da3.model_name) and da3.model_path is None:
        raise ValueError(
            "DA3-backed initialization requires init.da3.model_name or model_path"
        )
    if str(da3.quality) not in DA3_QUALITIES:
        raise ValueError(
            f"unsupported init.da3.quality: {da3.quality} "
            f"(expected one of {sorted(DA3_QUALITIES)})"
        )
    if da3.process_res is not None and int(da3.process_res) <= 0:
        raise ValueError("init.da3.process_res must be null or greater than zero")
    if str(da3.process_res_method) not in DA3_RESIZE_METHODS:
        raise ValueError(
            "unsupported init.da3.process_res_method: "
            f"{da3.process_res_method}"
        )
    if str(da3.depth_type) not in DA3_DEPTH_TYPES:
        raise ValueError(
            "init.da3.depth_type must be 'camera_z'; ray-distance/euclidean "
            "depth cannot use the camera-z backprojection"
        )
    if da3.max_views is not None and int(da3.max_views) <= 0:
        raise ValueError("init.da3.max_views must be null or greater than zero")
    if da3.fusion_views is not None and int(da3.fusion_views) <= 0:
        raise ValueError("init.da3.fusion_views must be null or greater than zero")
    if da3.max_points is not None and int(da3.max_points) <= 0:
        raise ValueError("init.da3.max_points must be null or greater than zero")
    voxel_size = getattr(da3, "voxel_size", None)
    if voxel_size is not None:
        voxel_size = float(voxel_size)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError(
                "init.da3.voxel_size must be null or a positive finite length"
            )
    confidence_percentile = float(da3.confidence_percentile)
    if not math.isfinite(confidence_percentile) or not 0.0 <= confidence_percentile <= 100.0:
        raise ValueError("init.da3.confidence_percentile must be in [0, 100]")
    if not 0.0 <= float(da3.mask_threshold) <= 1.0:
        raise ValueError("init.da3.mask_threshold must be in [0, 1]")
    if da3.min_consistent_views is not None and int(da3.min_consistent_views) <= 0:
        raise ValueError(
            "init.da3.min_consistent_views must be null/auto or greater than zero"
        )
    if float(da3.sync_tolerance_fraction) < 0.0:
        raise ValueError("init.da3.sync_tolerance_fraction must be nonnegative")
    if da3.absolute_depth_tolerance is not None and (
        not math.isfinite(float(da3.absolute_depth_tolerance))
        or float(da3.absolute_depth_tolerance) <= 0.0
    ):
        raise ValueError(
            "init.da3.absolute_depth_tolerance must be null or positive and finite"
        )
    if str(da3.consistency_scale_mode) not in DA3_CONSISTENCY_SCALE_MODES:
        raise ValueError(
            "unsupported init.da3.consistency_scale_mode: "
            f"{da3.consistency_scale_mode}"
        )
    if (
        str(da3.consistency_scale_mode) == "target_extent"
        and da3.target_bounds is None
    ):
        raise ValueError(
            "init.da3.consistency_scale_mode=target_extent requires target_bounds"
        )
    if str(da3.fusion_voxel_mode) not in DA3_FUSION_VOXEL_MODES:
        raise ValueError(
            f"unsupported init.da3.fusion_voxel_mode: {da3.fusion_voxel_mode}"
        )
    if str(da3.pixel_transform_convention) not in DA3_PIXEL_CONVENTIONS:
        raise ValueError(
            "unsupported init.da3.pixel_transform_convention: "
            f"{da3.pixel_transform_convention}"
        )
    for name in (
        "relative_depth_tolerance", "absolute_depth_tolerance_ratio",
        "fusion_voxel_ratio",
    ):
        value = float(getattr(da3, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"init.da3.{name} must be finite and greater than zero")
    if da3.target_bounds is not None:
        bounds = list(da3.target_bounds)
        if len(bounds) != 6 or not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError("init.da3.target_bounds must contain six finite values")
        if any(float(bounds[axis + 3]) <= float(bounds[axis]) for axis in range(3)):
            raise ValueError("init.da3.target_bounds maxima must exceed minima")
    if str(da3.ref_view_strategy) not in DA3_REF_VIEW_STRATEGIES:
        raise ValueError(
            "unsupported init.da3.ref_view_strategy: "
            f"{da3.ref_view_strategy}"
        )
    if str(da3.input_cameras) not in {"train", "all"}:
        raise ValueError("init.da3.input_cameras must be 'train' or 'all'")
    if config.init.keyframe_stride <= 0:
        raise ValueError("init.keyframe_stride must be greater than zero")
    if config.init.num_gaussians is None:
        if config.init.backend != "points":
            raise ValueError("init.num_gaussians: null requires init.backend=points")
    elif config.init.num_gaussians <= 0:
        raise ValueError("init.num_gaussians must be greater than zero")
    init_seed = getattr(config.init, "seed", None)
    if init_seed is not None:
        if int(init_seed) < 0:
            raise ValueError("init.seed must be non-negative")
        if int(init_seed) > 0xFFFFFFFF:
            raise ValueError("init.seed must fit the uint32 range")
    if int(config.init.num_velocity_nns) < 0:
        raise ValueError("init.num_velocity_nns must not be negative")
    if (
        not math.isfinite(float(config.init.duration_multiplier))
        or float(config.init.duration_multiplier) <= 0.0
    ):
        raise ValueError("init.duration_multiplier must be positive and finite")
    duration = float(config.init.duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("init.duration must be finite and greater than zero")
    if (
        math.isfinite(config.model.max_duration)
        and duration >= 2.0 * config.model.max_duration
    ):
        raise ValueError(
            "init.duration must be less than twice model.max_duration"
        )
    if config.train.iterations <= 0:
        raise ValueError("train.iterations must be greater than zero")
    if config.train.hard_separation and not config.model.marginal_gating:
        raise ValueError(
            "train.hard_separation requires model.marginal_gating=true"
        )
    hard_steps = config.train.hard_separation_steps
    if isinstance(hard_steps, bool) or not isinstance(hard_steps, int):
        raise ValueError("train.hard_separation_steps must be an integer")
    if hard_steps <= 0:
        raise ValueError("train.hard_separation_steps must be greater than zero")
    raw_hard_relocation_steps = config.train.hard_relocation_steps
    if raw_hard_relocation_steps is None or isinstance(
        raw_hard_relocation_steps, (str, bytes)
    ):
        raise ValueError("train.hard_relocation_steps must be a list")
    hard_relocation_steps = list(raw_hard_relocation_steps)
    if any(
        isinstance(step, bool) or not isinstance(step, int)
        for step in hard_relocation_steps
    ):
        raise ValueError("train.hard_relocation_steps must contain integers")
    if hard_relocation_steps != sorted(set(hard_relocation_steps)):
        raise ValueError(
            "train.hard_relocation_steps must be strictly increasing"
        )
    if any(step <= 0 or step >= hard_steps for step in hard_relocation_steps):
        raise ValueError(
            "train.hard_relocation_steps must be inside the hard phase"
        )
    if config.train.seed < 0:
        raise ValueError("train.seed must be non-negative")
    if config.train.seed > 0xFFFFFFFF:
        raise ValueError("train.seed must fit the native uint32 range")
    if config.train.seed + config.train.iterations > 0xFFFFFFFF:
        raise ValueError(
            "train.seed + train.iterations must fit the uint32 phase-seed range"
        )
    if config.train.checkpoint_interval < 0:
        raise ValueError("train.checkpoint_interval must not be negative")
    if config.train.log_interval <= 0:
        raise ValueError("train.log_interval must be greater than zero")
    if config.eval.frame_interval <= 0:
        raise ValueError("eval.frame_interval must be greater than zero")
    relocation = config.train.relocation
    if relocation.every <= 0:
        raise ValueError("train.relocation.every must be greater than zero")
    if relocation.start < 0:
        raise ValueError("train.relocation.start must not be negative")
    if relocation.stop <= relocation.start:
        raise ValueError(
            "train.relocation.stop must be greater than train.relocation.start"
        )
    if relocation.stop > config.train.iterations:
        raise ValueError("train.relocation.stop must not exceed train.iterations")
    opacity_threshold = float(relocation.opacity_threshold)
    # Native opacity is sigmoid(logit), hence strictly inside (0, 1).  Endpoint
    # thresholds are finite but degenerate: zero disables all dead selection,
    # while one marks every Gaussian dead.  Reject both before relocation can
    # silently become a no-op or terminate the topology.
    if not math.isfinite(opacity_threshold) or not 0.0 < opacity_threshold < 1.0:
        raise ValueError(
            "train.relocation.opacity_threshold must be finite and strictly "
            "between zero and one"
        )
    if config.train.relocation.mode not in RELOCATION_MODES:
        raise ValueError(
            f"unsupported train.relocation.mode: {config.train.relocation.mode}"
        )
    if config.train.relocation.score_mode not in SCORE_MODES:
        raise ValueError(
            "unsupported train.relocation.score_mode: "
            f"{config.train.relocation.score_mode}"
        )
    if not config.run.export_filename or Path(config.run.export_filename).name != config.run.export_filename:
        raise ValueError("run.export_filename must be a filename, not a path")
    if config.run.view_rgb_cache <= 0:
        raise ValueError("run.view_rgb_cache must be greater than zero")
    if config.run.view_load_workers < 0:
        raise ValueError("run.view_load_workers must not be negative")
    distributed = config.distributed
    if distributed.mode not in DISTRIBUTED_MODES:
        raise ValueError(
            f"unsupported distributed.mode: {distributed.mode} "
            f"(expected one of {sorted(DISTRIBUTED_MODES)})"
        )
    if distributed.reduction not in DISTRIBUTED_REDUCTIONS:
        raise ValueError(
            f"unsupported distributed.reduction: {distributed.reduction}; "
            "FTGSPP synchronous training requires an FP32 mean before Adam"
        )
    if distributed.reduction_dtype not in DISTRIBUTED_REDUCTION_DTYPES:
        raise ValueError(
            "distributed.reduction_dtype must be 'fp32' for exact "
            "pre-Adam collective reduction"
        )
    if distributed.backend not in DISTRIBUTED_BACKENDS:
        raise ValueError(
            f"unsupported distributed.backend: {distributed.backend} "
            f"(expected one of {sorted(DISTRIBUTED_BACKENDS)})"
        )
    clocks = distributed.clocks
    for field, expected in CLOCK_ABI.items():
        actual = str(getattr(clocks, field))
        if actual != expected:
            raise ValueError(
                f"distributed.clocks.{field} must be {expected!r}, got "
                f"{actual!r}"
            )
    collective = distributed.host_collective
    if collective.algorithm not in HOST_COLLECTIVE_ALGORITHMS:
        raise ValueError(
            "unsupported distributed.host_collective.algorithm: "
            f"{collective.algorithm}"
        )
    if collective.bucket_mb < 0:
        raise ValueError(
            "distributed.host_collective.bucket_mb must not be negative "
            "(zero enables P=2 startup calibration)"
        )
    if not math.isfinite(collective.timeout_seconds) or collective.timeout_seconds <= 0:
        raise ValueError(
            "distributed.host_collective.timeout_seconds must be greater than zero"
        )
    if collective.codec_policy not in HOST_COLLECTIVE_CODECS:
        raise ValueError(
            "unsupported distributed.host_collective.codec_policy: "
            f"{collective.codec_policy}"
        )
    if collective.reduction_tree not in HOST_COLLECTIVE_REDUCTION_TREES:
        raise ValueError(
            "unsupported distributed.host_collective.reduction_tree: "
            f"{collective.reduction_tree}"
        )
    if collective.topology_policy not in HOST_COLLECTIVE_TOPOLOGY_POLICIES:
        raise ValueError(
            "unsupported distributed.host_collective.topology_policy: "
            f"{collective.topology_policy}"
        )
    if distributed.sampling_policy not in DISTRIBUTED_SAMPLING_POLICIES:
        raise ValueError(
            "unsupported distributed.sampling_policy: "
            f"{distributed.sampling_policy} (expected one of "
            f"{sorted(DISTRIBUTED_SAMPLING_POLICIES)})"
        )
    if distributed.world_size_lr_scaling not in DISTRIBUTED_LR_SCALING:
        raise ValueError(
            "unsupported distributed.world_size_lr_scaling: "
            f"{distributed.world_size_lr_scaling} (expected one of "
            f"{sorted(DISTRIBUTED_LR_SCALING)})"
        )
    if distributed.wall_eval_interval_seconds < 0:
        raise ValueError(
            "distributed.wall_eval_interval_seconds must not be negative"
        )
    if distributed.verify_replica_interval < 0:
        raise ValueError("distributed.verify_replica_interval must not be negative")
    if any(device < 0 for device in distributed.device_ids):
        raise ValueError("distributed.device_ids must contain non-negative Vulkan device IDs")
    if len(set(distributed.device_ids)) != len(distributed.device_ids):
        raise ValueError("distributed.device_ids must not contain duplicates")
    if distributed.mode == "sync" and len(distributed.device_ids) < 2:
        raise ValueError("distributed.mode=sync requires at least two device_ids")
    if distributed.mode == "sync" and len(distributed.device_ids) > MAX_SUPPORTED_WORLD_SIZE:
        raise ValueError(
            "distributed.mode=sync supports at most two device_ids"
        )
    if distributed.mode == "sync":
        world_size = len(distributed.device_ids)
        if (
            world_size == 2
            and collective.algorithm
            not in {"auto", "pair_exchange_gpu_merge"}
        ):
            raise ValueError(
                "P=2 pair_host requires pair_exchange_gpu_merge"
            )


def validate_resolved_config(config: AppConfig, available_cameras: list[int]) -> None:
    train = config.data.train_cameras
    eval_ = config.data.eval_cameras or []
    if not train:
        raise ValueError("resolved config contains no training cameras")
    overlap = sorted(set(train) & set(eval_))
    if overlap:
        raise ValueError(f"training and evaluation cameras overlap: {overlap}")
    available = set(available_cameras)
    unknown = sorted((set(train) | set(eval_)) - available)
    if unknown:
        raise ValueError(f"resolved cameras are missing from images: {unknown}")
