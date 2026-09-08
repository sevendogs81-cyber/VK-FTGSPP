import math
from pathlib import Path

from ftgspp.config.schema import AppConfig, DEFAULT_FPS


RELOCATION_MODES = {"partial_copy", "exact_copy", "3d_mcmc"}
SCORE_MODES = {"default", "gate_included"}
PIPELINE_STAGES = {"init", "train", "export"}
VIEW_LOAD_MODES = {"lazy", "resident"}
IMAGE_CACHE_DEVICES = {"cpu", "gpu"}
DISTRIBUTED_MODES = {"off", "sync"}
DISTRIBUTED_REDUCTIONS = {"mean"}
DISTRIBUTED_BACKENDS = {"auto", "hierarchical_host"}
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
    "accumulation": "microsteps",
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
    fps = config.data.fps if config.data.fps > 0 else DEFAULT_FPS
    config.data.fps = fps
    num_frames = max(config.data.frames.stop - config.data.frames.start, 1)
    config.model.max_duration = num_frames / fps
    config.init.duration = max(config.init.keyframe_stride, 1) / fps
    world_size = (
        len(config.distributed.device_ids)
        if config.distributed.mode == "sync"
        else 1
    )
    microsteps = (
        int(config.distributed.accumulation.microsteps)
        if config.distributed.mode == "sync"
        else 1
    )
    config.distributed.global_batch_size = (
        world_size * microsteps * config.train.batch_size
    )
    apply_world_size_training_policy(config, world_size=world_size)
    return config


def apply_world_size_training_policy(
    config: AppConfig, *, world_size: int,
) -> None:
    """Apply explicit large-batch LR / step compensation in place."""

    policy = str(config.distributed.world_size_lr_scaling)
    if policy == "none" or world_size <= 1:
        return
    if policy in {"linear_lr", "sqrt_lr", "effective_batch_adam"}:
        batch = int(config.distributed.global_batch_size)
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
    if config.run.runtime != "native":
        raise ValueError(
            "run.runtime must be 'native' "
            "(Torch/legacy training path has been removed)"
        )
    if config.data.frames.stop <= config.data.frames.start:
        raise ValueError("data.frames.stop must be greater than data.frames.start")
    if config.init.keyframe_stride <= 0:
        raise ValueError("init.keyframe_stride must be greater than zero")
    if config.init.num_gaussians <= 0:
        raise ValueError("init.num_gaussians must be greater than zero")
    if config.train.iterations <= 0:
        raise ValueError("train.iterations must be greater than zero")
    if config.train.seed < 0:
        raise ValueError("train.seed must be non-negative")
    if config.train.seed > 0xFFFFFFFF:
        raise ValueError("train.seed must fit the native uint32 range")
    if config.train.seed + config.train.iterations > 0xFFFFFFFF:
        raise ValueError(
            "train.seed + train.iterations must fit the uint32 phase-seed range"
        )
    if config.train.batch_size <= 0:
        raise ValueError("train.batch_size must be greater than zero")
    if config.train.batch_size != 1:
        raise ValueError(
            "the native optimizer currently requires train.batch_size=1 per "
            "device; distributed global batch is the synchronous world size"
        )
    if config.train.checkpoint_interval < 0:
        raise ValueError("train.checkpoint_interval must not be negative")
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
    unknown_stages = sorted(set(config.pipeline.stages) - PIPELINE_STAGES)
    if unknown_stages:
        raise ValueError(f"unsupported pipeline stages: {unknown_stages}")
    if not config.run.export_filename or Path(config.run.export_filename).name != config.run.export_filename:
        raise ValueError("run.export_filename must be a filename, not a path")
    if config.run.view_load_mode not in VIEW_LOAD_MODES:
        raise ValueError(
            f"unsupported run.view_load_mode: {config.run.view_load_mode} "
            f"(expected one of {sorted(VIEW_LOAD_MODES)})"
        )
    if config.run.image_cache_device not in IMAGE_CACHE_DEVICES:
        raise ValueError(
            f"unsupported run.image_cache_device: {config.run.image_cache_device} "
            f"(expected one of {sorted(IMAGE_CACHE_DEVICES)})"
        )
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
    accumulation = distributed.accumulation
    if accumulation.microsteps not in {1, 2, 4}:
        raise ValueError(
            "distributed.accumulation.microsteps must be one of 1, 2, or 4"
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
    if collective.readback_ring_depth <= 0 or collective.upload_ring_depth <= 0:
        raise ValueError(
            "distributed.host_collective ring depths must be greater than zero"
        )
    if collective.timeout_seconds <= 0:
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
                "P=2 hierarchical_host requires pair_exchange_gpu_merge"
            )
    if distributed.mode == "sync" and config.train.lpips_loss:
        raise ValueError(
            "train.lpips_loss is a single-device Torch host plugin and is not "
            "supported by synchronous Vulkan training"
        )
    expected_global_batch = (
        len(distributed.device_ids) * accumulation.microsteps
        if distributed.mode == "sync" else 1
    ) * config.train.batch_size
    if distributed.global_batch_size != expected_global_batch:
        raise ValueError(
            "distributed.global_batch_size is derived from train.batch_size and "
            f"world size; expected {expected_global_batch}, got "
            f"{distributed.global_batch_size}"
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
