from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike as OSPathLike
from pathlib import Path
from typing import Self

from omegaconf import MISSING

type PathLike = str | OSPathLike[str]


@dataclass
class IntervalConfig:
    start: int = MISSING
    stop: int = MISSING

    def into_range(self) -> range:
        return range(self.start, self.stop)

    def into_slice(self) -> slice:
        return slice(self.start, self.stop)


DEFAULT_FPS = 60.0


@dataclass
class ModelConfig:
    marginal_gating: bool = False
    max_duration: float = 0.0


@dataclass
class DataConfig:
    name: str = MISSING
    fps: float = DEFAULT_FPS
    root: Path = MISSING
    video_path: Path = MISSING
    calibration_path: Path = MISSING
    extracted_path: Path = MISSING
    scale: float = MISSING
    frames: IntervalConfig = field(default_factory=IntervalConfig)
    train_cameras: list[int] | None = None
    eval_cameras: list[int] | None = None
    camera_split_source: str | None = None
    # If a synchronization file is present, every discovered camera must
    # have an explicit offset. Set false only to opt into a warned 0-second
    # fallback for incomplete legacy metadata.
    strict_time_offsets: bool = True


@dataclass
class InitConfig:
    points_path: Path = MISSING
    keyframe_stride: int = MISSING
    num_gaussians: int = MISSING
    sh_degree: int = MISSING
    scale: float = MISSING
    duration: float = 0.0
    opacity: float = 0.1
    num_velocity_nns: int = 0
    temporal_motion_adapted: bool = False
    temporal_flow_path: Path = Path("_flow")
    temporal_covis_thresh: float = 0.3


@dataclass
class RelocationConfig:
    start: int = MISSING
    stop: int = MISSING
    every: int = MISSING
    opacity_threshold: float = MISSING
    mode: str = "3d_mcmc"
    score_mode: str = "default"
    score_mode_start: int = 3000


@dataclass
class TrainConfig:
    iterations: int = MISSING
    batch_size: int = 1
    lrs: dict[str, float] = field(default_factory=dict)
    lr_schedules: dict[str, float] = field(default_factory=dict)
    relocation: RelocationConfig = field(default_factory=RelocationConfig)
    lpips_loss: bool = False
    lpips_weight: float = 0.01
    color_correction: bool = False
    color_correction_start: int = 3000
    color_corrector_weight: float = 1e-3
    hard_separation: bool = False
    seed: int = 0
    resume: bool = False
    resume_checkpoint: Path | None = None
    checkpoint_interval: int = 2000


@dataclass
class EvalConfig:
    frame_interval: int = 10


@dataclass
class DistributedAccumulationConfig:
    """Exact local micro-batch accumulation before one collective/Adam step."""

    # Production supports exact rank-local K-view means for K in {1, 2, 4}.
    microsteps: int = 1


@dataclass
class DistributedClockConfig:
    """Stable names for the four independent training clocks."""

    optimizer: str = "optimizer_steps"
    accumulation: str = "microsteps"
    samples: str = "views_seen"
    wall: str = "wall_seconds"


@dataclass
class HostCollectiveConfig:
    """Exact hierarchical-host collective policy and ABI-shaping options."""

    algorithm: str = "auto"
    # Zero benchmarks 8/16/32/64 MiB on both devices at startup. A positive
    # value is an explicit fixed override.
    bucket_mb: int = 0
    readback_ring_depth: int = 3
    upload_ring_depth: int = 3
    timeout_seconds: float = 120.0
    sparse_gradients: bool = True
    codec_policy: str = "auto_exact"
    reduction_tree: str = "fixed_rank_order_v1"
    topology_policy: str = "device_uuid_pci_numa_v1"


@dataclass
class DistributedConfig:
    """Synchronous replicated-data-parallel training options.

    The split native optimizer supports K=1/2/4 views per device and update.
    Consequently the effective global batch is ``P*K`` in ``sync`` mode and
    one in ``off`` mode.
    ``global_batch_size`` is derived and persisted for reproducibility.
    ``sampling_policy`` controls view sampling for both single-device and
    synchronous modes. Both modes default to ``flat_epoch`` so every training
    view is consumed exactly once per randomized epoch.
    """

    mode: str = "off"
    device_ids: list[int] = field(default_factory=list)
    reduction: str = "mean"
    reduction_dtype: str = "fp32"
    # auto is the only alias and always resolves to hierarchical_host.
    backend: str = "auto"
    accumulation: DistributedAccumulationConfig = field(
        default_factory=DistributedAccumulationConfig
    )
    clocks: DistributedClockConfig = field(default_factory=DistributedClockConfig)
    host_collective: HostCollectiveConfig = field(
        default_factory=HostCollectiveConfig
    )
    # Complete flat (frame, camera) epoch coverage is the default in every mode.
    sampling_policy: str = "flat_epoch"
    global_batch_size: int = 1
    verify_replica_interval: int = 0
    # Resume rejects a changed world size by default because it changes the
    # global batch and Adam trajectory. This flag is an explicit acknowledgement.
    allow_resume_world_size_change: bool = False
    # Large-batch compensation relative to single-device training. Default
    # ``none`` preserves today's contract (no automatic LR/step scaling).
    # ``linear_lr`` / ``sqrt_lr`` scale optimizer learning rates by P or
    # sqrt(P). ``effective_batch_adam`` follows FTGSPP make_optimizers with
    # sqrt(P*K) LR plus beta/epsilon scaling in the native Adam shader.
    # ``steps_div_p`` shortens train.iterations to ceil(base/P).
    world_size_lr_scaling: str = "none"
    # Wall-clock-equalized eval interval in seconds (0 disables). When set,
    # the runner records hold-out metrics on a wall timer in addition to
    # step-based eval.
    wall_eval_interval_seconds: float = 0.0


@dataclass
class RunConfig:
    runtime: str = "native"
    gpu: int = 0
    experiment_root: Path = MISSING
    output_path: Path = MISSING
    log_dir: Path = MISSING
    export_filename: str = "gaussians_4dgs.ply"
    overwrite_completed: bool = False
    # View RGB residency for native training:
    # - lazy: register cameras only; decode/bind per step (+ small LRU)
    # - resident: decode all views at startup into host/GPU buffers
    view_load_mode: str = "lazy"
    view_rgb_cache: int = 64
    # Only used when view_load_mode=resident: keep decoded RGBA on CPU or GPU.
    image_cache_device: str = "cpu"
    # Parallel JPEG decode workers for resident upload. In synchronous lazy
    # mode, zero also disables next-global-batch lookahead and makes current
    # batch decode deterministic/sequential; positive values enable the
    # coordinator's bounded per-rank lookahead pool.
    view_load_workers: int = 8


@dataclass
class PipelineConfig:
    stages: list[str] = field(
        default_factory=lambda: ["init", "train", "export"]
    )
    check_only: bool = False


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    init: InitConfig = field(default_factory=InitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        from ftgspp.config.compose import load_config

        return load_config(path)

    def save(self, path: PathLike) -> None:
        from ftgspp.config.compose import save_resolved_config

        save_resolved_config(self, path)


# Compatibility aliases.
Config = AppConfig
Interval = IntervalConfig
