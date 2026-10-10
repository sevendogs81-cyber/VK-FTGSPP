from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike as OSPathLike
from pathlib import Path

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
class PipelineConfig:
    """Select the public end-to-end task profile.

    ``scene`` performs mask-free full-scene reconstruction. ``foreground``
    makes masks mandatory and selects foreground-aware initialization,
    composite_v1 training, foreground_v1 evaluation, and model selection
    defaults as one profile.
    """

    mode: str = "scene"

    @property
    def image_objective(self) -> str:
        return "composite_v1" if self.mode == "foreground" else "scene_rgb_v1"

    @property
    def eval_protocol(self) -> str:
        return "foreground_v1" if self.mode == "foreground" else "scene_rgb_v1"


@dataclass
class ModelConfig:
    marginal_gating: bool = False
    max_duration: float = 0.0
    marginal_threshold: float = 0.05
    gate_init_logit: float = -1.0


@dataclass
class MaskConfig:
    """Foreground/validity inputs shared by initialization and training.

    ``none`` selects mask-free scene reconstruction, ``alpha`` consumes the
    source image alpha channel, and ``external`` resolves one grayscale mask
    per RGB view below ``root`` (or through ``manifest_path``).
    """

    source: str = "none"
    root: Path | None = None
    manifest_path: Path | None = None
    validity_root: Path | None = None
    required: bool = False
    rgb_encoding: str = "straight"
    fingerprint: str | None = None


@dataclass
class DataConfig:
    name: str = MISSING
    fps: float = DEFAULT_FPS
    root: Path = MISSING
    video_path: Path = MISSING
    calibration_path: Path = MISSING
    extracted_path: Path = MISSING
    # Optional canonical image index. This removes filename/layout assumptions
    # while calibration and mask manifests keep their existing contracts.
    view_manifest: Path | None = None
    scale: float = MISSING
    frames: IntervalConfig = field(default_factory=IntervalConfig)
    train_cameras: list[int] | None = None
    eval_cameras: list[int] | None = None
    camera_split_source: str | None = None
    masks: MaskConfig = field(default_factory=MaskConfig)
    # An optional locked final-test manifest is consumed only by the explicit
    # offline test command. It never participates in best-checkpoint choice.
    test_manifest: Path | None = None
    # If a synchronization file is present, every discovered camera must
    # have an explicit offset. Set false only to opt into a warned 0-second
    # fallback for incomplete metadata.
    strict_time_offsets: bool = True


@dataclass
class DA3InitConfig:
    """Optional DA3 geometry/color initializer settings.

    The upstream package is intentionally not imported by the core runtime.
    ``model_name`` may be a Hugging Face repo id; ``model_path`` is preferred
    for offline runs.  The small/base checkpoint is the default compromise;
    users can select ``depth-anything/DA3-LARGE-1.1`` for higher quality.
    """

    model_name: str = "depth-anything/DA3-BASE"
    model_path: Path | None = None
    device: str = "auto"
    # ``auto`` (default) derives a speed/quality preset from
    # ``num_gaussians // keyframes``. ``fast`` / ``balanced`` / ``high`` pin it.
    # Null process_res / max_views / max_points follow the resolved preset
    # and the native capture size (never upscaled).
    quality: str = "auto"
    process_res: int | None = None
    process_res_method: str = "upper_bound_resize"
    # Backprojection is defined only for optical-axis (camera-z) depth.  This
    # is explicit so a future Euclidean/ray-distance model cannot be consumed
    # with the wrong geometry silently.
    depth_type: str = "camera_z"
    # DA3 forward-batch size (VRAM). Null follows quality. Fusion still
    # unprojects every ``input_cameras`` view, split into chunks of this size.
    max_views: int | None = None
    # Number of representative cameras fused per keyframe. Null is automatic:
    # small rigs fuse every input camera, dense arrays pick a spread subset
    # from the per-keyframe Gaussian budget. Explicit quality presets retain
    # all input cameras unless this is set.
    fusion_views: int | None = None
    # Cap on the fused dense cache (3-NN reference), after voxel merge.
    # Null sizes it from num_gaussians // keyframes; it is not the Gaussian count.
    max_points: int | None = None
    # World-space fusion voxel. None infers 0.5 x median neighbor spacing.
    voxel_size: float | None = None
    confidence_percentile: float = 20.0
    # Null follows the schedule: on for monocular fusion, off once a second
    # view can veto sky floaters. Indoor walls are often labeled sky.
    use_sky_mask: bool | None = None
    use_ray_pose: bool = False
    ref_view_strategy: str = "saddle_balanced"
    input_cameras: str = "train"
    cache_path: Path | None = None
    foreground_only: bool = False
    mask_threshold: float = 0.5
    # ``None`` is resolved by the shared DA3 schedule: one view when only
    # one camera is fused, otherwise two. Scene and foreground share this.
    min_consistent_views: int | None = None
    sync_tolerance_fraction: float = 0.25
    relative_depth_tolerance: float = 0.02
    # Absolute cross-view depth tolerance is either explicit or scaled from a
    # robust median positive depth. It is intentionally independent of fusion
    # voxelization and predicted world-space cloud extent.
    absolute_depth_tolerance: float | None = None
    absolute_depth_tolerance_ratio: float = 0.002
    consistency_scale_mode: str = "median_depth"
    fusion_voxel_ratio: float = 0.002
    # ``spacing`` infers voxel size from local neighbor spacing. ``extent`` is
    # an explicit compatibility policy using fusion_voxel_ratio.
    fusion_voxel_mode: str = "spacing"
    # Optional locked world-space [xmin,ymin,zmin,xmax,ymax,zmax].  When set,
    # relative geometry tolerances and voxel sizes use this extent instead of
    # a prediction-dependent cloud extent.
    target_bounds: list[float] | None = None
    pixel_transform_convention: str = "source_index_scale"


@dataclass
class InitConfig:
    # Scene-only ``auto`` uses the configured PLY set, then DA3. Foreground
    # mode requires an explicit source: ``carving`` (default),
    # ``point_filtering``, ``points``, or ``da3``. The first two require a
    # versioned preparation manifest recording the selected method.
    backend: str = "auto"
    points_path: Path | None = MISSING
    keyframe_stride: int = MISSING
    # Null is an input-only request to count a user-supplied points set.
    # Resolution freezes a positive integer before a run plan is returned.
    num_gaussians: int | None = MISSING
    # Geometry/L3 seed is independent from train.seed when set.  Leaving it
    # null preserves the historical behavior for existing configurations.
    seed: int | None = None
    sh_degree: int = MISSING
    scale: float = MISSING
    # Initial temporal standard deviation in seconds. ``None`` derives it
    # from keyframe_stride / data.fps; an explicit value is never overwritten.
    duration: float | None = None
    duration_multiplier: float = 1.0
    opacity: float = 0.1
    num_velocity_nns: int = 0
    da3: DA3InitConfig = field(default_factory=DA3InitConfig)


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
class BackgroundConfig:
    """Versioned image-objective background policy."""

    mode: str = "fixed"
    color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    seed: int = 0
    rank_policy: str = "shared"
    hard_phase: str = "inherit"


@dataclass
class TrainConfig:
    # Task-specific input budget: preview / balanced / extended. Resolution
    # freezes all optimizer clocks; this label never reinterprets a resume.
    preset: str | None = None
    iterations: int = MISSING
    # One Adam update averages four sequential views on the same GPU.
    batch_size: int = 1
    lrs: dict[str, float] = field(default_factory=dict)
    lr_schedules: dict[str, float] = field(default_factory=dict)
    relocation: RelocationConfig = field(default_factory=RelocationConfig)
    color_correction: bool = False
    color_correction_start: int = 3000
    color_corrector_weight: float = 1e-3
    hard_separation: bool = False
    hard_separation_steps: int = 3000
    hard_relocation_steps: list[int] = field(
        default_factory=lambda: [1500]
    )
    seed: int = 0
    resume: bool = False
    resume_checkpoint: Path | None = None
    # Shared hold-out eval / optional optimizer-snapshot cadence. Zero
    # disables in-loop snapshots; the last step is still evaluated when
    # eval views exist. Writing ckpt files is gated by save_checkpoints.
    checkpoint_interval: int = 2000
    save_checkpoints: bool = True
    # Stable scalar/log reporting cadence for terminals, run.log and JSONL.
    log_interval: int = 2000
    # gaussians.npz / PLY come from the best hold-out joint PSNR+SSIM
    # score, not the last optimizer step. Falls back to last when there is
    # no eval.
    export_best: bool = True
    background: BackgroundConfig = field(default_factory=BackgroundConfig)


@dataclass
class EvalConfig:
    frame_interval: int = 10
    background_colors: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    # Joint ranking pairs task PSNR with its SSIM: scene uses full-frame
    # PSNR+SSIM, foreground uses foreground_psnr+mask-interior foreground_ssim.
    # Foreground always audits the exported model with the full diagnostic
    # suite; in-loop eval is GPU ranking scalars only.
    selector: str = "psnr"
    opacity_threshold: float = 0.5
    selector_min_silhouette_iou: float | None = None
    selector_max_background_opacity_leakage: float | None = None
    selector_max_foreground_holes: float | None = None
    selector_min_boundary_fscore: float | None = None
    # Fixed, camera/time-stratified subset for in-loop candidate ranking.
    # Zero evaluates the entire validation split at every checkpoint.
    candidate_max_views: int = 32
    temporal_cutoffs: list[float] = field(
        default_factory=lambda: [0.0, 0.005, 0.01, 0.02, 0.05]
    )


@dataclass
class DistributedClockConfig:
    """Stable names for the optimizer, sample, and wall clocks."""

    optimizer: str = "optimizer_steps"
    samples: str = "views_seen"
    wall: str = "wall_seconds"


@dataclass
class HostCollectiveConfig:
    """Exact pair-host collective policy and ABI-shaping options."""

    algorithm: str = "auto"
    # Zero benchmarks 8/16/32/64 MiB on both devices at startup. A positive
    # value is an explicit fixed override.
    bucket_mb: int = 0
    timeout_seconds: float = 120.0
    sparse_gradients: bool = True
    codec_policy: str = "auto_exact"
    reduction_tree: str = "fixed_rank_order_v1"
    topology_policy: str = "device_uuid_pci_numa_v1"


@dataclass
class DistributedConfig:
    """Synchronous replicated-data-parallel training options.

    Each optimizer update consumes exactly one view per device.  Consequently
    the effective global batch is ``P`` in ``sync`` mode and one in ``off``
    mode; local gradient accumulation is intentionally not part of the public
    training contract. The first community release documents single-device
    (``off``) training; two-device ``sync`` remains in development.
    ``sampling_policy`` controls view sampling for both single-device and
    synchronous modes. Both modes default to ``flat_epoch`` so every training
    view is consumed exactly once per randomized epoch.
    """

    mode: str = "off"
    device_ids: list[int] = field(default_factory=list)
    reduction: str = "mean"
    reduction_dtype: str = "fp32"
    # auto is the only alias and always resolves to pair_host.
    backend: str = "auto"
    clocks: DistributedClockConfig = field(default_factory=DistributedClockConfig)
    host_collective: HostCollectiveConfig = field(
        default_factory=HostCollectiveConfig
    )
    # Complete flat (frame, camera) epoch coverage is the default in every mode.
    sampling_policy: str = "flat_epoch"
    verify_replica_interval: int = 0
    # Resume rejects a changed world size by default because it changes the
    # global batch and Adam trajectory. This flag is an explicit acknowledgement.
    allow_resume_world_size_change: bool = False
    # Large-batch compensation relative to single-device training. Default
    # ``none`` preserves today's contract (no automatic LR/step scaling).
    # ``linear_lr`` / ``sqrt_lr`` scale optimizer learning rates by P or
    # sqrt(P). ``effective_batch_adam`` uses the fixed synchronous batch P.
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
    export_filename: str = "gaussians_4dgs.ply"
    overwrite_completed: bool = False
    # Native training always uses lazy view loading: register camera metadata,
    # then decode/bind the selected image per optimizer step.
    view_rgb_cache: int = 64
    # Parallel JPEG decode workers for bounded lazy lookahead. Zero disables
    # lookahead and decodes the current step synchronously.
    view_load_workers: int = 8


@dataclass
class AppConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    init: InitConfig = field(default_factory=InitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)
