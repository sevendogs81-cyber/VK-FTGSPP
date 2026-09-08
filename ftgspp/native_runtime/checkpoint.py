"""Versioned, explicit-only model and native-Adam checkpoints.

V5 checkpoints carry enough execution and dataset identity to reject unsafe
resume attempts.  A :class:`CheckpointSnapshot` owns one model/optimizer host
download; the same snapshot is written once to an immutable numbered NPZ and
``latest.json`` is atomically replaced with a small pointer.  This avoids the
historical second multi-gigabyte GPU download for ``latest.npz``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ftgspp.native_runtime.distributed import (
    HIERARCHICAL_HOST_COLLECTIVE_ABI,
    SAMPLING_PHASE_SEED_FORMULA,
    sampling_phase_seed,
)
from ftgspp.native_runtime.model import NativeGaussians

if TYPE_CHECKING:
    from ftgspp.config.schema import AppConfig
    from ftgspp.native_runtime.backend import NativeVulkanBackend
    from ftgspp.native_runtime.dataset import NativeDataset
    from ftgspp.native_runtime.distributed import DistributedPlan

FORMAT = 5
LEGACY_FORMATS = frozenset({3, 4})
LATEST_MANIFEST = "latest.json"
NATIVE_ABI = "ftgspp-vulkan-community-explicit-v5-abi-v1"
SINGLE_VIEW_SAMPLING = (
    "numpy-pcg64(phase_seed+phase_local_optimizer_step)-v2"
)
TORCH_RANDOM_SAMPLER_VIEW_SAMPLING = (
    "torch-mt19937-randomsampler-replacement-v1;"
    f"{SAMPLING_PHASE_SEED_FORMULA};"
    "policy=torch_random_sampler;"
    "step=phase_local_optimizer_step;"
    "slot=step*(world_size*microsteps)+rank*microsteps+microstep;"
    "num_samples=train_phase_steps*(world_size*microsteps)"
)
SYNC_VIEW_SAMPLING = (
    "numpy-seedsequence(phase_seed,phase_local_optimizer_step,FTGS)-choice-v2"
)
SYNC_FLAT_EPOCH_VIEW_SAMPLING = (
    "numpy-seedsequence(phase_seed,epoch,FEPM)-pcg64-permutation-v3;"
    f"{SAMPLING_PHASE_SEED_FORMULA};"
    "policy=flat_epoch;"
    "step=phase_local_optimizer_step;slot=step*world_size+rank;"
    "epoch=slot//num_train_views;"
    "offset=slot%num_train_views"
)
SYNC_K_FLAT_EPOCH_VIEW_SAMPLING = (
    "numpy-seedsequence(phase_seed,epoch,FEPM)-pcg64-permutation-v4;"
    f"{SAMPLING_PHASE_SEED_FORMULA};"
    "policy=flat_epoch;step=phase_local_optimizer_step;"
    "slot=step*(world_size*microsteps)+rank*microsteps+microstep;"
    "epoch=slot//num_train_views;offset=slot%num_train_views"
)
SYNC_CO_FRAME_VIEW_SAMPLING = (
    "numpy-seedsequence(phase_seed,phase_local_optimizer_step,COFR)-"
    "frame-then-distinct-camera-v1;"
    f"{SAMPLING_PHASE_SEED_FORMULA};"
    "policy=co_frame"
)


def checkpoint_sampler_abi(
    *, synchronous: bool, sampling_policy: str, microsteps: int = 1,
) -> str:
    """Return the exact stateless NumPy sampler contract persisted by V5.

    These strings identify the native runner's reproducible stream.  The
    ``torch_random_sampler`` policy intentionally identifies the FTGSPP
    RandomSampler-compatible stream reproduced in native Python without a
    torch runtime import.
    """

    microsteps = int(microsteps)
    if sampling_policy == "torch_random_sampler":
        suffix = (
            f";mode={'sync' if synchronous else 'off'};"
            f"microsteps={microsteps}"
        )
        return TORCH_RANDOM_SAMPLER_VIEW_SAMPLING + suffix
    if microsteps == 1:
        if sampling_policy == "flat_epoch":
            return SYNC_FLAT_EPOCH_VIEW_SAMPLING
        if sampling_policy == "co_frame":
            return SYNC_CO_FRAME_VIEW_SAMPLING
        base = SYNC_VIEW_SAMPLING if synchronous else SINGLE_VIEW_SAMPLING
        return (
            f"{base};{SAMPLING_PHASE_SEED_FORMULA};policy={sampling_policy};"
            "replacement_fallback=views_lt_world_size"
        )
    suffix = (
        f";microsteps={microsteps};"
        "slot_order=rank-major-contiguous-microsteps"
    )
    if sampling_policy == "flat_epoch":
        return SYNC_K_FLAT_EPOCH_VIEW_SAMPLING + suffix
    if sampling_policy == "co_frame":
        return SYNC_CO_FRAME_VIEW_SAMPLING + suffix
    base = SYNC_VIEW_SAMPLING if synchronous else SINGLE_VIEW_SAMPLING
    return (
        f"{base};{SAMPLING_PHASE_SEED_FORMULA};policy={sampling_policy};"
        "replacement_fallback=views_lt_global_slots" + suffix
    )


def validate_or_infer_training_clocks(
    metadata: dict[str, object],
    *,
    iteration: int,
    fallback_world_size: int = 1,
    fallback_microsteps: int = 1,
) -> tuple[dict[str, int | float], str | None]:
    """Validate four persisted clocks or apply the explicit K=1 legacy rule."""

    training_raw = metadata.get("training", {})
    if not isinstance(training_raw, Mapping):
        raise ResumeCompatibilityError("checkpoint training metadata is invalid")
    training = dict(training_raw)
    collective_raw = training.get("collective", {})
    collective = (
        dict(collective_raw) if isinstance(collective_raw, Mapping) else {}
    )
    raw_p = collective.get("P", training.get("world_size", fallback_world_size))
    raw_k = collective.get("K", fallback_microsteps)
    p = (
        int(raw_p)
        if isinstance(raw_p, (int, np.integer)) and not isinstance(raw_p, bool)
        else int(fallback_world_size)
    )
    k = (
        int(raw_k)
        if isinstance(raw_k, (int, np.integer)) and not isinstance(raw_k, bool)
        else int(fallback_microsteps)
    )
    expected_optimizer = int(iteration) + 1
    clocks_raw = training.get("clocks")
    warning: str | None = None
    if not isinstance(clocks_raw, Mapping):
        if k != 1:
            raise ResumeCompatibilityError(
                "clockless legacy checkpoint may only infer K=1"
            )
        clocks: dict[str, int | float] = {
            "optimizer_steps": expected_optimizer,
            "microsteps": expected_optimizer,
            "views_seen": expected_optimizer * p,
            "wall_seconds": 0.0,
        }
        training["clocks"] = {
            "abi": "ftgspp-four-training-clocks-v1",
            **clocks,
            "legacy_policy": (
                "infer optimizer_steps=iteration+1,K=1,"
                "views_seen=optimizer_steps*P,wall_seconds=0"
            ),
        }
        metadata["training"] = training
        warning = (
            "checkpoint had no four-clock record; applied the explicit K=1 "
            "legacy inference and reset wall_seconds to zero"
        )
        return clocks, warning

    raw = dict(clocks_raw)
    if raw.get("abi") != "ftgspp-four-training-clocks-v1":
        raise ResumeCompatibilityError("checkpoint training.clocks.abi is unsupported")
    integer_fields = ("optimizer_steps", "microsteps", "views_seen")
    clocks = {}
    for key in integer_fields:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ResumeCompatibilityError(
                f"checkpoint training.clocks.{key} must be an integer"
            )
        clocks[key] = int(value)
    wall = raw.get("wall_seconds")
    if isinstance(wall, bool) or not isinstance(
        wall, (int, float, np.integer, np.floating)
    ) or not np.isfinite(float(wall)):
        raise ResumeCompatibilityError(
            "checkpoint training.clocks.wall_seconds must be finite"
        )
    clocks["wall_seconds"] = float(wall)
    if (
        int(clocks["optimizer_steps"]) != expected_optimizer
        or int(clocks["microsteps"]) != expected_optimizer * k
        or int(clocks["views_seen"]) != expected_optimizer * k * p
        or float(clocks["wall_seconds"]) < 0.0
    ):
        raise ResumeCompatibilityError(
            "checkpoint four clocks are inconsistent with iteration/P/K"
        )
    return clocks, warning


class ResumeCompatibilityError(ValueError):
    """Raised when checkpoint metadata cannot safely resume this run."""


@dataclass(frozen=True)
class CheckpointSnapshot:
    model: NativeGaussians
    optimizer: Mapping[str, np.ndarray]
    iteration: int
    rng_state: dict
    metadata: Mapping[str, object]


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        # Recurse so NumPy floating scalars receive the same non-finite policy
        # as native Python floats before ``allow_nan=False`` serialization.
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "unknown"
    return value


def _array_layout(values: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        str(key): {
            "shape": [int(size) for size in np.asarray(value).shape],
            "dtype": np.asarray(value).dtype.str,
        }
        for key, value in sorted(values.items())
    }


def _contiguous_copy(value: np.ndarray) -> np.ndarray:
    """Copy to a C-contiguous array while preserving 0-d scalars.

    ``np.ascontiguousarray`` always returns ``ndim >= 1``, which would turn
    checkpoint scalars such as ``max_duration`` into shape ``[1]`` and fail the
    V5 layout manifest check on resume.
    """

    array = np.asarray(value)
    if array.ndim == 0:
        return np.array(array, copy=True, order="C")
    return np.ascontiguousarray(array)


def _git_revision(root: Path) -> dict[str, object]:
    try:
        def git(*arguments: str) -> str:
            result = subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=True, capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()

        commit = git("rev-parse", "HEAD")
        try:
            ref = git("symbolic-ref", "--quiet", "--short", "HEAD")
        except subprocess.CalledProcessError:
            ref = "detached"
        status = git("status", "--porcelain", "--untracked-files=normal")
        return {"commit": commit, "ref": ref, "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        # Installed wheels and minimal inference hosts may have no Git binary
        # or working tree. Fall back to parsing HEAD without adding a runtime
        # dependency; dirty state is then explicitly unknown.
        pass
    git_path = root / ".git"
    try:
        if git_path.is_file():
            pointer = git_path.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                raise ValueError("invalid .git pointer")
            git_path = (git_path.parent / pointer.split(":", 1)[1].strip()).resolve()
        head = (git_path / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            ref_path = git_path / ref
            if ref_path.is_file():
                commit = ref_path.read_text(encoding="ascii").strip()
            else:
                commit = "unknown"
                packed = git_path / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="ascii").splitlines():
                        if line and not line.startswith(("#", "^")):
                            value, name = line.split(" ", 1)
                            if name == ref:
                                commit = value
                                break
        else:
            ref, commit = "detached", head
        return {"commit": commit or "unknown", "ref": ref, "dirty": "unknown"}
    except (OSError, ValueError):
        return {"commit": "unknown", "ref": "unknown", "dirty": "unknown"}


def collect_runtime_provenance(
    *, device_ids: tuple[int, ...] | list[int],
) -> dict[str, object]:
    """Collect reproducibility data without importing/loading the native DLL."""

    root = Path(__file__).resolve().parents[2]
    shader_root = Path(os.environ.get(
        "FTGSPP_VK_SHADER_DIR", root / "native" / "shader"
    )).expanduser().resolve()
    shader_files: dict[str, str] = {}
    if shader_root.is_dir():
        # The native extension loads both generated/*.spv and precompiled
        # subtrees such as radix_sort/*.spv.  Use normalized relative paths as
        # keys so the digest covers the complete executable shader set and is
        # stable across Windows/Linux checkpoint hosts.
        paths = sorted(
            shader_root.rglob("*.spv"),
            key=lambda path: path.relative_to(shader_root).as_posix(),
        )
        for path in paths:
            relative = path.relative_to(shader_root).as_posix()
            try:
                shader_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                shader_files[relative] = "unknown"
    if shader_files:
        combined = hashlib.sha256(json.dumps(
            shader_files, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        shader_metadata: dict[str, object] = {
            "directory": str(shader_root),
            "set_sha256": combined,
            "files": shader_files,
        }
    else:
        shader_metadata = {
            "directory": str(shader_root),
            "set_sha256": "unknown",
            "files": {},
        }
    try:
        package_version = version("ftgspp-vulkan-community")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "package_version": package_version,
        "native_abi": NATIVE_ABI,
        "native_reported_abi": "unknown",
        "native_extension": {
            "path": "unknown", "sha256": "unknown", "bytes": "unknown",
        },
        "devices": [
            {"vulkan_index": int(value), "name": "unknown", "driver": "unknown"}
            for value in device_ids
        ],
        "shaders": shader_metadata,
        "git": _git_revision(root),
    }


def _algorithm_config(config: AppConfig) -> dict[str, object]:
    """Return only values that change the resumed optimization semantics.

    Device placement, output paths, checkpoint/log cadence and view-cache
    policy are deliberately absent: they may change without changing the
    mathematical training program. Dataset identity/calibration has its own
    complete manifest and is checked independently.
    """

    relocation = config.train.relocation
    values: dict[str, object] = {
        "model": {
            "velocity_model": "explicit",
            "marginal_gating": bool(config.model.marginal_gating),
            "max_duration": float(config.model.max_duration),
        },
        "representation": {
            "sh_degree": int(config.init.sh_degree),
        },
        "optimizer": {
            "iterations": int(config.train.iterations),
            "batch_size": int(config.train.batch_size),
            "lrs": dict(sorted(config.train.lrs.items())),
            "lr_schedules": dict(sorted(config.train.lr_schedules.items())),
        },
        "loss": {
            "lpips_loss": bool(config.train.lpips_loss),
            "lpips_weight": float(config.train.lpips_weight),
            "color_correction": bool(config.train.color_correction),
            "color_correction_start": int(
                config.train.color_correction_start
            ),
            "color_corrector_weight": float(
                config.train.color_corrector_weight
            ),
        },
        # These values are currently native/Python pipeline constants rather
        # than Hydra options. Persist them in the semantic hash so changing a
        # shader/C++ constant cannot silently resume with a different objective
        # merely because the user-facing config stayed identical.
        "native_optimization_contract": {
            "image": {
                "l1_weight": 0.8,
                "ssim_weight": 0.2,
                "reference_alpha": "rgb-loss-and-gradient-mask",
                "alpha_is_not_an_optimized_output": True,
            },
            "adam": {
                "beta1": (
                    f"0.9^{config.distributed.global_batch_size}"
                    if config.distributed.world_size_lr_scaling
                    == "effective_batch_adam" else 0.9
                ),
                "beta2": (
                    f"0.999^{config.distributed.global_batch_size}"
                    if config.distributed.world_size_lr_scaling
                    == "effective_batch_adam" else 0.999
                ),
                "epsilon": (
                    f"1e-15/sqrt({config.distributed.global_batch_size})"
                    if config.distributed.world_size_lr_scaling
                    == "effective_batch_adam" else 1e-15
                ),
                "synchronous_reduction": "raw-fp32-mean-before-adam",
                "world_size_lr_scaling": str(
                    config.distributed.world_size_lr_scaling
                ),
                "world_size_beta_scaling": (
                    "effective_global_batch"
                    if config.distributed.world_size_lr_scaling
                    == "effective_batch_adam" else "none"
                ),
            },
            "lpips": {
                "host_plugin": True,
                "net_type": "alex",
                "torchmetrics_normalize_0_1": True,
                "prediction_clamp": "clamp-0-1-with-jacobian",
                "rgba_mask": "reference-alpha-premultiply-both-images-in-graph",
                "training_total_includes_weighted_term": True,
                "synchronous_training_supported": False,
            },
            "main_regularization": {
                "scale": 0.01,
                "opacity": 0.01,
                "gate": 0.001 if config.model.marginal_gating else 0.0,
            },
            "relocation_numerics": {
                "reference": "gsplat-1.5.3-compute_relocation",
                "ratio_cap": 51,
                "binomial_table": "float32-math.comb-n0-through-50",
                "alternating_sign": "integer-parity",
                "opacity_max": 1.0 - float(np.finfo(np.float32).eps),
            },
            "hard_separation": {
                "steps": 5000,
                "means_lr": 0.001 * 1.6e-4,
                "scale_regularization": 0.001,
                "opacity_regularization": 0.001,
                "gate_regularization": 0.0,
                "color_regularization": 0.0,
                "freeze_explicit_velocity_gate": True,
                "freeze_color_parameters": True,
                "phase_local_sampling_step": True,
                "phase_local_relocation_window": "500<step<3000,every=100",
            },
        },
        "relocation": {
            "start": int(relocation.start),
            "stop": int(relocation.stop),
            "every": int(relocation.every),
            "opacity_threshold": float(relocation.opacity_threshold),
            "mode": str(relocation.mode),
            "score_mode": str(relocation.score_mode),
            "score_mode_start": int(relocation.score_mode_start),
        },
        "hard_separation": bool(config.train.hard_separation),
    }
    return dict(_jsonable(values))


def _algorithm_manifest(config: AppConfig) -> dict[str, object]:
    values = _algorithm_config(config)
    canonical = json.dumps(
        values, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "config": values,
    }


def _collective_topology(
    provenance: Mapping[str, object],
    *,
    policy: str,
) -> dict[str, object]:
    raw_devices = provenance.get("devices", [])
    devices = list(raw_devices) if isinstance(raw_devices, (list, tuple)) else []
    identity_fields = (
        "uuid", "device_uuid", "pci_bus_id", "pci_bdf", "numa_node",
    )
    identities = [
        {
            key: _jsonable(device.get(key, "unknown"))
            for key in identity_fields
        }
        for device in devices
        if isinstance(device, Mapping)
    ]
    canonical = json.dumps(
        {"policy": str(policy), "devices": identities},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "policy": str(policy),
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
        "devices": identities,
    }


def build_checkpoint_metadata(
    config: AppConfig,
    dataset: NativeDataset,
    distributed: DistributedPlan,
    *,
    phase: str,
    phase_step: int,
    provenance: Mapping[str, object],
    transport: str | None = None,
    effective_sparse: bool | None = None,
    clocks: Mapping[str, int | float] | None = None,
) -> dict[str, object]:
    """Build the run-dependent portion of a V5 checkpoint manifest."""

    # ``flat_epoch`` deliberately owns a new ABI. Do not relabel the
    # historical step-local ``unique``/``with_replacement`` sampler.
    sampler_abi = checkpoint_sampler_abi(
        synchronous=distributed.enabled,
        sampling_policy=distributed.sampling_policy,
        microsteps=distributed.microsteps,
    )
    if distributed.sampling_policy == "flat_epoch":
        sampler_alignment = (
            "quality-oriented-flat-view-random-reshuffling;uniform-marginal;"
            "not-torch-bitstream"
        )
    elif distributed.sampling_policy == "co_frame":
        sampler_alignment = (
            "quality-oriented-co-frame-multi-camera;uniform-frame-marginal;"
            "not-torch-bitstream"
        )
    elif distributed.sampling_policy == "torch_random_sampler":
        sampler_alignment = (
            "legacy-ftgspp-torch-randomsampler-bitstream;"
            "torch-free-native-reimplementation"
        )
    else:
        sampler_alignment = (
            "uniform-native-numpy-stream;distribution-only-torch-alignment;"
            "not-torch-bitstream"
        )
    selected_transport = str(
        distributed.backend if transport is None else transport
    )
    optimizer_steps = max(int(phase_step) + 1, 0)
    clock_values: dict[str, int | float] = {
        "optimizer_steps": optimizer_steps,
        "microsteps": optimizer_steps * int(distributed.microsteps),
        "views_seen": (
            optimizer_steps
            * int(distributed.microsteps)
            * int(distributed.world_size)
        ),
        "wall_seconds": 0.0,
    }
    if clocks is not None:
        for key in clock_values:
            if key in clocks:
                clock_values[key] = clocks[key]
    topology = _collective_topology(
        provenance,
        policy=distributed.collective_topology_policy,
    )
    collective = {
        "abi": (
            HIERARCHICAL_HOST_COLLECTIVE_ABI
            if distributed.enabled else "single-device"
        ),
        "algorithm": (
            distributed.collective_algorithm
            if distributed.enabled else "none"
        ),
        "P": int(distributed.world_size),
        "K": int(distributed.microsteps),
        "reduction_tree": str(distributed.collective_reduction_tree),
        "topology": topology,
        "bucket_bytes": int(distributed.collective_bucket_bytes),
        "readback_ring_depth": int(
            distributed.collective_readback_ring_depth
        ),
        "upload_ring_depth": int(distributed.collective_upload_ring_depth),
        "codec_policy": str(distributed.collective_codec_policy),
        "sparse_gradients": bool(
            distributed.collective_sparse
            if effective_sparse is None else effective_sparse
        ),
    }
    return {
        "checkpoint_schema": "ftgspp-vulkan-v5",
        "phase": str(phase),
        "phase_step": int(phase_step),
        "training": {
            "seed": int(config.train.seed),
            "world_size": int(distributed.world_size),
            "device_ids": [int(value) for value in distributed.device_ids],
            "per_device_batch_size": int(distributed.per_device_batch_size),
            "global_batch_size": int(distributed.global_batch_size),
            "reduction": str(distributed.reduction),
            "requested_backend": str(distributed.backend),
            "transport": selected_transport,
            "collective": collective,
            "clocks": {
                "abi": "ftgspp-four-training-clocks-v1",
                "consumers": {
                    "adam_bias": "optimizer_steps",
                    "sampling": "views_seen advances sampler slots; flat_epoch thresholds stay phase-local optimizer_steps for K=1-compatible configs",
                    "sh_schedule": "views_seen is the sample clock; threshold units remain optimizer_steps for K=1-compatible configs",
                    "relocation": "views_seen is the sample clock; threshold units remain phase-local optimizer_steps for K=1-compatible configs",
                    "hard_separation": "views_seen is the sample clock; threshold units remain phase-local optimizer_steps for K=1-compatible configs",
                    "lr": "optimizer_steps unless world_size_lr_scaling explicitly opts into a quality candidate",
                },
                **clock_values,
            },
            "sampling_policy": str(distributed.sampling_policy),
            "world_size_lr_scaling": str(
                getattr(distributed, "world_size_lr_scaling", "none")
            ),
            "sampler_abi": sampler_abi,
            "sampler_alignment": sampler_alignment,
            "phase_seed_formula": SAMPLING_PHASE_SEED_FORMULA,
            "phase_seeds": {
                "main": sampling_phase_seed(
                    seed=config.train.seed,
                    phase="main",
                    train_iterations=config.train.iterations,
                ),
                "hard_separation": sampling_phase_seed(
                    seed=config.train.seed,
                    phase="hard_separation",
                    train_iterations=config.train.iterations,
                ),
            },
        },
        "dataset": dataset.checkpoint_metadata(),
        "model_config": {
            "velocity_model": "explicit",
            "marginal_gating": bool(config.model.marginal_gating),
            "max_duration": float(config.model.max_duration),
            "sh_degree": int(config.init.sh_degree),
            "color_correction": bool(config.train.color_correction),
            "hard_separation": bool(config.train.hard_separation),
        },
        "algorithm": _algorithm_manifest(config),
        "runtime": _jsonable(provenance),
    }


def capture_checkpoint_snapshot(
    backend: NativeVulkanBackend,
    iteration: int,
    rng: np.random.Generator,
    *,
    optimizer_state: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CheckpointSnapshot:
    """Download model/optimizer exactly once and freeze a host snapshot."""

    model = backend.model()
    optimizer = backend.optimizer_state() if optimizer_state is None else optimizer_state
    legacy_field_keys = sorted(
        str(key) for key in optimizer if str(key).startswith("field_")
    )
    if legacy_field_keys:
        raise ValueError(
            "V5 checkpoints are explicit-only and cannot serialize field "
            f"state: {legacy_field_keys}"
        )
    manifest = dict(_jsonable(metadata or {}))
    model_arrays = model.arrays()
    manifest["layout"] = {
        "model": _array_layout(model_arrays),
        "optimizer": _array_layout(optimizer),
        "color": _array_layout({
            key: value for key, value in optimizer.items()
            if str(key).startswith("color_")
        }),
    }
    return CheckpointSnapshot(
        model=model,
        optimizer=optimizer,
        iteration=int(iteration),
        rng_state=_jsonable(rng.bit_generator.state),
        metadata=manifest,
    )


def save_model(path: Path, model: NativeGaussians) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **model.arrays())
    temporary.replace(path)


def load_model(path: Path) -> NativeGaussians:
    with np.load(path, allow_pickle=False) as archive:
        return NativeGaussians.from_mapping({
            key: np.array(archive[key], copy=True) for key in archive.files
        })


def save_checkpoint_snapshot(
    path: Path,
    snapshot: CheckpointSnapshot,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically serialize an already captured host snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    arrays = {
        f"model__{key}": value for key, value in snapshot.model.arrays().items()
    }
    arrays.update({
        f"optimizer__{key}": value for key, value in snapshot.optimizer.items()
    })
    arrays["runtime__format"] = np.asarray(FORMAT, np.int32)
    arrays["runtime__iteration"] = np.asarray(snapshot.iteration, np.int64)
    arrays["runtime__rng"] = np.asarray(json.dumps(
        _jsonable(snapshot.rng_state), sort_keys=True, allow_nan=False,
    ))
    arrays["runtime__metadata"] = np.asarray(json.dumps(
        _jsonable(snapshot.metadata), sort_keys=True, allow_nan=False,
    ))
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    backend: NativeVulkanBackend,
    iteration: int,
    rng: np.random.Generator,
    *,
    optimizer_state: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CheckpointSnapshot:
    """Compatibility convenience wrapper; prefer capture + publish in runner."""

    snapshot = capture_checkpoint_snapshot(
        backend, iteration, rng,
        optimizer_state=optimizer_state, metadata=metadata,
    )
    save_checkpoint_snapshot(path, snapshot)
    return snapshot


def publish_checkpoint(
    directory: Path,
    snapshot: CheckpointSnapshot,
    *,
    filename: str | None = None,
    update_latest: bool = True,
) -> Path:
    """Write one immutable checkpoint and atomically advance latest.json."""

    directory.mkdir(parents=True, exist_ok=True)
    if int(snapshot.iteration) < 0:
        raise ValueError("published training checkpoint iteration must be non-negative")
    name = filename or f"{snapshot.iteration:06d}.npz"
    if Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("checkpoint filename must be a local .npz filename")
    target = directory / name
    if target.exists():
        raise FileExistsError(
            f"immutable checkpoint already exists: {target}; refusing to "
            "advance latest.json to parameters that were not serialized"
        )
    save_checkpoint_snapshot(target, snapshot, overwrite=False)
    if update_latest:
        pointer = {
            "format": FORMAT,
            "iteration": int(snapshot.iteration),
            "checkpoint": name,
        }
        manifest = directory / LATEST_MANIFEST
        temporary = manifest.with_name(manifest.name + ".tmp")
        temporary.write_text(
            json.dumps(pointer, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest)
    return target


def _legacy_metadata(format_value: int) -> dict[str, object]:
    return {
        "checkpoint_schema": f"legacy-v{format_value}",
        "legacy": True,
        "phase": "unknown",
        "phase_step": -1,
        "training": {
            "world_size": "unknown",
            "global_batch_size": "unknown",
            "sampling_policy": "unknown",
            "sampler_abi": "unknown",
        },
        "dataset": {"fingerprint": "unknown"},
        "model_config": {},
        "runtime": {
            "native_abi": "unknown", "shaders": {"set_sha256": "unknown"},
            "git": {"commit": "unknown", "dirty": "unknown"},
        },
    }


def load_checkpoint_full(
    path: Path, *, allow_v4_migration: bool = False,
) -> CheckpointSnapshot:
    with np.load(path, allow_pickle=False) as archive:
        format_array = np.asarray(archive["runtime__format"])
        if format_array.size != 1 or format_array.dtype.kind not in "iu":
            raise ValueError(f"checkpoint format tag is not an integer scalar: {path}")
        format_value = int(format_array.reshape(-1)[0])
        if format_value not in LEGACY_FORMATS and format_value != FORMAT:
            raise ValueError(f"unsupported native checkpoint format in {path}")
        model_values = {
            key.removeprefix("model__"): _contiguous_copy(archive[key])
            for key in archive.files if key.startswith("model__")
        }
        optimizer = {
            key.removeprefix("optimizer__"): _contiguous_copy(archive[key])
            for key in archive.files if key.startswith("optimizer__")
        }
        iteration_array = np.asarray(archive["runtime__iteration"])
        if iteration_array.size != 1 or iteration_array.dtype.kind not in "iu":
            raise ValueError(
                f"checkpoint iteration is not an integer scalar: {path}"
            )
        iteration = int(iteration_array.reshape(-1)[0])
        if iteration < -1:
            raise ValueError(f"checkpoint iteration must be >= -1: {path}")
        rng_state = json.loads(str(archive["runtime__rng"]))
        if not isinstance(rng_state, Mapping):
            raise ValueError(f"checkpoint RNG state is not a mapping: {path}")
        rng_state = dict(rng_state)
        if format_value in {4, FORMAT}:
            if "runtime__metadata" not in archive.files:
                raise ValueError(
                    f"V{format_value} checkpoint has no runtime metadata: {path}"
                )
            metadata = json.loads(str(archive["runtime__metadata"]))
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    f"V{format_value} checkpoint metadata is not a mapping: {path}"
                )
            metadata = dict(metadata)
        else:
            metadata = _legacy_metadata(format_value)
    if format_value == 4 and not allow_v4_migration:
        field_keys = sorted(
            str(key) for key in optimizer if str(key).startswith("field_")
        )
        if field_keys:
            raise ValueError(
                "legacy V4 field checkpoints cannot resume in the explicit-only "
                "runtime; convert offline with "
                "`ftgspp-convert-checkpoint SOURCE DEST --field-mode sample`"
            )
        raise ValueError(
            "explicit V4 checkpoints require a one-time V5 migration; run "
            "`ftgspp-convert-checkpoint SOURCE DEST --field-mode reject`"
        )
    if format_value in {4, FORMAT}:
        raw_layout = metadata.get("layout", {})
        if not isinstance(raw_layout, Mapping):
            raise ValueError(f"checkpoint layout manifest is not a mapping: {path}")
        recorded_layout = dict(raw_layout)
        expected_model = recorded_layout.get("model")
        expected_optimizer = recorded_layout.get("optimizer")
        actual_model = _array_layout(model_values)
        actual_optimizer = _array_layout(optimizer)
        if expected_model != actual_model:
            raise ValueError(f"checkpoint model layout manifest is inconsistent: {path}")
        if expected_optimizer != actual_optimizer:
            raise ValueError(
                f"checkpoint optimizer layout manifest is inconsistent: {path}"
            )
    return CheckpointSnapshot(
        NativeGaussians.from_mapping(model_values), optimizer,
        iteration, rng_state, metadata,
    )


def load_checkpoint(
    path: Path,
) -> tuple[NativeGaussians, dict[str, np.ndarray], int, dict]:
    """Load V3/V5 while retaining the historical four-value public API."""

    snapshot = load_checkpoint_full(path)
    return (
        snapshot.model, dict(snapshot.optimizer), snapshot.iteration,
        dict(snapshot.rng_state),
    )


def validate_resume_compatibility(
    saved: Mapping[str, object],
    current: Mapping[str, object],
    *,
    allow_world_size_change: bool,
) -> list[str]:
    """Reject semantic resume mismatches and return non-fatal warnings."""

    if bool(saved.get("legacy", False)) or bool(saved.get("legacy_migration", False)):
        return [
            "legacy/migrated checkpoint has no dataset/distributed provenance; "
            "resume compatibility could only be checked from array layouts"
        ]
    errors: list[str] = []
    warnings: list[str] = []
    saved_training = dict(saved.get("training", {}))
    current_training = dict(current.get("training", {}))
    legacy_transport = str(saved_training.get("transport", ""))
    legacy_backend = str(saved_training.get("requested_backend", ""))
    unsupported_legacy = sorted(
        {legacy_transport, legacy_backend}
        & {"packed_host", "host_mailbox", "device_group"}
    )
    if unsupported_legacy:
        raise ResumeCompatibilityError(
            "checkpoint is unsupported: legacy distributed backend/transport "
            f"{unsupported_legacy} cannot resume under hierarchical_host; "
            "no implicit collective migration is defined"
        )
    saved_collective_raw = saved_training.get("collective", {})
    current_collective_raw = current_training.get("collective", {})
    saved_collective = (
        dict(saved_collective_raw)
        if isinstance(saved_collective_raw, Mapping) else {}
    )
    current_collective = (
        dict(current_collective_raw)
        if isinstance(current_collective_raw, Mapping) else {}
    )
    for key in (
        "abi", "algorithm", "K", "reduction_tree", "codec_policy",
        "sparse_gradients",
    ):
        if saved_collective.get(key) != current_collective.get(key):
            errors.append(
                f"training.collective.{key} changed from "
                f"{saved_collective.get(key)!r} to "
                f"{current_collective.get(key)!r}"
            )
    if saved_collective.get("P") != current_collective.get("P"):
        message = (
            "training.collective.P changed from "
            f"{saved_collective.get('P')!r} to "
            f"{current_collective.get('P')!r}"
        )
        if allow_world_size_change:
            warnings.append(message + " under explicit world-size acknowledgement")
        else:
            errors.append(message)
    saved_topology_raw = saved_collective.get("topology", {})
    current_topology_raw = current_collective.get("topology", {})
    saved_topology = (
        dict(saved_topology_raw)
        if isinstance(saved_topology_raw, Mapping) else {}
    )
    current_topology = (
        dict(current_topology_raw)
        if isinstance(current_topology_raw, Mapping) else {}
    )
    if (
        saved_topology.get("fingerprint")
        != current_topology.get("fingerprint")
    ):
        errors.append(
            "training.collective.topology fingerprint differs from the "
            "checkpoint"
        )
    saved_clocks_raw = saved_training.get("clocks", {})
    current_clocks_raw = current_training.get("clocks", {})
    saved_clocks = (
        dict(saved_clocks_raw)
        if isinstance(saved_clocks_raw, Mapping) else {}
    )
    current_clocks = (
        dict(current_clocks_raw)
        if isinstance(current_clocks_raw, Mapping) else {}
    )
    if saved_clocks.get("abi") != current_clocks.get("abi"):
        errors.append(
            "training.clocks.abi differs from the checkpoint"
        )
    saved_world = saved_training.get("world_size")
    current_world = current_training.get("world_size")
    if saved_world != current_world:
        message = f"world_size changed from {saved_world} to {current_world}"
        if allow_world_size_change:
            warnings.append(
                message + "; global batch and stochastic optimizer trajectory will change"
            )
        else:
            errors.append(
                message + "; set distributed.allow_resume_world_size_change=true "
                "to acknowledge this algorithmic change"
            )
    for key in (
        "seed", "sampling_policy", "sampler_abi", "phase_seed_formula",
        "phase_seeds", "per_device_batch_size", "reduction",
    ):
        if saved_training.get(key) != current_training.get(key):
            errors.append(
                f"training.{key} changed from {saved_training.get(key)!r} "
                f"to {current_training.get(key)!r}"
            )
    if saved_world == current_world and (
        saved_training.get("global_batch_size")
        != current_training.get("global_batch_size")
    ):
        errors.append("training.global_batch_size changed without a world-size change")

    saved_dataset = dict(saved.get("dataset", {}))
    current_dataset = dict(current.get("dataset", {}))
    for key in (
        "fingerprint", "frames", "fps", "time_offsets",
        "camera_dense_mapping", "train_cameras", "eval_cameras",
    ):
        if saved_dataset.get(key) != current_dataset.get(key):
            errors.append(f"dataset.{key} differs from the checkpoint")
    saved_model = dict(saved.get("model_config", {}))
    current_model = dict(current.get("model_config", {}))
    for key in (
        "velocity_model", "marginal_gating", "max_duration", "sh_degree",
        "color_correction", "hard_separation",
    ):
        if saved_model.get(key) != current_model.get(key):
            errors.append(
                f"model_config.{key} changed from {saved_model.get(key)!r} "
                f"to {current_model.get(key)!r}"
            )
    saved_algorithm = dict(saved.get("algorithm", {}))
    current_algorithm = dict(current.get("algorithm", {}))
    if saved_algorithm.get("sha256") != current_algorithm.get("sha256"):
        errors.append(
            "algorithm configuration changed (loss/LR/relocation/color/"
            "hard-separation contract differs from the checkpoint)"
        )
    saved_runtime = dict(saved.get("runtime", {}))
    current_runtime = dict(current.get("runtime", {}))
    if saved_runtime.get("native_abi") != current_runtime.get("native_abi"):
        errors.append("native checkpoint/gradient ABI differs")
    if (
        saved_runtime.get("native_reported_abi")
        != current_runtime.get("native_reported_abi")
    ):
        errors.append(
            "native extension ABI differs from the checkpoint provenance"
        )
    saved_extension_raw = saved_runtime.get("native_extension", {})
    current_extension_raw = current_runtime.get("native_extension", {})
    saved_extension = (
        dict(saved_extension_raw)
        if isinstance(saved_extension_raw, Mapping) else {}
    )
    current_extension = (
        dict(current_extension_raw)
        if isinstance(current_extension_raw, Mapping) else {}
    )
    saved_extension_hash = saved_extension.get("sha256")
    current_extension_hash = current_extension.get("sha256")
    if (
        saved_extension_hash not in (None, "", "unknown")
        and current_extension_hash not in (None, "", "unknown")
        and saved_extension_hash != current_extension_hash
    ):
        warnings.append(
            "native extension binary SHA256 changed from the checkpoint; "
            "ABI compatibility is preserved, but numerical/code behavior may differ"
        )
    saved_devices = list(saved_runtime.get("devices", []))
    current_devices = list(current_runtime.get("devices", []))
    device_differences: list[str] = []
    for rank, (saved_device, current_device) in enumerate(zip(
        saved_devices, current_devices, strict=False,
    )):
        if not isinstance(saved_device, Mapping) or not isinstance(
            current_device, Mapping
        ):
            continue
        for key in ("uuid", "name", "driver_version"):
            old, new = saved_device.get(key), current_device.get(key)
            if (
                old not in (None, "", "unknown")
                and new not in (None, "", "unknown")
                and old != new
            ):
                device_differences.append(
                    f"rank {rank} {key} {old!r}->{new!r}"
                )
    if len(saved_devices) != len(current_devices):
        device_differences.append(
            f"device count {len(saved_devices)}->{len(current_devices)}"
        )
    if device_differences:
        warnings.append(
            "Vulkan device provenance changed (allowed, but numerical ULP/"
            "performance behavior may differ): " + "; ".join(device_differences)
        )
    saved_shaders = dict(saved_runtime.get("shaders", {}))
    current_shaders = dict(current_runtime.get("shaders", {}))
    if saved_shaders.get("set_sha256") != current_shaders.get("set_sha256"):
        warnings.append(
            "compiled shader set differs from checkpoint provenance; run a "
            "single-step equivalence check before trusting resumed metrics"
        )
    saved_git = dict(saved_runtime.get("git", {})).get("commit")
    current_git = dict(current_runtime.get("git", {})).get("commit")
    if saved_git != current_git:
        warnings.append(
            f"Git revision changed from {saved_git!r} to {current_git!r}"
        )
    if errors:
        raise ResumeCompatibilityError(
            "checkpoint is incompatible with this run:\n- " + "\n- ".join(errors)
        )
    return warnings


def latest_checkpoint(directory: Path) -> Path | None:
    manifest = directory / LATEST_MANIFEST
    if manifest.is_file():
        try:
            pointer = json.loads(manifest.read_text(encoding="utf-8"))
            if int(pointer["format"]) != FORMAT:
                raise ValueError(
                    f"latest pointer format is {pointer['format']}, expected {FORMAT}"
                )
            name = str(pointer["checkpoint"])
            if Path(name).name != name or not name.endswith(".npz"):
                raise ValueError("latest checkpoint pointer escapes its directory")
            target = directory / name
            if not target.is_file():
                raise FileNotFoundError(target)
            with np.load(target, allow_pickle=False) as archive:
                checkpoint_format = int(archive["runtime__format"])
                checkpoint_iteration = int(archive["runtime__iteration"])
            if checkpoint_format != FORMAT:
                raise ValueError(
                    "latest pointer target is not a V5 checkpoint; migrate it "
                    "with ftgspp-convert-checkpoint before resume"
                )
            if checkpoint_iteration != int(pointer["iteration"]):
                raise ValueError(
                    "latest pointer iteration does not match its checkpoint"
                )
            return target
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint pointer {manifest}: {exc}") from exc
    legacy_latest = directory / "latest.npz"
    if legacy_latest.is_file():
        return legacy_latest
    candidates = [
        path for path in directory.glob("[0-9]*.npz")
        if path.stem.isdigit()
    ]
    return max(candidates, key=lambda path: int(path.stem)) if candidates else None


def resolve_checkpoint_path(path: Path) -> Path:
    """Resolve an explicit latest.json while leaving NPZ paths unchanged."""

    path = Path(path)
    if path.name != LATEST_MANIFEST:
        return path
    resolved = latest_checkpoint(path.parent)
    if resolved is None:
        raise FileNotFoundError(
            f"{path} does not point to an existing checkpoint"
        )
    return resolved
