"""Synchronous replicated multi-device orchestration for native Vulkan.

Every backend owns a complete model and Adam state. A macro-step executes K
rank-major views per rank, finalizes the local FP32 sum by K, reduces across P
before Adam, and applies the same global mean on every replica. Multi-device
production has one user-visible
``hierarchical_host`` backend; P=2 uses the native pair-exchange GPU merge.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import (
    Executor,
    Future,
    ThreadPoolExecutor,
)
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

from ftgspp.config.schema import AppConfig
from ftgspp.native_runtime.distributed_contract import (
    RELOCATION_MAX_DONOR_COPIES,
    assert_replicas_consistent,
    cap_relocation_donor_copies,
    shard_resumed_relocation_statistics,
)

logger = logging.getLogger("ftgspp.native.distributed")

T = TypeVar("T")


def _log_geometry_grads_enabled() -> bool:
    value = os.environ.get("FTGSPP_TRAIN_LOG_GRAD", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


HIERARCHICAL_HOST_COLLECTIVE_ABI = "ftgspp-hierarchical-host-v1"
# Temporary native capability ABI used only by the P=2 implementation. It is
# intentionally not a configured backend or checkpoint transport.
NATIVE_PAIR_EXCHANGE_ABI = "ftgspp-host-mailbox-v3"
PAIR_EXCHANGE_GPU_MERGE = "pair_exchange_gpu_merge"
MAX_SAMPLING_SEED = 0xFFFFFFFF
SAMPLING_PHASE_SEED_FORMULA = "phase_seed(main)=seed; phase_seed(hard_separation)=seed+train_iterations"


def sampling_phase_seed(
    *, seed: int, phase: str, train_iterations: int,
) -> int:
    """Return the stateless view-sampler seed for one optimizer phase."""

    seed = int(seed)
    train_iterations = int(train_iterations)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if seed > MAX_SAMPLING_SEED:
        raise ValueError("seed must fit uint32")
    if train_iterations <= 0:
        raise ValueError("train_iterations must be positive")
    if phase == "main":
        return seed
    if phase == "hard_separation":
        return (seed + train_iterations) & MAX_SAMPLING_SEED
    raise ValueError(f"unsupported sampling phase: {phase}")


class SynchronousBackend(Protocol):
    """Backend surface required by :class:`SynchronousBackendGroup`.

    The protocol is deliberately duck-typed so unit tests can use a fake
    backend and so a future Vulkan peer reducer can preserve this coordinator.
    """

    def backward_view(
        self, view_index: int, step: int, *, collect_loss: bool = True,
    ) -> None: ...
    def finalize_local_gradient(self, microsteps: int) -> None: ...
    def apply_sync_tile_capacity(self, exact_count: int) -> None: ...
    def clear_sync_collectives_for_replay(self) -> None: ...
    def drain_for_structure_change(self) -> None: ...
    def inject_sync_tile_overflow(self, exact_count: int) -> None: ...
    def prefetch_train_views(
        self, view_indices: Sequence[int], *, executor: Executor | None = None,
    ) -> None: ...
    def schedule_train_views(
        self, view_indices: Sequence[int], *, executor: Executor,
    ) -> None: ...
    def prefetch_eval_views(
        self, view_indices: Sequence[int], *,
        executor: Executor | None = None,
    ) -> None: ...
    def schedule_eval_views(
        self, view_indices: Sequence[int], *, executor: Executor,
    ) -> None: ...
    def supports_pair_exchange_gpu_merge(self) -> bool: ...
    def create_pair_exchange_gpu_merge_owner(
        self, *, world_size: int, bucket_bytes: int, sparse: bool,
    ) -> Mapping[str, object]: ...
    def configure_pair_exchange_gpu_merge(
        self, *, sparse: bool, codec_policy: str = "auto_exact",
    ) -> None: ...
    def exchange_pair_gradient(
        self, *, rank: int, generation: int,
    ) -> Mapping[str, object]: ...
    def optimizer_step_pair_exchange_gpu_merge(
        self, step: int, timestamps: Sequence[float],
    ) -> Mapping[str, float]: ...
    def exchange_pair_relocation(
        self, *, rank: int, generation: int,
    ) -> Mapping[str, object]: ...
    def train_view_timestamp(self, view_index: int) -> float: ...
    def release_pair_exchange_gpu_merge_owner(self) -> None: ...
    def reset_pair_exchange_gpu_merge_attachments(self) -> None: ...
    def optimizer_step(self, step: int) -> None: ...
    def set_collect_geometry_grad_stats(self, enabled: bool) -> None: ...
    def geometry_gradient_stats(self) -> Mapping[str, float]: ...
    def last_loss(self) -> Mapping[str, float]: ...
    def relocation_stats_state(self) -> np.ndarray: ...
    def set_relocation_stats_state(self, values: np.ndarray) -> None: ...
    def relocate_native(
        self, step: int, mode: str, include_gate: bool, seed: int,
    ) -> Mapping[str, int | bool]: ...
    def plan_relocation_native(
        self, step: int, include_gate: bool, seed: int,
    ) -> Mapping[str, object]: ...
    def apply_planned_relocation(self, mode: str) -> None: ...
    def clear_relocation_stats_state(self) -> None: ...
    def relocate(
        self,
        mapping: np.ndarray,
        counts: np.ndarray,
        mode: str,
        *,
        require_dead_destinations: bool = True,
        require_alive_donors: bool = True,
    ) -> None: ...
    def begin_hard_separation(self) -> dict[str, int]: ...
    def resume_hard_separation(self) -> dict[str, int]: ...
    def hard_separation_digest(self) -> str: ...
    def model(self): ...
    def optimizer_state(self) -> Mapping[str, np.ndarray]: ...
    def performance_snapshot(self) -> Mapping[str, object]: ...
    def benchmark_pair_transfer_buckets(
        self, candidate_bytes: list[int] | tuple[int, ...], *, rounds: int = 3,
    ) -> list[dict[str, object]]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class DistributedPlan:
    mode: str
    device_ids: tuple[int, ...]
    reduction: str
    backend: str
    sampling_policy: str
    per_device_batch_size: int
    global_batch_size: int
    verify_replica_interval: int
    microsteps: int = 1
    collective_algorithm: str = PAIR_EXCHANGE_GPU_MERGE
    collective_bucket_bytes: int = 32 * 1024 * 1024
    collective_sparse: bool = True
    collective_codec_policy: str = "auto_exact"
    collective_reduction_tree: str = "fixed_rank_order_v1"
    collective_topology_policy: str = "device_uuid_pci_numa_v1"
    collective_readback_ring_depth: int = 3
    collective_upload_ring_depth: int = 3
    world_size_lr_scaling: str = "none"
    wall_eval_interval_seconds: float = 0.0

    @classmethod
    def from_config(cls, config: AppConfig) -> "DistributedPlan":
        distributed = config.distributed
        device_ids = (
            tuple(int(value) for value in distributed.device_ids)
            if distributed.mode == "sync"
            else (int(config.run.gpu),)
        )
        backend = (
            "hierarchical_host"
            if str(distributed.backend) == "auto"
            else str(distributed.backend)
        )
        collective = distributed.host_collective
        world_size = len(device_ids)
        if distributed.mode == "sync" and world_size != 2:
            raise ValueError(
                "distributed.mode=sync is limited to exactly two Vulkan devices"
            )
        collective_algorithm = (
            PAIR_EXCHANGE_GPU_MERGE
            if str(collective.algorithm) == "auto"
            else str(collective.algorithm)
        )
        return cls(
            mode=str(distributed.mode),
            device_ids=device_ids,
            reduction=str(distributed.reduction),
            backend=backend,
            sampling_policy=str(distributed.sampling_policy),
            per_device_batch_size=int(config.train.batch_size),
            global_batch_size=int(distributed.global_batch_size),
            verify_replica_interval=int(distributed.verify_replica_interval),
            microsteps=(
                int(distributed.accumulation.microsteps)
                if distributed.mode == "sync" else 1
            ),
            collective_algorithm=collective_algorithm,
            collective_bucket_bytes=int(collective.bucket_mb) * 1024 * 1024,
            collective_sparse=bool(collective.sparse_gradients),
            collective_codec_policy=str(collective.codec_policy),
            collective_reduction_tree=str(collective.reduction_tree),
            collective_topology_policy=str(collective.topology_policy),
            collective_readback_ring_depth=int(
                collective.readback_ring_depth
            ),
            collective_upload_ring_depth=int(collective.upload_ring_depth),
            world_size_lr_scaling=str(distributed.world_size_lr_scaling),
            wall_eval_interval_seconds=float(
                distributed.wall_eval_interval_seconds
            ),
        )

    @property
    def world_size(self) -> int:
        return len(self.device_ids)

    @property
    def enabled(self) -> bool:
        return self.mode == "sync"


@lru_cache(maxsize=8)
def _flat_epoch_permutation(
    seed: int, epoch: int, num_views: int,
) -> np.ndarray:
    """Return the immutable PCG64 permutation for one flat-view epoch.

    Keeping only the most recent epochs bounds host memory while avoiding a
    full ``num_views`` shuffle on every optimizer step.  The cache is an
    optimization only: the permutation is completely reconstructible from a
    domain-separated ``SeedSequence(seed, epoch, FEPM)`` and does not become
    checkpoint state. Keeping seed and epoch in separate entropy lanes avoids
    cross-run aliasing such as ``seed=7, epoch=1`` versus ``seed=8, epoch=0``.
    """

    entropy = [
        int(seed) & 0xFFFFFFFF,
        int(epoch) & 0xFFFFFFFF,
        0x4645504D,  # "FEPM": flat-epoch permutation domain.
    ]
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(entropy))
    )
    permutation = generator.permutation(int(num_views))
    permutation.flags.writeable = False
    return permutation


class _TorchMT19937:
    """Small CPU MT19937 engine matching PyTorch's RandomSampler randint path."""

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UMASK = 0x80000000
    _LMASK = 0x7FFFFFFF
    _MASK32 = 0xFFFFFFFF
    __slots__ = ("_state", "_left", "_next")

    def __init__(self, seed: int) -> None:
        state = [0] * self._N
        state[0] = int(seed) & self._MASK32
        for index in range(1, self._N):
            previous = state[index - 1]
            state[index] = (
                1812433253 * (previous ^ (previous >> 30)) + index
            ) & self._MASK32
        self._state = state
        self._left = 1
        self._next = 0

    @classmethod
    def _twist(cls, first: int, second: int) -> int:
        mixed = (first & cls._UMASK) | (second & cls._LMASK)
        value = mixed >> 1
        if second & 1:
            value ^= cls._MATRIX_A
        return value & cls._MASK32

    def _next_state(self) -> None:
        state = self._state
        self._left = self._N
        self._next = 0
        offset = 0
        for _ in range(self._N - self._M):
            state[offset] = (
                state[offset + self._M]
                ^ self._twist(state[offset], state[offset + 1])
            ) & self._MASK32
            offset += 1
        for _ in range(self._M - 1):
            state[offset] = (
                state[offset + self._M - self._N]
                ^ self._twist(state[offset], state[offset + 1])
            ) & self._MASK32
            offset += 1
        state[offset] = (
            state[offset + self._M - self._N]
            ^ self._twist(state[offset], state[0])
        ) & self._MASK32

    def random_uint32(self) -> int:
        self._left -= 1
        if self._left == 0:
            self._next_state()
        value = self._state[self._next]
        self._next += 1
        value ^= value >> 11
        value &= self._MASK32
        value ^= (value << 7) & 0x9D2C5680
        value &= self._MASK32
        value ^= (value << 15) & 0xEFC60000
        value &= self._MASK32
        value ^= value >> 18
        return value & self._MASK32

    def random_uint64(self) -> int:
        high = self.random_uint32()
        low = self.random_uint32()
        return ((high << 32) | low) & 0xFFFFFFFFFFFFFFFF


@lru_cache(maxsize=32)
def _torch_random_sampler_stream(
    seed: int, num_views: int, total_slots: int,
) -> np.ndarray:
    """Return PyTorch ``RandomSampler(replacement=True)`` flat indices.

    PyTorch 2.5 CPU ``RandomSampler`` yields chunks of
    ``torch.randint(high=n, size=(32,), dtype=int64, generator=...)``.  For
    ``n < 2**32`` that kernel consumes one MT19937 uint32 per sample and maps
    it with modulo.  We reproduce that tiny path directly so native training
    keeps its torch-free runtime dependency and pays only one cached startup
    cost per phase.
    """

    seed = int(seed)
    num_views = int(num_views)
    total_slots = int(total_slots)
    if seed < 0 or seed > MAX_SAMPLING_SEED:
        raise ValueError("torch_random_sampler seed must fit uint32")
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if total_slots <= 0:
        raise ValueError("torch_random_sampler total_slots must be positive")
    generator = _TorchMT19937(seed)
    values = np.empty(total_slots, dtype=np.int64)
    if num_views >= 1 << 32:
        for index in range(total_slots):
            values[index] = generator.random_uint64() % num_views
    else:
        for index in range(total_slots):
            values[index] = generator.random_uint32() % num_views
    values.flags.writeable = False
    return values


def sample_rank_views(
    *, seed: int, step: int, world_size: int, num_views: int,
    microsteps: int = 1,
    sampling_policy: str = "flat_epoch",
    view_frames: Sequence[int] | None = None,
    view_cameras: Sequence[int] | None = None,
    sample_count: int | None = None,
) -> tuple[int, ...]:
    """Return deterministic, rank-addressable uniform view samples.

    Sampling is deterministic and rank-addressable rather than tied to mutable
    process RNG state, so scheduling order does not alter the selected views.
    One call returns all ``P*K`` slots in rank-major order and therefore cannot
    accidentally give every rank the rank-0 stream. ``flat_epoch`` lays
    optimizer-step/rank/microstep slots onto cached random permutations of the
    complete flat ``(frame, camera)`` view table. It therefore covers every
    view exactly once per epoch without forcing ranks onto the same synchronized
    frame.
    ``co_frame`` instead samples one frame, then assigns distinct cameras from
    that frame across ranks (with replacement only when fewer cameras exist
    than ranks). ``unique`` avoids duplicate views within a step when possible,
    while ``with_replacement`` matches Torch RandomSampler's
    uniform-with-replacement *distribution*, not its generator or bit-exact
    sample stream. ``torch_random_sampler`` reproduces FTGSPP's
    ``RandomSampler(replacement=True)`` index stream without importing torch;
    callers pass ``sample_count`` so the stream is cached once per phase.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if microsteps <= 0:
        raise ValueError("microsteps must be positive")
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    global_slots = int(world_size) * int(microsteps)
    if sampling_policy not in {
        "flat_epoch", "unique", "with_replacement", "co_frame",
        "torch_random_sampler",
    }:
        raise ValueError(
            "sampling_policy must be 'flat_epoch', 'unique', "
            "'with_replacement', 'co_frame', or 'torch_random_sampler'"
        )
    if sampling_policy == "torch_random_sampler":
        if step < 0:
            raise ValueError(
                "step must be non-negative for torch_random_sampler sampling"
            )
        if sample_count is None:
            raise ValueError(
                "torch_random_sampler requires sample_count so the legacy "
                "stream can be cached once per phase"
            )
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError("torch_random_sampler sample_count must be positive")
        total_slots = sample_count * global_slots
        first_slot = int(step) * global_slots
        if first_slot + global_slots > total_slots:
            raise ValueError(
                "torch_random_sampler step is outside sample_count"
            )
        stream = _torch_random_sampler_stream(
            int(seed), int(num_views), int(total_slots)
        )
        return tuple(
            int(stream[first_slot + rank * int(microsteps) + microstep])
            for rank in range(int(world_size))
            for microstep in range(int(microsteps))
        )
    if sampling_policy == "flat_epoch":
        if step < 0:
            raise ValueError("step must be non-negative for flat_epoch sampling")
        values: list[int] = []
        for rank in range(int(world_size)):
            for microstep in range(int(microsteps)):
                global_slot = (
                    int(step) * global_slots
                    + rank * int(microsteps)
                    + microstep
                )
                epoch, offset = divmod(global_slot, int(num_views))
                permutation = _flat_epoch_permutation(
                    int(seed), int(epoch), int(num_views)
                )
                values.append(int(permutation[offset]))
        return tuple(values)
    if sampling_policy == "co_frame":
        return _sample_co_frame_views(
            seed=int(seed),
            step=int(step),
            world_size=global_slots,
            num_views=int(num_views),
            view_frames=view_frames,
            view_cameras=view_cameras,
        )
    if world_size == 1 and microsteps == 1:
        # Keep the historical keyed NumPy stream only for the two explicit
        # step-local legacy policies.  ``flat_epoch`` and ``co_frame`` must go
        # through their real implementations even for a single device.
        return (int(np.random.default_rng(int(seed) + int(step)).integers(num_views)),)
    # SeedSequence avoids collisions inherent in the old seed+step+rank form.
    entropy = [int(seed) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, 0x46544753]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    values = generator.choice(
        int(num_views),
        size=global_slots,
        replace=(
            sampling_policy == "with_replacement"
            or int(num_views) < global_slots
        ),
    )
    return tuple(int(value) for value in values)


def _sample_co_frame_views(
    *,
    seed: int,
    step: int,
    world_size: int,
    num_views: int,
    view_frames: Sequence[int] | None,
    view_cameras: Sequence[int] | None,
) -> tuple[int, ...]:
    """Bind ranks to one synchronized frame with distinct cameras when possible."""

    if view_frames is None or view_cameras is None:
        raise ValueError(
            "co_frame sampling requires view_frames and view_cameras metadata"
        )
    if len(view_frames) != num_views or len(view_cameras) != num_views:
        raise ValueError(
            "view_frames/view_cameras length must equal num_views for co_frame"
        )
    frames_by_id: dict[int, list[int]] = {}
    for index, frame in enumerate(view_frames):
        frames_by_id.setdefault(int(frame), []).append(int(index))
    frame_ids = sorted(frames_by_id)
    if not frame_ids:
        raise ValueError("co_frame sampling found no frames")
    entropy = [int(seed) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, 0x434F4652]  # COFR
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    frame = int(frame_ids[int(generator.integers(len(frame_ids)))])
    candidates = list(frames_by_id[frame])
    # Prefer diversity across cameras: shuffle then take unique cameras first.
    generator.shuffle(candidates)
    chosen: list[int] = []
    seen_cameras: set[int] = set()
    leftovers: list[int] = []
    for index in candidates:
        camera = int(view_cameras[index])
        if camera in seen_cameras:
            leftovers.append(index)
            continue
        seen_cameras.add(camera)
        chosen.append(index)
        if len(chosen) >= world_size:
            break
    while len(chosen) < world_size:
        pool = leftovers if leftovers else candidates
        chosen.append(int(pool[int(generator.integers(len(pool)))]))
    return tuple(int(value) for value in chosen[:world_size])




def _mean_scalars(per_rank: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not per_rank:
        return {}
    keys = set(per_rank[0])
    if any(set(values) != keys for values in per_rank[1:]):
        raise RuntimeError("native loss scalar keys differ between replicas")
    result: dict[str, float] = {}
    for key in sorted(keys):
        values = np.asarray([rank[key] for rank in per_rank], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"native loss scalar {key!r} is non-finite")
        result[key] = float(np.mean(values))
    if {"l1", "ssim"} <= result.keys():
        result["image_loss"] = result["l1"] + result["ssim"]
    objective_terms = (
        "l1", "ssim", "reg_opacity", "reg_scale", "gate", "color_cc",
    )
    if all(key in result for key in objective_terms):
        # Reconstruct derived values from the globally averaged primitive terms
        # so the logged identity is exact, rather than relying on two separately
        # rounded reductions of mathematically equivalent expressions.
        result["total_loss"] = sum(result[key] for key in objective_terms)
    return result


def _rank_backward_timing(
    rank_backward_seconds: Sequence[float],
) -> dict[str, float]:
    """Summarize per-rank backward times without changing the mean field.

    ``backward_seconds`` stays the cross-rank mean for backward compatibility.
    The step's compute barrier is closer to ``backward_seconds_max`` because the
    faster rank blocks inside exchange until the slower rank arrives.
    """

    values = [float(value) for value in rank_backward_seconds]
    if not values:
        return {
            "backward_seconds": 0.0,
            "backward_seconds_max": 0.0,
            "backward_seconds_min": 0.0,
            "backward_seconds_imbalance": 0.0,
        }
    maximum = float(max(values))
    minimum = float(min(values))
    summary: dict[str, float] = {
        "backward_seconds": float(np.mean(values)),
        "backward_seconds_max": maximum,
        "backward_seconds_min": minimum,
        "backward_seconds_imbalance": maximum - minimum,
    }
    for rank, value in enumerate(values):
        summary[f"backward_seconds_rank{rank}"] = float(value)
    return summary


def _exchange_timing_summary(
    exchanges: Sequence[Mapping[str, object]],
    *,
    download_seconds: float,
    upload_seconds: float,
) -> dict[str, float]:
    """Split exchange wall into peer-wait vs transfer for overlap analysis.

    ``gradient_exchange_wall_seconds`` remains max(rank wall) and still includes
    peer wait. Prefer ``gradient_transfer_overlap`` (= (dl+ul)/transfer_wall)
    over comparing (dl+ul) to the full exchange wall.
    """

    walls = [float(value.get("wall_seconds", 0.0)) for value in exchanges]
    peer_waits: list[float] = []
    transfer_walls: list[float] = []
    sparse_packs: list[float] = []
    for value in exchanges:
        peer_wait = float(value.get("peer_wait_seconds", 0.0))
        peer_waits.append(peer_wait)
        if "transfer_wall_seconds" in value:
            transfer_wall = float(value.get("transfer_wall_seconds", 0.0))
        else:
            # Old native builds: approximate transfer as wall minus peer wait.
            transfer_wall = max(
                0.0, float(value.get("wall_seconds", 0.0)) - peer_wait,
            )
        transfer_walls.append(transfer_wall)
        sparse_packs.append(float(value.get("sparse_pack_seconds", 0.0)))

    exchange_wall = float(max(walls)) if walls else 0.0
    peer_wait_max = float(max(peer_waits)) if peer_waits else 0.0
    peer_wait_mean = float(np.mean(peer_waits)) if peer_waits else 0.0
    transfer_wall = float(max(transfer_walls)) if transfer_walls else 0.0
    transfer_wall_mean = float(np.mean(transfer_walls)) if transfer_walls else 0.0
    copy_seconds = float(download_seconds) + float(upload_seconds)
    transfer_denom = max(transfer_wall_mean, 1e-12)
    exchange_denom = max(exchange_wall, 1e-12)
    return {
        "gradient_exchange_wall_seconds": exchange_wall,
        "gradient_peer_wait_seconds": peer_wait_max,
        "gradient_peer_wait_seconds_mean": peer_wait_mean,
        "gradient_sparse_pack_seconds": (
            float(np.mean(sparse_packs)) if sparse_packs else 0.0
        ),
        "gradient_transfer_wall_seconds": transfer_wall,
        "gradient_transfer_wall_seconds_mean": transfer_wall_mean,
        # True copy-pipeline overlap (excludes peer idle time).
        "gradient_transfer_overlap": copy_seconds / transfer_denom,
        # Legacy ratio; drops below 1.0 when peer wait dominates (full-res).
        "gradient_exchange_overlap": copy_seconds / exchange_denom,
        "pcie_d2h_bytes": float(np.mean([
            float(value.get("pcie_d2h_bytes", value.get("payload_bytes", 0)))
            for value in exchanges
        ])) if exchanges else 0.0,
        "pcie_h2d_bytes": float(np.mean([
            float(value.get("pcie_h2d_bytes", value.get("payload_bytes", 0)))
            for value in exchanges
        ])) if exchanges else 0.0,
    }


class SynchronousBackendGroup:
    """Own and coordinate one native backend per Vulkan device."""

    def __init__(
        self,
        backends: Sequence[SynchronousBackend],
        *,
        device_ids: Sequence[int] | None = None,
        reduction: str = "mean",
        transport: str = "hierarchical_host",
        collective_algorithm: str = "auto",
        collective_bucket_bytes: int = 32 * 1024 * 1024,
        collective_sparse: bool = True,
        collective_codec_policy: str = "auto_exact",
        local_microsteps: int = 1,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        if reduction != "mean":
            raise ValueError("only pre-Adam mean reduction is supported")
        self.backends = tuple(backends)
        self.device_ids = tuple(
            range(len(backends)) if device_ids is None else (int(value) for value in device_ids)
        )
        if len(self.device_ids) != len(self.backends):
            raise ValueError("device_ids and backends must have the same length")
        self.reduction = reduction
        requested_transport = str(transport)
        if requested_transport not in {"auto", "hierarchical_host"}:
            raise ValueError(
                "unsupported synchronous transport "
                f"{requested_transport!r}; expected 'auto' or "
                "'hierarchical_host'"
            )
        self.transport = "hierarchical_host"
        self.collective_algorithm = str(collective_algorithm)
        if self.collective_algorithm == "auto" and len(backends) == 2:
            self.collective_algorithm = PAIR_EXCHANGE_GPU_MERGE
        self.collective_bucket_bytes = int(collective_bucket_bytes)
        self.collective_sparse = bool(collective_sparse)
        self.collective_codec_policy = (
            str(collective_codec_policy)
            if self.collective_sparse else "dense"
        )
        if self.collective_codec_policy not in {
            "auto_exact", "dense", "sorted_sparse",
        }:
            raise ValueError(
                "collective_codec_policy must be auto_exact, dense, or "
                "sorted_sparse"
            )
        if self.collective_codec_policy == "dense":
            self.collective_sparse = False
        self.local_microsteps = int(local_microsteps)
        if self.local_microsteps not in {1, 2, 4}:
            raise ValueError("local_microsteps must be one of 1, 2, or 4")
        if self.world_size != 2:
            raise ValueError(
                "hierarchical_host supports only a dual-device world"
            )
        if (
            self.collective_bucket_bytes < 0
            or self.collective_bucket_bytes % 4
        ):
            raise ValueError(
                "collective_bucket_bytes must be non-negative and 4-byte aligned"
            )
        self.collective_bucket_benchmark: dict[str, object] = {
            "executed": False,
            "reason": "explicit_override",
            "selected_bucket_bytes": self.collective_bucket_bytes,
        }
        if self.collective_bucket_bytes == 0:
            self.collective_bucket_bytes = (
                self._calibrate_pair_transfer_bucket()
            )
        self.pair_exchange_probe: dict[str, object] = {}
        if self.collective_algorithm != PAIR_EXCHANGE_GPU_MERGE:
            raise RuntimeError(
                "hierarchical_host requires pair_exchange_gpu_merge"
            )
        pair_supported = all(
            callable(
                getattr(backend, "supports_pair_exchange_gpu_merge", None)
            )
            and bool(backend.supports_pair_exchange_gpu_merge())
            for backend in self.backends
        )
        if not pair_supported:
            raise RuntimeError(
                "hierarchical_host requires the native pair_exchange_gpu_merge "
                "capability; rebuild the native extension with the pair "
                "transfer-ticket ABI"
            )
        try:
            self.pair_exchange_probe = dict(
                self.leader.create_pair_exchange_gpu_merge_owner(
                    world_size=self.world_size,
                    bucket_bytes=self.collective_bucket_bytes,
                    sparse=self.collective_sparse,
                )
            )
            self._validate_pair_exchange_probe(self.pair_exchange_probe)
            self.collective_sparse = bool(
                self.pair_exchange_probe.get(
                    "sparse", self.collective_sparse
                )
            )
            for backend in self.backends:
                backend.configure_pair_exchange_gpu_merge(
                    sparse=self.collective_sparse,
                    codec_policy=self.collective_codec_policy,
                )
        except Exception:
            self.leader.release_pair_exchange_gpu_merge_owner()
            raise
        sparse_bucket = int(
            self.pair_exchange_probe.get("sparse_bucket_bytes", 0) or 0
        )
        if sparse_bucket:
            logger.info(
                "selected hierarchical_host/%s bucket=%d MiB sparse_bucket=%d MiB",
                self.collective_algorithm,
                self.collective_bucket_bytes // (1024 * 1024),
                sparse_bucket // (1024 * 1024),
            )
        else:
            logger.info(
                "selected hierarchical_host/%s bucket=%d MiB",
                self.collective_algorithm,
                self.collective_bucket_bytes // (1024 * 1024),
            )
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.backends), thread_name_prefix="ftgspp-vk-rank"
        )
        # JPEG decode/undistort for future global batches must not occupy the
        # rank executor while Vulkan backward is running. The shared cache only
        # commits completed lookahead images when that batch is consumed.
        self._lookahead_executor_workers = max(1, len(self.backends))
        self._lookahead_executor = ThreadPoolExecutor(
            max_workers=self._lookahead_executor_workers,
            thread_name_prefix="ftgspp-rgb-lookahead",
        )
        self._relocation_collective_generation = 0
        self._closed = False

    def _calibrate_pair_transfer_bucket(self) -> int:
        """Select 8/16/32/64 MiB by the slower concurrently measured rank."""

        candidates = tuple(
            value * 1024 * 1024 for value in (8, 16, 32, 64)
        )
        rounds = 3
        methods = [
            getattr(backend, "benchmark_pair_transfer_buckets", None)
            for backend in self.backends
        ]
        if not all(callable(method) for method in methods):
            raise RuntimeError(
                "automatic pair bucket calibration requires a rebuilt native "
                "backend on every rank"
            )
        with ThreadPoolExecutor(
            max_workers=self.world_size,
            thread_name_prefix="ftgspp-bucket-calibration",
        ) as executor:
            futures = [
                executor.submit(method, candidates, rounds=rounds)
                for method in methods
            ]
            rank_results = [future.result() for future in futures]
        rank_by_bucket: list[dict[int, dict[str, object]]] = []
        for rank, values in enumerate(rank_results):
            indexed = {
                int(value["bucket_bytes"]): dict(value) for value in values
            }
            if set(indexed) != set(candidates):
                raise RuntimeError(
                    f"rank {rank} bucket calibration returned candidates "
                    f"{sorted(indexed)}, expected {list(candidates)}"
                )
            for candidate, value in indexed.items():
                wall = float(value.get("wall_seconds", float("nan")))
                if not np.isfinite(wall) or wall <= 0.0:
                    raise RuntimeError(
                        "non-finite/non-positive pair bucket calibration for "
                        f"rank {rank}, bucket {candidate}: {wall}"
                    )
            rank_by_bucket.append(indexed)
        scores = {
            candidate: max(
                float(values[candidate]["wall_seconds"])
                for values in rank_by_bucket
            )
            for candidate in candidates
        }
        selected = min(candidates, key=lambda value: (scores[value], value))
        self.collective_bucket_benchmark = {
            "executed": True,
            "schema": (
                "ftgspp-pair-bucket-calibration-v1"
                if self.world_size == 2
                else "ftgspp-hierarchical-bucket-calibration-v1"
            ),
            "rounds": rounds,
            "fixed_payload_bytes": max(candidates),
            "candidates": [
                {
                    "bucket_bytes": candidate,
                    "slower_rank_wall_seconds": scores[candidate],
                }
                for candidate in candidates
            ],
            "rank_results": rank_results,
            "selected_bucket_bytes": selected,
        }
        return selected

    @staticmethod
    def _validate_pair_exchange_probe(
        report: Mapping[str, object],
    ) -> None:
        if str(report.get("abi", "")) != NATIVE_PAIR_EXCHANGE_ABI:
            raise RuntimeError(
                "stale native pair-exchange ABI: Python requires "
                f"{NATIVE_PAIR_EXCHANGE_ABI!r}, native reported "
                f"{report.get('abi', 'missing')!r}"
            )
        if not bool(report.get("async_pipeline", False)) or not bool(
            report.get("borrowed_readback", False)
        ):
            raise RuntimeError(
                "pair_exchange_gpu_merge requires async transfer tickets and "
                "borrowed Vulkan readback mappings"
            )

    def _recreate_collective_owner(self) -> None:
        """Rebind phase-specific dense layouts after a global phase barrier."""

        for backend in self.backends:
            backend.reset_pair_exchange_gpu_merge_attachments()
        self.leader.release_pair_exchange_gpu_merge_owner()
        try:
            self.pair_exchange_probe = dict(
                self.leader.create_pair_exchange_gpu_merge_owner(
                    world_size=self.world_size,
                    bucket_bytes=self.collective_bucket_bytes,
                    sparse=self.collective_sparse,
                )
            )
            self._validate_pair_exchange_probe(self.pair_exchange_probe)
            self.collective_sparse = bool(
                self.pair_exchange_probe.get(
                    "sparse", self.collective_sparse
                )
            )
            for backend in self.backends:
                backend.configure_pair_exchange_gpu_merge(
                    sparse=self.collective_sparse,
                    codec_policy=self.collective_codec_policy,
                )
        except Exception:
            self.leader.release_pair_exchange_gpu_merge_owner()
            raise
        self._relocation_collective_generation = 0

    @property
    def world_size(self) -> int:
        return len(self.backends)

    @property
    def leader(self) -> SynchronousBackend:
        return self.backends[0]

    def configure_view_lookahead(self, *, workers: int) -> None:
        """Resize the shared RGB decode pool after view options resolve."""

        workers = max(1, int(workers))
        if workers == self._lookahead_executor_workers:
            return
        self._lookahead_executor.shutdown(wait=True, cancel_futures=True)
        self._lookahead_executor_workers = workers
        self._lookahead_executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="ftgspp-rgb-lookahead",
        )
        logger.info(
            "sync lazy RGB lookahead executor resized: workers=%d",
            workers,
        )

    def view_lookahead_workers(self) -> int:
        return int(self._lookahead_executor_workers)

    def performance_snapshot(self) -> dict[str, object]:
        """Collect per-rank native counters at a quiescent step boundary."""

        replicas = [
            dict(backend.performance_snapshot()) for backend in self.backends
        ]
        host_storage = max(
            int(self.pair_exchange_probe.get("host_storage_bytes", 0)),
            max(
                (
                    int(value.get("host_staging_bytes", 0))
                    for value in replicas
                ),
                default=0,
            ),
        )
        return {
            "world_size": self.world_size,
            "transport": self.transport,
            "collective_algorithm": self.collective_algorithm,
            "collective_sparse": self.collective_sparse,
            "collective_bucket_bytes": self.collective_bucket_bytes,
            "collective_sparse_bucket_bytes": int(
                self.pair_exchange_probe.get("sparse_bucket_bytes", 0) or 0
            ),
            "collective_bucket_benchmark": dict(
                self.collective_bucket_benchmark
            ),
            "host_staging_bytes": host_storage,
            "replicas": replicas,
        }

    def _parallel(self, operation: Callable[[int, SynchronousBackend], T]) -> list[T]:
        if self._closed:
            raise RuntimeError("synchronous backend group is closed")
        futures: list[Future[T]] = [
            self._executor.submit(operation, rank, backend)
            for rank, backend in enumerate(self.backends)
        ]
        # Resolve in rank order for deterministic error attribution/results.
        results: list[T] = []
        try:
            for rank, future in enumerate(futures):
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return results

    def _complete_generation_with_overflow_replay(
        self,
        views: tuple[int, ...],
        step: int,
        *,
        collect_loss: bool,
        exchange_op: Callable[[int, SynchronousBackend], Mapping[str, object]],
        parallel_op: Callable[
            [Callable[[int, SynchronousBackend], T]], list[T]
        ],
    ) -> tuple[list[tuple[float, dict[str, object], list[dict[str, float]]]], int]:
        """Run rank-local backward+exchange, replaying a globally aborted overflow."""

        retries = 0
        while True:
            def _backward_then_exchange(
                rank: int, backend: SynchronousBackend,
            ) -> tuple[float, dict[str, object], list[dict[str, float]]]:
                backward_begin = time.perf_counter()
                first = rank * self.local_microsteps
                rank_losses: list[dict[str, float]] = []
                for view in views[first:first + self.local_microsteps]:
                    backend.backward_view(
                        view, int(step), collect_loss=collect_loss)
                    if collect_loss and self.local_microsteps > 1:
                        rank_losses.append(dict(backend.last_loss()))
                backend.finalize_local_gradient(self.local_microsteps)
                backward_seconds = time.perf_counter() - backward_begin
                # First gradient bucket D2H starts inside this rank worker as
                # soon as local compute retires. Python does not join the peer
                # backward before exchange.
                exchange = dict(exchange_op(rank, backend))
                return backward_seconds, exchange, rank_losses

            rank_results = parallel_op(_backward_then_exchange)
            aborted = [
                dict(value[1]) for value in rank_results
                if bool(value[1].get("aborted", False))
            ]
            if not aborted:
                return rank_results, retries
            if any(bool(value.get("fatal", False)) for value in aborted):
                reason = next(
                    (
                        str(value.get("reason", "tile invariant abort"))
                        for value in aborted if bool(value.get("fatal", False))
                    ),
                    "tile invariant abort",
                )
                raise RuntimeError(reason)
            retries += 1
            if retries >= 4:
                raise RuntimeError(
                    "consecutive tile overflow across ranks; "
                    "capacity growth did not converge"
                )
            max_exact = max(
                int(value[1].get("max_exact_count", 0))
                for value in rank_results
            )
            logger.warning(
                "sync generation %s aborted for tile overflow; "
                "growing all ranks to exact_count>=%s and replaying",
                int(step) + 1, max_exact,
            )
            parallel_op(
                lambda _rank, backend: backend.apply_sync_tile_capacity(
                    max_exact),
            )
            self.leader.clear_sync_collectives_for_replay()

    def train_step(
        self,
        view_indices: Sequence[int],
        step: int,
        *,
        phase: str = "main",
        collect_loss: bool = True,
        next_view_indices: Sequence[int] | None = None,
    ) -> dict[str, object]:
        """Execute one globally synchronous optimizer update."""

        expected_views = self.world_size * self.local_microsteps
        if len(view_indices) != expected_views:
            raise ValueError(
                f"expected {expected_views} rank-grouped views, "
                f"got {len(view_indices)}"
            )
        views = tuple(int(value) for value in view_indices)
        next_views = (
            None if next_view_indices is None
            else tuple(int(value) for value in next_view_indices)
        )
        if next_views is not None and (
            len(next_views) == 0 or len(next_views) % expected_views != 0
        ):
            raise ValueError(
                f"expected a non-empty multiple of {expected_views} "
                f"lookahead views, got {len(next_views)}"
            )
        if phase not in {"main", "hard_separation"}:
            raise ValueError(f"unknown synchronous optimizer phase: {phase!r}")
        return self._train_step_pair_exchange_gpu_merge(
            views, int(step), phase=phase, collect_loss=collect_loss,
            next_views=next_views,
        )

    def _train_step_pair_exchange_gpu_merge(
        self,
        views: tuple[int, ...],
        step: int,
        *,
        phase: str,
        collect_loss: bool,
        next_views: tuple[int, ...] | None,
    ) -> dict[str, object]:
        """Dual-GPU Host traversal with Vulkan pair reduction.

        Each rank queues readback into a persistently mapped Vulkan ticket,
        publishes that borrowed mapping, and streams rank-ordered mappings into
        its private upload stager. No packed gradient becomes a NumPy object;
        CPU reduction time and the former readback-to-mailbox memcpy are both
        identically zero.
        """

        data_begin = time.perf_counter()
        self.leader.prefetch_train_views(
            views, executor=self._lookahead_executor,
        )
        data_seconds = time.perf_counter() - data_begin
        if next_views is not None:
            self.leader.schedule_train_views(
                next_views, executor=self._lookahead_executor,
            )
        wave_begin = time.perf_counter()
        rank_results, overflow_retries = self._complete_generation_with_overflow_replay(
            views, int(step), collect_loss=collect_loss,
            exchange_op=lambda rank, backend: backend.exchange_pair_gradient(
                rank=rank, generation=int(step) + 1,
            ),
            parallel_op=self._parallel,
        )
        wave_seconds = time.perf_counter() - wave_begin
        exchanges = [value[1] for value in rank_results]
        download_seconds = float(np.mean([
            float(value.get("download_seconds", 0.0)) for value in exchanges
        ]))
        upload_seconds = float(np.mean([
            float(value.get("upload_seconds", 0.0)) for value in exchanges
        ]))
        host_memcpy_seconds = float(np.mean([
            float(value.get("host_memcpy_seconds", 0.0)) for value in exchanges
        ]))
        exchange_timing = _exchange_timing_summary(
            exchanges,
            download_seconds=download_seconds,
            upload_seconds=upload_seconds,
        )
        persistent_host_mapping = bool(exchanges) and all(
            bool(value.get("vulkan_persistent_mapping", False))
            for value in exchanges
        )
        async_pipeline = bool(exchanges) and all(
            bool(value.get("async_pipeline", False)) for value in exchanges
        )
        timestamps = tuple(
            float(self.backends[rank].train_view_timestamp(view))
            for rank in range(self.world_size)
            for view in views[
                rank * self.local_microsteps:
                (rank + 1) * self.local_microsteps
            ]
        )
        collect_grads = collect_loss and _log_geometry_grads_enabled()
        if collect_grads:
            self._parallel(
                lambda _rank, backend: backend.set_collect_geometry_grad_stats(
                    True),
            )
        try:
            optimizer_reports = self._parallel(
                lambda _rank, backend:
                    backend.optimizer_step_pair_exchange_gpu_merge(
                        int(step), timestamps,
                    )
            )
            reduction_seconds = float(np.mean([
                float(value.get("gpu_reduction_seconds", 0.0))
                for value in optimizer_reports
            ]))
            optimizer_seconds = float(np.mean([
                float(value.get("optimizer_seconds", 0.0))
                for value in optimizer_reports
            ]))
            loss = None
            if collect_loss:
                post_optimizer_losses = self._parallel(
                    lambda _rank, backend: dict(backend.last_loss())
                )
                if self.local_microsteps == 1:
                    loss = _mean_scalars(post_optimizer_losses)
                else:
                    loss = _mean_scalars([
                        item for value in rank_results for item in value[2]
                    ])
                    loss["color_cc"] = _mean_scalars(
                        post_optimizer_losses
                    )["color_cc"]
                    loss["total_loss"] = sum(loss[key] for key in (
                        "l1", "ssim", "reg_opacity", "reg_scale", "gate",
                        "color_cc",
                    ))
                if collect_grads:
                    grad_stats = self._parallel(
                        lambda _rank, backend: dict(
                            backend.geometry_gradient_stats()),
                    )
                    loss.update(_mean_scalars(grad_stats))
        finally:
            if collect_grads:
                self._parallel(
                    lambda _rank, backend:
                        backend.set_collect_geometry_grad_stats(False),
                )
        return {
            "loss": loss,
            "views": views,
            "data_seconds": data_seconds,
            **_rank_backward_timing([value[0] for value in rank_results]),
            "gradient_download_seconds": download_seconds,
            "reduction_seconds": reduction_seconds,
            "gradient_upload_seconds": upload_seconds,
            "gradient_host_memcpy_seconds": host_memcpy_seconds,
            **exchange_timing,
            "gradient_async_pipeline": async_pipeline,
            "gradient_persistent_host_mapping": persistent_host_mapping,
            "optimizer_seconds": optimizer_seconds,
            "wave_backward_download_seconds": wave_seconds,
            "pair_exchange": exchanges[0] if exchanges else {},
            "transport": "hierarchical_host",
            "collective_algorithm": PAIR_EXCHANGE_GPU_MERGE,
            "sync_abort_retries": overflow_retries,
        }

    def parallel_shards(
        self,
        items: Sequence[int],
        operation: Callable[[int, SynchronousBackend, tuple[int, ...]], T],
    ) -> list[T]:
        """Run deterministic rank-strided evaluation shards in parallel."""

        shards = tuple(
            tuple(int(items[index]) for index in range(rank, len(items), self.world_size))
            for rank in range(self.world_size)
        )
        return self._parallel(
            lambda rank, backend: operation(rank, backend, shards[rank])
        )

    def begin_hard_separation(self) -> dict[str, int]:
        """Apply the deterministic gate split on every replica."""
        results = self._parallel(
            lambda _rank, backend: backend.begin_hard_separation()
        )
        if not results:
            return {"marginalized": 0, "kept": 0}
        if any(dict(value) != dict(results[0]) for value in results[1:]):
            raise RuntimeError("hard-separation split differs between replicas")
        return {str(key): int(value) for key, value in results[0].items()}

    def resume_hard_separation(self) -> dict[str, int]:
        """Restore hard-phase optimizer settings on every replica."""
        results = self._parallel(
            lambda _rank, backend: backend.resume_hard_separation()
        )
        if not results:
            return {"marginalized": 0, "kept": 0}
        if any(dict(value) != dict(results[0]) for value in results[1:]):
            raise RuntimeError("hard-separation resume differs between replicas")
        return {str(key): int(value) for key, value in results[0].items()}

    def hard_separation_digest(self) -> str:
        """Verify that all replicas have the same post-split gate state."""
        digests = self._parallel(
            lambda _rank, backend: backend.hard_separation_digest()
        )
        if digests and any(value != digests[0] for value in digests[1:]):
            raise RuntimeError("hard-separation gate state differs between replicas")
        return str(digests[0]) if digests else ""

    def _sum_relocation_statistics(self) -> None:
        """Add per-rank relocation observations onto every replica."""

        self._relocation_collective_generation += 1
        generation = self._relocation_collective_generation
        # exchange_*_relocation already waits for compute and packed transfers.
        self._parallel(
            lambda rank, backend: backend.exchange_pair_relocation(
                rank=rank, generation=generation,
            )
        )

    def _global_relocation_statistics(self) -> np.ndarray:
        """Return the additive global statistic from the active collective."""

        self._sum_relocation_statistics()
        # Every device now contains the same native-computed sum. Only rank 0 is
        # read back because the checkpoint writer has one authoritative owner.
        return np.ascontiguousarray(
            self.leader.relocation_stats_state(), dtype=np.float32,
        )

    def consolidate_relocation_statistics(self) -> np.ndarray:
        """Place global history on rank 0 and zero followers for checkpointing.

        This single-owner representation remains additive: subsequent local
        observations can be summed without multiplying checkpoint history by
        ``world_size``.
        """

        global_statistics = self._global_relocation_statistics()
        sharded = shard_resumed_relocation_statistics(
            global_statistics, self.world_size, owner_rank=0
        )
        self._parallel(
            lambda rank, backend: backend.set_relocation_stats_state(sharded[rank])
        )
        return global_statistics

    def relocate(
        self, step: int, mode: str, include_gate: bool, seed: int,
    ) -> dict[str, int | bool]:
        """Sum observations, plan once on rank 0, and broadcast topology."""

        self._sum_relocation_statistics()
        plan = dict(self.leader.plan_relocation_native(
            int(step), bool(include_gate), int(seed)
        ))
        if "mapping" not in plan or "counts" not in plan:
            raise RuntimeError("native relocation planner did not return mapping/counts")
        mapping = np.ascontiguousarray(plan.pop("mapping"), dtype=np.int32)
        counts = np.ascontiguousarray(plan.pop("counts"), dtype=np.int32)
        if mapping.ndim != 1 or counts.shape != mapping.shape:
            raise RuntimeError(
                f"invalid native relocation plan: mapping={mapping.shape}, counts={counts.shape}"
            )
        donors = mapping[mapping >= 0]
        if np.any(mapping < -1):
            raise RuntimeError("native relocation mapping contains an invalid negative index")
        if donors.size and int(donors.max()) >= mapping.size:
            raise RuntimeError("native relocation plan contains an out-of-range donor")
        expected_counts = np.bincount(donors, minlength=mapping.size).astype(np.int32)
        if not np.array_equal(counts, expected_counts):
            raise RuntimeError("native relocation counts do not match its mapping")
        relocated = int(np.count_nonzero(mapping >= 0))
        all_dead = bool(plan.get("all_dead", False))
        reported_relocated = int(plan.get("relocated", relocated))
        if reported_relocated != relocated:
            raise RuntimeError(
                "native relocation plan relocated count does not match mapping"
            )
        if all_dead:
            if relocated != 0 or np.any(counts != 0):
                raise RuntimeError(
                    "all-dead relocation plan must not select a donor"
                )
            if int(plan.get("alive", 0)) != 0:
                raise RuntimeError(
                    "all-dead relocation plan reports a non-zero alive count"
                )
            self._parallel(
                lambda rank, backend: (
                    None
                    if rank == 0
                    else backend.clear_relocation_stats_state()
                )
            )
            return {
                "relocated": 0,
                "alive": 0,
                "all_dead": True,
            }
        reported_alive = int(plan.get("alive", mapping.size - reported_relocated))
        if reported_alive + reported_relocated != mapping.size:
            raise RuntimeError(
                "native relocation plan alive count does not match mapping"
            )
        # The GPU sampler does not cap MCMC copies.  Drop extras here so the
        # broadcast mapping stays inside relocate_dynamic's [0, 50] contract.
        mapping, counts = cap_relocation_donor_copies(mapping, counts)
        relocated = int(np.count_nonzero(mapping >= 0))
        capped = relocated != reported_relocated
        if capped:
            logger.info(
                "capped relocation donor copies: planned=%d replayed=%d max_copies=%d",
                reported_relocated,
                relocated,
                RELOCATION_MAX_DONOR_COPIES,
            )

        def _apply(rank: int, backend: SynchronousBackend) -> None:
            if rank == 0 and not capped:
                backend.apply_planned_relocation(str(mode))
                return
            backend.relocate(
                mapping, counts, str(mode),
                require_dead_destinations=False,
                require_alive_donors=False,
            )

        self._parallel(_apply)
        result: dict[str, int | bool] = {
            "relocated": relocated,
            "alive": reported_alive,
            "all_dead": False,
        }
        return result

    def checkpoint_optimizer_state(self) -> dict[str, np.ndarray]:
        """Return leader Adam state with globally consolidated relocation stats."""

        global_statistics = self.consolidate_relocation_statistics()
        # optimizer_state() already returns newly owned host arrays from the
        # leader. Transfer that ownership directly into the immutable
        # checkpoint snapshot; copying every field/Adam array here would add a
        # second multi-gigabyte peak allocation for no isolation benefit.
        state = {
            str(key): value
            for key, value in self.leader.optimizer_state().items()
        }
        state["relocation_stats"] = global_statistics
        return state

    def assert_replicas(self, *, include_optimizer: bool = False) -> None:
        """Expensive exact diagnostic; relocation statistics are excluded."""

        models = self._parallel(
            lambda _rank, backend: {
                str(key): np.asarray(value)
                for key, value in backend.model().arrays().items()
            }
        )
        assert_replicas_consistent(models)
        if include_optimizer:
            optimizer = self._parallel(
                lambda _rank, backend: {
                    str(key): np.asarray(value)
                    for key, value in backend.optimizer_state().items()
                    if key != "relocation_stats"
                }
            )
            assert_replicas_consistent(optimizer)

    def close(self) -> None:
        if self._closed:
            return
        collective_released = False
        try:
            # Decode futures retain NativeView objects. Finish/cancel them
            # before destroying the native backends and shared dataset state.
            self._lookahead_executor.shutdown(
                wait=True, cancel_futures=True,
            )
            # The owner is module-global in the native extension. Release it
            # while every backend still has a live native trainer handle.
            for backend in self.backends:
                backend.reset_pair_exchange_gpu_merge_attachments()
            self.leader.release_pair_exchange_gpu_merge_owner()
            collective_released = True
            futures = [self._executor.submit(backend.close) for backend in self.backends]
            for future in futures:
                future.result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if not collective_released:
                try:
                    self.leader.release_pair_exchange_gpu_merge_owner()
                except Exception:
                    logger.exception(
                        "failed to release hierarchical collective owner"
                    )
            self._closed = True

    def __enter__(self) -> "SynchronousBackendGroup":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
