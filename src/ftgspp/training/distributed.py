"""Synchronous replicated multi-device orchestration for native Vulkan.

Every backend owns a complete model and Adam state. A macro-step consumes one
view per rank, reduces the rank-local FP32 gradients across P before Adam, and
applies the same global mean on every replica. Multi-device production has one
user-visible ``pair_host`` backend; P=2 uses the native pair-exchange
GPU merge.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import (
    as_completed,
    Executor,
    Future,
    ThreadPoolExecutor,
)
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

from ftgspp.config.schema import AppConfig
from ftgspp.io.model import NativeGaussians
from ftgspp.training.distributed_contract import (
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


PAIR_HOST_COLLECTIVE_ABI = "ftgspp-pair-host-v1"
# Temporary native capability ABI used only by the P=2 implementation. It is
# intentionally not a configured backend or checkpoint transport.
NATIVE_PAIR_EXCHANGE_ABI = "ftgspp-host-mailbox-v3"
PAIR_EXCHANGE_GPU_MERGE = "pair_exchange_gpu_merge"
class SynchronousBackend(Protocol):
    """Backend surface required by :class:`SynchronousBackendGroup`.

    The protocol is deliberately duck-typed so unit tests can use a fake
    backend and so a future Vulkan peer reducer can preserve this coordinator.
    """

    def backward_view(
        self, view_index: int, step: int, *, collect_loss: bool = True,
        background=(0.0, 0.0, 0.0),
    ) -> None: ...
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
    def upload_model(self, model: NativeGaussians) -> None: ...
    def supports_pair_exchange_gpu_merge(self) -> bool: ...
    def create_pair_exchange_gpu_merge_owner(
        self, *, world_size: int, bucket_bytes: int, sparse: bool, timeout_seconds: float,
    ) -> Mapping[str, object]: ...
    def join_pair_session(self, leader: SynchronousBackend) -> None: ...
    def abort_pair_session(self, reason: str) -> None: ...
    def configure_pair_exchange_gpu_merge(
        self, *, sparse: bool, codec_policy: str = "auto_exact",
    ) -> None: ...
    def exchange_pair_gradient(
        self, *, rank: int, generation: int,
    ) -> Mapping[str, object]: ...
    def optimizer_step_pair_exchange_gpu_merge(
        self, step: int, rank0_timestamp: float, rank1_timestamp: float,
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
    verify_replica_interval: int
    collective_algorithm: str = PAIR_EXCHANGE_GPU_MERGE
    collective_bucket_bytes: int = 32 * 1024 * 1024
    collective_timeout_seconds: float = 120.0
    collective_sparse: bool = True
    collective_codec_policy: str = "auto_exact"
    collective_reduction_tree: str = "fixed_rank_order_v1"
    collective_topology_policy: str = "device_uuid_pci_numa_v1"
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
            "pair_host"
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
            verify_replica_interval=int(distributed.verify_replica_interval),
            collective_algorithm=collective_algorithm,
            collective_bucket_bytes=int(collective.bucket_mb) * 1024 * 1024,
            collective_timeout_seconds=float(collective.timeout_seconds),
            collective_sparse=bool(collective.sparse_gradients),
            collective_codec_policy=str(collective.codec_policy),
            collective_reduction_tree=str(collective.reduction_tree),
            collective_topology_policy=str(collective.topology_policy),
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
        transport: str = "pair_host",
        collective_algorithm: str = "auto",
        collective_bucket_bytes: int = 32 * 1024 * 1024,
        collective_timeout_seconds: float = 120.0,
        collective_sparse: bool = True,
        collective_codec_policy: str = "auto_exact",
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
        if requested_transport not in {"auto", "pair_host"}:
            raise ValueError(
                "unsupported synchronous transport "
                f"{requested_transport!r}; expected 'auto' or "
                "'pair_host'"
            )
        self.transport = "pair_host"
        self.collective_algorithm = str(collective_algorithm)
        if self.collective_algorithm == "auto" and len(backends) == 2:
            self.collective_algorithm = PAIR_EXCHANGE_GPU_MERGE
        self.collective_bucket_bytes = int(collective_bucket_bytes)
        self.collective_timeout_seconds = float(collective_timeout_seconds)
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
        if self.world_size != 2:
            raise ValueError(
                "pair_host supports only a dual-device world"
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
                "pair_host requires pair_exchange_gpu_merge"
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
                "pair_host requires the native pair_exchange_gpu_merge "
                "capability; rebuild the native extension with the pair "
                "transfer-ticket ABI"
            )
        try:
            self.pair_exchange_probe = dict(
                self.leader.create_pair_exchange_gpu_merge_owner(
                    world_size=self.world_size,
                    bucket_bytes=self.collective_bucket_bytes,
                    sparse=self.collective_sparse,
                    timeout_seconds=self.collective_timeout_seconds,
                )
            )
            self._validate_pair_exchange_probe(self.pair_exchange_probe)
            self.collective_sparse = bool(
                self.pair_exchange_probe.get(
                    "sparse", self.collective_sparse
                )
            )
            for backend in self.backends:
                if backend is not self.leader:
                    backend.join_pair_session(self.leader)
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
                "selected pair_host/%s bucket=%d MiB sparse_bucket=%d MiB",
                self.collective_algorithm,
                self.collective_bucket_bytes // (1024 * 1024),
                sparse_bucket // (1024 * 1024),
            )
        else:
            logger.info(
                "selected pair_host/%s bucket=%d MiB",
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
                else "ftgspp-pair-bucket-calibration-v1"
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
                    timeout_seconds=self.collective_timeout_seconds,
                )
            )
            self._validate_pair_exchange_probe(self.pair_exchange_probe)
            self.collective_sparse = bool(
                self.pair_exchange_probe.get(
                    "sparse", self.collective_sparse
                )
            )
            for backend in self.backends:
                if backend is not self.leader:
                    backend.join_pair_session(self.leader)
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
        # Observe failures in completion order so a waiting lower rank cannot
        # hide a failed peer. Abort native waits before joining workers.
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException as error:
            self.leader.abort_pair_session(str(error))
            for future in futures:
                future.cancel()
            for future in futures:
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:
                        pass
            raise
        return [future.result() for future in futures]

    def _complete_generation_with_overflow_replay(
        self,
        views: tuple[int, ...],
        step: int,
        *,
        collect_loss: bool,
        backgrounds: tuple[tuple[float, float, float], ...] | None,
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
                if backgrounds is None:
                    backend.backward_view(
                        views[rank], int(step), collect_loss=collect_loss)
                else:
                    backend.backward_view(
                        views[rank], int(step), collect_loss=collect_loss,
                        background=backgrounds[rank])
                backward_seconds = time.perf_counter() - backward_begin
                # First gradient bucket D2H starts inside this rank worker as
                # soon as local compute retires. Python does not join the peer
                # backward before exchange.
                exchange = dict(exchange_op(rank, backend))
                return backward_seconds, exchange, []

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
        backgrounds: Sequence[Sequence[float]] | None = None,
    ) -> dict[str, object]:
        """Execute one globally synchronous optimizer update."""

        expected_views = self.world_size
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
        resolved_backgrounds = None
        if backgrounds is not None:
            if len(backgrounds) != expected_views:
                raise ValueError(
                    f"expected {expected_views} rank backgrounds, got {len(backgrounds)}"
                )
            resolved_backgrounds = tuple(
                tuple(float(channel) for channel in color)
                for color in backgrounds
            )
            if any(len(color) != 3 for color in resolved_backgrounds):
                raise ValueError("each rank background must contain RGB")
        return self._train_step_pair_exchange_gpu_merge(
            views, int(step), phase=phase, collect_loss=collect_loss,
            next_views=next_views, backgrounds=resolved_backgrounds,
        )

    def _train_step_pair_exchange_gpu_merge(
        self,
        views: tuple[int, ...],
        step: int,
        *,
        phase: str,
        collect_loss: bool,
        next_views: tuple[int, ...] | None,
        backgrounds: tuple[tuple[float, float, float], ...] | None,
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
            backgrounds=backgrounds,
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
            float(self.backends[rank].train_view_timestamp(views[rank]))
            for rank in range(self.world_size)
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
                        int(step), timestamps[0], timestamps[1],
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
                loss = _mean_scalars(post_optimizer_losses)
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
            "transport": "pair_host",
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

    def upload_model(self, model: NativeGaussians) -> None:
        """Upload one immutable export snapshot to every eval replica."""

        self._parallel(lambda _rank, backend: backend.upload_model(model))

    def begin_hard_separation(self) -> dict[str, int]:
        """Apply the deterministic gate split on every replica."""
        results = self._parallel(
            lambda _rank, backend: backend.begin_hard_separation()
        )
        if not results:
            return {"marginalized": 0, "kept": 0}
        if any(dict(value) != dict(results[0]) for value in results[1:]):
            raise RuntimeError("hard-separation split differs between replicas")
        self._recreate_collective_owner()
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
        self._recreate_collective_owner()
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
            # Decode futures retain View objects. Finish/cancel them
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
                        "failed to release pair collective owner"
                    )
            self._closed = True

    def __enter__(self) -> "SynchronousBackendGroup":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
