"""Test-only mathematical references for hierarchical host collectives.

Nothing in this module is registered as a training backend.  The routines use
fixed rank order and explicit FP32 operations so native pair, central, sharded,
sparse and K-view implementations can be compared against one contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ftgspp.native_runtime.distributed_contract import DistributedContractError


COLLECTIVE_ABI = "ftgspp-hierarchical-host-v1"


def _finite_float32(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise DistributedContractError(f"{label} must use float32")
    if not array.flags.c_contiguous:
        raise DistributedContractError(f"{label} must be C contiguous")
    if not np.all(np.isfinite(array)):
        raise DistributedContractError(f"{label} contains non-finite values")
    return array


def fixed_rank_dense_mean(per_rank: Sequence[np.ndarray]) -> np.ndarray:
    """Reduce equal dense buffers in rank order without a float64 detour."""
    if not per_rank:
        raise DistributedContractError("at least one rank is required")
    arrays = [
        _finite_float32(value, label=f"rank {rank} dense gradient")
        for rank, value in enumerate(per_rank)
    ]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays[1:]):
        raise DistributedContractError("dense gradient shapes differ")
    output = np.zeros(shape, dtype=np.float32)
    for value in arrays:
        np.add(output, value, out=output)
    np.multiply(output, np.float32(1.0 / len(arrays)), out=output)
    if not np.all(np.isfinite(output)):
        raise DistributedContractError("dense reduction overflowed float32")
    return np.ascontiguousarray(output)


def finalize_local_gradient(
    microstep_gradients: Sequence[np.ndarray],
) -> np.ndarray:
    """Return ``sum(local microsteps) / K`` at one unchanged parameter point."""
    if not microstep_gradients:
        raise DistributedContractError("K must be positive")
    return fixed_rank_dense_mean(microstep_gradients)


@dataclass(frozen=True)
class SortedSparseGradient:
    """Strictly ID-sorted exact sparse rows for one rank and bucket."""

    ids: np.ndarray
    values: np.ndarray

    def validate(self, *, row_count: int | None = None) -> None:
        ids = np.asarray(self.ids)
        values = _finite_float32(self.values, label="sparse values")
        if ids.dtype != np.uint32 or ids.ndim != 1 or not ids.flags.c_contiguous:
            raise DistributedContractError(
                "sparse ids must be contiguous uint32 shape (M,)"
            )
        if values.ndim != 2 or values.shape[0] != ids.size:
            raise DistributedContractError(
                "sparse values must have shape (M, record_width)"
            )
        if ids.size > 1 and np.any(ids[1:] <= ids[:-1]):
            raise DistributedContractError(
                "sparse Gaussian/entry ids must be strictly increasing"
            )
        if row_count is not None and ids.size and int(ids[-1]) >= int(row_count):
            raise DistributedContractError("sparse id is outside the bucket")


def sorted_sparse_from_dense(
    dense: np.ndarray,
    touched: np.ndarray,
) -> SortedSparseGradient:
    values = _finite_float32(dense, label="dense gradient")
    mask = np.asarray(touched)
    if values.ndim != 2:
        raise DistributedContractError("dense gradient must have shape (N, W)")
    if mask.dtype != np.bool_ or mask.shape != (values.shape[0],):
        raise DistributedContractError("touched must be bool shape (N,)")
    ids = np.ascontiguousarray(np.flatnonzero(mask), dtype=np.uint32)
    records = SortedSparseGradient(
        ids=ids,
        values=np.ascontiguousarray(values[mask], dtype=np.float32),
    )
    records.validate(row_count=values.shape[0])
    return records


def exact_sorted_sparse_mean(
    per_rank: Sequence[SortedSparseGradient],
    *,
    row_count: int,
    record_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """K-way merge exact sparse rows, treating absent ranks as zeros.

    The divisor is always the full world size, never the number of ranks that
    touched an ID.  Returns the sorted global union and its reduced values.
    """
    if not per_rank:
        raise DistributedContractError("at least one sparse rank is required")
    if row_count < 0 or record_width <= 0:
        raise DistributedContractError("invalid sparse bucket dimensions")
    for records in per_rank:
        records.validate(row_count=row_count)
        if records.values.shape[1] != record_width:
            raise DistributedContractError("sparse record widths differ")

    positions = [0] * len(per_rank)
    union_ids: list[int] = []
    union_values: list[np.ndarray] = []
    scale = np.float32(1.0 / len(per_rank))
    while True:
        current = [
            int(records.ids[position])
            for records, position in zip(per_rank, positions, strict=True)
            if position < records.ids.size
        ]
        if not current:
            break
        next_id = min(current)
        reduced = np.zeros(record_width, dtype=np.float32)
        for rank, records in enumerate(per_rank):
            position = positions[rank]
            if position < records.ids.size and int(records.ids[position]) == next_id:
                np.add(reduced, records.values[position], out=reduced)
                positions[rank] += 1
        np.multiply(reduced, scale, out=reduced)
        if not np.all(np.isfinite(reduced)):
            raise DistributedContractError("sparse reduction overflowed float32")
        union_ids.append(next_id)
        union_values.append(reduced)

    ids = np.ascontiguousarray(union_ids, dtype=np.uint32)
    if union_values:
        values = np.ascontiguousarray(np.stack(union_values), dtype=np.float32)
    else:
        values = np.empty((0, record_width), dtype=np.float32)
    return ids, values


def scatter_sparse_rows(
    ids: np.ndarray,
    values: np.ndarray,
    *,
    row_count: int,
) -> np.ndarray:
    records = SortedSparseGradient(
        np.ascontiguousarray(ids),
        np.ascontiguousarray(values),
    )
    records.validate(row_count=row_count)
    output = np.zeros((row_count, records.values.shape[1]), dtype=np.float32)
    output[records.ids] = records.values
    return output


def sharded_dense_mean_reference(
    per_rank: Sequence[np.ndarray],
    *,
    shard_count: int,
) -> np.ndarray:
    """Reference reduce-scatter/allgather with fixed rank order per shard."""
    if shard_count <= 0:
        raise DistributedContractError("shard_count must be positive")
    if not per_rank:
        raise DistributedContractError("at least one rank is required")
    arrays = [
        _finite_float32(value, label=f"rank {rank} sharded gradient")
        for rank, value in enumerate(per_rank)
    ]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays[1:]):
        raise DistributedContractError("sharded gradient shapes differ")
    flattened = [value.reshape(-1) for value in arrays]
    output = np.empty(flattened[0].size, dtype=np.float32)
    for indices in np.array_split(np.arange(output.size), shard_count):
        if indices.size == 0:
            continue
        output[indices] = fixed_rank_dense_mean(
            [value[indices] for value in flattened]
        )
    return np.ascontiguousarray(output.reshape(shape))


def choose_exact_bucket_codec(
    *,
    dense_seconds: float,
    sparse_seconds: float,
    sparse_exact: bool,
    minimum_sparse_gain: float = 0.05,
) -> str:
    """Choose sparse only when exact and measurably faster than dense."""
    values = (dense_seconds, sparse_seconds, minimum_sparse_gain)
    if not all(np.isfinite(value) for value in values):
        raise DistributedContractError("bucket calibration values must be finite")
    if dense_seconds <= 0.0 or sparse_seconds <= 0.0:
        raise DistributedContractError("bucket calibration times must be positive")
    if minimum_sparse_gain < 0.0 or minimum_sparse_gain >= 1.0:
        raise DistributedContractError("minimum sparse gain must be in [0, 1)")
    if sparse_exact and sparse_seconds <= dense_seconds * (1.0 - minimum_sparse_gain):
        return "exact_sorted_sparse"
    return "dense"
