"""CPU/NumPy references for validating synchronous Vulkan math."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np

from ftgspp.native_runtime.distributed_contract import (
    DistributedContractError,
    _as_finite_float_array,
    assert_replicas_consistent,
    sum_relocation_statistics,
)


def packed_fp32_mean(
    per_rank: Sequence[np.ndarray],
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Test-only fixed-rank-order mean of packed FP32 gradient arenas.

    This mirrors the historical native packed reducer exactly: every element is
    accumulated in rank order with FP32 rounding after each addition, then
    multiplied by an FP32 reciprocal.  Validation and reduction complete in a
    temporary arena before ``output`` is touched, so exact or partial aliasing
    with any input is safe and an invalid/non-finite reduction is failure
    atomic.  This is a numerical reference, not a runtime transport backend.
    """

    if not per_rank:
        raise DistributedContractError("packed FP32 mean requires at least one rank")
    arrays: list[np.ndarray] = []
    count: int | None = None
    for rank, value in enumerate(per_rank):
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise DistributedContractError(
                f"rank {rank} packed gradient must have dtype float32"
            )
        if array.ndim != 1 or not array.flags.c_contiguous:
            raise DistributedContractError(
                f"rank {rank} packed gradient must be contiguous and one-dimensional"
            )
        if count is None:
            count = int(array.size)
            if count == 0:
                raise DistributedContractError(
                    "packed FP32 mean does not accept empty arenas"
                )
        elif array.size != count:
            raise DistributedContractError("packed rank gradient sizes differ")
        if not np.isfinite(array).all():
            raise DistributedContractError(
                f"rank {rank} packed gradient contains a non-finite value"
            )
        arrays.append(array)

    destination: np.ndarray | None = None
    if output is not None:
        destination = np.asarray(output)
        if destination.dtype != np.float32:
            raise DistributedContractError("packed mean output must have dtype float32")
        if (
            destination.ndim != 1
            or destination.size != count
            or not destination.flags.c_contiguous
        ):
            raise DistributedContractError(
                "packed mean output must be a matching contiguous 1D arena"
            )
        if not destination.flags.writeable:
            raise DistributedContractError("packed mean output must be writable")

    accumulated = np.zeros(count, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for array in arrays:
            np.add(accumulated, array, out=accumulated)
        np.multiply(
            accumulated,
            np.float32(1.0) / np.float32(len(arrays)),
            out=accumulated,
        )
    if not np.isfinite(accumulated).all():
        raise DistributedContractError(
            "packed FP32 reduction encountered non-finite input or overflow"
        )
    if destination is not None:
        np.copyto(destination, accumulated)
        return destination
    return accumulated


def mean_raw_gradients(
    per_rank: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Return the global-batch mean of rank-local raw data gradients.

    All ranks must expose the same keyed buffers with identical shapes.  The
    result must be produced *before* Adam updates ``m``, ``v`` or parameters.
    The function intentionally rejects missing/non-finite data instead of
    silently allowing one rank to take a different optimizer step.
    """

    if not per_rank:
        raise DistributedContractError("at least one rank is required")
    expected_keys = set(per_rank[0])
    if not expected_keys:
        raise DistributedContractError("raw gradient state must not be empty")
    result: dict[str, np.ndarray] = {}
    for rank, values in enumerate(per_rank):
        if set(values) != expected_keys:
            missing = sorted(expected_keys - set(values))
            extra = sorted(set(values) - expected_keys)
            raise DistributedContractError(
                f"rank {rank} raw gradient keys differ; missing={missing}, extra={extra}"
            )
    for key in sorted(expected_keys):
        arrays = [
            _as_finite_float_array(values[key], label=f"rank {rank} gradient {key}")
            for rank, values in enumerate(per_rank)
        ]
        shape = arrays[0].shape
        if any(value.shape != shape for value in arrays[1:]):
            shapes = [value.shape for value in arrays]
            raise DistributedContractError(f"gradient {key} shapes differ: {shapes}")
        accumulated = np.zeros(shape, dtype=np.float64)
        for value in arrays:
            accumulated += value.astype(np.float64, copy=False)
        result[key] = np.ascontiguousarray(accumulated / len(arrays), dtype=np.float32)
    return result


def add_shared_gradients(
    data_mean: Mapping[str, np.ndarray],
    shared: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Add a once-per-global-step regularizer/distillation gradient.

    ``shared`` must not be divided by ``world_size``.  This is appropriate for
    gate/scale/opacity regularizers shared by all ranks. A regularizer already
    present, identically, in
    every rank-local gradient may instead be included before
    :func:`mean_raw_gradients`; averaging identical copies also leaves it once.
    """

    result = {
        key: np.ascontiguousarray(
            _as_finite_float_array(value, label=f"data gradient {key}"),
            dtype=np.float32,
        ).copy()
        for key, value in data_mean.items()
    }
    unknown = sorted(set(shared) - set(result))
    if unknown:
        raise DistributedContractError(
            f"shared gradients have no corresponding data buffer: {unknown}"
        )
    for key, value in shared.items():
        addition = _as_finite_float_array(value, label=f"shared gradient {key}")
        if addition.shape != result[key].shape:
            raise DistributedContractError(
                f"shared gradient {key} has shape {addition.shape}, "
                f"expected {result[key].shape}"
            )
        result[key] = np.ascontiguousarray(
            result[key].astype(np.float64) + addition.astype(np.float64),
            dtype=np.float32,
        )
    return result


def reduce_color_gradients(
    per_rank_data_gradients: Sequence[np.ndarray],
    params: np.ndarray,
    regularizer_weight: float,
) -> np.ndarray:
    """Mean dense per-camera image gradients and add L2 gradient once.

    Each rank supplies the full ``(num_cameras, 12)`` layout, with zeros for
    cameras absent from that rank's view.  This preserves uniform per-view
    weighting.  FTGSPP-main defines the color penalty as
    ``regularizer_weight * sum(params**2)``, so the L2 gradient is
    ``2 * regularizer_weight * params`` exactly once, matching
    ``NativeColorCorrectors.adam_step``.
    """

    reduced = mean_raw_gradients([
        {"color": value} for value in per_rank_data_gradients
    ])["color"]
    parameters = _as_finite_float_array(params, label="color parameters")
    if parameters.shape != reduced.shape:
        raise DistributedContractError(
            f"color parameters have shape {parameters.shape}, expected {reduced.shape}"
        )
    weight = float(regularizer_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise DistributedContractError("color regularizer weight must be finite and non-negative")
    return np.ascontiguousarray(
        reduced.astype(np.float64) + 2.0 * weight * parameters.astype(np.float64),
        dtype=np.float32,
    )

def relocation_scores(statistics: np.ndarray) -> np.ndarray:
    """Return the native per-Gaussian score ``sum_magnitude/max(count, 1)``."""

    stats = _as_finite_float_array(statistics, label="relocation statistics")
    if stats.ndim != 2 or stats.shape[1] != 2:
        raise DistributedContractError("relocation statistics must have shape (N, 2)")
    if np.any(stats < 0.0):
        raise DistributedContractError("relocation magnitude sums and counts must be non-negative")
    return np.ascontiguousarray(
        stats[:, 0] / np.maximum(stats[:, 1], np.float32(1.0)),
        dtype=np.float32,
    )


def _pcg_hash(value: np.ndarray) -> np.ndarray:
    """Vectorized uint32 PCG hash matching ``ftgspp_relocation_plan.slang``."""

    mask = np.uint64(0xFFFFFFFF)
    source = np.asarray(value, dtype=np.uint32).astype(np.uint64)
    state = (source * np.uint64(747796405) + np.uint64(2891336453)) & mask
    shift = (state >> np.uint64(28)) + np.uint64(4)
    word = (((state >> shift) ^ state) * np.uint64(277803737)) & mask
    return np.asarray(((word >> np.uint64(22)) ^ word) & mask, dtype=np.uint32)


@dataclass(frozen=True)
class RelocationPlan:
    """One authoritative topology-changing operation for all replicas."""

    mapping: np.ndarray
    counts: np.ndarray
    raw_scores: np.ndarray
    donor_cdf: np.ndarray
    relocated: int
    alive: int
    all_dead: bool

    def validate(self) -> None:
        n = int(self.mapping.size)
        if self.mapping.shape != (n,) or self.mapping.dtype != np.int32:
            raise DistributedContractError("relocation mapping must be int32 shape (N,)")
        if self.counts.shape != (n,) or self.counts.dtype != np.int32:
            raise DistributedContractError("relocation counts must be int32 shape (N,)")
        if self.raw_scores.shape != (n, 4):
            raise DistributedContractError("relocation raw scores must have shape (N, 4)")
        if self.donor_cdf.shape != (n,):
            raise DistributedContractError("relocation donor CDF must have shape (N,)")
        donors = self.mapping[self.mapping >= 0]
        if donors.size and (int(donors.min()) < 0 or int(donors.max()) >= n):
            raise DistributedContractError("relocation mapping has an out-of-range donor")
        expected = np.bincount(donors, minlength=n).astype(np.int32)
        if not np.array_equal(self.counts, expected):
            raise DistributedContractError("relocation counts do not match mapping")
        if int(donors.size) != int(self.relocated):
            raise DistributedContractError("relocated count does not match mapping")


def deterministic_relocation_plan(
    opacity_logits: np.ndarray,
    gate_raw: np.ndarray,
    statistics: np.ndarray,
    *,
    opacity_threshold: float,
    include_gate: bool,
    seed: int,
    step: int,
) -> RelocationPlan:
    """Host reference for the three-phase native Vulkan relocation planner.

    This mirrors the opacity/gradient/gate score normalization, whole-vector
    alive-uniform fallback, and PCG donor sampling in
    ``ftgspp_relocation_plan.slang``.  Production distributed
    training should still execute the planner on rank 0 and broadcast the
    resulting ``mapping`` and ``counts``; independently recomputing a CDF on
    every GPU is not deterministic because float atomic-add order may differ.
    """

    opacity_values = _as_finite_float_array(opacity_logits, label="opacity logits").reshape(-1)
    gate_values = _as_finite_float_array(gate_raw, label="gate logits").reshape(-1)
    stats = _as_finite_float_array(statistics, label="relocation statistics")
    n = int(opacity_values.size)
    if gate_values.size != n or stats.shape != (n, 2):
        raise DistributedContractError(
            "opacity, gate and relocation statistics must describe the same N Gaussians"
        )
    threshold = float(opacity_threshold)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold >= 1.0:
        raise DistributedContractError("opacity threshold must be finite in [0, 1)")

    # Stable host evaluation of the native sigmoid expressions.
    opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_values.astype(np.float64), -80.0, 80.0)))
    gate = 1.0 / (1.0 + np.exp(-np.clip(20.0 * gate_values.astype(np.float64), -80.0, 80.0)))
    gradient = relocation_scores(stats).astype(np.float64)
    alive_mask = opacity > threshold
    raw = np.stack((opacity, gradient, 1.0 - gate, alive_mask.astype(np.float64)), axis=1)

    alive = int(np.count_nonzero(alive_mask))
    mapping = np.full(n, -1, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)
    cdf = np.zeros(n, dtype=np.float64)
    if n and alive:
        totals = raw[:, :3].sum(axis=0, dtype=np.float64)
        score = np.zeros(n, dtype=np.float64)
        enabled_totals = totals if include_gate else totals[:2]
        uniform_fallback = bool(np.any(~np.isfinite(enabled_totals))) or bool(
            np.any(enabled_totals <= 0.0)
        )
        if uniform_fallback:
            # FTGSPP replaces the complete non-finite probability vector with
            # ones before selecting alive donors.  It does not repair only the
            # degenerate component while retaining other score biases.
            score[alive_mask] = 1.0
        else:
            score[alive_mask] = (
                raw[alive_mask, 0] / totals[0]
                + raw[alive_mask, 1] / totals[1]
            )
            if include_gate:
                score[alive_mask] += raw[alive_mask, 2] / totals[2]
        cdf = np.cumsum(score, dtype=np.float64)
        if cdf[-1] > 0.0:
            cdf /= cdf[-1]

        dead_indices = np.flatnonzero(~alive_mask)
        if dead_indices.size:
            effective_seed = np.uint32(int(seed) & 0xFFFFFFFF) ^ np.uint32(int(step) & 0xFFFFFFFF)
            random_bits = _pcg_hash(effective_seed ^ dead_indices.astype(np.uint32))
            targets = (random_bits.astype(np.float64) + 0.5) / 4294967296.0
            donors = np.searchsorted(cdf, targets, side="left")
            donors = np.minimum(donors, n - 1).astype(np.int32)
            mapping[dead_indices] = donors
            counts = np.bincount(donors, minlength=n).astype(np.int32)

    plan = RelocationPlan(
        mapping=mapping,
        counts=counts,
        raw_scores=np.ascontiguousarray(raw, dtype=np.float32),
        donor_cdf=np.ascontiguousarray(cdf, dtype=np.float32),
        relocated=int(np.count_nonzero(mapping >= 0)),
        alive=alive,
        all_dead=alive == 0,
    )
    plan.validate()
    return plan


def state_digest(state: Mapping[str, np.ndarray]) -> str:
    """Return a canonical digest over keys, dtype, shape and C-order bytes."""

    digest = sha256()
    for key in sorted(state):
        array = np.asarray(state[key])
        if array.dtype.hasobject:
            raise DistributedContractError(f"state {key} has unsupported object dtype")
        contiguous = np.ascontiguousarray(array)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(
            (contiguous.ndim, *contiguous.shape), dtype=np.int64
        ).tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()

def consolidate_optimizer_state_for_checkpoint(
    per_rank: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Build one resumable optimizer state from synchronous replicas.

    Parameters and Adam moments must be replicated exactly.  The sole
    exception is ``relocation_stats``: between relocation events each rank owns
    observations from its local views, so checkpointing must sum those pairs.
    Broadcasting rank 0's local statistics would lose observations; summing
    replicas that all hold an earlier global history would count that history
    ``world_size`` times.  Use :func:`shard_resumed_relocation_statistics` when
    restoring the returned checkpoint.
    """

    if not per_rank:
        raise DistributedContractError("at least one optimizer replica is required")
    if any("relocation_stats" not in state for state in per_rank):
        raise DistributedContractError("every optimizer replica needs relocation_stats")
    replicated = [
        {key: value for key, value in state.items() if key != "relocation_stats"}
        for state in per_rank
    ]
    assert_replicas_consistent(replicated)
    result = {
        key: np.array(value, copy=True)
        for key, value in per_rank[0].items()
        if key != "relocation_stats"
    }
    result["relocation_stats"] = sum_relocation_statistics([
        state["relocation_stats"] for state in per_rank
    ])
    return result
