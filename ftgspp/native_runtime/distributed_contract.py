"""Production state invariants for synchronous replicated FTGSPP training.

Transport and numerical reference functions live in ``distributed.py`` and
``ftgspp.reference.distributed`` respectively.  This module owns only replica,
relocation-history and checkpoint-state validation used by the runtime.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


BASE_OPTIMIZER_KEYS = frozenset({
    "g_xyz_ws",
    "g_sh_coeffs_1",
    "g_sh_coeffs_2",
    "g_rotations",
    "g_scales_opacs",
    "g_times_durations",
    "g_velocities_gates",
    "relocation_stats",
})

COLOR_STATE_KEYS = frozenset({"color_params", "color_m", "color_v"})


class DistributedContractError(ValueError):
    """Raised when rank-local values cannot represent one synchronous step."""


class ReplicaDivergenceError(DistributedContractError):
    """Raised when replicated model/optimizer state is no longer identical."""


def _as_finite_float_array(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise DistributedContractError(
            f"{label} must be a real floating-point array"
        )
    if not np.isfinite(array).all():
        raise DistributedContractError(f"{label} contains a non-finite value")
    return array

# Native replay (`relocate_dynamic`) and the MCMC binomial table both reject
# more than 50 copies of one donor.  The GPU sampler does not cap itself, so
# multi-rank plans with many dead slots must drop the extra destinations
# before broadcast.  Matches `relocation_max_ratio = 51` (allowed copies 0..50).
RELOCATION_MAX_DONOR_COPIES = 50


def cap_relocation_donor_copies(
    mapping: np.ndarray,
    counts: np.ndarray,
    *,
    max_copies: int = RELOCATION_MAX_DONOR_COPIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most ``max_copies`` destinations per donor, lowest index first.

    The native planner assigns every dead Gaussian independently, so a popular
    donor can exceed the MCMC copy limit when many slots die in one step.
    Dropping extras is deterministic and identical on every rank that starts
    from the same mapping.
    """

    mapping = np.ascontiguousarray(mapping, dtype=np.int32).copy()
    counts = np.ascontiguousarray(counts, dtype=np.int32)
    limit = int(max_copies)
    if limit < 0:
        raise DistributedContractError("max_copies must be non-negative")
    if mapping.ndim != 1 or counts.shape != mapping.shape:
        raise DistributedContractError(
            f"relocation mapping/counts must be 1-D of equal length, "
            f"got {mapping.shape} vs {counts.shape}"
        )
    if mapping.size == 0 or not np.any(counts > limit):
        return mapping, counts.copy()
    valid = mapping >= 0
    dests = np.flatnonzero(valid)
    donors = mapping[dests]
    order = np.argsort(donors, kind="stable")
    dests = dests[order]
    donors = donors[order]
    starts = np.empty(donors.size, dtype=bool)
    starts[0] = True
    if donors.size > 1:
        starts[1:] = donors[1:] != donors[:-1]
    group_id = np.cumsum(starts) - 1
    start_pos = np.flatnonzero(starts)
    rank_in_group = np.arange(donors.size) - start_pos[group_id]
    mapping[dests[rank_in_group >= limit]] = -1
    kept = mapping[mapping >= 0]
    capped = np.bincount(kept, minlength=mapping.size).astype(np.int32)
    return mapping, capped


def sum_relocation_statistics(per_rank: Sequence[np.ndarray]) -> np.ndarray:
    """Sum rank-local ``(gradient-magnitude sum, observation count)`` pairs.

    Vector gradients must be converted to magnitudes on the rank that observed
    the view.  Taking a norm after all-reducing vectors is incorrect because
    gradients from different projections can cancel.  Counts are sums, never
    means: averaging a single observation over multiple ranks would interact
    incorrectly with the native ``max(count, 1)`` denominator.
    """

    if not per_rank:
        raise DistributedContractError("at least one relocation-stat rank is required")
    arrays = [
        _as_finite_float_array(value, label=f"rank {rank} relocation statistics")
        for rank, value in enumerate(per_rank)
    ]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] != 2:
        raise DistributedContractError(
            f"relocation statistics must have shape (N, 2), got {shape}"
        )
    if any(value.shape != shape for value in arrays[1:]):
        raise DistributedContractError(
            f"relocation statistic shapes differ: {[value.shape for value in arrays]}"
        )
    if any(np.any(value < 0.0) for value in arrays):
        raise DistributedContractError("relocation magnitude sums and counts must be non-negative")
    accumulated = np.zeros(shape, dtype=np.float64)
    for value in arrays:
        accumulated += value.astype(np.float64, copy=False)
    return np.ascontiguousarray(accumulated, dtype=np.float32)

def assert_replicas_consistent(
    per_rank: Sequence[Mapping[str, np.ndarray]],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Verify identical keys/layout and numeric values across all replicas."""

    if not per_rank:
        raise DistributedContractError("at least one replica is required")
    reference = per_rank[0]
    keys = set(reference)
    for rank, state in enumerate(per_rank[1:], start=1):
        if set(state) != keys:
            raise ReplicaDivergenceError(f"rank {rank} state keys differ from rank 0")
        for key in sorted(keys):
            expected = np.asarray(reference[key])
            actual = np.asarray(state[key])
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise ReplicaDivergenceError(
                    f"rank {rank} state {key} layout differs: "
                    f"{actual.dtype}{actual.shape} != {expected.dtype}{expected.shape}"
                )
            if expected.dtype.kind in "fc":
                equal = np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
            else:
                equal = np.array_equal(actual, expected)
            if not equal:
                if expected.dtype.kind in "fc":
                    difference = float(np.max(np.abs(
                        actual.astype(np.float64) - expected.astype(np.float64)
                    )))
                    detail = f"; max_abs={difference:.9g}"
                else:
                    detail = ""
                raise ReplicaDivergenceError(
                    f"rank {rank} state {key} differs from rank 0{detail}"
                )




def shard_resumed_relocation_statistics(
    global_statistics: np.ndarray,
    world_size: int,
    *,
    owner_rank: int = 0,
) -> tuple[np.ndarray, ...]:
    """Assign checkpointed relocation history to exactly one resumed rank.

    Normal parameters and Adam moments are broadcast to every rank.  Historical
    relocation statistics are additive state and therefore have a single
    owner; other ranks start with zero statistics.  The next sum reduction then
    equals ``checkpoint_history + all_new_local_observations`` instead of
    ``world_size * checkpoint_history + ...``.
    """

    size = int(world_size)
    owner = int(owner_rank)
    if size <= 0:
        raise DistributedContractError("world_size must be positive")
    if owner < 0 or owner >= size:
        raise DistributedContractError("owner_rank must be in [0, world_size)")
    stats = _as_finite_float_array(global_statistics, label="global relocation statistics")
    if stats.ndim != 2 or stats.shape[1] != 2 or np.any(stats < 0.0):
        raise DistributedContractError(
            "global relocation statistics must be non-negative shape (N, 2)"
        )
    return tuple(
        np.ascontiguousarray(stats, dtype=np.float32).copy()
        if rank == owner
        else np.zeros(stats.shape, dtype=np.float32)
        for rank in range(size)
    )


def validate_checkpoint_optimizer_state(
    optimizer: Mapping[str, np.ndarray],
    *,
    num_gaussians: int,
    color_num_cameras: int,
) -> None:
    """Validate that a checkpoint can initialize every replica atomically.

    Gaussian and color parameters/moments are indivisible state groups.
    Omitting one member would let a resumed rank follow a different trajectory
    even when Gaussian arrays are identical.
    """

    keys = set(optimizer)
    missing_base = sorted(BASE_OPTIMIZER_KEYS - keys)
    if missing_base:
        raise DistributedContractError(f"checkpoint is missing base optimizer state: {missing_base}")
    n = int(num_gaussians)
    if n < 0:
        raise DistributedContractError("num_gaussians must be non-negative")

    expected_sizes = {
        "g_xyz_ws": 6 * n,
        "g_rotations": 8 * n,
        "g_scales_opacs": 8 * n,
        "g_times_durations": 4 * n,
        "g_velocities_gates": 8 * n,
        "relocation_stats": 2 * n,
    }
    for key, size in expected_sizes.items():
        value = _as_finite_float_array(optimizer[key], label=f"checkpoint {key}")
        if value.size != size:
            raise DistributedContractError(
                f"checkpoint {key} has {value.size} floats, expected {size}"
            )
        if key == "relocation_stats" and np.any(value < 0.0):
            raise DistributedContractError(
                "checkpoint relocation_stats must be non-negative"
            )
    # Native SH moments keep the exact reordered float4 layout and pad the
    # Gaussian dimension to the fixed 32-lane subgroup. Accepting merely any
    # 48-float multiple would postpone a malformed checkpoint failure until
    # native upload, and could obscure which ABI dimension was wrong.
    padded_gaussians = ((n + 31) // 32) * 32
    expected_sh_size = 48 * padded_gaussians
    for key in ("g_sh_coeffs_1", "g_sh_coeffs_2"):
        value = _as_finite_float_array(optimizer[key], label=f"checkpoint {key}")
        if value.size != expected_sh_size:
            raise DistributedContractError(
                f"checkpoint {key} has {value.size} floats, expected "
                f"{expected_sh_size} for 48*ceil({n}/32)*32 native SH padding"
            )

    field_keys = sorted(key for key in keys if str(key).startswith("field_"))
    if field_keys:
        raise DistributedContractError(
            "production checkpoints are explicit-only; convert the legacy "
            f"field state offline before resume: {field_keys}"
        )

    camera_count = int(color_num_cameras)
    if camera_count < 0:
        raise DistributedContractError("color_num_cameras must be non-negative")
    present_color = keys & COLOR_STATE_KEYS
    if camera_count and present_color != COLOR_STATE_KEYS:
        missing = sorted(COLOR_STATE_KEYS - present_color)
        raise DistributedContractError(f"checkpoint is missing color state: {missing}")
    if not camera_count and present_color:
        raise DistributedContractError("checkpoint unexpectedly contains color-correction state")
    if camera_count:
        expected = 12 * camera_count
        for key in COLOR_STATE_KEYS:
            value = _as_finite_float_array(optimizer[key], label=f"checkpoint {key}")
            if value.size != expected:
                raise DistributedContractError(
                    f"checkpoint {key} has {value.size} floats, expected {expected}"
                )
