"""Spatial point sampling for Gaussian init and DA3 fusion.

Random per-view caps are avoided: multi-view clouds are voxel-fused first, then
an adaptive surface grid draws the per-frame budget from the fused dense cloud.
3-NN velocity still queries the dense fused points.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _row_inverse(keys: np.ndarray) -> np.ndarray:
    """Group fixed-width integer rows without NumPy's slower ``axis=0`` path."""

    contiguous = np.ascontiguousarray(keys)
    packed = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1])))
    _unique, inverse = np.unique(packed.reshape(-1), return_inverse=True)
    return inverse


def auto_voxel_size(
    xyz: np.ndarray,
    *,
    rng: np.random.Generator,
    sample_size: int = 32_768,
    diagnostics: dict[str, object] | None = None,
) -> float:
    """Infer a fusion voxel from median neighbor spacing.

    A bounded subsample always queries the complete reference tree so the
    estimate does not coarsen just because more candidates arrived.
    """

    points = np.asarray(xyz, dtype=np.float32)
    report = {
        "sample_size": 0,
        "neighbor_count": 0,
        "median_spacing": None,
        "voxel_size": 1e-3,
    }
    if len(points) < 2:
        if diagnostics is not None:
            diagnostics.update(report)
        return 1e-3
    query = points
    reference = points
    if len(points) > sample_size:
        chosen = rng.choice(len(points), size=int(sample_size), replace=False)
        query = points[chosen]
    report["sample_size"] = int(len(query))
    neighbors = min(4, len(reference))
    distances, _ = cKDTree(reference).query(query, k=neighbors, workers=-1)
    if distances.ndim == 1:
        nearest = distances
    else:
        nearest = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    positive = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    report["neighbor_count"] = int(len(positive))
    if len(positive) == 0:
        span = float(np.max(points.max(axis=0) - points.min(axis=0)))
        size = max(span / 256.0, 1e-6)
    else:
        median = float(np.median(positive))
        size = float(max(0.5 * median, 1e-6))
        report["median_spacing"] = median
    report["voxel_size"] = float(size)
    if diagnostics is not None:
        diagnostics.update(report)
    return float(size)


def voxel_fuse_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep one real candidate per occupied voxel.

    The candidate nearest the voxel center wins deterministically. Positions
    are never averaged across surfaces.
    """

    points = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(rgb)
    if len(points) == 0:
        empty = np.empty(0, dtype=np.int64)
        return points, colors, empty
    size = float(voxel_size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("voxel_size must be a positive finite length")
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / np.float32(size)).astype(np.int64)
    inverse = _row_inverse(keys)
    voxel_count = int(inverse.max()) + 1
    centers = (keys.astype(np.float32) + 0.5) * np.float32(size) + origin
    distance = np.linalg.norm(points - centers, axis=1)
    rank = -distance
    order = np.argsort(rank, kind="mergesort")
    winner = np.empty(voxel_count, dtype=np.int64)
    winner[inverse[order]] = order
    return points[winner], colors[winner], winner


def _voxel_inverse(points: np.ndarray, voxel: float) -> np.ndarray:
    """Return an occupied-voxel id for every point."""

    origin = points.min(axis=0)
    keys = np.floor((points - origin) / np.float32(voxel)).astype(np.int64)
    return _row_inverse(keys)


def _voxel_winners(
    points: np.ndarray,
    voxel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one center-nearest point and its voxel id for every occupied cell."""

    origin = points.min(axis=0)
    keys = np.floor((points - origin) / np.float32(voxel)).astype(np.int64)
    inverse = _row_inverse(keys)
    voxel_count = int(inverse.max()) + 1
    # Keep the winner on the original surface instead of averaging points
    # across a thin sheet.
    centers = (keys.astype(np.float32) + 0.5) * np.float32(voxel) + origin
    distance = np.sum((points - centers) ** 2, axis=1)
    # Assign farthest-to-nearest so the final write for each voxel is its
    # center-nearest source point.  Sorting only by distance and looking for
    # adjacent voxel ids is incorrect because equal voxels are interleaved.
    order = np.argsort(-distance, kind="mergesort")
    winners = np.empty(voxel_count, dtype=np.int64)
    winners[inverse[order]] = order
    return winners, inverse


def _adaptive_surface_voxel(
    points: np.ndarray,
    budget: int,
    *,
    diagnostics: dict[str, object] | None = None,
) -> tuple[float, int]:
    """Choose a voxel edge from occupied cells rather than bbox volume.

    A bounding-box volume assumes a filled 3-D distribution.  Depth clouds are
    usually sheets or thin shells, so that estimate can produce far fewer
    occupied cells than the requested budget and force random fill.  Searching
    the occupied-cell count makes the grid adapt to the intrinsic surface
    dimension while retaining a cheap, deterministic spatial hash.
    """

    span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-8)
    extent = max(float(np.max(span)), 1e-8)
    lower = max(extent / float(2**20), 1e-8)
    upper = max(extent * 2.0, lower * 2.0)
    upper_count = 1

    # Keep a lower bracket that still has enough occupied cells.  Repeated
    # coordinates can make the number of distinct cells smaller than budget;
    # in that case the caller will use the exact-budget fallback below.
    full_passes = 1
    lower_count = int(_voxel_inverse(points, lower).max()) + 1
    for _ in range(8):
        if lower_count >= budget:
            break
        lower *= 0.25
        lower_count = int(_voxel_inverse(points, lower).max()) + 1
        full_passes += 1

    best_voxel = lower
    best_count = lower_count

    # Estimate the useful part of the search interval on a deterministic,
    # bounded subset.  The old implementation rebuilt a full N-point spatial
    # hash 16 times; this keeps those cheap probes independent of cloud size.
    sample_limit = 8_192
    if len(points) > sample_limit:
        sample_ids = np.linspace(
            0, len(points) - 1, sample_limit, dtype=np.int64,
        )
        search_points = points[sample_ids]
    else:
        search_points = points
    sample_target = max(
        1,
        min(
            len(search_points),
            int(np.ceil(budget * len(search_points) / len(points))),
        ),
    )
    sample_lower, sample_upper = lower, upper
    candidate = lower
    for _ in range(12):
        middle = float(np.sqrt(sample_lower * sample_upper))
        count = int(_voxel_inverse(search_points, middle).max()) + 1
        if count >= sample_target:
            candidate = middle
            sample_lower = middle
        else:
            sample_upper = middle

    # Calibrate the approximate result with only a few full-cloud passes.
    # Keep a valid lower bracket throughout so discontinuous voxel counts still
    # return at least the requested number of representatives when possible.
    if candidate > lower:
        count = int(_voxel_inverse(points, candidate).max()) + 1
        full_passes += 1
        if count >= budget:
            if count < best_count or best_count < budget:
                best_voxel, best_count = candidate, count
            lower = candidate
            lower_count = count
        else:
            upper = candidate
            upper_count = count
    for _ in range(5):
        # Surface clouds are locally close to two-dimensional, so occupied
        # count scales approximately with edge^-2.  Use that relation to aim
        # directly at the budget, while the geometric bracket keeps the search
        # safe for volumes, curves, and discontinuous counts.
        if lower_count == len(points) and upper_count < budget:
            # The tiny lower bracket often puts every point in its own cell and
            # has no useful local slope. Approach from the measured upper
            # bracket and bias slightly toward enough cells.
            predicted = (
                upper * np.sqrt(max(upper_count / budget, 1e-12)) * 0.92
            )
        else:
            dimension = 2.0
            if (
                0 < upper_count < lower_count < len(points)
                and upper > lower
            ):
                dimension = float(np.clip(
                    np.log(lower_count / upper_count) / np.log(upper / lower),
                    0.5,
                    3.0,
                ))
            predicted = lower * max(
                lower_count / budget, 1.0,
            ) ** (1.0 / dimension)
        middle = float(np.clip(
            predicted,
            lower * 1.01,
            max(lower * 1.01, upper / 1.01),
        ))
        if not lower < middle < upper:
            middle = float(np.sqrt(lower * upper))
        count = int(_voxel_inverse(points, middle).max()) + 1
        full_passes += 1
        if count >= budget:
            if count < best_count or best_count < budget:
                best_voxel, best_count = middle, count
            lower = middle
            lower_count = count
        else:
            upper = middle
            upper_count = count
        if best_count >= budget and best_count <= int(np.ceil(1.05 * budget)):
            break
    if diagnostics is not None:
        diagnostics.update({
            "voxel_search_full_passes": int(full_passes),
            "voxel_search_sample_points": int(len(search_points)),
            "voxel_search_sample_target": int(sample_target),
        })
    return float(best_voxel), int(best_count)


def spatial_sample_indices(
    rng: np.random.Generator,
    xyz: np.ndarray,
    budget: int,
    *,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """Draw ``budget`` points with an adaptive surface-aware grid.

    The grid edge is selected from the number of *occupied* cells.  This is
    different from deriving it from the 3-D bounding-box volume: for a thin
    surface, the latter systematically underestimates the number of useful
    cells and fills most of the budget with random points.  One real point is
    retained per cell, and a deterministic random fill is used only when the
    input has fewer distinct occupied cells than the budget.
    """

    points = np.asarray(xyz)
    count = int(len(points))
    budget = int(budget)
    if count <= budget:
        if diagnostics is not None:
            diagnostics.update({
                "strategy": "all",
                "selected_points": count,
                "occupied_voxels": count,
                "voxel_size": None,
                "fallback_points": 0,
            })
        return np.arange(count, dtype=np.int64)

    search_diagnostics: dict[str, object] = {}
    voxel, estimated_cells = _adaptive_surface_voxel(
        points, budget, diagnostics=search_diagnostics,
    )
    winners, _inverse = _voxel_winners(points, voxel)
    voxel_count = int(len(winners))
    if voxel_count >= budget:
        chosen = rng.choice(voxel_count, size=budget, replace=False)
        selected = winners[chosen].astype(np.int64, copy=False)
        fallback_points = 0
    else:
        # This only occurs for duplicate-heavy clouds or a very discontinuous
        # voxel-count curve.  It preserves the exact budget without replacing
        # the spatial representatives already selected.
        used = np.zeros(count, dtype=bool)
        used[winners] = True
        leftover = np.flatnonzero(~used)
        missing = budget - voxel_count
        if missing > len(leftover):
            raise RuntimeError("spatial sampler cannot satisfy the requested budget")
        extra = rng.choice(leftover, size=missing, replace=False)
        selected = np.concatenate([winners, extra]).astype(np.int64, copy=False)
        fallback_points = int(missing)

    if diagnostics is not None:
        diagnostics.update({
            "strategy": "adaptive_surface_grid",
            "selected_points": int(len(selected)),
            "occupied_voxels": voxel_count,
            "estimated_occupied_voxels": int(estimated_cells),
            "voxel_size": float(voxel),
            "occupied_fraction": float(voxel_count / max(budget, 1)),
            "fallback_points": fallback_points,
            **search_diagnostics,
        })
    return selected


def select_point_indices(
    rng: np.random.Generator,
    population_size: int,
    budget: int,
    xyz: np.ndarray | None = None,
    *,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """Select source points without replacement, spatially when coordinates exist."""

    population_size = int(population_size)
    budget = int(budget)
    if population_size <= 0:
        raise ValueError("point cloud must contain at least one point")
    if budget <= 0:
        raise ValueError("point budget must be greater than zero")
    if population_size <= budget:
        selected = np.arange(population_size, dtype=np.int64)
        if diagnostics is not None:
            diagnostics.update({
                "strategy": "all",
                "selected_points": population_size,
            })
        return selected
    if xyz is None:
        selected = rng.choice(population_size, size=budget, replace=False)
        if diagnostics is not None:
            diagnostics.update({"strategy": "random", "selected_points": budget})
        return selected
    if len(xyz) != population_size:
        raise ValueError(
            f"xyz length {len(xyz)} does not match population_size {population_size}"
        )
    selected = spatial_sample_indices(rng, xyz, budget, diagnostics=diagnostics)
    if diagnostics is not None:
        diagnostics.setdefault("strategy", "adaptive_surface_grid")
        diagnostics.setdefault("selected_points", budget)
    return selected
