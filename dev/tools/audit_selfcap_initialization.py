"""Read-only audit of cached SelfCap initialization; JSON is printed to stdout.

Replays the selected initializer's sampling and time-jitter RNG on recorded
clouds, without training or rewriting caches. Ratios diagnose geometry, not
correspondence correctness. v1/v2 preserve the historical sampler and scale
formula; v3/v4 use adaptive surface sampling and non-self spacing. v4 adds the
bounded-sample voxel search used by current initialization.
Run with the checkout's src directory on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from ftgspp.io.ply import read_point_ply
from ftgspp.training.init_plan import initialization_seed
from ftgspp.training.initialize import (
    _spacing,
    _velocity,
    keyframe_neighbor_indices,
    sample_keyframe_points,
)


def historical_spacing(points: np.ndarray) -> np.ndarray:
    """Reproduce the pre-v3 scale estimate for old manifests."""

    if len(points) == 1:
        return np.ones(1, dtype=np.float32) * 1e-3
    k = min(3, len(points))
    distances, _ = cKDTree(points).query(points, k=k)
    if distances.ndim == 1:
        distances = distances[:, None]
    return np.sqrt(np.maximum(np.mean(distances**2, axis=1), 1e-7)).astype(np.float32)


def historical_select(
    rng: np.random.Generator, xyz: np.ndarray, budget: int,
) -> np.ndarray:
    """Reproduce the pre-v3 bbox-volume grid for old manifests."""

    count = int(len(xyz))
    if count <= int(budget):
        return np.arange(count, dtype=np.int64)
    span = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1e-8)
    volume = float(np.prod(span))
    voxel = (volume / float(max(budget, 1))) ** (1.0 / 3.0)
    voxel = max(
        voxel,
        float(np.min(span)) / float(max(budget, 1)) ** (1.0 / 3.0),
        1e-8,
    )
    origin = xyz.min(axis=0)
    keys = np.floor((xyz - origin) / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1
    order = rng.permutation(count)
    winner = np.empty(voxel_count, dtype=np.int64)
    winner[inverse[order]] = order
    if voxel_count >= budget:
        chosen = rng.choice(voxel_count, size=budget, replace=False)
        return winner[chosen].astype(np.int64, copy=False)
    used = np.zeros(count, dtype=bool)
    used[winner] = True
    leftover = np.flatnonzero(~used)
    extra = rng.choice(leftover, size=budget - voxel_count, replace=False)
    return np.concatenate([winner, extra]).astype(np.int64, copy=False)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(zip(("mean", "p50", "p90", "p99"),
                    map(float, [x.mean(), *np.percentile(x, [50, 90, 99])]), strict=True))


def audit(run: Path, *, replay_version: str = "auto") -> dict:
    manifest_path = run / "initialization_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Read the recorded values without merging today's schema/defaults into
    # historical runs (older configs may contain subsequently removed fields).
    node = OmegaConf.load(run / "config.resolved.yaml")
    config = node.get("config", node)
    frames = manifest["keyframes"]
    if len(frames) < 2:
        raise ValueError("this temporal audit requires at least two keyframes")
    for frame in frames:
        provenance = frame["provenance"]
        source_manifest = Path(provenance["root"]) / "manifest.json"
        if (provenance.get("manifest_sha256") is not None
                and hashlib.sha256(source_manifest.read_bytes()).hexdigest()
                != provenance["manifest_sha256"]):
            raise ValueError(f"source manifest changed: {source_manifest}")
    version = str(replay_version)
    if version == "auto":
        algorithm = str(manifest.get("algorithm", ""))
        version = "v4" if algorithm.endswith("l3-v4") else (
            "v3" if algorithm.endswith("l3-v3") else (
            "v2" if algorithm.endswith("l3-v2") else "v1"
        ))
    if version not in {"v1", "v2", "v3", "v4"}:
        raise ValueError("replay-version must be auto, v1, v2, v3, or v4")
    clouds = [read_point_ply(Path(f["provenance"]["root"]) / f"f{f['frame']:06d}.ply")[0]
              for f in frames]
    if any(len(xyz) != frame["available_points"]
           for frame, xyz in zip(frames, clouds, strict=True)):
        raise ValueError("source population differs from recorded initialization")
    current = version in {"v3", "v4"}
    seed = initialization_seed(config) if current else int(config.train.seed)
    rng = np.random.default_rng(seed)
    neighbors = keyframe_neighbor_indices([len(xyz) for xyz in clouds])
    collected = {key: [] for key in (
        "sigma", "speed", "jitter_shift_sigma", "frame_motion_sigma", "source_restore_mae",
    )}
    grid_fractions = []
    cloud_hashes = []
    time_anchor = version in {"v2", "v3", "v4"} and bool(
        getattr(config.init, "time_anchor", True)
    )
    isolate = version in {"v2", "v3", "v4"} and bool(
        getattr(config.init, "isolate_source_scales", True)
    )
    for i, (frame, xyz) in enumerate(zip(frames, clouds, strict=True)):
        cloud_hashes.append(hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest())
        budget = int(frame["selected_points"])
        if current and (budget <= 0 or len(xyz) == 0):
            continue
        unique_budget = min(budget, len(xyz))
        sample_diagnostics: dict[str, object] = {}
        if current:
            selected, source, spacing, sample_diagnostics = sample_keyframe_points(rng, xyz, budget)
            unique_sigma = config.init.scale * spacing
        else:
            selected = historical_select(rng, xyz, unique_budget)
            source = xyz[selected]
            unique_sigma = config.init.scale * historical_spacing(source)
        j = neighbors[i] if current else i + 1 if i + 1 < len(frames) else i - 1
        t_ref = float(frame["reference_timestamp"])
        dt = 0.0 if j is None else frames[j]["reference_timestamp"] - t_ref
        times = (
            np.full((len(selected), 1), t_ref, np.float32)
            if j is None else (rng.random((len(selected), 1), dtype=np.float32) - 0.5) * dt + t_ref
        )
        velocity = (
            np.zeros_like(source) if j is None
            else _velocity(source, clouds[j], dt, config.init.num_velocity_nns)
        )
        speed = np.linalg.norm(velocity, axis=1)
        if time_anchor:
            means = source + (times - np.float32(t_ref)) * velocity
        else:
            means = source
        sigma = unique_sigma if isolate else config.init.scale * (
            _spacing(means) if current else historical_spacing(means)
        )
        restored = means - (times - np.float32(t_ref)) * velocity
        collected["sigma"].append(sigma)
        collected["speed"].append(speed)
        collected["jitter_shift_sigma"].append(
            np.abs(times[:, 0] - t_ref) * speed / np.maximum(sigma, 1e-12)
        )
        collected["frame_motion_sigma"].append(abs(dt) * speed / np.maximum(sigma, 1e-12))
        collected["source_restore_mae"].append(np.mean(np.abs(restored - source), axis=1))
        if current:
            grid_fractions.append(float(
                sample_diagnostics.get("occupied_fraction", 1.0)
            ))
        else:
            span = np.maximum(xyz.max(0) - xyz.min(0), 1e-8)
            voxel = max(
                (float(np.prod(span)) / max(unique_budget, 1)) ** (1 / 3),
                float(np.min(span)) / max(unique_budget, 1) ** (1 / 3),
                1e-8,
            )
            keys = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
            grid_fractions.append(
                min(len(np.unique(keys, axis=0)), unique_budget)
                / max(unique_budget, 1)
            )
    return {
        "run": str(run), "seed": seed, "train_seed": int(config.train.seed),
        "replay_version": version,
        "time_anchor": time_anchor,
        "isolate_source_scales": isolate,
        "gaussians": sum(f["selected_points"] for f in frames),
        "candidate_points": sum(map(len, clouds)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "xyz_sha256_by_frame": cloud_hashes,
        "statistics": {key: stats(np.concatenate(parts)) for key, parts in collected.items()},
        "spatial_grid_fraction_of_budget": stats(grid_fractions),
        "caveats": [
            "Speed is KNN displacement, not validated physical motion.",
            "Sigma is the initialized isotropic standard deviation in world units.",
            "v1 replays the historical un-anchored initializer; v2/v3/v4 use time anchoring.",
            "This tool does not rewrite caches or reinterpret old PSNR tables.",
            (
                "v1/v2 grid fraction uses the historical bbox-volume grid; "
                "v3/v4 report the portion represented by adaptive occupied cells."
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--replay-version", choices=("auto", "v1", "v2", "v3", "v4"), default="auto",
        help="v4 is current; v1-v3 replay historical initializers",
    )
    parser.add_argument("--output", type=Path, help="new JSONL file; refuses overwrite")
    args = parser.parse_args()
    output = args.output.open("x") if args.output is not None else None
    try:
        for run in args.runs:
            result = audit(run.resolve(), replay_version=args.replay_version)
            line = json.dumps(result, allow_nan=False)
            if output is not None:
                print(line, file=output, flush=True)
                print(json.dumps({key: value for key, value in result.items()
                                  if key != "xyz_sha256_by_frame"}), flush=True)
            else:
                print(line, flush=True)
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
