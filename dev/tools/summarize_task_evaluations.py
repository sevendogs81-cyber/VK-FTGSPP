"""Compare non-mutating full-validation reports from evaluate_task_run.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCENE_METRICS = ("psnr", "ssim")
FOREGROUND_METRICS = (
    "foreground_psnr",
    "foreground_roi_ssim",
    "foreground_roi_lpips",
    "silhouette_iou",
    "boundary_fscore",
    "foreground_holes",
    "background_opacity_leakage",
    "psnr",
)


def _finite(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _load(directory: Path, label: str, scene: str) -> dict[str, object]:
    path = directory / f"{label}_{scene}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_scope") != "full_validation":
        raise ValueError(f"report is not full validation: {path}")
    return payload


def _metric_row(
    baseline: dict[str, object],
    candidate: dict[str, object],
    keys: tuple[str, ...],
) -> dict[str, object]:
    base_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    if not isinstance(base_metrics, dict) or not isinstance(
        candidate_metrics, dict,
    ):
        raise ValueError("evaluation report metrics must be objects")
    row: dict[str, object] = {}
    for key in keys:
        old = _finite(base_metrics.get(key))
        new = _finite(candidate_metrics.get(key))
        row[key] = {
            "baseline": old,
            "candidate": new,
            "delta": (
                None if old is None or new is None else float(new - old)
            ),
        }
    return row


def _foreground_frame_deltas(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> list[dict[str, object]]:
    base_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    if not isinstance(base_metrics, dict) or not isinstance(
        candidate_metrics, dict,
    ):
        return []
    base_rows = {
        (int(row["frame"]), int(row["camera"])): row
        for row in base_metrics.get("foreground_views", [])
    }
    candidate_rows = {
        (int(row["frame"]), int(row["camera"])): row
        for row in candidate_metrics.get("foreground_views", [])
    }
    result: list[dict[str, object]] = []
    for frame_camera in sorted(set(base_rows) & set(candidate_rows)):
        old = _finite(base_rows[frame_camera].get("foreground_psnr"))
        new = _finite(candidate_rows[frame_camera].get("foreground_psnr"))
        result.append({
            "frame": frame_camera[0],
            "camera": frame_camera[1],
            "baseline_foreground_psnr": old,
            "candidate_foreground_psnr": new,
            "delta_foreground_psnr": (
                None if old is None or new is None else float(new - old)
            ),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("scene", "foreground"), required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=("corgi1", "corgi2", "dance1", "dance2", "bike1", "bike2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    keys = (
        SCENE_METRICS
        if args.mode == "scene" else FOREGROUND_METRICS
    )
    rows: list[dict[str, object]] = []
    for scene in args.scenes:
        baseline = _load(args.directory, args.baseline_label, scene)
        candidate = _load(args.directory, args.candidate_label, scene)
        row: dict[str, object] = {
            "scene": scene,
            "baseline_run": baseline["run"],
            "candidate_run": candidate["run"],
            "metrics": _metric_row(baseline, candidate, keys),
        }
        if args.mode == "foreground":
            row["per_frame"] = _foreground_frame_deltas(
                baseline, candidate,
            )
        rows.append(row)

    means: dict[str, object] = {}
    for key in keys:
        deltas = [
            row["metrics"][key]["delta"]
            for row in rows
            if row["metrics"][key]["delta"] is not None
        ]
        means[key] = {
            "mean_delta": (
                None if not deltas else float(np.mean(deltas))
            ),
            "comparable_scenes": len(deltas),
        }
    report = {
        "schema": "ftgspp-task-evaluation-comparison-v1",
        "mode": args.mode,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "rows": rows,
        "means": means,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "means": means,
    }), flush=True)


if __name__ == "__main__":
    main()
