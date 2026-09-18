"""Evaluate a selected scene or foreground export without mutating its run.

The output is a single JSON object with current-code, full-validation metrics.
All eligible held-out views are used regardless of the candidate subset used
during training.  Historical metrics.jsonl files are never opened for writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

from ftgspp.config.compose import load_config
from ftgspp.io.checkpoint import load_model
from ftgspp.io.export_selection import load_export_selection
from ftgspp.training.backend import NativeVulkanBackend, camera_extent
from ftgspp.training.runner import NativeRunner, primary_metric_name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compute foreground ROI LPIPS (scene LPIPS uses evaluate_scene_run.py)",
    )
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    config_path = run / "config.resolved.yaml"
    selection_path = run / "export_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    model_path = run / "gaussians.npz"
    config = load_config(config_path)
    selection = load_export_selection(run)
    model_digest = _sha256(model_path)
    declared_model = selection.get("model_sha256")
    if declared_model is not None and str(declared_model) != model_digest:
        raise ValueError("model does not match export_selection.json")

    # The comparison contract is every held-out frame under the current
    # evaluator.  Candidate subset limits remain recorded in the source run.
    config.eval.frame_interval = 1
    config.eval.candidate_max_views = 0
    config.eval.lpips = bool(args.lpips and config.pipeline.mode == "foreground")
    config.run.output_path = output.parent
    runner = NativeRunner(config)
    dataset = runner._preflight()
    if not dataset.eval_views:
        raise ValueError("run has no held-out evaluation views")

    backend = NativeVulkanBackend(args.device, honor_device_env=False)
    temporary = output.parent / (
        f".{output.name}.{uuid.uuid4().hex}.metrics.jsonl"
    )
    try:
        metadata = backend.device_metadata()
        backend.configure(config, camera_extent(dataset))
        backend.upload_dataset(
            dataset,
            rgb_cache=int(config.run.view_rgb_cache),
            workers=int(config.run.view_load_workers),
        )
        backend.upload_model(load_model(model_path))
        metrics = runner._evaluate(
            backend,
            dataset,
            int(selection.get("step", config.train.iterations - 1)),
            result_filename=temporary.name,
            full_validation=True,
        )
    finally:
        backend.close()
        temporary.unlink(missing_ok=True)

    primary_metric = primary_metric_name(config.eval.selector)
    report = {
        "schema": "ftgspp-task-evaluation-v1",
        "run": str(run),
        "pipeline_mode": str(config.pipeline.mode),
        "protocol": str(config.pipeline.eval_protocol),
        "primary_metric": primary_metric,
        "primary_score": metrics.get(primary_metric),
        "evaluation_scope": "full_validation",
        "model_sha256": model_digest,
        "config_sha256": _sha256(config_path),
        "selection_sha256": _sha256(selection_path),
        "color_snapshot_sha256": None,
        "color_correction": "disabled_for_holdout_cameras",
        "selected_step": int(selection.get(
            "step", config.train.iterations - 1,
        )),
        "source_candidate_views": selection.get("views"),
        "metrics": metrics,
        "device": metadata,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "output": str(output),
        "primary_metric": primary_metric,
        "primary_score": report["primary_score"],
        "views": metrics.get("views"),
    }), flush=True)


if __name__ == "__main__":
    main()
