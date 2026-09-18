"""Evaluate a selected scene export without changing its run or NAS leaderboard.

Produces per-frame PSNR, SSIM_2 and LPIPS-VGG with the existing benchmark's
[0,1], normalize=False LPIPS convention. Uses only held-out eval views, never
applies trained color correction to them, and closes Vulkan before starting VGG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ftgspp.config.compose import load_config
from ftgspp.data.scene import Scene
from ftgspp.io.checkpoint import load_model
from ftgspp.training.backend import NativeVulkanBackend, camera_extent
from ftgspp.training.runner import _EvalSsimWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lpips-device", default="auto")
    parser.add_argument("--lpips-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.lpips_batch_size <= 0:
        parser.error("--lpips-batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run.resolve()
    config = load_config(run / "config.resolved.yaml")
    if config.pipeline.mode != "scene":
        raise ValueError("this evaluator is for full scene RGB, not foreground")
    if config.train.hard_separation:
        raise ValueError("this diagnostic does not restore hard-separation phase state")
    selection = json.loads((run / "export_selection.json").read_text())
    model_path = run / "gaussians.npz"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != selection["model_sha256"]:
        raise ValueError("model does not match the recorded export selection")
    dataset = Scene.from_config(config)
    if not dataset.eval_views:
        raise ValueError("no held-out evaluation views")
    workspace = _EvalSsimWorkspace()
    pairs, rows = [], []
    backend = NativeVulkanBackend(args.device, honor_device_env=False)
    try:
        metadata = backend.device_metadata()
        backend.configure(config, camera_extent(dataset))
        backend.upload_dataset(dataset, rgb_cache=64, workers=8)
        backend.upload_model(load_model(model_path))
        for index, view in enumerate(dataset.eval_views):
            pred = np.clip(np.asarray(backend.render_eval(index), np.float32), 0, 1)
            reference = view.load_rgb()[..., :3].astype(np.float32) / 255.0
            if pred.shape != reference.shape or not np.isfinite(pred).all():
                raise ValueError(f"invalid render at evaluation view {index}")
            mse = float(np.mean((pred - reference) ** 2))
            rows.append({"frame": int(view.frame), "camera": int(view.camera_id),
                         "psnr": float(-10 * np.log10(max(mse, 1e-12))),
                         "ssim2": float(workspace.ssim(pred, reference))})
            pairs.append((pred, reference))
    finally:
        backend.close()
    from ftgspp.optional.lpips import evaluate_lpips_vgg_unnormalized
    lpips = evaluate_lpips_vgg_unnormalized(
        pairs, device=args.lpips_device, batch_size=args.lpips_batch_size,
        vulkan_metadata=metadata,
    )
    for row, value in zip(rows, lpips, strict=True):
        row["lpips_vgg"] = float(value)
    result = {
        "run": str(run), "model_sha256": model_hash,
        "config_sha256": hashlib.sha256((run / "config.resolved.yaml").read_bytes()).hexdigest(),
        "color_snapshot_sha256": None,
        "color_correction": "disabled_for_holdout_cameras",
        "step": selection["step"], "selection_source": selection["source"],
        "protocol": "scene_rgb_v1", "ssim_variant": "SSIM_2", "ssim_data_range": 2.0,
        "lpips_convention": "VGG [0,1] normalize=False (legacy benchmark convention)",
        "views": len(rows), "device": metadata, "per_frame": rows,
        "means": {key: float(np.mean([r[key] for r in rows]))
                  for key in ("psnr", "ssim2", "lpips_vgg")},
        "caveat": (
            "The held-out camera also selected the checkpoint; "
            "this is not a locked final-test result."
        ),
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "means": result["means"]}), flush=True)


if __name__ == "__main__":
    main()
