"""Installed-wheel end-to-end check; creates its own tiny calibrated scene.

Run outside the repository using the wheel environment's Python. This is a
correctness fixture, not a representative reconstruction quality benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from omegaconf import OmegaConf


def scene(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for kind in ("intri", "extri"):
        storage = cv2.FileStorage(str(root / f"{kind}.yml"), cv2.FILE_STORAGE_WRITE)
        storage.write("names", ["0", "1", "2"])
        for camera in range(3):
            if kind == "intri":
                storage.write(f"K_{camera}", np.array([[30.,0,16],[0,30.,16],[0,0,1]]))
                storage.write(f"D_{camera}", np.zeros((5,1)))
                storage.write(f"H_{camera}", 32)
                storage.write(f"W_{camera}", 32)
            else:
                storage.write(f"Rot_{camera}", np.eye(3))
                storage.write(f"T_{camera}", np.array([[0.1*camera],[0.],[0.]]))
        storage.release()
    for camera in range(3):
        directory = root / "images" / str(camera)
        directory.mkdir(parents=True, exist_ok=True)
        for frame in range(2):
            bgr = np.full((32,32,3), 30+camera*10+frame*5, np.uint8)
            cv2.circle(bgr, (16+frame,16), 6, (120,170,210), -1)
            yy, xx = np.mgrid[:32, :32]
            alpha = np.where(
                (xx - (16 + frame)) ** 2 + (yy - 16) ** 2 < 7 ** 2,
                230, 25,
            ).astype(np.uint8)
            cv2.imwrite(
                str(directory / f"{frame:06d}.png"),
                np.concatenate((bgr, alpha[..., None]), axis=-1),
            )
    points = root / "points_stride1"
    points.mkdir(exist_ok=True)
    rng = np.random.default_rng(5)
    xyz = rng.uniform(-0.3,0.3,(32,3))
    xyz[:,2] += 3
    header = ("ply\nformat ascii 1.0\nelement vertex 32\nproperty float x\n"
              "property float y\nproperty float z\nproperty uchar red\n"
              "property uchar green\nproperty uchar blue\nend_header\n")
    text = header + "".join(f"{x} {y} {z} 120 170 210\n" for x,y,z in xyz)
    for frame in range(2):
        (points / f"f{frame:06d}.ply").write_text(text)
    return root


def compare(label, expected, expected_state, actual, actual_state):
    differences = {}
    for prefix, before, after in (
        ("model", expected.arrays(), actual.arrays()),
        ("optimizer", expected_state.optimizer, actual_state.optimizer),
    ):
        if before.keys() != after.keys():
            raise AssertionError(f"{prefix} keys changed")
        for key, value in before.items():
            result = after[key]
            if result.shape != value.shape:
                raise AssertionError(f"{prefix}.{key} shape changed")
            differences[f"{prefix}.{key}"] = {
                "max_abs": float(np.max(np.abs(result - value), initial=0)),
                "passed": bool(np.allclose(result, value, rtol=1e-6, atol=1e-7)),
            }
    clocks_match = (
        actual_state.metadata["training"]["clocks"]["views_seen"] ==
        expected_state.metadata["training"]["clocks"]["views_seen"]
    )
    return {"case": label, "clocks_match": clocks_match,
            "passed": clocks_match and all(item["passed"] for item in differences.values()),
            "arrays": differences}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--repeat", action="store_true", help="also measure independent full-run repeatability")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"use a fresh output path: {args.output}")
    root = scene(args.output / "scene")
    config = {
        "pipeline": {"mode": "foreground"},
        "data": {"root": str(root), "frames": {"start":0,"stop":2},
                 "fps":2., "scale":1., "train_cameras":[0,1], "eval_cameras":[2],
                 "masks":{"source":"alpha", "required":True,
                          "rgb_encoding":"straight"}},
        "init": {
            "backend": "points", "points_path": str(root / "points_stride1"),
            "num_gaussians":32, "keyframe_stride":1, "sh_degree":3,
        },
        "model": {"marginal_gating":args.hard},
        "train": {"iterations":args.steps, "checkpoint_interval":max(1,args.steps//2),
                  "background":{"mode":"random_solid", "seed":19,
                                "rank_policy":"shared", "hard_phase":"inherit"},
                  "hard_separation":args.hard, "color_correction":True,
                  "color_correction_start":0,
                  "hard_separation_steps":args.steps if args.hard else 3000,
                  "hard_relocation_steps":(
                      [max(1, args.steps//2)] if args.hard and args.steps > 1 else []
                  ),
                  "relocation":{"start":1,"stop":args.steps,"every":10}},
        "distributed": {"host_collective":{"bucket_mb":8}},
        "eval":{"frame_interval":1,
                "background_colors":[[0.,0.,0.],[1.,1.,1.]]},
        "run":{"view_load_workers":0,"view_rgb_cache":4},
    }
    config_path = args.output / "scene.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    full = args.output / "full"
    def run(*argv):
        subprocess.run([sys.executable, "-m", "ftgspp.cli", *map(str,argv)], check=True)
    run("doctor", "--devices", *args.devices)
    run("train", "--config", config_path, "--output", full, "--devices", *args.devices)
    run("eval", "--run", full, "--device", args.devices[0])
    run("export", "--model", full / "gaussians.npz", "--output", args.output / "export.ply")
    from ftgspp.io.checkpoint import load_checkpoint_full, latest_checkpoint
    from ftgspp.config.compose import load_run_config, save_resolved_config
    from ftgspp.data.scene import Scene

    # Exercise the installed final-test path with explicit geometry, timestamp,
    # and content hashes. It must append to its own file without altering the
    # ordinary validation history used for checkpoint selection.
    saved_config = load_run_config(full)
    training_scene = Scene.from_config(saved_config)
    locked_root = args.output / "locked-final-test"
    rgb_root = locked_root / "rgb"
    mask_root = locked_root / "masks"
    rgb_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    final_rows = []
    for source_frame in range(2):
        frame = source_frame + 2
        source_path = root / "images" / "2" / f"{source_frame:06d}.png"
        decoded = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 4:
            raise AssertionError(f"expected synthetic BGRA source: {source_path}")
        decoded = decoded.copy()
        decoded[0, 0, 0] = np.uint8(int(decoded[0, 0, 0]) + 1)
        rgb_path = rgb_root / f"{frame:06d}.png"
        if not cv2.imwrite(str(rgb_path), decoded):
            raise OSError(rgb_path)
        mask_path = mask_root / f"{frame:06d}.png"
        if not cv2.imwrite(str(mask_path), decoded[..., 3]):
            raise OSError(mask_path)
        final_rows.append({
            "camera_id": 2,
            "frame": frame,
            "timestamp": float(
                np.float32(frame / training_scene.fps)
                - np.float32(training_scene.time_offsets[2])
            ),
            "rgb_path": str(rgb_path.relative_to(locked_root)),
            "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            "mask_path": str(mask_path.relative_to(locked_root)),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "original_size": list(decoded.shape[:2]),
        })
    camera = training_scene.view(0, 2)
    final_manifest = locked_root / "manifest.json"
    final_manifest.write_text(json.dumps({
        "schema": "ftgspp-final-test-v1",
        "training_dataset_fingerprint": training_scene.checkpoint_metadata()["fingerprint"],
        "fps": float(training_scene.fps),
        "rgb_encoding": "straight",
        "mask_source": "external",
        "cameras": {"2": {
            "w2c": np.asarray(camera.w2c).tolist(),
            "intrinsic": np.asarray(camera.intrinsic).tolist(),
            "source_intrinsic": np.asarray(camera.source_intrinsic).tolist(),
            "distortion": np.asarray(camera.distortion).tolist(),
            "size": list(camera.size),
            "time_offset": float(training_scene.time_offsets[2]),
        }},
        "views": final_rows,
    }, indent=2) + "\n")
    validation_before = (full / "metrics.jsonl").read_bytes()
    run("eval", "--run", full, "--device", args.devices[0],
        "--test-manifest", final_manifest)
    if (full / "metrics.jsonl").read_bytes() != validation_before:
        raise AssertionError("locked final test modified validation metrics")
    final_metrics = [
        json.loads(line) for line in (full / "final_test_metrics.jsonl").read_text().splitlines()
        if line
    ]
    identity = final_metrics[-1].get("evaluation_identity", {})
    if not (
        identity.get("kind") == "final_test"
        and identity.get("model_sha256")
        and identity.get("selection_manifest_sha256")
        and identity.get("manifest_sha256")
        and identity.get("content_sha256")
        and identity.get("model_step") is not None
    ):
        raise AssertionError(f"incomplete final-test identity: {identity}")
    selected_manifest = json.loads(
        (full / "export_selection.json").read_text(encoding="utf-8")
    )
    selected_checkpoint = full / selected_manifest["checkpoint_artifact"]
    if not selected_checkpoint.is_file():
        raise AssertionError(
            f"selected checkpoint is missing: {selected_checkpoint}"
        )
    expected_selected_name = (
        "hard_final.npz" if args.hard else "main_selected.npz"
    )
    if selected_checkpoint.name != expected_selected_name:
        raise AssertionError(
            f"unexpected selected checkpoint: {selected_checkpoint.name}"
        )
    if args.hard and int(selected_manifest["step"]) != int(
        selected_manifest["last_step"]
    ):
        raise AssertionError("hard separation must export its terminal model")
    checkpoints = sorted(
        path for path in (full / "ckpt").glob("*.npz")
        if path.stem.isdigit()
    )
    # Resume from an interior main step; with --hard also test a hard-phase cursor.
    selected = [checkpoints[0]]
    if args.hard:
        selected.append(checkpoints[len(checkpoints)//2])
    expected_state = load_checkpoint_full(latest_checkpoint(full / "ckpt"))
    # Resume equivalence compares the final optimizer snapshot. The exported
    # model may intentionally be an earlier best-validation snapshot.
    expected = expected_state.model
    comparisons = []
    for index, checkpoint in enumerate(selected):
        resumed = args.output / f"resumed{index}"
        saved = load_run_config(full)
        saved.train.resume_checkpoint = checkpoint
        saved.run.output_path = resumed
        save_resolved_config(saved, resumed / "config.resolved.yaml")
        run("train", "--resume", resumed)
        actual_state = load_checkpoint_full(latest_checkpoint(resumed / "ckpt"))
        actual = actual_state.model
        comparisons.append(compare(f"resume{index}", expected, expected_state, actual, actual_state))
    if args.repeat:
        repeated = args.output / "repeated"
        run("train", "--config", config_path, "--output", repeated, "--devices", *args.devices)
        repeated_state = load_checkpoint_full(latest_checkpoint(repeated / "ckpt"))
        comparisons.append(compare(
            "full_repeat", expected, expected_state,
            repeated_state.model, repeated_state,
        ))
    report = {"passed": all(case["passed"] for case in comparisons),
              "steps": args.steps, "hard": args.hard, "devices": args.devices,
              "rtol": 1e-6, "atol": 1e-7, "comparisons": comparisons,
              "locked_final_test": {
                  "passed": True,
                  "views": len(final_rows),
                  "identity": identity,
              }}
    (args.output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
