"""Installed command line entry point. Optional dependencies load on demand."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from ftgspp.config.compose import load_run_config, resolve_config
from ftgspp.io.export_selection import (
    export_selection_path,
    load_export_selection,
    restore_exported_color_correction,
)

_PREPARE_SUBCOMMANDS = {
    "dataset": "index a rectangular image grid or validate a view manifest",
    "masks": "generate a BiRefNet mask cache (optional extra)",
    "points": "calibrated visual-hull + IBR point clouds (optional extra)",
    "selfcap": "SelfCap layout adapter",
    "final-test": "validate a locked final-test manifest against a run",
}


def _prepare_help() -> str:
    names = ",".join(_PREPARE_SUBCOMMANDS)
    lines = [
        f"usage: ftgspp prepare {{{names}}} ...",
        "",
        "Prepare captures, masks, point clouds, or a locked final-test split.",
        "Optional model packages load only for the chosen subcommand.",
        "",
    ]
    width = max(len(name) for name in _PREPARE_SUBCOMMANDS)
    for name, help_text in _PREPARE_SUBCOMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {help_text}")
    lines.extend(["", "See also: ftgspp prepare <command> --help"])
    return "\n".join(lines) + "\n"


def _dispatch_prepare(rest: list[str]) -> None:
    if not rest or rest[0] in ("-h", "--help"):
        print(_prepare_help(), end="")
        return
    name = rest[0]
    if name == "masks":
        from ftgspp.optional.birefnet import main as prepare_masks
        prepare_masks(rest[1:])
        return
    if name == "points":
        from ftgspp.optional.visual_hull import main as prepare_points
        prepare_points(rest[1:])
        return
    if name == "dataset":
        from ftgspp.data.prepare_dataset import main as prepare_dataset
        prepare_dataset(rest[1:])
        return
    if name == "final-test":
        from ftgspp.data.prepare_final_test import main as prepare_final_test
        prepare_final_test(rest[1:])
        return
    if name == "selfcap":
        from ftgspp.data.prepare_selfcap import main as prepare_selfcap
        prepare_selfcap(rest[1:])
        return
    if name.startswith("-"):
        print(
            "warning: flag-only `ftgspp prepare` is a SelfCap compatibility "
            "alias; use `ftgspp prepare selfcap`",
            file=sys.stderr,
        )
        from ftgspp.data.prepare_selfcap import main as prepare_selfcap
        prepare_selfcap(rest)
        return
    print(f"error: unknown prepare command {name!r}", file=sys.stderr)
    print(_prepare_help(), end="", file=sys.stderr)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare":
        _dispatch_prepare(argv[1:])
        return
    parser = argparse.ArgumentParser(
        prog="ftgspp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "prepare subcommands: dataset, masks, points, selfcap, final-test. "
            "See: ftgspp prepare --help"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check native devices without creating a run")
    doctor.add_argument("--devices", nargs="+", type=int, default=[0])
    train = commands.add_parser("train", help="train from YAML or resume a V6 run")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", type=Path)
    train.add_argument("--output", type=Path)
    train.add_argument(
        "--devices", nargs="+", type=int,
        help="Vulkan device id from `ftgspp doctor` (first release: one device)",
    )
    train.add_argument(
        "--pipeline", choices=("scene", "foreground"),
        help="select the full-scene or foreground-target profile",
    )
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    evaluate = commands.add_parser("eval", help="evaluate a completed run")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--device", type=int, default=0)
    evaluate.add_argument(
        "--test-manifest", type=Path,
        help="locked ftgspp-final-test-v1/v2 manifest; never used for model selection",
    )
    export = commands.add_parser("export", help="export an explicit model to viewer PLY")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "export":
        from ftgspp.io.checkpoint import load_model
        from ftgspp.io.ply import export_4dgs
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_4dgs(args.output, load_model(args.model))
        return
    if args.command == "doctor":
        from ftgspp.training.backend import NativeVulkanBackend
        for device in args.devices:
            backend = NativeVulkanBackend(device, honor_device_env=False)
            try:
                print(json.dumps(backend.device_metadata(), sort_keys=True))
            finally:
                backend.close()
        return
    from ftgspp.training.runner import NativeRunner
    if args.command == "train":
        if args.resume:
            if args.output or args.devices or args.pipeline or args.set:
                parser.error("resume uses the saved effective plan; output/devices/overrides are not accepted")
            config = load_run_config(args.resume.resolve())
            config.run.output_path = args.resume.resolve()
            config.train.resume = True
            config.run.overwrite_completed = True
        else:
            path = args.config.expanduser().resolve()
            node = OmegaConf.load(path)
            overrides = OmegaConf.from_dotlist(args.set)
            if args.output:
                overrides = OmegaConf.merge(overrides, {"run": {"output_path": str(args.output.expanduser().resolve())}})
            if args.devices:
                if len(args.devices) == 2:
                    print(
                        "warning: two-device training is in development and is "
                        "not part of the first-release supported surface",
                        file=sys.stderr,
                    )
                elif len(args.devices) != 1:
                    parser.error(
                        "train accepts one Vulkan device; two-device sync is "
                        "in development"
                    )
                overrides = OmegaConf.merge(overrides, {
                    "run": {"gpu": args.devices[0]},
                    "distributed": {"mode": "sync" if len(args.devices) == 2 else "off",
                                    "device_ids": args.devices},
                })
            if args.pipeline:
                overrides = OmegaConf.merge(
                    overrides, {"pipeline": {"mode": args.pipeline}},
                )
            # YAML path overrides passed via CLI are relative to the invoking cwd.
            path_keys = ("data.root", "data.video_path", "data.calibration_path",
                         "data.extracted_path", "data.view_manifest",
                         "data.test_manifest",
                         "data.masks.root", "data.masks.manifest_path",
                         "data.masks.validity_root", "init.points_path",
                         "init.da3.model_path", "init.da3.cache_path",
                         "train.resume_checkpoint", "run.output_path", "run.experiment_root")
            for key in path_keys:
                value = OmegaConf.select(overrides, key)
                if value is not None:
                    OmegaConf.update(overrides, key, str(Path(value).expanduser().resolve()))
            config = resolve_config(OmegaConf.merge(node, overrides), base_dir=path.parent).config
        NativeRunner(config).run()
        return
    from ftgspp.training.backend import NativeVulkanBackend, camera_extent
    from ftgspp.data.scene import load_final_test_scene
    from ftgspp.io.checkpoint import load_model, latest_checkpoint, load_checkpoint_full
    config = load_run_config(args.run.resolve())
    config.run.output_path = args.run.resolve()
    runner = NativeRunner(config)
    dataset = runner._preflight()
    selection_path = export_selection_path(args.run)
    selection = load_export_selection(args.run)
    final_test = False
    if args.command == "eval":
        test_manifest = args.test_manifest
        if test_manifest is None:
            configured = getattr(config.data, "test_manifest", None)
            test_manifest = Path(configured) if configured is not None else None
        if test_manifest is not None:
            training_dataset = dataset
            dataset = load_final_test_scene(training_dataset, test_manifest)
            if (
                (dataset.evaluation_identity or {}).get("schema")
                == "ftgspp-final-test-v2"
            ):
                training_fingerprint = str(
                    training_dataset.checkpoint_metadata()["fingerprint"]
                )
                locked_fingerprint = selection.get(
                    "training_dataset_fingerprint"
                )
                if locked_fingerprint is None:
                    raise ValueError(
                        "final-test-v2 requires an export_selection.json with "
                        "training_dataset_fingerprint"
                    )
                if str(locked_fingerprint) != training_fingerprint:
                    raise ValueError(
                        "current training data does not match the exported "
                        "model's locked dataset fingerprint"
                    )
            # Novel test cameras have no learned per-camera affine row. The
            # protocol evaluates the model-generalizable raw color output.
            config.train.color_correction = False
            final_test = True
    backend = NativeVulkanBackend(args.device, honor_device_env=False)
    try:
        backend.configure(config, camera_extent(dataset))
        backend.upload_dataset(dataset, rgb_cache=config.run.view_rgb_cache, workers=config.run.view_load_workers)
        model_path = args.run / "gaussians.npz"
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        declared_model_hash = selection.get("model_sha256")
        if declared_model_hash is not None and declared_model_hash != model_sha256:
            raise ValueError(
                "gaussians.npz does not match export_selection.json model_sha256"
            )
        selected_step = int(selection.get("step", config.train.iterations - 1))
        model = load_model(model_path)
        backend.upload_model(model)
        checkpoint = (
            None if final_test else latest_checkpoint(args.run / "ckpt")
        )
        if checkpoint is not None:
            snapshot = load_checkpoint_full(checkpoint)
            backend.restore_optimizer(snapshot.optimizer)
            if str(snapshot.metadata.get("phase", "")).startswith("hard_separation"):
                backend.resume_hard_separation()
        if selection and not final_test:
            restore_exported_color_correction(
                backend, args.run, selection, dataset,
            )
        if final_test:
            identity = dict(dataset.evaluation_identity or {})
            identity.update({
                "model_path": str(model_path.resolve()),
                "model_sha256": model_sha256,
                "model_step": selected_step,
                "selection_manifest": (
                    str(selection_path.resolve()) if selection_path.is_file() else None
                ),
                "selection_manifest_sha256": (
                    hashlib.sha256(selection_path.read_bytes()).hexdigest()
                    if selection_path.is_file() else None
                ),
                "color_correction": "disabled_for_novel_final_test_cameras",
            })
            dataset.evaluation_identity = identity
        runner._evaluate(backend, dataset, selected_step)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
