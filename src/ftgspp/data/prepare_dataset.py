"""Create a canonical dataset manifest and starter configuration.

This is the format-neutral community entry point. It indexes an existing
rectangular image grid without copying pixels, or validates a hand-authored
``ftgspp-dataset-v1`` manifest for arbitrary filenames and locations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from ftgspp.data.multiview import load_image_manifest

_FLAT = re.compile(r"^c(?P<camera>[0-9]+)-f(?P<frame>[0-9]{6})\.[A-Za-z0-9]+$")


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _discover_images(root: Path) -> dict[tuple[int, int], Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    flat: dict[tuple[int, int], Path] = {}
    for path in sorted(candidate for candidate in root.iterdir() if candidate.is_file()):
        match = _FLAT.fullmatch(path.name)
        if match is None:
            continue
        key = int(match.group("frame")), int(match.group("camera"))
        if key in flat:
            raise ValueError(f"duplicate image for frame/camera {key}")
        flat[key] = path
    if flat:
        return flat

    directories = sorted(
        (
            path for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: (
            0, int(path.name.removeprefix("c"))
        ) if path.name.removeprefix("c").isdigit() else (1, path.name),
    )
    rows: dict[tuple[int, int], Path] = {}
    for position, directory in enumerate(directories):
        camera_text = directory.name.removeprefix("c")
        camera = int(camera_text) if camera_text.isdigit() else position
        for path in sorted(candidate for candidate in directory.iterdir() if candidate.is_file()):
            if not path.stem.isdigit():
                continue
            key = int(path.stem), camera
            if key in rows:
                raise ValueError(f"duplicate image for frame/camera {key}")
            rows[key] = path
    if not rows:
        raise FileNotFoundError(
            f"no images found in {root}; use camera/<numeric-frame> files, "
            "c<camera>-f<six-digit-frame> files, or --manifest"
        )
    return rows


def _rectangular_range(
    rows: dict[tuple[int, int], Path],
    *,
    start: int | None,
    stop: int | None,
) -> tuple[list[int], list[int]]:
    available_frames = sorted({frame for frame, _camera in rows})
    begin = available_frames[0] if start is None else int(start)
    end = available_frames[-1] + 1 if stop is None else int(stop)
    frames = list(range(begin, end))
    if not frames:
        raise ValueError("selected frame range is empty")
    cameras = sorted(camera for frame, camera in rows if frame == frames[0])
    if not cameras:
        raise ValueError(f"no images exist for selected start frame {frames[0]}")
    expected = set(cameras)
    for frame in frames:
        observed = {camera for value, camera in rows if value == frame}
        if observed != expected:
            raise ValueError(
                f"image grid is not rectangular at frame {frame}: "
                f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )
    return frames, cameras


def _camera_split(
    cameras: Sequence[int],
    train: Sequence[int] | None,
    evaluate: Sequence[int] | None,
) -> tuple[list[int], list[int]]:
    available = {int(value) for value in cameras}
    eval_ids = [] if evaluate is None else [int(value) for value in evaluate]
    train_ids = (
        sorted(available - set(eval_ids))
        if train is None else [int(value) for value in train]
    )
    if not train_ids:
        raise ValueError("at least one training camera is required")
    if len(set(train_ids)) != len(train_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("camera lists must not contain duplicates")
    overlap = sorted(set(train_ids) & set(eval_ids))
    unknown = sorted((set(train_ids) | set(eval_ids)) - available)
    if overlap:
        raise ValueError(f"training and evaluation cameras overlap: {overlap}")
    if unknown:
        raise ValueError(f"camera lists contain undiscovered IDs: {unknown}")
    omitted = sorted(available - set(train_ids) - set(eval_ids))
    if omitted:
        raise ValueError(f"camera split omits discovered IDs: {omitted}")
    return train_ids, eval_ids


def _write_manifest(
    rows: dict[tuple[int, int], Path],
    frames: Sequence[int],
    cameras: Sequence[int],
    output: Path,
    *,
    fps: float,
    include_hashes: bool,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite dataset manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, object]] = []
    for frame in frames:
        for camera in cameras:
            path = rows[(int(frame), int(camera))]
            row: dict[str, object] = {
                "frame": int(frame),
                "camera_id": int(camera),
                "rgb_path": _relative(path, output.parent),
            }
            if include_hashes:
                row["rgb_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            views.append(row)
    payload = {
        "schema": "ftgspp-dataset-v1",
        "fps": float(fps),
        "views": views,
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary.replace(output)


def _write_config(
    output: Path,
    *,
    name: str,
    pipeline: str,
    root: Path,
    images: Path,
    manifest: Path,
    calibration: Path,
    mask_source: str,
    mask_manifest: Path | None,
    mask_root: Path | None,
    frames: Sequence[int],
    train_cameras: Sequence[int],
    eval_cameras: Sequence[int],
    fps: float,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite starter config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    def rel(path: Path) -> str:
        return _relative(path, output.parent)

    data: dict[str, object] = {
        "name": name,
        "root": rel(root),
        "extracted_path": rel(images),
        "calibration_path": rel(calibration),
        "view_manifest": rel(manifest),
        "fps": float(fps),
        "frames": {"start": int(frames[0]), "stop": int(frames[-1]) + 1},
        "train_cameras": [int(value) for value in train_cameras],
        "eval_cameras": [int(value) for value in eval_cameras],
    }
    if pipeline == "foreground":
        masks: dict[str, object] = {
            "source": mask_source,
            "rgb_encoding": "straight",
        }
        if mask_manifest is not None:
            masks["manifest_path"] = rel(mask_manifest)
        if mask_root is not None:
            masks["root"] = rel(mask_root)
        data["masks"] = masks
    keyframe_stride = min(10, len(frames))
    init: dict[str, object] = {
        "seed": 0,
        "keyframe_stride": keyframe_stride,
    }
    if pipeline == "foreground":
        init.update({
            "backend": "carving",
            "points_path": rel(
                root / f"points_vhull_ibr_stride{keyframe_stride}"
            ),
        })
    node = {
        "pipeline": {"mode": pipeline},
        "data": data,
        "init": init,
        "run": {"output_path": rel(output.parent / "runs" / name)},
    }
    OmegaConf.save(OmegaConf.create(node), output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ftgspp prepare dataset",
        description="index a generic multiview dataset and write a starter YAML",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="rectangular camera/frame image root")
    source.add_argument("--manifest", type=Path, help="existing ftgspp-dataset-v1 JSON")
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--pipeline", choices=("scene", "foreground"), default="scene")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--train-cameras", type=int, nargs="+")
    parser.add_argument(
        "--eval-cameras", type=int, nargs="*",
        help="hold-out camera ids that exist in the grid; omit for no hold-out",
    )
    parser.add_argument(
        "--mask-source", choices=("external", "alpha"), default="external",
    )
    parser.add_argument("--mask-manifest", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.fps is not None and (
        not math.isfinite(float(args.fps)) or not float(args.fps) > 0.0
    ):
        parser.error("--fps must be finite and greater than zero")
    if args.images is not None:
        images = args.images.expanduser().resolve()
        rows = _discover_images(images)
        frames, cameras = _rectangular_range(
            rows, start=args.start, stop=args.stop,
        )
        root = images.parent
        manifest = (
            args.output_manifest.expanduser().resolve()
            if args.output_manifest is not None
            else root / "manifests" / "dataset.json"
        )
        fps = 60.0 if args.fps is None else float(args.fps)
        _write_manifest(
            rows, frames, cameras, manifest, fps=fps,
            include_hashes=bool(args.hash_images), overwrite=bool(args.overwrite),
        )
    else:
        if args.output_manifest is not None:
            parser.error("--output-manifest applies only with --images")
        manifest = args.manifest.expanduser().resolve()
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            raw_views = payload["views"]
            all_frames = sorted({int(row["frame"]) for row in raw_views})
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"cannot inspect dataset manifest: {exc}")
        if not all_frames:
            parser.error("dataset manifest has no views")
        start = all_frames[0] if args.start is None else int(args.start)
        stop = all_frames[-1] + 1 if args.stop is None else int(args.stop)
        frames, cameras, paths, _timestamps = load_image_manifest(
            manifest, start=start, stop=stop,
        )
        images = Path(os.path.commonpath([str(path.parent) for row in paths for path in row]))
        root = manifest.parent.parent if manifest.parent.name == "manifests" else manifest.parent
        manifest_fps = payload.get("fps")
        fps = float(
            args.fps if args.fps is not None
            else (manifest_fps if manifest_fps is not None else 60.0)
        )
        if not math.isfinite(fps) or fps <= 0.0:
            parser.error("manifest/--fps must be finite and greater than zero")

    train_cameras, eval_cameras = _camera_split(
        cameras, args.train_cameras, args.eval_cameras,
    )
    config = (
        args.config.expanduser().resolve()
        if args.config is not None else root / "ftgspp.yaml"
    )
    calibration = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None else root
    )
    mask_manifest = (
        args.mask_manifest.expanduser().resolve()
        if args.mask_manifest is not None else (
            root / "manifests" / "masks.json"
            if args.pipeline == "foreground" and args.mask_source == "external"
            and args.mask_root is None else None
        )
    )
    mask_root = (
        args.mask_root.expanduser().resolve()
        if args.mask_root is not None else None
    )
    _write_config(
        config,
        name=str(args.name or root.name),
        pipeline=str(args.pipeline),
        root=root,
        images=images,
        manifest=manifest,
        calibration=calibration,
        mask_source=str(args.mask_source),
        mask_manifest=mask_manifest,
        mask_root=mask_root,
        frames=frames,
        train_cameras=train_cameras,
        eval_cameras=eval_cameras,
        fps=fps,
        overwrite=bool(args.overwrite),
    )
    print(f"dataset manifest: {manifest}")
    print(f"starter config: {config}")
    if args.pipeline == "foreground":
        stride = min(10, len(frames))
        point_output = root / f"points_vhull_ibr_stride{stride}"
        if args.mask_source == "external":
            print(
                "masks: ftgspp prepare masks "
                f"--images {images} --output {root / 'masks' / 'birefnet_hr'}"
            )
        print(
            "points: ftgspp prepare points --method carving "
            f"--config {config} --output {point_output}"
        )
    else:
        print(
            "initialize: provide points_stride*/fXXXXXX.ply or configure the "
            "DA3 route"
        )
    print(f"train after initialization: ftgspp train --config {config}")


if __name__ == "__main__":
    main()
