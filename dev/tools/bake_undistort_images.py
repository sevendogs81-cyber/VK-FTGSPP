#!/usr/bin/env python3
"""Bake SelfCap frames to undistorted, scaled JPEGs for faster lazy training.

Reuses the native runtime geometry / undistort maps so training with
``data.scale=1.0`` and zero distortion only needs JPEG decode (no remap).

Example:
  python dev/tools/bake_undistort_images.py \\
    --source /path/to/dataset \\
    --output /path/to/dataset_undistorted \\
    --scale 1.0 --jpeg-quality 90 --workers 16 --start 0 --stop 300
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ftgspp.data.scene import (
    Camera,
    _geometry,
    clear_undistort_cache,
    read_cameras,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="prepared SelfCap root")
    parser.add_argument("--output", type=Path, required=True, help="output root")
    parser.add_argument("--images-subdir", type=str, default="images")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 8)))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=300)
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="optional camera name filter, e.g. 0000 0001",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="skip frames whose output JPEG already exists (default)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
    )
    parser.add_argument(
        "--write-data-yaml",
        type=Path,
        default=None,
        help="optional data YAML fragment to write",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="corgi_first300_train21_eval3_scale1_undist",
    )
    parser.add_argument(
        "--keep-root",
        type=Path,
        default=None,
        help="data.root for points (default: --source)",
    )
    return parser.parse_args()


def _matrix_yaml(name: str, matrix: np.ndarray) -> str:
    arr = np.asarray(matrix, dtype=np.float64)
    flat = ", ".join(f"{float(v):.10f}" for v in arr.reshape(-1))
    return (
        f"{name}: !!opencv-matrix\n"
        f"  rows: {arr.shape[0]}\n"
        f"  cols: {arr.shape[1] if arr.ndim > 1 else 1}\n"
        f"  dt: d\n"
        f"  data: [{flat}]\n"
    )


def _write_intri(
    path: Path,
    names: list[str],
    target_ks: dict[str, np.ndarray],
    sizes: dict[str, tuple[int, int]],
    source_intri: Path,
) -> None:
    """Write EasyMocap-style intri.yml with D=0 and baked K/H/W."""

    src = cv2.FileStorage(str(source_intri), cv2.FILE_STORAGE_READ)
    lines = ["%YAML:1.0", "---", "names:"]
    for name in names:
        lines.append(f'  - "{name}"')
    lines.append("")
    for name in names:
        h, w = sizes[name]
        k = np.asarray(target_ks[name], dtype=np.float64)
        d = np.zeros((5, 1), dtype=np.float64)
        lines.append(_matrix_yaml(f"K_{name}", k).rstrip())
        lines.append(f"H_{name}: {float(h):.10f}")
        lines.append(f"W_{name}: {float(w):.10f}")
        lines.append(_matrix_yaml(f"D_{name}", d).rstrip())
        ccm = src.getNode(f"ccm_{name}").mat()
        if ccm is None:
            ccm = np.eye(3, dtype=np.float64)
        lines.append(_matrix_yaml(f"ccm_{name}", ccm).rstrip())
        lines.append("")
    src.release()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _prepare_camera_maps(
    camera: Camera,
    sample: Path,
    scale: float,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray | None, np.ndarray | None]:
    source_k, target_k, size = _geometry(camera, sample, scale)
    distortion = np.asarray(camera.distortion, dtype=np.float32).reshape(-1)[:4]
    if np.any(distortion):
        map_x, map_y = cv2.initUndistortRectifyMap(
            source_k,
            distortion,
            None,
            target_k,
            (int(size[1]), int(size[0])),
            cv2.CV_32FC1,
        )
    else:
        map_x = map_y = None
    return target_k, size, map_x, map_y


def _process_one(
    *,
    src_path: Path,
    dst_path: Path,
    map_x: np.ndarray | None,
    map_y: np.ndarray | None,
    size: tuple[int, int],
    jpeg_quality: int,
    skip_existing: bool,
    overwrite: bool,
) -> tuple[str, str]:
    if dst_path.exists() and skip_existing and not overwrite:
        return ("skip", str(dst_path))
    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(src_path)
    if map_x is not None and map_y is not None:
        rgb = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    else:
        height, width = size
        if image.shape[:2] != (height, width):
            rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        else:
            rgb = image
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.name + ".tmp.jpg")
    ok = cv2.imwrite(
        str(tmp),
        rgb,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to write {dst_path}")
    tmp.replace(dst_path)
    return ("write", str(dst_path))


def _write_data_yaml(
    path: Path,
    *,
    name: str,
    source_root: Path,
    output_root: Path,
    scale: float,
    start: int,
    stop: int,
) -> None:
    text = f"""name: {name}
fps: 60.0
strict_time_offsets: true
# Keep original root so points_stride* resolve without copying PLYs.
root: {source_root}
video_path: {source_root}
calibration_path: {output_root}
extracted_path: {output_root}/images
scale: {scale}
frames:
  start: {start}
  stop: {stop}
train_cameras: null
eval_cameras: [0, 8, 16]
camera_split_source: null
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    images_src = source / args.images_subdir
    images_dst = output / "images"
    if not images_src.is_dir():
        raise FileNotFoundError(images_src)

    cameras = read_cameras(source)
    names = list(cameras.keys())
    if args.cameras:
        wanted = set(args.cameras)
        names = [name for name in names if name in wanted]
        missing = wanted - set(names)
        if missing:
            raise ValueError(f"unknown cameras: {sorted(missing)}")

    clear_undistort_cache()
    output.mkdir(parents=True, exist_ok=True)
    images_dst.mkdir(parents=True, exist_ok=True)

    target_ks: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    maps: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}

    print(f"source images : {images_src}")
    print(f"output root   : {output}")
    print(f"scale/quality : {args.scale} / jpeg={args.jpeg_quality}")
    print(f"frames        : [{args.start}, {args.stop})")
    print(f"cameras       : {len(names)}")
    print(f"workers       : {args.workers}")

    def _sample_for(name: str) -> Path:
        cam_dir = images_src / name
        sample = cam_dir / f"{args.start:06d}.jpg"
        if sample.is_file():
            return sample
        found = sorted(cam_dir.glob("*.jpg"))
        if not found:
            raise FileNotFoundError(f"no jpgs under {cam_dir}")
        return found[0]

    # Always bake K/H/W for every camera so intri.yml stays complete even if
    # this invocation only materializes a camera subset.
    for name, camera in cameras.items():
        target_k, size, map_x, map_y = _prepare_camera_maps(
            camera, _sample_for(name), float(args.scale),
        )
        target_ks[name] = np.asarray(target_k, dtype=np.float64)
        sizes[name] = (int(size[0]), int(size[1]))
        if name in names:
            maps[name] = (map_x, map_y)
            print(f"  cam {name}: baked size {sizes[name][0]}x{sizes[name][1]}")

    tasks: list[tuple[Path, Path, str]] = []
    for name in names:
        for frame in range(int(args.start), int(args.stop)):
            src = images_src / name / f"{frame:06d}.jpg"
            dst = images_dst / name / f"{frame:06d}.jpg"
            if not src.is_file():
                raise FileNotFoundError(src)
            tasks.append((src, dst, name))

    wrote = skipped = 0
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                _process_one,
                src_path=src,
                dst_path=dst,
                map_x=maps[name][0],
                map_y=maps[name][1],
                size=sizes[name],
                jpeg_quality=int(args.jpeg_quality),
                skip_existing=bool(args.skip_existing),
                overwrite=bool(args.overwrite),
            )
            for src, dst, name in tasks
        ]
        with tqdm(total=len(futures), desc="bake frames") as bar:
            for future in as_completed(futures):
                status, _path = future.result()
                if status == "skip":
                    skipped += 1
                else:
                    wrote += 1
                bar.update(1)

    elapsed = time.perf_counter() - began
    print(f"done frames: wrote={wrote} skipped={skipped} in {elapsed:.1f}s")

    _write_intri(
        output / "intri.yml",
        names=list(cameras.keys()),
        target_ks=target_ks,
        sizes=sizes,
        source_intri=source / "intri.yml",
    )

    _symlink_or_copy(source / "extri.yml", output / "extri.yml")
    if (source / "t_offsets.json").is_file():
        _symlink_or_copy(source / "t_offsets.json", output / "t_offsets.json")

    meta = {
        "source": str(source),
        "output": str(output),
        "scale": float(args.scale),
        "jpeg_quality": int(args.jpeg_quality),
        "frames": [int(args.start), int(args.stop)],
        "cameras_processed": names,
        "sizes": {name: {"h": sizes[name][0], "w": sizes[name][1]} for name in sizes},
        "wrote": wrote,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
    }
    (output / "bake_manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )

    keep_root = (args.keep_root or source).resolve()
    data_yaml = args.write_data_yaml
    if data_yaml is None:
        data_yaml = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "data"
            / f"{args.data_name}.yaml"
        )
    _write_data_yaml(
        data_yaml,
        name=args.data_name,
        source_root=keep_root,
        output_root=output,
        scale=float(args.scale),
        start=int(args.start),
        stop=int(args.stop),
    )
    print(f"wrote calibration : {output / 'intri.yml'}")
    print(f"wrote data yaml   : {data_yaml}")
    print(f"manifest          : {output / 'bake_manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
