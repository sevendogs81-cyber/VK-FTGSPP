"""Generate a small, deterministic Corgi-compatible SelfCap scene.

This is a fallback for machines that do not have the proprietary Corgi release.
The generated scene keeps the native loader contract (images/<camera>/<frame>,
OpenCV intri/extri, time offsets and points_stride5) while using a compact
moving multi-colour object so a full 20k Vulkan run remains practical.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack([right, up, forward], axis=0).astype(np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ position
    return matrix


def write_calibration(root: Path, cameras: int, width: int, height: int) -> None:
    intri = cv2.FileStorage(str(root / "intri.yml"), cv2.FILE_STORAGE_WRITE)
    extri = cv2.FileStorage(str(root / "extri.yml"), cv2.FILE_STORAGE_WRITE)
    names = [str(index) for index in range(cameras)]
    intri.write("names", np.asarray(names, dtype=object))
    extri.write("names", np.asarray(names, dtype=object))
    focal = float(width) * 0.88
    k = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], np.float64)
    for camera in range(cameras):
        angle = 2.0 * math.pi * camera / cameras
        position = np.array([4.0 * math.sin(angle), 1.2 + 0.35 * math.cos(angle), 4.0 * math.cos(angle)], np.float32)
        matrix = look_at(position, np.array([0.0, 1.0, 0.0], np.float32))
        name = str(camera)
        intri.write(f"K_{name}", k)
        intri.write(f"D_{name}", np.zeros((5, 1), np.float64))
        intri.write(f"H_{name}", float(height))
        intri.write(f"W_{name}", float(width))
        extri.write(f"Rot_{name}", matrix[:3, :3])
        extri.write(f"T_{name}", matrix[:3, 3:4])
    intri.release()
    extri.release()


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(xyz)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    vertices = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--cameras", type=int, default=6)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--points", type=int, default=512)
    args = parser.parse_args()
    if args.frames <= 1 or args.cameras < 2 or args.points < 8:
        raise ValueError("frames>1, cameras>=2 and points>=8 are required")

    root = args.output.resolve()
    image_root = root / "images"
    points_root = root / "points_stride5"
    image_root.mkdir(parents=True, exist_ok=True)
    points_root.mkdir(parents=True, exist_ok=True)
    write_calibration(root, args.cameras, args.width, args.height)
    (root / "t_offsets.json").write_text(
        "{" + ", ".join(f'\"{camera}\": 0.0' for camera in range(args.cameras)) + "}\n",
        encoding="utf-8",
    )

    rng = np.random.default_rng(20260907)
    # A compact static shell plus a moving warm-colour lobe.  The same points
    # are projected into the generated RGB views, making the scene self-consistent.
    shell = rng.normal(size=(args.points, 3)).astype(np.float32)
    shell /= np.maximum(np.linalg.norm(shell, axis=1, keepdims=True), 1e-6)
    shell *= rng.uniform(0.18, 0.85, size=(args.points, 1)).astype(np.float32)
    shell[:, 1] += 1.0
    base_rgb = np.clip((shell - shell.min(0)) / np.maximum(np.ptp(shell, axis=0), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    focal = float(args.width) * 0.88
    intrinsic = np.array([[focal, 0.0, args.width / 2.0], [0.0, focal, args.height / 2.0], [0.0, 0.0, 1.0]], np.float32)
    camera_w2c = []
    for camera in range(args.cameras):
        angle = 2.0 * math.pi * camera / args.cameras
        position = np.array([4.0 * math.sin(angle), 1.2 + 0.35 * math.cos(angle), 4.0 * math.cos(angle)], np.float32)
        camera_w2c.append(look_at(position, np.array([0.0, 1.0, 0.0], np.float32)))

    for frame in range(args.frames):
        phase = 2.0 * math.pi * frame / args.frames
        motion = np.array([0.55 * math.sin(phase), 0.12 * math.cos(phase), 0.0], np.float32)
        xyz = shell + motion
        if frame % 5 == 0 or frame == args.frames - 1:
            write_ply(points_root / f"f{frame:06d}.ply", xyz, base_rgb)
        for camera, w2c in enumerate(camera_w2c):
            view_dir = xyz @ w2c[:3, :3].T + w2c[:3, 3]
            projected = view_dir @ intrinsic.T
            uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
            valid = (view_dir[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < args.width) & (uv[:, 1] >= 0) & (uv[:, 1] < args.height)
            image = np.zeros((args.height, args.width, 3), np.uint8)
            yy, xx = np.mgrid[0:args.height, 0:args.width]
            image[..., 0] = np.clip(12 + yy * 0.20 + camera * 2, 0, 255).astype(np.uint8)
            image[..., 1] = np.clip(16 + xx * 0.14, 0, 255).astype(np.uint8)
            image[..., 2] = 24
            for point, colour in zip(uv[valid], base_rgb[valid]):
                cv2.circle(image, tuple(np.rint(point).astype(int)), 2, tuple(int(v) for v in colour[::-1]), -1, cv2.LINE_AA)
            directory = image_root / str(camera)
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / f"{frame:06d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"generated {args.frames} frames x {args.cameras} cameras at {root}")


if __name__ == "__main__":
    main()
