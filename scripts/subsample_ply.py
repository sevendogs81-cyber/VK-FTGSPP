"""Deterministically subsample a binary/ascii point PLY for local staging."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ftgspp.native_runtime.ply import read_point_ply


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header", "",
    ]).encode("ascii")
    vertices = np.empty(len(xyz), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=18000)
    args = parser.parse_args()
    xyz, rgb = read_point_ply(args.input)
    if args.count <= 0:
        raise ValueError("count must be positive")
    if len(xyz) > args.count:
        # Evenly spread indices preserve the full spatial extent and avoid the
        # duplicate-heavy random sampling used by the old initialization path.
        indices = np.linspace(0, len(xyz) - 1, args.count, dtype=np.int64)
        xyz, rgb = xyz[indices], rgb[indices]
    write_ply(args.output, xyz, rgb)
    print(f"wrote {len(xyz)} points to {args.output}")


if __name__ == "__main__":
    main()
