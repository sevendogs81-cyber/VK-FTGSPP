"""PLY input and challenge-compatible 4DGS output without torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ftgspp.native_runtime.model import NativeGaussians

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_point_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"invalid PLY: {path}")
        fmt, count, current = None, None, None
        properties: list[tuple[str, str]] = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"missing PLY end_header: {path}")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts or parts[0] == "comment":
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current = parts[1]
                if current == "vertex":
                    count = int(parts[2])
            elif parts[0] == "property" and current == "vertex":
                if parts[1] == "list":
                    raise ValueError("list-valued vertex properties are unsupported")
                properties.append((parts[2], parts[1]))
        if fmt is None or count is None:
            raise ValueError(f"incomplete PLY header: {path}")
        required = {"x", "y", "z", "red", "green", "blue"}
        if missing := required - {name for name, _ in properties}:
            raise ValueError(f"PLY {path} is missing {sorted(missing)}")
        if fmt == "ascii":
            matrix = np.loadtxt(stream, max_rows=count, ndmin=2)
            values = {name: matrix[:, index] for index, (name, _) in enumerate(properties)}
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            order = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, order + PLY_TYPES[kind]) for name, kind in properties])
            vertices = np.fromfile(stream, dtype=dtype, count=count)
            values = {name: vertices[name] for name, _ in properties}
        else:
            raise ValueError(f"unsupported PLY format: {fmt}")
    xyz = np.stack([values[axis] for axis in "xyz"], axis=1).astype(np.float32)
    rgb = np.stack([values[name] for name in ("red", "green", "blue")], axis=1).astype(np.uint8)
    return xyz, rgb


def export_4dgs(path: Path, model: NativeGaussians) -> Path:
    n = len(model)
    rest = model.sh_coefficients[:, 1:].transpose(0, 2, 1).reshape(n, -1)
    names = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{index}" for index in range(rest.shape[1])]
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]
    names += ["t", "t_scale", "motion_0", "motion_1", "motion_2"]
    vertices = np.empty(n, dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = model.means[:, index]
    vertices["opacity"] = model.opacity_logits[:, 0]
    for index in range(3):
        vertices[f"f_dc_{index}"] = model.sh_coefficients[:, 0, index]
    for index in range(rest.shape[1]):
        vertices[f"f_rest_{index}"] = rest[:, index]
    for index in range(3):
        vertices[f"scale_{index}"] = model.log_scales[:, index]
        vertices[f"motion_{index}"] = model.velocities[:, index]
    for index in range(4):
        vertices[f"rot_{index}"] = model.quaternions[:, index]
    vertices["t"] = model.times[:, 0]
    vertices["t_scale"] = model.temporal_log_scale()[:, 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", "comment exported from FTGSPP native Vulkan", f"element vertex {n}"]
    header.extend(f"property float {name}" for name in names)
    header.append("end_header")
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.tofile(stream)
    return path
