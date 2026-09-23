"""Deterministic camera paths: path seconds are independent of scene seconds."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .asset import camera, scaled_camera, atomic_json

PATH_SCHEMA = "ftgspp-camera-path-v1"


def validate_path(data):
    if data.get("schema") != PATH_SCHEMA:
        raise ValueError("Unsupported camera path schema")
    keys = data.get("keyframes", [])
    if not keys:
        raise ValueError("Camera path needs at least one keyframe")
    keys = [
        dict(path_time=float(k["path_time"]), camera=camera(k["camera"])) for k in keys
    ]
    ts = np.array([k["path_time"] for k in keys])
    if not np.isfinite(ts).all() or ts[0] < 0 or np.any(np.diff(ts) <= 0):
        raise ValueError("Path times must be non-negative and strictly increasing")
    return dict(schema=PATH_SCHEMA, keyframes=keys)


def from_state(state):
    return validate_path(
        dict(
            schema=PATH_SCHEMA,
            keyframes=[
                dict(path_time=t, camera=c)
                for t, c in zip(state["path_times"], state["bookmarks"], strict=True)
            ],
        )
    )


def load_path(path):
    return validate_path(json.loads(Path(path).read_text()))


def save_path(path, data):
    atomic_json(Path(path), validate_path(data))


class CameraPath:
    def __init__(self, data):
        self.data = validate_path(data)
        self.times = np.array([k["path_time"] for k in self.data["keyframes"]])
        self.cameras = [k["camera"] for k in self.data["keyframes"]]
        self.poses = np.array([np.linalg.inv(c["w2c"]) for c in self.cameras])
        self.slerp = (
            Slerp(self.times, Rotation.from_matrix(self.poses[:, :3, :3]))
            if len(self.times) > 1
            else None
        )

    @property
    def duration(self):
        return float(self.times[-1] - self.times[0])

    def sample(self, t, width=None, height=None):
        t = np.clip(float(t), self.times[0], self.times[-1])
        if self.slerp is None:
            return scaled_camera(self.cameras[0], width, height)
        i = int(
            np.clip(
                np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2
            )
        )
        f = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        # Use one output extent; interpolate normalized intrinsics after rescaling.
        width = width or self.cameras[0]["width"]
        height = height or round(
            width * self.cameras[0]["height"] / self.cameras[0]["width"]
        )
        a, b = (scaled_camera(c, width, height) for c in self.cameras[i : i + 2])
        pose = np.eye(4)
        pose[:3, :3] = self.slerp(t).as_matrix()
        pose[:3, 3] = (1 - f) * self.poses[i, :3, 3] + f * self.poses[i + 1, :3, 3]
        return camera(
            dict(
                a,
                w2c=np.linalg.inv(pose),
                K=(1 - f) * a["K"] + f * b["K"],
                timestamp=(1 - f) * a["timestamp"] + f * b["timestamp"],
                name="Path",
            )
        )
