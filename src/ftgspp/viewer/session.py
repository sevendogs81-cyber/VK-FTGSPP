"""Validated JSON sessions, bound to a model hash."""

import json
import math
from pathlib import Path
import numpy as np
from .asset import atomic_json, camera
from .path import from_state


def validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("Session state must be an object")
    state = dict(state)
    if "camera" in state:
        state["camera"] = camera(state["camera"])
    for name in (
        "playing", "loop", "snap", "lock_camera", "checker", "transparent",
        "interactive",
    ):
        if name in state and not isinstance(state[name], bool):
            raise ValueError(f"Invalid session {name}")
    if "selected_camera" in state:
        selected = state["selected_camera"]
        if type(selected) is not int or selected < -1:
            raise ValueError("Invalid session selected_camera")
    limits = {
        "time": (-1e12, 1e12),
        "speed": (0.01, 100),
        "exposure": (-20, 20),
        "target_fps": (1, 1000),
        "sh_degree": (0, 3),
        "mode": (0, 2),
        "marginal_threshold": (0, 1),
    }
    for name, (low, high) in limits.items():
        if name in state:
            if isinstance(state[name], (bool, np.bool_)):
                raise ValueError(f"Invalid session {name}")
            n = float(state[name])
            if (
                not math.isfinite(n)
                or not low <= n <= high
                or name in ("sh_degree", "mode")
                and int(n) != n
            ):
                raise ValueError(f"Invalid session {name}")
            state[name] = int(n) if name in ("sh_degree", "mode") else n
    for key in ("background", "world_up"):
        if key in state:
            a = np.array(state[key], dtype=float, copy=True)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid session {key}")
            if key == "background" and (a.min() < 0 or a.max() > 1):
                raise ValueError("Background must lie in [0,1]")
            if key == "world_up":
                norm = np.linalg.norm(a)
                if norm < 1e-6:
                    raise ValueError("World up cannot be zero")
                a /= norm
            state[key] = a.tolist()
    bookmarks = state.get("bookmarks", [])
    path_times = state.get("path_times", [])
    if not isinstance(bookmarks, list) or not isinstance(path_times, list):
        raise ValueError("Bookmarks and path times must be arrays")
    if bookmarks:
        if not path_times:
            raise ValueError("Bookmarks without path times")
        path = from_state(state)
        state["bookmarks"] = [k["camera"] for k in path["keyframes"]]
        state["path_times"] = [k["path_time"] for k in path["keyframes"]]
    elif path_times:
        raise ValueError("Path times without bookmarks")
    return state


def save_session(path, asset, state):
    atomic_json(
        Path(path),
        dict(
            schema="ftgspp-viewer-session-v1",
            model_sha256=asset.sha256,
            model=str(asset.model_path),
            state=validate_state(state),
        ),
    )


def load_session(path, asset):
    data = json.loads(Path(path).read_text())
    if (
        not isinstance(data, dict)
        or data.get("schema") != "ftgspp-viewer-session-v1"
        or data.get("model_sha256") != asset.sha256
    ):
        raise ValueError("Session schema/model mismatch")
    # Explicitly pause on restore; opening a session must be deterministic.
    return dict(validate_state(data["state"]), playing=False)
