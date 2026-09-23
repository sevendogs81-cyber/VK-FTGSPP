"""Portable viewer assets. No training preflight, image cache, or model downloads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ftgspp.io.hashing import file_sha256
from ftgspp.io.model import NativeGaussians

SCHEMA = "ftgspp-viewer-v1"


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, default=json_value, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        if overwrite:
            os.replace(name, path)
        else:
            os.link(name, path)  # atomic no-clobber publication
    finally:
        Path(name).unlink(missing_ok=True)


def camera(value):
    c = dict(value)
    for key in ("width", "height"):
        n = int(c[key])
        if n != c[key] or not 1 <= n <= 16384:
            raise ValueError(f"Invalid camera {key}: {c[key]}")
        c[key] = n
    k, w = (np.asarray(c[key], dtype=np.float32) for key in ("K", "w2c"))
    if (
        k.shape != (3, 3)
        or w.shape != (4, 4)
        or not np.isfinite(k).all()
        or not np.isfinite(w).all()
    ):
        raise ValueError("Camera requires finite K[3,3], w2c[4,4]")
    if (
        k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or abs(k[0, 1]) > 1e-5
        or abs(k[1, 0]) > 1e-5
    ):
        raise ValueError("Viewer requires positive pinhole intrinsics without skew")
    if (
        not np.allclose(w[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(w[:3, :3] @ w[:3, :3].T, np.eye(3), atol=2e-3)
        or not np.isclose(np.linalg.det(w[:3, :3]), 1, atol=2e-3)
    ):
        raise ValueError(
            "w2c must be a rigid OpenCV pose (+X right, +Y down, +Z forward)"
        )
    c.update(
        K=k,
        w2c=w,
        timestamp=float(c.get("timestamp", 0)),
        name=str(c.get("name", "Camera")),
    )
    if not math.isfinite(c["timestamp"]):
        raise ValueError("Non-finite camera timestamp")
    return c


def camera_hash(cameras):
    return hashlib.sha256(
        json.dumps(
            cameras,
            default=json_value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def fit_camera_extent(c, width, height):
    """Fit the original camera aspect into a pixel box (nearest whole pixels)."""
    # Browser extents have already been rounded. Preserve them on repeated
    # fitting so wide/portrait frames don't lose pixels at each protocol layer.
    if (abs(height - width * c["height"] / c["width"]) <= 0.5 + 1e-9
            or abs(width - height * c["width"] / c["height"]) <= 0.5 + 1e-9):
        return width, height
    scale = min(width / c["width"], height / c["height"])
    return (max(1, int(c["width"] * scale + 0.5)),
            max(1, int(c["height"] * scale + 0.5)))


def scaled_camera(c, width=None, height=None):
    c = camera(c)
    if width is None and height is None:
        return c
    width = width or round(height * c["width"] / c["height"])
    height = height or round(width * c["height"] / c["width"])
    k = c["K"].copy()
    k[0] *= width / c["width"]
    k[1] *= height / c["height"]
    return camera(dict(c, K=k, width=width, height=height))


def inferred_metadata(model):
    # Bounded deterministic sample; duration is a temporal scale, not video length.
    ids = np.linspace(0, len(model) - 1, min(len(model), 100000), dtype=int)
    points = model.means[ids]
    if len(points):
        lo, hi = np.percentile(points, [1, 99], axis=0)
        center = (lo + hi) / 2
        radius = max(float(np.linalg.norm(hi - lo) / 2), 0.1)
        times = np.unique(model.times[ids, 0]).tolist()
        start, end = float(times[0]), float(times[-1])
    else:
        center, radius, times, start, end = np.zeros(3), 1.0, [], 0.0, 0.0
    w = np.eye(4, dtype=np.float32)
    w[:3, 3] = -(center - np.array([0, 0, radius * 3]))
    initial = camera(
        dict(
            width=1280,
            height=720,
            K=[[900, 0, 640], [0, 900, 360], [0, 0, 1]],
            w2c=w,
            timestamp=start,
            name="Fit (inferred)",
        )
    )
    return dict(
        initial_camera=initial,
        cameras=[],
        sample_times=times,
        center=center,
        radius=radius,
        time_start=start,
        time_end=end,
        fps=30.0,
        sh_degree=3,
        marginal_threshold=0.05,
        world_up=[0, -1, 0],
        inferred=["camera", "time_range", "fps", "sh_degree", "marginal_threshold"],
    )


def validate_metadata(meta):
    m = dict(meta)
    m["initial_camera"] = camera(m["initial_camera"])
    m["cameras"] = [camera(c) for c in m.get("cameras", [])]
    for key in ("time_start", "time_end", "fps", "radius", "marginal_threshold"):
        m[key] = float(m[key])
        if not math.isfinite(m[key]):
            raise ValueError(f"Non-finite {key}")
    if (
        m["time_end"] < m["time_start"]
        or m["fps"] <= 0
        or m["radius"] <= 0
        or not 0 <= m["marginal_threshold"] <= 1
    ):
        raise ValueError("Invalid time range / FPS / radius / marginal threshold")
    if int(m["sh_degree"]) != m["sh_degree"] or not 0 <= m["sh_degree"] <= 3:
        raise ValueError("SH degree must be 0..3")
    for key in ("center", "world_up"):
        a = np.asarray(m[key], dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid {key}")
        if key == "world_up":
            if np.linalg.norm(a) < 1e-6:
                raise ValueError("world_up cannot be zero")
            a /= np.linalg.norm(a)
        m[key] = a.tolist()
    t = np.asarray(m.get("sample_times", []), dtype=np.float32)
    if t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError("Invalid sample times")
    m["sample_times"] = np.unique(t).tolist()
    return m


@dataclass
class Asset:
    model_path: Path
    model: NativeGaussians | None
    metadata: dict
    sha256: str


def load_asset(
    source: Path, *, time_range=None, fps=None, sh_degree=None, marginal_threshold=None
):
    source = Path(source).expanduser().resolve()
    model_path = source / "gaussians.npz" if source.is_dir() else source
    if model_path.suffix.lower() != ".npz":
        raise ValueError("Use native gaussians.npz; PLY omits dynamic gate parameters")
    with np.load(model_path, allow_pickle=False) as z:
        model = NativeGaussians.from_mapping(z)
    if math.isnan(model.max_duration) or model.max_duration <= 0:
        raise ValueError("max_duration must be positive or +infinity")
    digest = file_sha256(model_path)
    meta = inferred_metadata(model)
    manifest_path = model_path.parent / "viewer_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported viewer manifest schema")
        if manifest["model"]["sha256"] != digest:
            raise ValueError(
                "Viewer manifest/model hash mismatch; regenerate metadata for this model"
            )
        if manifest["camera_sha256"] != camera_hash(manifest["metadata"]["cameras"]):
            raise ValueError("Viewer camera metadata hash mismatch")
        meta.update(manifest["metadata"])
    elif source.is_dir() and (source / "config.resolved.yaml").exists():
        try:
            meta.update(legacy_metadata(source))
        except (ValueError, KeyError, OSError) as exc:
            warnings.warn(
                f"Cannot recover calibrated cameras: {exc}. Using inferred view; provide a viewer manifest for faithful capture views."
            )
    if time_range is not None:
        meta["time_start"], meta["time_end"] = time_range
    for key, value in (
        ("fps", fps),
        ("sh_degree", sh_degree),
        ("marginal_threshold", marginal_threshold),
    ):
        if value is not None:
            meta[key] = value
    meta["label"] = f"{model_path.parent.name} / {len(model):,} Gaussians"
    meta = validate_metadata(meta)
    if set(meta.get("inferred", [])) - {"bounds"}:
        warnings.warn("Viewer inferred defaults: " + ", ".join(meta["inferred"]))
    return Asset(model_path, model, meta, digest)


def legacy_metadata(run):
    """Read existing calibration/cache only; decode one size reference per camera."""
    from omegaconf import OmegaConf
    from ftgspp.data.scene import (
        _scan_images,
        _camera_setup,
        _camera_name,
        _geometry,
        _load_da3_camera_manifest,
        _time_offsets,
    )
    from ftgspp.data.multiview import load_image_manifest

    cfg = OmegaConf.load(run / "config.resolved.yaml")
    cfg = cfg.get("config", cfg)
    root = Path(cfg.data.extracted_path)
    explicit = None
    if cfg.data.get("view_manifest"):
        frames, ids, paths, explicit, _ = load_image_manifest(
            Path(cfg.data.view_manifest),
            start=cfg.data.frames.start,
            stop=cfg.data.frames.stop,
            reject_empty=True,
            return_hashes=True,
        )
    else:
        frames, ids, paths = _scan_images(
            root, cfg.data.frames.start, cfg.data.frames.stop
        )
    names = {int(cid): paths[0][i].parent.name for i, cid in enumerate(ids)}
    offsets = (
        {i: 0 for i in ids}
        if explicit is not None
        else _time_offsets(cfg, ids, root, names)
    )
    calibration_root = Path(cfg.data.calibration_path)
    calibrated = any(
        (calibration_root / name).is_file()
        for name in (
            "intri.yml",
            "extri.yml",
            "train_intri.yml",
            "train_extri.yml",
            "test_intri.yml",
            "test_extri.yml",
        )
    )
    cache = None if calibrated else _load_da3_camera_manifest(cfg, ids)
    if not calibrated and not cache:
        raise ValueError(
            "No calibrated rig or verified completed DA3 camera cache; placeholder cameras are not viewer calibration"
        )
    named = _camera_setup(cfg, ids)[0] if calibrated else None
    cs, samples = [], []
    for i, cid in enumerate(ids):
        if cache:
            w, k, size = cache[cid]
        else:
            original = named[
                _camera_name(named, cid, allow_positional=ids == list(range(len(ids))))
            ]
            _, k, size = _geometry(original, paths[0][i], float(cfg.data.scale))
            w = np.eye(4, dtype=np.float32)
            w[:3] = original.w2c[:3]
        ts = [
            float(explicit[(frame, cid)])
            if explicit is not None
            else float(np.float32(frame / cfg.data.fps) - np.float32(offsets[cid]))
            for frame in frames
        ]
        samples.extend(ts)
        cs.append(
            camera(
                dict(
                    width=size[1],
                    height=size[0],
                    K=k,
                    w2c=w,
                    timestamp=ts[0],
                    name=str(cid),
                )
            )
        )
    return dict(
        cameras=cs,
        initial_camera=cs[0],
        sample_times=sorted(set(samples)),
        time_start=min(samples),
        time_end=max(samples),
        fps=float(cfg.data.fps),
        sh_degree=int(cfg.init.sh_degree),
        marginal_threshold=float(cfg.model.marginal_threshold),
        inferred=["bounds"],
        camera_source="existing DA3 cache" if cache else "existing calibration",
    )
