"""NumPy/OpenCV SelfCap loader for the native Vulkan runtime.

This is the only training dataset path. It deliberately has no torch,
torchvision, pycolmap, or CUDA dependency.
"""

from __future__ import annotations

import json
import hashlib
import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pycolmap

from ftgspp.data.multiview import (
    load_time_offsets,
    resolve_camera_name as _camera_name,
    scan_image_grid,
    select_camera_ids as _selection,
)

class FileStorage:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self._storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not self._storage.isOpened():
            raise OSError(f"failed to open OpenCV calibration: {path}")

    def close(self) -> None:
        self._storage.release()

    def matrix(self, key: str) -> np.ndarray | None:
        value = self._storage.getNode(key).mat()
        return None if value is None else np.asarray(value)

    def number(self, key: str) -> float:
        return float(self._storage.getNode(key).real())

    def strings(self, key: str) -> list[str]:
        node = self._storage.getNode(key)
        result: list[str] = []
        for index in range(node.size()):
            value = node.at(index).string()
            if not value:
                value = str(int(node.at(index).real()))
            if value != "none":
                result.append(value)
        return result


@dataclass(frozen=True)
class Camera:
    name: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    w2c: np.ndarray
    height: int
    width: int


def read_cameras(intrinsics: Path, extrinsics: Path | None = None) -> dict[str, Camera]:
    if extrinsics is None:
        root = intrinsics
        intrinsics, extrinsics = root / "intri.yml", root / "extri.yml"
    intri, extri = FileStorage(intrinsics), FileStorage(extrinsics)
    try:
        result: dict[str, Camera] = {}
        for name in intri.strings("names"):
            k = intri.matrix(f"K_{name}")
            if k is None:
                raise ValueError(f"missing K_{name} in {intrinsics}")
            d = intri.matrix(f"D_{name}")
            if d is None:
                d = intri.matrix(f"dist_{name}")
            if d is None:
                d = np.zeros((5, 1), np.float64)
            d = np.pad(np.asarray(d, np.float64).reshape(-1), (0, max(0, 5-len(d))))[:5]
            t = extri.matrix(f"T_{name}")
            if t is None:
                raise ValueError(f"missing T_{name} in {extrinsics}")
            r = extri.matrix(f"Rot_{name}")
            if r is None:
                rvec = extri.matrix(f"R_{name}")
                if rvec is None:
                    raise ValueError(f"missing R_{name}/Rot_{name} in {extrinsics}")
                r = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = np.asarray(r, np.float32)
            w2c[:3, 3] = np.asarray(t, np.float32).reshape(3)
            result[name] = Camera(
                name=name,
                intrinsic=np.asarray(k, np.float64),
                distortion=d,
                w2c=w2c,
                height=int(intri.number(f"H_{name}")) or -1,
                width=int(intri.number(f"W_{name}")) or -1,
            )
        return result
    finally:
        intri.close()
        extri.close()


def _scan_images(image_root: Path, start: int, stop: int) -> tuple[list[int], list[int], list[list[Path]]]:
    return scan_image_grid(
        image_root, start=start, stop=stop, reject_empty=True,
    )


def route_camera_split(camera_id: int, train_ids: Sequence[int], eval_ids: Sequence[int]) -> str | None:
    """Return ``\"train\"``, ``\"eval\"``, or ``None`` if the camera is unassigned."""
    if camera_id in train_ids:
        return "train"
    if camera_id in eval_ids:
        return "eval"
    return None


def _camera_setup(config, cameras: Sequence[int]) -> tuple[dict[str, Camera], list[int], list[int], str]:
    root = Path(config.data.calibration_path)
    allow_positional = list(cameras) == list(range(len(cameras)))
    split = [root / "train_intri.yml", root / "train_extri.yml", root / "test_intri.yml", root / "test_extri.yml"]
    exists = [path.is_file() for path in split]
    full_pair = [root / "intri.yml", root / "extri.yml"]
    full_exists = [path.is_file() for path in full_pair]
    if any(exists) and not all(exists):
        raise FileNotFoundError("incomplete train/test calibration split: " + ", ".join(str(p) for p, ok in zip(split, exists) if not ok))
    if all(exists):
        train_named = read_cameras(split[0], split[1])
        test_named = read_cameras(split[2], split[3])
        overlap = sorted(set(train_named) & set(test_named))
        if overlap:
            raise ValueError(f"train/test calibration cameras overlap: {overlap}")
        if any(full_exists) and not all(full_exists):
            missing = next(
                path for path, exists in zip(full_pair, full_exists) if not exists
            )
            raise FileNotFoundError(f"incomplete full calibration pair; missing: {missing}")
        full = read_cameras(root) if all(full_exists) else {**train_named, **test_named}
        unknown_split_names = sorted(
            (set(train_named) | set(test_named)) - set(full)
        )
        if unknown_split_names:
            raise ValueError(
                "train/test calibration cameras missing from full "
                f"calibration: {unknown_split_names}"
            )
        if config.data.camera_split_source is not None and config.data.train_cameras is not None and config.data.eval_cameras is not None:
            train = _selection(config.data.train_cameras, cameras, "data.train_cameras")
            evaluate = _selection(config.data.eval_cameras, cameras, "data.eval_cameras")
            source = str(config.data.camera_split_source)
        else:
            resolved_names = {
                c: _camera_name(
                    full, c, allow_positional=allow_positional,
                )
                for c in cameras
            }
            missing_images = sorted(
                (set(train_named) | set(test_named)) - set(resolved_names.values())
            )
            if missing_images:
                raise ValueError(
                    "train/test calibration contains cameras without image "
                    f"directories: {missing_images}"
                )
            train = [
                c for c in cameras
                if resolved_names[c] in train_named
            ]
            evaluate = [
                c for c in cameras
                if resolved_names[c] in test_named
            ]
            source = "train/test calibration files"
    else:
        full = read_cameras(root)
        if config.data.eval_cameras is None:
            raise ValueError("data.eval_cameras must be set when split calibrations are absent; use [] for no hold-out")
        evaluate = _selection(config.data.eval_cameras, cameras, "data.eval_cameras")
        train = _selection(config.data.train_cameras, cameras, "data.train_cameras") if config.data.train_cameras is not None else [c for c in cameras if c not in set(evaluate)]
        source = "full calibration, no hold-out" if not evaluate else "full calibration plus config eval cameras"
    if not train or set(train) & set(evaluate):
        raise ValueError("camera split must contain training cameras and no overlap")
    assigned = set(train) | set(evaluate)
    discovered = set(int(camera) for camera in cameras)
    if assigned != discovered:
        raise ValueError(
            "camera split must assign every discovered camera exactly once: "
            f"unassigned={sorted(discovered - assigned)}, "
            f"unknown={sorted(assigned - discovered)}"
        )
    return full, train, evaluate, source


def _time_offsets(config, cameras: Sequence[int], image_root: Path) -> dict[int, float]:
    data = config.data
    roots: list[Path] = []
    for value in (
        getattr(data, "calibration_path", None),
        getattr(data, "root", None),
        getattr(data, "video_path", None),
        image_root,
        image_root.parent,
    ):
        if value is None:
            continue
        path = Path(value)
        if path not in roots:
            roots.append(path)
    result, _source = load_time_offsets(
        cameras, roots,
        strict=bool(getattr(data, "strict_time_offsets", True)),
    )
    return result


def _colmap_undistorted_geometry(
    height: int,
    width: int,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    *,
    blank_pixels: bool = False,
) -> tuple[int, int, np.ndarray]:
    """Match ``data.utils.undistort.compute_undistorted_geometry`` exactly."""

    k = np.asarray(intrinsic)
    d = np.pad(
        np.asarray(distortion).reshape(-1),
        (0, max(0, 5 - np.asarray(distortion).size)),
    )
    fx, fy, cx, cy = (
        float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]),
    )
    k1, k2, p1, p2 = (float(value) for value in d[:4])
    camera = pycolmap.Camera.create(
        0, pycolmap.CameraModelId.OPENCV, 0, width, height,
    )
    undistorted_camera = pycolmap.Camera.create(
        0, pycolmap.CameraModelId.PINHOLE, 0, width, height,
    )
    camera.params = [fx, fy, cx, cy, k1, k2, p1, p2]
    undistorted_camera.params = [fx, fy, cx, cy]

    # Torch arange + 0.5 in the reference path produces float32 points.
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    left = np.stack((np.full_like(ys, 0.5), ys + np.float32(0.5)), axis=-1)
    right = np.stack((
        np.full_like(ys, width - 0.5), ys + np.float32(0.5),
    ), axis=-1)
    top = np.stack((xs + np.float32(0.5), np.full_like(xs, 0.5)), axis=-1)
    bottom = np.stack((
        xs + np.float32(0.5), np.full_like(xs, height - 0.5),
    ), axis=-1)
    left, right, top, bottom = (
        undistorted_camera.img_from_cam(camera.cam_from_img(points))
        for points in (left, right, top, bottom)
    )

    min_scale_x = min(
        cx / (cx - left[:, 0].min()),
        (width - 0.5 - cx) / (right[:, 0].max() - cx),
    )
    min_scale_y = min(
        cy / (cy - top[:, 1].min()),
        (height - 0.5 - cy) / (bottom[:, 1].max() - cy),
    )
    max_scale_x = max(
        cx / (cx - left[:, 0].max()),
        (width - 0.5 - cx) / (right[:, 0].min() - cx),
    )
    max_scale_y = max(
        cy / (cy - top[:, 1].max()),
        (height - 0.5 - cy) / (bottom[:, 1].min() - cy),
    )
    scale_x = 1.0 / (
        min_scale_x * blank_pixels + max_scale_x * (1 - blank_pixels)
    )
    scale_y = 1.0 / (
        min_scale_y * blank_pixels + max_scale_y * (1 - blank_pixels)
    )
    target_width = int(max(1.0, scale_x * width))
    target_height = int(max(1.0, scale_y * height))
    target_intrinsic = np.asarray([
        [fx, 0.0, target_width / width * cx],
        [0.0, fy, target_height / height * cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return target_height, target_width, target_intrinsic


def _colmap_undistort_maps(
    height: int,
    width: int,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    *,
    blank_pixels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the same float32 pixel-center map as ``colmap_undistort``."""

    target_height, target_width, target_k = _colmap_undistorted_geometry(
        height, width, intrinsic, distortion, blank_pixels=blank_pixels,
    )
    k = np.asarray(intrinsic, dtype=np.float32)
    d = np.pad(
        np.asarray(distortion, dtype=np.float32).reshape(-1),
        (0, max(0, 5 - np.asarray(distortion).size)),
    )
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    k1, k2, p1, p2 = d[:4]
    x, y = np.meshgrid(
        np.arange(target_width, dtype=np.float32),
        np.arange(target_height, dtype=np.float32),
        indexing="xy",
    )
    u = (x + np.float32(0.5) - target_k[0, 2]) / fx
    v = (y + np.float32(0.5) - target_k[1, 2]) / fy
    u2 = u * u
    uv = u * v
    v2 = v * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2 * r2
    du = u * radial + np.float32(2.0) * p1 * uv + p2 * (
        r2 + np.float32(2.0) * u2
    )
    dv = v * radial + np.float32(2.0) * p2 * uv + p1 * (
        r2 + np.float32(2.0) * v2
    )
    map_x = (u + du) * fx + cx - np.float32(0.5)
    map_y = (v + dv) * fy + cy - np.float32(0.5)
    return (
        np.ascontiguousarray(map_x, dtype=np.float32),
        np.ascontiguousarray(map_y, dtype=np.float32),
    )


def _geometry(camera: Camera, sample: Path, scale: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(sample), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(sample)
    height, width = image.shape[:2]
    k = camera.intrinsic.copy()
    if camera.height > 0 and camera.width > 0 and (height != camera.height or width != camera.width):
        k[0] *= width / camera.width
        k[1] *= height / camera.height
        k[2, 2] = 1.0
    distortion = camera.distortion.copy()
    if len(distortion) > 4:
        distortion[4:] = 0.0  # current FTGSPP/PyCOLMAP path uses OPENCV k1,k2,p1,p2
    if np.any(distortion):
        undist_h, undist_w, target_k = _colmap_undistorted_geometry(
            height, width, k, distortion, blank_pixels=False,
        )
    else:
        undist_h, undist_w = height, width
        target_k = k.astype(np.float32)
    out_h, out_w = max(1, int(scale*undist_h)), max(1, int(scale*undist_w))
    target_k[0] *= np.float32(out_w / undist_w)
    target_k[1] *= np.float32(out_h / undist_h)
    target_k[2, 2] = 1.0
    return k.astype(np.float32), target_k.astype(np.float32), (out_h, out_w)


# Per-camera undistort maps are identical across frames; cache them so
# registration of N frames does not rebuild initUndistortRectifyMap N times.
_UndistortKey = tuple[int, int, int, str]
_UndistortMaps = dict[_UndistortKey, tuple[np.ndarray, np.ndarray]]
_UNDISTORT_MAPS: _UndistortMaps = {}
_UNDISTORT_MAPS_LOCK = threading.RLock()


def clear_undistort_cache() -> None:
    with _UNDISTORT_MAPS_LOCK:
        _UNDISTORT_MAPS.clear()


def _undistort_maps_for(
    view: "NativeView", source_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.sha256()
    for value in (
        view.source_intrinsic, view.intrinsic, view.distortion,
    ):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    height, width = (int(value) for value in source_size)
    key: _UndistortKey = (
        int(view.camera_id), height, width, digest.hexdigest(),
    )
    with _UNDISTORT_MAPS_LOCK:
        cached = _UNDISTORT_MAPS.get(key)
        if cached is not None:
            return cached
        map_x, map_y = _colmap_undistort_maps(
            height, width, view.source_intrinsic, view.distortion,
            blank_pixels=False,
        )
        _UNDISTORT_MAPS[key] = (map_x, map_y)
        return map_x, map_y


@dataclass(frozen=True)
class NativeView:
    path: Path
    frame: int
    camera_id: int
    timestamp: float
    w2c: np.ndarray
    source_intrinsic: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray
    size: tuple[int, int]

    def load_rgb(self) -> np.ndarray:
        """Decode RGB or RGBA without discarding an on-disk alpha mask.

        The historical method name is retained for callers, but a four-channel
        PNG is returned as RGBA so the native loss can consume its alpha.  JPEG
        and ordinary PNG inputs continue to return RGB.
        """

        image = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(self.path)
        if image.dtype == np.uint16:
            image = ((image.astype(np.uint32) + 128) // 257).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError(
                f"unsupported image dtype {image.dtype} for {self.path}; "
                "expected 8-bit or 16-bit integer pixels"
            )
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            rgb = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 2:
            gray = image[..., 0]
            rgb = np.stack((gray, gray, gray, image[..., 1]), axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            raise ValueError(
                f"unsupported image channel layout {image.shape} for {self.path}"
            )
        height, width = self.size
        if np.any(self.distortion):
            map_x, map_y = _undistort_maps_for(self, image.shape[:2])
            rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(
                f"decoded image must be HxWx3 or HxWx4, got {rgb.shape} for {self.path}"
            )
        if rgb.shape[2] == 3:
            rgba = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = 255
            return np.ascontiguousarray(rgba)
        return rgb


@dataclass
class NativeDataset:
    frames: list[int]
    cameras: list[int]
    train_cameras: list[int]
    eval_cameras: list[int]
    camera_split_source: str
    train_views: list[NativeView]
    eval_views: list[NativeView]
    fps: float
    source_root: Path | None = None
    time_offsets: dict[int, float] = field(default_factory=dict)
    _checkpoint_metadata_cache: dict[str, object] | None = field(
        default=None, init=False, repr=False,
    )

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    def view(self, frame: int, camera_id: int) -> NativeView:
        for view in (*self.train_views, *self.eval_views):
            if view.frame == frame and view.camera_id == camera_id:
                return view
        raise KeyError(f"no view for frame={frame}, camera={camera_id}")

    def camera_dense_mapping(self) -> dict[int, int]:
        """Return the stable sparse-camera-id to dense-color-row mapping."""

        camera_ids = sorted({int(value) for value in self.cameras})
        return {camera_id: index for index, camera_id in enumerate(camera_ids)}

    def validate_view_manifest(self) -> None:
        """Validate every registered view before creating Vulkan replicas.

        This deliberately avoids decoding the complete dataset.  It verifies
        existence/non-empty files, unique split/frame/camera keys, finite
        timestamps/calibration, positive sizes and complete configured camera
        coverage. The train/eval partition must exactly cover the discovered
        camera IDs. Decode/undistort errors are still surfaced lazily for the
        selected view, while missing files fail once during preflight instead
        of deadlocking a synchronous rank later.
        """

        if not self.frames:
            raise ValueError("dataset manifest contains no frames")
        if not self.train_views:
            raise ValueError("dataset manifest contains no training views")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("dataset FPS must be finite and positive")
        frame_ids = set(int(value) for value in self.frames)
        camera_ids = set(int(value) for value in self.cameras)
        if len(frame_ids) != len(self.frames):
            raise ValueError("dataset manifest contains duplicate frame IDs")
        if len(camera_ids) != len(self.cameras):
            raise ValueError("dataset manifest contains duplicate camera IDs")
        if set(int(value) for value in self.time_offsets) != camera_ids:
            raise ValueError(
                "dataset time-offset keys must exactly cover all camera IDs"
            )
        if not np.isfinite(np.asarray(
            list(self.time_offsets.values()), dtype=np.float64,
        )).all():
            raise ValueError("dataset time offsets must be finite")
        expected_train = set(int(value) for value in self.train_cameras)
        expected_eval = set(int(value) for value in self.eval_cameras)
        if len(expected_train) != len(self.train_cameras):
            raise ValueError("dataset manifest train camera split has duplicates")
        if len(expected_eval) != len(self.eval_cameras):
            raise ValueError("dataset manifest eval camera split has duplicates")
        if expected_train & expected_eval:
            raise ValueError("dataset manifest train/eval camera sets overlap")
        assigned = expected_train | expected_eval
        if not expected_train or assigned != camera_ids:
            raise ValueError(
                "dataset manifest camera split must exactly cover discovered "
                f"cameras: unassigned={sorted(camera_ids - assigned)}, "
                f"unknown={sorted(assigned - camera_ids)}"
            )

        seen: set[tuple[str, int, int]] = set()
        observed = {"train": set(), "eval": set()}
        for split, views in (("train", self.train_views), ("eval", self.eval_views)):
            allowed = expected_train if split == "train" else expected_eval
            for index, view in enumerate(views):
                key = (split, int(view.frame), int(view.camera_id))
                if key in seen:
                    raise ValueError(f"duplicate dataset view manifest key: {key}")
                seen.add(key)
                observed[split].add(int(view.camera_id))
                if int(view.frame) not in frame_ids:
                    raise ValueError(f"{split} view {index} has unknown frame {view.frame}")
                if int(view.camera_id) not in allowed:
                    raise ValueError(
                        f"{split} view {index} has camera {view.camera_id} "
                        "outside its configured split"
                    )
                symlink_fast_path = (
                    os.environ.get("FTGSPP_SYMLINK_PREFLIGHT") == "1"
                    and view.path.is_symlink()
                )
                if not symlink_fast_path and not view.path.is_file():
                    raise FileNotFoundError(view.path)
                # A local symlink's lstat avoids an expensive remote NAS stat;
                # image decoding still follows the link when the view is used.
                file_size = (
                    view.path.lstat().st_size if symlink_fast_path
                    else view.path.stat().st_size
                )
                if file_size <= 0:
                    raise ValueError(f"dataset view is empty: {view.path}")
                height, width = (int(value) for value in view.size)
                if height <= 0 or width <= 0:
                    raise ValueError(f"dataset view has invalid size {view.size}: {view.path}")
                if not np.isfinite(float(view.timestamp)):
                    raise ValueError(f"dataset view has non-finite timestamp: {view.path}")
                arrays = {
                    "w2c": (view.w2c, (4, 4)),
                    "source_intrinsic": (view.source_intrinsic, (3, 3)),
                    "intrinsic": (view.intrinsic, (3, 3)),
                }
                for label, (value, shape) in arrays.items():
                    array = np.asarray(value)
                    if array.shape != shape or not np.isfinite(array).all():
                        raise ValueError(
                            f"dataset view {view.path} has invalid {label}"
                        )
                if not np.isfinite(np.asarray(view.distortion)).all():
                    raise ValueError(
                        f"dataset view {view.path} has invalid distortion"
                    )
        if observed["train"] != expected_train:
            raise ValueError(
                "training view manifest camera coverage differs from split: "
                f"observed={sorted(observed['train'])}, expected={sorted(expected_train)}"
            )
        if observed["eval"] != expected_eval:
            raise ValueError(
                "evaluation view manifest camera coverage differs from split: "
                f"observed={sorted(observed['eval'])}, expected={sorted(expected_eval)}"
            )
        expected_keys = {
            ("train", frame, camera)
            for frame in frame_ids for camera in expected_train
        } | {
            ("eval", frame, camera)
            for frame in frame_ids for camera in expected_eval
        }
        if seen != expected_keys:
            missing = sorted(expected_keys - seen)[:8]
            extra = sorted(seen - expected_keys)[:8]
            raise ValueError(
                "dataset view manifest is not the complete frame/camera grid: "
                f"missing={missing}, extra={extra}"
            )

    @staticmethod
    def _array_fingerprint(value: np.ndarray) -> str:
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray((array.ndim, *array.shape), np.int64).tobytes())
        digest.update(array.tobytes())
        return digest.hexdigest()

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return stable identity plus synchronization diagnostics for resume.

        Offset range and per-logical-frame adjusted timestamp spread are kept
        as compact summaries, so a resumed run can audit camera synchronization
        without re-reading every view record from the checkpoint manifest.
        """

        if self._checkpoint_metadata_cache is not None:
            return deepcopy(self._checkpoint_metadata_cache)

        root = self.source_root.resolve() if self.source_root is not None else None

        def record(split: str, index: int, view: NativeView) -> dict[str, object]:
            # Resolving thousands of local links to an NFS image tree can turn
            # metadata preflight into an hours-long operation.  Keep the link
            # path under the explicit opt-in fast staging mode; decode still
            # follows it later when the view is sampled.
            if (
                os.environ.get("FTGSPP_SYMLINK_PREFLIGHT") == "1"
                and view.path.is_symlink()
            ):
                path = Path(os.path.abspath(view.path))
            else:
                path = view.path.resolve()
            try:
                logical_path = path.relative_to(root).as_posix() if root else path.as_posix()
            except ValueError:
                logical_path = path.as_posix()
            return {
                "split": split,
                "index": int(index),
                "path": logical_path,
                "bytes": int(
                    path.lstat().st_size
                    if os.environ.get("FTGSPP_SYMLINK_PREFLIGHT") == "1"
                    and path.is_symlink()
                    else path.stat().st_size
                ),
                "frame": int(view.frame),
                "camera_id": int(view.camera_id),
                "timestamp": float(view.timestamp),
                "size": [int(view.size[0]), int(view.size[1])],
                "w2c": self._array_fingerprint(view.w2c),
                "source_intrinsic": self._array_fingerprint(view.source_intrinsic),
                "intrinsic": self._array_fingerprint(view.intrinsic),
                "distortion": self._array_fingerprint(view.distortion),
            }

        records = [
            record("train", index, view)
            for index, view in enumerate(self.train_views)
        ] + [
            record("eval", index, view)
            for index, view in enumerate(self.eval_views)
        ]
        encoded = json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        mapping = self.camera_dense_mapping()
        offset_values = np.asarray(
            [self.time_offsets[camera] for camera in sorted(self.time_offsets)],
            dtype=np.float64,
        )
        if offset_values.size == 0 or not np.isfinite(offset_values).all():
            raise ValueError("cannot checkpoint empty/non-finite camera time offsets")
        timestamps_by_frame: dict[int, list[float]] = {
            int(frame): [] for frame in self.frames
        }
        for view in (*self.train_views, *self.eval_views):
            timestamps_by_frame.setdefault(int(view.frame), []).append(
                float(view.timestamp)
            )
        empty_timestamp_frames = sorted(
            frame for frame, values in timestamps_by_frame.items() if not values
        )
        if empty_timestamp_frames:
            raise ValueError(
                "cannot checkpoint logical frames without adjusted timestamps: "
                f"{empty_timestamp_frames[:8]}"
            )
        timestamp_spreads = np.asarray([
            max(values) - min(values)
            for _, values in sorted(timestamps_by_frame.items())
        ], dtype=np.float64)
        metadata: dict[str, object] = {
            "fingerprint": hashlib.sha256(encoded).hexdigest(),
            "manifest_entries": len(records),
            "train_views": len(self.train_views),
            "eval_views": len(self.eval_views),
            "frames": [int(value) for value in self.frames],
            "fps": float(self.fps),
            "time_offsets": {
                str(key): float(value)
                for key, value in sorted(self.time_offsets.items())
            },
            "time_offset_range_seconds": {
                "minimum": float(offset_values.min()),
                "maximum": float(offset_values.max()),
                "span": float(offset_values.max() - offset_values.min()),
            },
            "same_frame_adjusted_timestamp_spread_seconds": {
                "definition": (
                    "max(timestamp)-min(timestamp) across all cameras for "
                    "one logical frame"
                ),
                "minimum": float(timestamp_spreads.min()),
                "maximum": float(timestamp_spreads.max()),
                "mean": float(timestamp_spreads.mean()),
            },
            "timestamp_convention": "logical_frame/fps-time_offset",
            "camera_dense_mapping": {
                str(key): int(value) for key, value in sorted(mapping.items())
            },
            "train_cameras": [int(value) for value in self.train_cameras],
            "eval_cameras": [int(value) for value in self.eval_cameras],
            "camera_split_source": str(self.camera_split_source),
            "path_policy": "relative_to_extracted_path_when_possible",
        }
        # NativeDataset/view records are immutable after construction. Keep the
        # expensive all-view stat/calibration hash out of every checkpoint and
        # hand callers a copy so metadata mutation cannot poison the cache.
        self._checkpoint_metadata_cache = metadata
        return deepcopy(metadata)

    @classmethod
    def from_config(cls, config) -> "NativeDataset":
        root = Path(config.data.extracted_path)
        frames, camera_ids, paths = _scan_images(root, config.data.frames.start, config.data.frames.stop)
        fps = float(config.data.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"dataset FPS must be finite and positive, got {fps!r}")
        cameras, train_ids, eval_ids, source = _camera_setup(config, camera_ids)
        offsets = _time_offsets(config, camera_ids, root)
        geometries: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
        camera_values: dict[int, Camera] = {}
        allow_positional = list(camera_ids) == list(range(len(camera_ids)))
        for index, camera_id in enumerate(camera_ids):
            camera = cameras[_camera_name(
                cameras, camera_id, allow_positional=allow_positional,
            )]
            camera_values[camera_id] = camera
            geometries[camera_id] = _geometry(camera, paths[0][index], float(config.data.scale))

        train_views: list[NativeView] = []
        eval_views: list[NativeView] = []
        train_set = set(train_ids)
        eval_set = set(eval_ids)
        for frame_index, frame in enumerate(frames):
            for camera_index, camera_id in enumerate(camera_ids):
                camera = camera_values[camera_id]
                source_k, target_k, size = geometries[camera_id]
                view = NativeView(
                    path=paths[frame_index][camera_index], frame=frame,
                    camera_id=camera_id,
                    # Match Main's float32 tensor expression
                    # ``frame / fps - time_offsets`` before shader upload.
                    timestamp=float(
                        np.float32(frame / fps)
                        - np.float32(offsets[camera_id])
                    ),
                    w2c=np.ascontiguousarray(camera.w2c, dtype=np.float32),
                    source_intrinsic=source_k, intrinsic=target_k,
                    distortion=np.ascontiguousarray(
                        np.concatenate([camera.distortion[:4], np.zeros(max(0, len(camera.distortion)-4))]),
                        dtype=np.float32,
                    ), size=size,
                )
                split = route_camera_split(camera_id, train_set, eval_set)
                if split == "train":
                    train_views.append(view)
                elif split == "eval":
                    eval_views.append(view)
                else:  # _camera_setup requires an exact discovered-camera split.
                    raise RuntimeError(
                        f"camera {camera_id} escaped validated train/eval split"
                    )
        dataset = cls(
            frames, list(camera_ids), train_ids, eval_ids, source,
            train_views, eval_views, fps,
            source_root=root, time_offsets=dict(offsets),
        )
        dataset.validate_view_manifest()
        dataset.checkpoint_metadata()
        return dataset
