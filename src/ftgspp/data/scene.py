"""NumPy/OpenCV SelfCap loader for the native Vulkan runtime.

This is the only training dataset path. It deliberately has no torch,
torchvision, or CUDA dependency; pycolmap is used only for calibrated
undistortion geometry.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pycolmap

from ftgspp.data.multiview import (
    load_image_manifest,
    load_time_offsets,
    resolve_camera_name as _camera_name,
    scan_image_grid,
    select_camera_ids as _selection,
)
from ftgspp.io.init_cache import resolve_published_cache_root, published_cache_candidates

logger = logging.getLogger("ftgspp.native")


def _decode_unit_map(path: Path, *, label: str) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    if value.ndim == 3:
        if value.shape[2] == 1:
            value = value[..., 0]
        elif value.shape[2] in (3, 4):
            # Masks are scalar data. Reject colored disagreement instead of
            # silently selecting a channel from a mislabeled RGB image.
            channels = value[..., :3]
            if not np.array_equal(channels[..., 0], channels[..., 1]) or not np.array_equal(
                channels[..., 0], channels[..., 2]
            ):
                raise ValueError(f"{label} must be grayscale: {path}")
            value = channels[..., 0]
        else:
            raise ValueError(f"unsupported {label} layout {value.shape}: {path}")
    if value.ndim != 2:
        raise ValueError(f"{label} must be a single-channel image: {path}")
    if value.dtype == np.uint8:
        result = value.astype(np.float32) * np.float32(1.0 / 255.0)
    elif value.dtype == np.uint16:
        result = value.astype(np.float32) * np.float32(1.0 / 65535.0)
    elif np.issubdtype(value.dtype, np.floating):
        result = value.astype(np.float32)
    else:
        raise ValueError(
            f"unsupported {label} dtype {value.dtype}: {path}; expected uint8/uint16/float"
        )
    if not np.isfinite(result).all() or result.min() < 0.0 or result.max() > 1.0:
        raise ValueError(f"{label} values must be finite and in [0, 1]: {path}")
    return result


def _transform_scalar_map(
    value: np.ndarray, view: "View", *, interpolation: int,
) -> np.ndarray:
    if np.any(view.distortion):
        map_x, map_y = _undistort_maps_for(view, value.shape[:2])
        value = cv2.remap(
            value, map_x, map_y, interpolation,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    height, width = view.size
    if value.shape != (height, width):
        value = cv2.resize(value, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(value, dtype=np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


VIEW_HASH_CACHE_SCHEMA = "ftgspp-view-sha256-v1"
VIEW_HASH_CACHE_RELATIVE = Path(".ftgspp") / "view_sha256_v1.json"
DEFAULT_HASH_WORKERS = 8
_VIEW_HASH_LOCK = threading.Lock()


def _hash_worker_count() -> int:
    raw = os.environ.get("FTGSPP_HASH_WORKERS")
    if raw is None or not str(raw).strip():
        return DEFAULT_HASH_WORKERS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_HASH_WORKERS


def _content_identity(path: Path) -> tuple[int, int, int]:
    """Cheap followed-link identity used only to validate a digest cache hit."""

    stat = path.stat()
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    ctime_ns = int(getattr(stat, "st_ctime_ns", int(stat.st_ctime * 1_000_000_000)))
    return int(stat.st_size), mtime_ns, ctime_ns


def _content_probe(path: Path, size: int, *, chunk_size: int = 4096) -> str:
    """Hash bounded head/tail bytes for filesystems with coarse stat clocks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(chunk_size))
        if size > chunk_size:
            stream.seek(max(chunk_size, size-chunk_size))
            digest.update(stream.read(chunk_size))
    return digest.hexdigest()


def _view_hash_cache_root(source_root: Path | None) -> Path | None:
    """Sidecar next to the scene, never inside the camera-grid folder.

    ``data.extracted_path`` is typically ``<scene>/images``. Resolving a
    foreground symlink would also drop the cache into the shared RGB tree and
    make the next directory scan treat ``.ftgspp`` as a camera.
    """

    if source_root is None:
        return None
    root = Path(source_root)
    if root.name == "images":
        return root.parent
    return root


def _view_hash_cache_path(root: Path) -> Path:
    return Path(root) / VIEW_HASH_CACHE_RELATIVE


def _load_view_hash_cache(root: Path | None) -> dict[str, dict[str, object]]:
    if root is None:
        return {}
    path = _view_hash_cache_path(root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("schema") != VIEW_HASH_CACHE_SCHEMA:
        return {}
    entries = payload.get("entries")
    return dict(entries) if isinstance(entries, dict) else {}


def _store_view_hash_cache(
    root: Path | None, entries: dict[str, dict[str, object]],
) -> None:
    if root is None:
        return
    path = _view_hash_cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": VIEW_HASH_CACHE_SCHEMA, "entries": entries}
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _file_sha256_many(
    paths: Sequence[Path | None],
    *,
    cache_root: Path | None,
) -> dict[str, str]:
    """Map ``Path.as_posix()`` to the same digest ``_file_sha256`` would return.

    Hits use path/stat identity plus a bounded content probe, so coarse-clock
    filesystems cannot reuse a stale digest after a same-size rewrite. Misses
    are fully hashed concurrently; the returned digest is still SHA-256 of all
    file contents.
    """

    unique: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        return {}

    with _VIEW_HASH_LOCK:
        cache = _load_view_hash_cache(cache_root)
        results: dict[str, str] = {}
        missing: list[Path] = []
        for path in unique:
            cache_key = path.as_posix()
            try:
                size, mtime_ns, ctime_ns = _content_identity(path)
            except OSError:
                missing.append(path)
                continue
            entry = cache.get(cache_key)
            digest = entry.get("sha256") if isinstance(entry, dict) else None
            probe = (
                _content_probe(path, size)
                if isinstance(entry, dict)
                and int(entry.get("size", -1)) == size
                and int(entry.get("mtime_ns", -1)) == mtime_ns
                and int(entry.get("ctime_ns", -1)) == ctime_ns
                else None
            )
            if (
                isinstance(entry, dict)
                and int(entry.get("size", -1)) == size
                and int(entry.get("mtime_ns", -1)) == mtime_ns
                and int(entry.get("ctime_ns", -1)) == ctime_ns
                and entry.get("content_probe") == probe
                and isinstance(digest, str)
                and len(digest) == 64
            ):
                results[cache_key] = digest
            else:
                missing.append(path)

    computed: dict[str, str] = {}
    workers = 0
    if missing:
        workers = min(_hash_worker_count(), len(missing))
        if workers == 1:
            computed = {path.as_posix(): _file_sha256(path) for path in missing}
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_file_sha256, path): path for path in missing}
                for future, path in futures.items():
                    computed[path.as_posix()] = future.result()
    logger.info(
        "view content hashes: cached=%d computed=%d workers=%d",
        len(unique) - len(missing), len(missing), workers,
    )

    results.update(computed)
    if computed and cache_root is not None:
        with _VIEW_HASH_LOCK:
            cache = _load_view_hash_cache(cache_root)
            for path in missing:
                cache_key = path.as_posix()
                size, mtime_ns, ctime_ns = _content_identity(path)
                cache[cache_key] = {
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "ctime_ns": ctime_ns,
                    "content_probe": _content_probe(path, size),
                    "sha256": results[cache_key],
                }
            _store_view_hash_cache(cache_root, cache)
    return results


def _identity_rgb_path(view: View) -> Path:
    if (
        os.environ.get("FTGSPP_SYMLINK_PREFLIGHT") == "1"
        and view.path.is_symlink()
    ):
        return Path(os.path.abspath(view.path))
    return view.path.resolve()


@dataclass(frozen=True)
class _MaskManifest:
    rows: dict[tuple[int, int], dict[str, Any]]
    strict: bool = False
    schema: str | None = None


def _declared_sha256(value: object, *, field_name: str, manifest: Path) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA256 in {manifest}")
    return digest


def _read_mask_manifest(path: Path | None, base: Path) -> _MaskManifest:
    if path is None:
        return _MaskManifest({})
    if not path.is_file():
        raise FileNotFoundError(
            f"foreground mask manifest not found: {path}; generate it with "
            "'ftgspp prepare masks --images <scene>/images "
            "--output <scene>/masks/birefnet_v1' or configure an existing "
            "data.masks.root/source=alpha"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema = str(raw.get("schema")) if isinstance(raw, dict) and raw.get("schema") else None
    strict = schema == "ftgspp-birefnet-mask-v1"
    if strict:
        marker_value = raw.get("completion_marker")
        if not isinstance(marker_value, str) or not marker_value:
            raise ValueError(
                f"BiRefNet mask manifest lacks completion_marker: {path}; regenerate the cache"
            )
        marker_path = Path(marker_value).expanduser()
        if not marker_path.is_absolute():
            marker_path = base / marker_path
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"BiRefNet mask cache is incomplete or has an invalid completion marker: {marker_path}"
            ) from exc
        if not isinstance(marker, dict) or marker.get("schema") != schema:
            raise ValueError(f"BiRefNet completion marker schema mismatch: {marker_path}")
        expected = _file_sha256(path)
        if marker.get("manifest_sha256") != expected:
            raise ValueError(
                f"BiRefNet completion marker does not match mask manifest: {marker_path}"
            )
    rows: Any = raw.get("views", raw) if isinstance(raw, dict) else raw
    result: dict[tuple[int, int], dict[str, Any]] = {}
    if isinstance(rows, dict):
        normalized = []
        for key, value in rows.items():
            if not isinstance(value, (str, dict)):
                continue
            tokens = str(key).replace(":", "/").split("/")
            if len(tokens) != 2:
                continue
            item = {"camera_id": int(tokens[0]), "frame": int(tokens[1])}
            item.update({"mask_path": value} if isinstance(value, str) else value)
            normalized.append(item)
        rows = normalized
    if not isinstance(rows, list):
        raise ValueError(f"mask manifest must contain a list or view mapping: {path}")
    for row in rows:
        if not isinstance(row, dict) or "camera_id" not in row or "frame" not in row:
            raise ValueError(f"invalid mask manifest row in {path}: {row!r}")
        key = (int(row["camera_id"]), int(row["frame"]))
        if key in result:
            raise ValueError(f"duplicate mask manifest row {key}: {path}")
        resolved: dict[str, Any] = {}
        for source_key, target_key in (("mask_path", "mask"), ("rgb_path", "rgb"),
                                       ("validity_path", "validity")):
            if row.get(source_key) is None:
                continue
            candidate = Path(str(row[source_key])).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved[target_key] = candidate.resolve()
        for field_name in ("rgb_sha256", "mask_sha256", "validity_sha256"):
            if row.get(field_name) is not None:
                resolved[field_name] = _declared_sha256(
                    row[field_name], field_name=field_name, manifest=path,
                )
        if row.get("original_size") is not None:
            size = row["original_size"]
            if (
                not isinstance(size, (list, tuple)) or len(size) != 2
                or any(isinstance(value, bool) or int(value) <= 0 for value in size)
            ):
                raise ValueError(f"invalid original_size for mask row {key}: {path}")
            resolved["original_size"] = (int(size[0]), int(size[1]))
        if strict:
            missing = [
                name for name in ("rgb", "mask", "rgb_sha256", "mask_sha256", "original_size")
                if name not in resolved
            ]
            if missing:
                raise ValueError(
                    f"BiRef mask row {key} lacks required fields {missing}: {path}"
                )
        for file_key, hash_key in (
            ("rgb", "rgb_sha256"), ("mask", "mask_sha256"),
            ("validity", "validity_sha256"),
        ):
            file_path = resolved.get(file_key)
            declared = resolved.get(hash_key)
            if file_path is not None and not Path(file_path).is_file():
                raise FileNotFoundError(file_path)
            if declared is not None and _file_sha256(Path(file_path)) != declared:
                raise ValueError(
                    f"mask manifest {hash_key} mismatch for camera/frame {key}: {file_path}"
                )
        if "original_size" in resolved and "rgb" in resolved:
            image = cv2.imread(str(resolved["rgb"]), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"cannot decode mask-manifest RGB: {resolved['rgb']}")
            if tuple(int(value) for value in image.shape[:2]) != resolved["original_size"]:
                raise ValueError(
                    f"mask manifest original_size mismatch for camera/frame {key}: "
                    f"{image.shape[:2]} != {resolved['original_size']}"
                )
        result[key] = resolved
    if strict and marker.get("views") != len(result):
        raise ValueError(
            f"BiRefNet completion marker view count does not match manifest: {marker_path}"
        )
    return _MaskManifest(result, strict=strict, schema=schema)


def _derived_map_path(root: Path | None, image_root: Path, rgb_path: Path) -> Path | None:
    if root is None:
        return None
    relative = rgb_path.relative_to(image_root)
    exact = root / relative
    png = exact.with_suffix(".png")
    return png if png.is_file() or not exact.is_file() else exact

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


def _uncalibrated_camera_setup(
    config, cameras: Sequence[int],
) -> tuple[list[int], list[int], str]:
    """Resolve a camera split when calibration files are intentionally absent.

    DA3 will estimate the actual camera geometry during initialization.  Until
    then the Scene still needs a complete manifest, so camera IDs and the train
    / eval partition are resolved from the image grid and configuration alone.
    """

    discovered = [int(camera) for camera in cameras]
    eval_ids = _selection(
        config.data.eval_cameras, discovered, "data.eval_cameras",
    ) if config.data.eval_cameras is not None else []
    train_ids = _selection(
        config.data.train_cameras, discovered, "data.train_cameras",
    ) if config.data.train_cameras is not None else [
        camera for camera in discovered if camera not in set(eval_ids)
    ]
    if not train_ids:
        raise ValueError(
            "uncalibrated DA3 datasets require at least one training camera; "
            "set data.train_cameras or leave at least one camera out of eval_cameras"
        )
    overlap = sorted(set(train_ids) & set(eval_ids))
    if overlap:
        raise ValueError(f"training and evaluation cameras overlap: {overlap}")
    assigned = set(train_ids) | set(eval_ids)
    if assigned != set(discovered):
        raise ValueError(
            "uncalibrated camera split must exactly cover discovered cameras: "
            f"unassigned={sorted(set(discovered)-assigned)}, "
            f"unknown={sorted(assigned-set(discovered))}"
        )
    return train_ids, eval_ids, "DA3 estimated cameras (no calibration files)"


def _guess_intrinsic(height: int, width: int, scale: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a provisional pinhole camera for the pre-initialization manifest."""

    focal = float(max(height, width))
    intrinsic = np.asarray([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    out_h = max(1, int(round(float(scale) * height)))
    out_w = max(1, int(round(float(scale) * width)))
    target = intrinsic.copy()
    target[0] *= np.float32(out_w / width)
    target[1] *= np.float32(out_h / height)
    target[2, 2] = 1.0
    return target, (out_h, out_w)


def _da3_cache_root(config) -> Path:
    configured = getattr(getattr(config.init, "da3", None), "cache_path", None)
    base = (
        Path(configured) if configured is not None
        else Path(config.data.root) / f"points_da3_stride{int(config.init.keyframe_stride)}"
    )
    return resolve_published_cache_root(base)


def _load_da3_camera_manifest(
    config, camera_ids: Sequence[int],
    *, cache_root: Path | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] | None:
    """Load the camera rig saved alongside an uncalibrated DA3 point cache."""

    cache_root = _da3_cache_root(config) if cache_root is None else Path(cache_root)
    path = cache_root / "camera_manifest.npz"
    if not path.is_file():
        return None
    manifest_path = cache_root / "manifest.json"
    completed_path = cache_root / "completed.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "ignoring DA3 camera rig without a complete manifest: %s", path,
        )
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "ftgspp-da3-foreground-v1"
        or not isinstance(completed, dict)
        or completed.get("manifest_fingerprint") != manifest.get("fingerprint")
    ):
        logger.warning("ignoring DA3 camera rig with invalid cache metadata: %s", path)
        return None
    unsigned = dict(manifest)
    stored_fingerprint = unsigned.pop("fingerprint", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    camera_record = manifest.get("camera_manifest")
    if (
        stored_fingerprint != hashlib.sha256(encoded).hexdigest()
        or not isinstance(camera_record, dict)
        or camera_record.get("path") != path.name
        or camera_record.get("sha256") != _file_sha256(path)
    ):
        logger.warning("ignoring DA3 camera rig with failed integrity checks: %s", path)
        return None
    # Callers validate the request and stage identity with _valid_da3_cache
    # against a provisional Scene before applying these camera parameters.
    try:
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["camera_ids"], dtype=np.int64).reshape(-1)
            extrinsics = np.asarray(archive["extrinsics"], dtype=np.float32)
            intrinsics = np.asarray(archive["intrinsics"], dtype=np.float32)
            sizes = np.asarray(archive["sizes"], dtype=np.int64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid DA3 camera manifest: {path}") from exc
    expected = [int(camera) for camera in camera_ids]
    if (
        ids.shape != (len(expected),)
        or extrinsics.shape != (len(expected), 4, 4)
        or intrinsics.shape != (len(expected), 3, 3)
        or sizes.shape != (len(expected), 2)
        or ids.tolist() != expected
    ):
        raise ValueError(
            f"DA3 camera manifest {path} does not cover the current camera grid"
        )
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError(f"DA3 camera manifest {path} contains non-finite values")
    result: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for index, camera in enumerate(expected):
        size = tuple(int(value) for value in sizes[index])
        if min(size) <= 0:
            raise ValueError(f"DA3 camera manifest has invalid size for camera {camera}")
        if intrinsics[index, 0, 0] <= 0.0 or intrinsics[index, 1, 1] <= 0.0:
            raise ValueError(
                f"DA3 camera manifest has invalid focal length for camera {camera}"
            )
        result[camera] = (
            np.ascontiguousarray(extrinsics[index]),
            np.ascontiguousarray(intrinsics[index]),
            size,
        )
    return result


def _time_offsets(
    config,
    cameras: Sequence[int],
    image_root: Path,
    camera_names: dict[int, str] | None = None,
) -> dict[int, float]:
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
        camera_names=camera_names,
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
    view: "View", source_size: tuple[int, int],
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
class View:
    path: Path
    frame: int
    camera_id: int
    timestamp: float
    w2c: np.ndarray
    source_intrinsic: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray
    size: tuple[int, int]
    mask_path: Path | None = None
    validity_path: Path | None = None
    mask_source: str = "none"
    rgb_encoding: str = "straight"

    def load_mask(self) -> np.ndarray:
        """Return the transformed soft foreground mask in ``[0, 1]``."""

        if self.mask_source == "external":
            if self.mask_path is None:
                raise FileNotFoundError(
                    f"external foreground mask is missing for camera={self.camera_id} frame={self.frame}"
                )
            return _transform_scalar_map(
                _decode_unit_map(self.mask_path, label="foreground mask"),
                self, interpolation=cv2.INTER_LINEAR,
            )
        if self.mask_source == "alpha":
            image = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(self.path)
            if image.ndim != 3 or image.shape[2] not in (2, 4):
                raise ValueError(
                    "data.masks.source=alpha requires a two- or four-channel image: "
                    f"{self.path}"
                )
            alpha = image[..., -1]
            scale = 65535.0 if alpha.dtype == np.uint16 else 255.0
            if alpha.dtype not in (np.uint8, np.uint16):
                raise ValueError(f"unsupported alpha dtype {alpha.dtype}: {self.path}")
            return _transform_scalar_map(
                alpha.astype(np.float32) / np.float32(scale),
                self, interpolation=cv2.INTER_LINEAR,
            )
        return np.ones(self.size, dtype=np.float32)

    def load_validity(self) -> np.ndarray:
        """Return geometry/loss validity independently of foreground."""

        if self.validity_path is None:
            return np.ones(self.size, dtype=np.float32)
        transformed = _transform_scalar_map(
            _decode_unit_map(self.validity_path, label="validity mask"),
            self, interpolation=cv2.INTER_NEAREST,
        )
        return (transformed >= 0.5).astype(np.float32)

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
        if self.mask_source in {"external", "alpha"}:
            mask = self.load_mask()
            if self.rgb_encoding == "premultiplied":
                # Preparation metadata explicitly identifies premultiplied
                # RGB. Convert once to straight RGB before storing the mask in
                # reference alpha; the native composite objective then never
                # applies the foreground factor twice.
                denominator = np.maximum(mask[..., None], 1e-6)
                straight = np.where(
                    mask[..., None] > 1e-6,
                    rgb[..., :3].astype(np.float32) / denominator,
                    0.0,
                )
                rgb = np.clip(np.rint(straight), 0.0, 255.0).astype(np.uint8)
            rgba = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
            rgba[..., :3] = rgb[..., :3]
            rgba[..., 3] = np.clip(
                np.rint(mask * np.float32(255.0)), 0, 255,
            ).astype(np.uint8)
            return np.ascontiguousarray(rgba)
        if rgb.shape[2] == 3:
            rgba = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = 255
            return np.ascontiguousarray(rgba)
        return rgb


def _as_homogeneous_np(ext: np.ndarray) -> np.ndarray:
    """Accept DA3's 3x4 extrinsics as well as homogeneous 4x4 matrices."""

    value = np.asarray(ext, dtype=np.float32)
    if value.shape == (4, 4):
        return value
    if value.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3] = value
        return result
    raise ValueError(f"expected a 3x4 or 4x4 extrinsic, got {value.shape}")


@dataclass
class Scene:
    frames: list[int]
    cameras: list[int]
    train_cameras: list[int]
    eval_cameras: list[int]
    camera_split_source: str
    train_views: list[View]
    eval_views: list[View]
    fps: float
    source_root: Path | None = None
    time_offsets: dict[int, float] = field(default_factory=dict)
    timestamp_convention: str = "logical_frame/fps-time_offset"
    calibration_available: bool = True
    # Present only for a locked, evaluation-only manifest. The runner copies
    # this identity into a dedicated final-test result stream.
    evaluation_identity: dict[str, object] | None = None
    _checkpoint_metadata_cache: dict[str, object] | None = field(
        default=None, init=False, repr=False,
    )

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    def view(self, frame: int, camera_id: int) -> View:
        for view in (*self.train_views, *self.eval_views):
            if view.frame == frame and view.camera_id == camera_id:
                return view
        raise KeyError(f"no view for frame={frame}, camera={camera_id}")

    def views_nearest_timestamp(
        self,
        reference_timestamp: float,
        camera_ids: Sequence[int],
        *,
        tolerance_seconds: float,
        train_only: bool = True,
    ) -> list[View]:
        """Select one nearest actual capture per camera and enforce sync."""

        pool = self.train_views if train_only else (*self.train_views, *self.eval_views)
        selected: list[View] = []
        for camera_id in camera_ids:
            candidates = [view for view in pool if int(view.camera_id) == int(camera_id)]
            if not candidates:
                raise KeyError(f"no views for camera={camera_id}")
            view = min(
                candidates,
                key=lambda item: (abs(float(item.timestamp) - reference_timestamp), int(item.frame)),
            )
            residual = abs(float(view.timestamp) - float(reference_timestamp))
            if residual > float(tolerance_seconds) + 1e-12:
                raise ValueError(
                    "camera synchronization residual exceeds tolerance: "
                    f"camera={camera_id} frame={view.frame} residual={residual:.9g}s "
                    f"tolerance={tolerance_seconds:.9g}s"
                )
            selected.append(view)
        return selected

    def update_camera_geometry(
        self,
        extrinsics: dict[int, np.ndarray],
        intrinsics: dict[int, np.ndarray],
        sizes: dict[int, tuple[int, int]],
    ) -> None:
        """Replace provisional camera geometry with DA3-estimated geometry.

        This is intentionally a Scene operation rather than a second dataset
        object: initialization runs immediately before native upload, so all
        subsequent training/evaluation views observe the same estimated rig.
        """

        expected = set(int(camera) for camera in self.cameras)
        if set(extrinsics) != expected or set(intrinsics) != expected or set(sizes) != expected:
            raise ValueError("estimated camera geometry must cover every dataset camera")

        def update(view: View) -> View:
            camera = int(view.camera_id)
            ext = np.ascontiguousarray(_as_homogeneous_np(extrinsics[camera]), dtype=np.float32)
            K = np.ascontiguousarray(np.asarray(intrinsics[camera], np.float32))
            size = tuple(int(value) for value in sizes[camera])
            if ext.shape != (4, 4) or K.shape != (3, 3):
                raise ValueError(f"invalid estimated camera geometry for camera {camera}")
            if (
                min(size) <= 0
                or not np.isfinite(ext).all()
                or not np.isfinite(K).all()
                or K[0, 0] <= 0.0
                or K[1, 1] <= 0.0
            ):
                raise ValueError(f"invalid estimated camera geometry for camera {camera}")
            return replace(
                view,
                w2c=ext,
                source_intrinsic=K,
                intrinsic=K,
                distortion=np.zeros(5, np.float32),
                size=size,
            )

        self.train_views = [update(view) for view in self.train_views]
        self.eval_views = [update(view) for view in self.eval_views]
        self.calibration_available = True
        self._checkpoint_metadata_cache = None
        self.validate_view_manifest()


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
                if view.mask_source == "external":
                    if view.mask_path is None or not view.mask_path.is_file():
                        raise FileNotFoundError(
                            view.mask_path or Path(
                                f"missing-mask-camera-{view.camera_id}-frame-{view.frame}"
                            )
                        )
                    if view.mask_path.stat().st_size <= 0:
                        raise ValueError(f"dataset foreground mask is empty: {view.mask_path}")
                if view.validity_path is not None and not view.validity_path.is_file():
                    raise FileNotFoundError(view.validity_path)
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
        hash_paths: list[Path | None] = []
        for view in (*self.train_views, *self.eval_views):
            hash_paths.append(_identity_rgb_path(view))
            hash_paths.append(view.mask_path)
            hash_paths.append(view.validity_path)
        content_hashes = _file_sha256_many(
            hash_paths, cache_root=_view_hash_cache_root(self.source_root),
        )

        def record(split: str, index: int, view: View) -> dict[str, object]:
            # Resolving thousands of local links to an NFS image tree can turn
            # metadata preflight into an hours-long operation.  Keep the link
            # path under the explicit opt-in fast staging mode; decode still
            # follows it later when the view is sampled.
            path = _identity_rgb_path(view)
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
                "rgb_sha256": content_hashes[path.as_posix()],
                "mask_source": str(view.mask_source),
                "rgb_encoding": str(view.rgb_encoding),
                "mask_path": (
                    view.mask_path.as_posix() if view.mask_path is not None else None
                ),
                "mask_sha256": (
                    content_hashes[view.mask_path.as_posix()]
                    if view.mask_path is not None else None
                ),
                "validity_sha256": (
                    content_hashes[view.validity_path.as_posix()]
                    if view.validity_path is not None else None
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
            "timestamp_convention": str(self.timestamp_convention),
            "camera_dense_mapping": {
                str(key): int(value) for key, value in sorted(mapping.items())
            },
            "train_cameras": [int(value) for value in self.train_cameras],
            "eval_cameras": [int(value) for value in self.eval_cameras],
            "camera_split_source": str(self.camera_split_source),
            "path_policy": "relative_to_extracted_path_when_possible",
        }
        # Scene/view records are immutable after construction. Keep the
        # expensive all-view stat/calibration hash out of every checkpoint and
        # hand callers a copy so metadata mutation cannot poison the cache.
        self._checkpoint_metadata_cache = metadata
        return deepcopy(metadata)

    @classmethod
    def from_config(cls, config) -> "Scene":
        root = Path(config.data.extracted_path)
        view_manifest = getattr(config.data, "view_manifest", None)
        if view_manifest is None:
            frames, camera_ids, paths = _scan_images(
                root, config.data.frames.start, config.data.frames.stop,
            )
            explicit_timestamps = None
        else:
            frames, camera_ids, paths, explicit_timestamps = load_image_manifest(
                Path(view_manifest),
                start=config.data.frames.start,
                stop=config.data.frames.stop,
                reject_empty=True,
            )
        fps = float(config.data.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"dataset FPS must be finite and positive, got {fps!r}")
        calibration_root = Path(config.data.calibration_path)
        calibration_files = (
            calibration_root / "intri.yml",
            calibration_root / "extri.yml",
            calibration_root / "train_intri.yml",
            calibration_root / "train_extri.yml",
            calibration_root / "test_intri.yml",
            calibration_root / "test_extri.yml",
        )
        source_calibration_available = any(path.is_file() for path in calibration_files)
        if source_calibration_available:
            cameras, train_ids, eval_ids, source = _camera_setup(config, camera_ids)
        else:
            cameras = {}
            train_ids, eval_ids, source = _uncalibrated_camera_setup(config, camera_ids)
        calibration_available = source_calibration_available
        # ``prepare_selfcap`` preserves arbitrary video stems as directory
        # names when no calibration is available.  Include those names as
        # synchronization aliases so a generated t_offsets.json remains
        # usable even though the internal camera IDs are positional.
        camera_names = (
            {}
            if view_manifest is not None else {
                int(camera_id): paths[0][index].parent.name
                for index, camera_id in enumerate(camera_ids)
                if paths[0][index].parent != root
            }
        )
        offsets = (
            {int(camera): 0.0 for camera in camera_ids}
            if explicit_timestamps is not None
            else _time_offsets(config, camera_ids, root, camera_names)
        )
        geometries: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
        camera_values: dict[int, Camera] = {}
        allow_positional = list(camera_ids) == list(range(len(camera_ids)))
        for index, camera_id in enumerate(camera_ids):
            if source_calibration_available:
                camera = cameras[_camera_name(
                    cameras, camera_id, allow_positional=allow_positional,
                )]
                camera_values[camera_id] = camera
                geometries[camera_id] = _geometry(
                    camera, paths[0][index], float(config.data.scale),
                )
            else:
                image = cv2.imread(str(paths[0][index]), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(paths[0][index])
                height, width = image.shape[:2]
                target_k, size = _guess_intrinsic(
                    height, width, float(config.data.scale),
                )
                geometries[camera_id] = (
                    target_k.copy(), target_k, size,
                )

        train_views: list[View] = []
        eval_views: list[View] = []
        mask_options = getattr(config.data, "masks", None)
        mask_source = str(getattr(mask_options, "source", "none"))
        mask_root = getattr(mask_options, "root", None)
        validity_root = getattr(mask_options, "validity_root", None)
        manifest_path = getattr(mask_options, "manifest_path", None)
        declared_mask_fingerprint = getattr(mask_options, "fingerprint", None)
        if declared_mask_fingerprint is not None and manifest_path is None:
            raise ValueError(
                "data.masks.fingerprint requires data.masks.manifest_path"
            )
        if declared_mask_fingerprint is not None:
            actual_mask_fingerprint = _file_sha256(Path(manifest_path))
            if str(declared_mask_fingerprint) != actual_mask_fingerprint:
                raise ValueError(
                    "data.masks.fingerprint does not match the mask manifest"
                )
        mask_manifest = _read_mask_manifest(
            Path(manifest_path) if manifest_path is not None else None,
            Path(manifest_path).parent if manifest_path is not None else root,
        )
        mask_rows = mask_manifest.rows
        train_set = set(train_ids)
        eval_set = set(eval_ids)
        for frame_index, frame in enumerate(frames):
            for camera_index, camera_id in enumerate(camera_ids):
                source_k, target_k, size = geometries[camera_id]
                camera = camera_values.get(camera_id)
                distortion = (
                    np.concatenate([
                        camera.distortion[:4],
                        np.zeros(max(0, len(camera.distortion)-4)),
                    ])
                    if camera is not None else np.zeros(5)
                )
                rgb_path = paths[frame_index][camera_index]
                mask_row = mask_rows.get((int(camera_id), int(frame)), {})
                if "rgb" in mask_row and mask_row["rgb"].resolve() != rgb_path.resolve():
                    raise ValueError(
                        "mask manifest RGB path does not match discovered view for "
                        f"camera={camera_id} frame={frame}: {mask_row['rgb']} != {rgb_path}"
                    )
                mask_path = mask_row.get("mask") or _derived_map_path(
                    Path(mask_root) if mask_root is not None else None, root, rgb_path,
                )
                validity_path = mask_row.get("validity") or _derived_map_path(
                    Path(validity_root) if validity_root is not None else None,
                    root, rgb_path,
                )
                view = View(
                    path=rgb_path, frame=frame,
                    camera_id=camera_id,
                    # Match Main's float32 tensor expression
                    # ``frame / fps - time_offsets`` before shader upload.
                    timestamp=(
                        float(explicit_timestamps[(int(frame), int(camera_id))])
                        if explicit_timestamps is not None else float(
                            np.float32(frame / fps)
                            - np.float32(offsets[camera_id])
                        )
                    ),
                    # Uncalibrated scenes use identity placeholders until DA3
                    # estimates a shared rig during initialization.
                    w2c=np.ascontiguousarray(
                        camera.w2c if camera is not None else np.eye(4),
                        dtype=np.float32,
                    ),
                    source_intrinsic=source_k, intrinsic=target_k,
                    distortion=np.ascontiguousarray(
                        distortion,
                        dtype=np.float32,
                    ), size=size,
                    mask_path=mask_path,
                    validity_path=validity_path,
                    mask_source=mask_source,
                    rgb_encoding=str(getattr(mask_options, "rgb_encoding", "straight")),
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
            timestamp_convention=(
                "explicit_per_view_manifest"
                if explicit_timestamps is not None
                else "logical_frame/fps-time_offset"
            ),
            calibration_available=calibration_available,
        )
        if mask_manifest.strict:
            discovered = {
                (int(view.camera_id), int(view.frame))
                for view in (*train_views, *eval_views)
            }
            declared = set(mask_rows)
            if declared != discovered:
                raise ValueError(
                    "BiRefNet mask manifest must exactly cover the loaded RGB grid: "
                    f"missing={sorted(discovered - declared)[:8]}, "
                    f"extra={sorted(declared - discovered)[:8]}"
                )
        dataset.validate_view_manifest()
        dataset.checkpoint_metadata()
        if not source_calibration_available:
            # Construct the requested raster/placeholder geometry first. A
            # cached rig must match these inputs (including scale and image
            # contents), rather than defining its own apparently valid inputs.
            from ftgspp.optional.da3 import _cache_root
            from ftgspp.training.initialize import _valid_da3_cache
            positions = list(range(0, len(frames), int(config.init.keyframe_stride)))
            if positions[-1] != len(frames) - 1:
                positions.append(len(frames) - 1)
            selected_frames = [int(frames[position]) for position in positions]
            for cache_root in published_cache_candidates(_cache_root(config)):
                if not (cache_root / "camera_manifest.npz").is_file():
                    continue
                try:
                    if not _valid_da3_cache(
                        config, dataset, cache_root, expected_frames=selected_frames,
                        frame_positions=positions,
                    ):
                        continue
                    geometry = _load_da3_camera_manifest(config, camera_ids, cache_root=cache_root)
                    if geometry is None:
                        continue
                    dataset.update_camera_geometry(
                        {camera: value[0] for camera, value in geometry.items()},
                        {camera: value[1] for camera, value in geometry.items()},
                        {camera: value[2] for camera, value in geometry.items()},
                    )
                except (OSError, ValueError) as exc:
                    logger.warning("ignoring invalid DA3 camera cache: %s", exc)
                    continue
                dataset.camera_split_source = "DA3 estimated cameras (cached; no calibration files)"
                break
        return dataset


def load_final_test_scene(training_scene: Scene, manifest_path: Path) -> Scene:
    """Load a locked evaluation-only manifest in the training coordinate frame.

    Final-test views never enter the training scene or checkpoint selector. The
    manifest carries the training dataset fingerprint and explicit calibrated
    geometry so test images cannot silently change the model's coordinate or
    timestamp convention.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid final test manifest: {manifest_path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") not in {
            "ftgspp-final-test-v1", "ftgspp-final-test-v2",
        }
    ):
        raise ValueError(
            "final test manifest must use schema ftgspp-final-test-v1/v2: "
            f"{manifest_path}"
        )
    if payload.get("schema") == "ftgspp-final-test-v2" and (
        payload.get("split") != "final_test"
        or payload.get("independent_from_selection") is not True
    ):
        raise ValueError(
            "ftgspp-final-test-v2 requires split=final_test and "
            "independent_from_selection=true"
        )
    expected_fingerprint = training_scene.checkpoint_metadata()["fingerprint"]
    if payload.get("training_dataset_fingerprint") != expected_fingerprint:
        raise ValueError(
            "final test manifest does not match the locked training dataset fingerprint"
        )
    fps = float(payload.get("fps", training_scene.fps))
    if not np.isfinite(fps) or abs(fps - float(training_scene.fps)) > 1e-9:
        raise ValueError("final test manifest FPS must match the training dataset")
    rgb_encoding = str(payload.get("rgb_encoding", "straight"))
    if rgb_encoding not in {"straight", "premultiplied"}:
        raise ValueError(
            "final test rgb_encoding must be straight or premultiplied"
        )
    raw_cameras = payload.get("cameras")
    raw_views = payload.get("views")
    if not isinstance(raw_cameras, dict) or not isinstance(raw_views, list) or not raw_views:
        raise ValueError("final test manifest requires non-empty cameras and views")
    base = manifest_path.parent
    selection_views = (
        *getattr(training_scene, "train_views", ()),
        *getattr(training_scene, "eval_views", ()),
    )
    selection_keys = {
        (int(view.frame), int(view.camera_id)) for view in selection_views
    }
    selection_paths = {Path(view.path).resolve() for view in selection_views}
    selection_rgb_hashes = {
        _file_sha256(Path(view.path)) for view in selection_views
    }

    def verify_hash(file_path: Path, value: object, label: str) -> str:
        declared = _declared_sha256(
            value, field_name=f"final test {label}_sha256", manifest=manifest_path,
        )
        actual = _file_sha256(file_path)
        if actual != declared:
            raise ValueError(
                f"final test {label} content hash mismatch: {file_path}"
            )
        return actual

    def resolve_file(value: object, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"final test {label} must be a non-empty path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        return path

    views: list[View] = []
    seen: set[tuple[int, int]] = set()
    content_rows: list[dict[str, object]] = []
    for index, row in enumerate(raw_views):
        if not isinstance(row, dict):
            raise ValueError(f"invalid final test view row {index}")
        try:
            frame = int(row["frame"])
            camera_id = int(row["camera_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"final test view row {index} lacks frame/camera_id") from exc
        if frame < 0 or camera_id < 0:
            raise ValueError(
                f"final test view row {index} has a negative frame/camera_id"
            )
        key = (frame, camera_id)
        if key in seen:
            raise ValueError(f"duplicate final test view frame/camera: {key}")
        if key in selection_keys:
            raise ValueError(
                "final-test frame/camera overlaps checkpoint-selection views: "
                f"{key}"
            )
        seen.add(key)
        camera = raw_cameras.get(str(camera_id), raw_cameras.get(camera_id))
        if not isinstance(camera, dict):
            raise ValueError(f"final test camera {camera_id} has no geometry")
        w2c = np.asarray(camera.get("w2c"), np.float32)
        intrinsic = np.asarray(camera.get("intrinsic"), np.float32)
        source_intrinsic = np.asarray(
            camera.get("source_intrinsic", intrinsic), np.float32,
        )
        distortion = np.asarray(
            camera.get("distortion", np.zeros(5)), np.float32,
        ).reshape(-1)
        if (
            w2c.shape != (4, 4) or intrinsic.shape != (3, 3)
            or source_intrinsic.shape != (3, 3)
            or not np.isfinite(w2c).all() or not np.isfinite(intrinsic).all()
            or not np.isfinite(source_intrinsic).all()
            or not np.isfinite(distortion).all()
            or intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0
        ):
            raise ValueError(f"invalid final test geometry for camera {camera_id}")
        rgb_path = resolve_file(row.get("rgb_path"), "rgb_path")
        if rgb_path in selection_paths:
            raise ValueError(
                f"final-test RGB path overlaps checkpoint-selection data: {rgb_path}"
            )
        rgb_sha256 = verify_hash(rgb_path, row.get("rgb_sha256"), "rgb")
        if rgb_sha256 in selection_rgb_hashes:
            raise ValueError(
                "final-test RGB content overlaps checkpoint-selection data: "
                f"{rgb_path}"
            )
        decoded = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"cannot decode final test image: {rgb_path}")
        original_size_value = row.get("original_size")
        if (
            not isinstance(original_size_value, (list, tuple))
            or len(original_size_value) != 2
        ):
            raise ValueError(
                f"final test view {key} requires original_size=[height,width]"
            )
        original_size = tuple(int(value) for value in original_size_value)
        if min(original_size) <= 0 or original_size != tuple(decoded.shape[:2]):
            raise ValueError(
                f"final test view {key} original_size does not match RGB: "
                f"{original_size} != {decoded.shape[:2]}"
            )
        size_value = camera.get("size")
        if size_value is None:
            size = tuple(int(value) for value in decoded.shape[:2])
        else:
            size = tuple(int(value) for value in size_value)
            if len(size) != 2 or min(size) <= 0:
                raise ValueError(f"invalid final test size for camera {camera_id}")
        timestamp = row.get("timestamp")
        if timestamp is None:
            if camera_id not in training_scene.time_offsets:
                raise ValueError(
                    f"final test view {key} requires an explicit training-coordinate timestamp"
                )
            timestamp = float(
                np.float32(frame / fps)
                - np.float32(training_scene.time_offsets[camera_id])
            )
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError(f"final test view {key} has a non-finite timestamp")
        mask_source = str(
            row.get("mask_source", payload.get("mask_source", "external"))
        )
        if mask_source not in {"external", "alpha"}:
            raise ValueError("final test masks must use external or alpha source")
        if mask_source == "alpha" and (
            decoded.ndim != 3 or decoded.shape[2] != 4
        ):
            raise ValueError(
                f"final test alpha mask requires RGBA input: {rgb_path}"
            )
        mask_path = (
            resolve_file(row.get("mask_path"), "mask_path")
            if mask_source == "external" else None
        )
        mask_sha256 = (
            verify_hash(mask_path, row.get("mask_sha256"), "mask")
            if mask_path is not None else None
        )
        if mask_path is not None:
            decoded_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if (
                decoded_mask is None
                or tuple(decoded_mask.shape[:2]) != tuple(decoded.shape[:2])
            ):
                raise ValueError(
                    f"final test mask dimensions do not match RGB: {mask_path}"
                )
        validity_path = (
            resolve_file(row.get("validity_path"), "validity_path")
            if row.get("validity_path") is not None else None
        )
        validity_sha256 = (
            verify_hash(validity_path, row.get("validity_sha256"), "validity")
            if validity_path is not None else None
        )
        if validity_path is not None:
            decoded_validity = cv2.imread(
                str(validity_path), cv2.IMREAD_UNCHANGED,
            )
            if (
                decoded_validity is None
                or tuple(decoded_validity.shape[:2])
                != tuple(decoded.shape[:2])
            ):
                raise ValueError(
                    "final test validity dimensions do not match RGB: "
                    f"{validity_path}"
                )
        content_rows.append({
            "camera_id": camera_id,
            "frame": frame,
            "timestamp": timestamp,
            "rgb_sha256": rgb_sha256,
            "mask_sha256": mask_sha256,
            "validity_sha256": validity_sha256,
            "original_size": list(original_size),
        })
        views.append(View(
            path=rgb_path, frame=frame, camera_id=camera_id,
            timestamp=timestamp, w2c=np.ascontiguousarray(w2c),
            source_intrinsic=np.ascontiguousarray(source_intrinsic),
            intrinsic=np.ascontiguousarray(intrinsic),
            distortion=np.ascontiguousarray(distortion), size=size,
            mask_path=mask_path, validity_path=validity_path,
            mask_source=mask_source,
            rgb_encoding=rgb_encoding,
        ))
    camera_ids = sorted({int(view.camera_id) for view in views})
    frames = sorted({int(view.frame) for view in views})
    offsets = {
        camera: float(
            raw_cameras.get(str(camera), raw_cameras.get(camera, {})).get(
                "time_offset", 0.0,
            )
        )
        for camera in camera_ids
    }
    if not all(np.isfinite(value) for value in offsets.values()):
        raise ValueError("final test camera time offsets must be finite")
    content_sha256 = hashlib.sha256(json.dumps(
        content_rows, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return Scene(
        frames=frames, cameras=camera_ids, train_cameras=[],
        eval_cameras=camera_ids,
        camera_split_source=f"locked final test manifest: {manifest_path}",
        train_views=[], eval_views=views, fps=fps, source_root=base,
        time_offsets=offsets, calibration_available=True,
        evaluation_identity={
            "kind": "final_test",
            "schema": str(payload["schema"]),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _file_sha256(manifest_path),
            "training_dataset_fingerprint": expected_fingerprint,
            "content_sha256": content_sha256,
            "views": len(views),
        },
    )
