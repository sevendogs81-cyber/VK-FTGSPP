"""Torch-free multiview manifest and synchronization core.

Both the native Vulkan training adapter and the optional Torch UFM adapter use
this module.  Camera calibration tensors remain adapter-specific, while image
identity, frame/camera coverage and time-offset semantics have one owner.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeVar

import numpy as np

FRAME_INDEX_WIDTH = 6
TIME_OFFSET_FILENAMES = ("timecode.json", "t_offsets.json", "time_offsets.json")
_FLAT_IMAGE_RE = re.compile(
    rf"^c(?P<camera>[0-9]+)-f(?P<frame>[0-9]{{{FRAME_INDEX_WIDTH}}})"
    r"(?P<suffix>\.[A-Za-z0-9]+)$"
)
CameraT = TypeVar("CameraT")


def camera_aliases(camera: int) -> tuple[str, ...]:
    aliases = (
        str(camera), f"{camera:02d}", f"{camera:03d}", f"{camera:04d}",
        f"{camera:06d}", f"c{camera}", f"c{camera:02d}", f"c{camera:03d}",
        f"c{camera:04d}", f"c{camera:06d}",
    )
    return tuple(dict.fromkeys(aliases))


def resolve_camera_name(
    cameras: Mapping[str, CameraT],
    camera_id: int,
    *,
    allow_positional: bool = True,
) -> str:
    for name in camera_aliases(camera_id):
        if name in cameras:
            return name
    if allow_positional:
        try:
            return list(cameras)[camera_id]
        except IndexError:
            pass
    raise KeyError(
        f"no calibration alias for camera {camera_id}; available names are "
        f"{list(cameras)[:8]}"
    )


def select_camera_ids(
    selection: Iterable[int] | None,
    available: Sequence[int],
    label: str,
) -> list[int]:
    if selection is None:
        return []
    selected = [int(camera) for camera in selection]
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label} contains duplicate camera IDs")
    unknown = sorted(set(selected) - {int(camera) for camera in available})
    if unknown:
        raise ValueError(f"{label} contains unknown cameras: {unknown}")
    return selected


def _camera_directory_id(name: str, fallback: int) -> int:
    if name.isdigit():
        return int(name)
    match = re.fullmatch(r"c([0-9]+)", name)
    return int(match.group(1)) if match is not None else fallback


def _camera_directory_key(path: Path) -> tuple[int, int | str]:
    if path.name.isdigit():
        return 0, int(path.name)
    match = re.fullmatch(r"c([0-9]+)", path.name)
    return (0, int(match.group(1))) if match is not None else (1, path.name)


def _numeric_stem_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def scan_image_grid(
    image_root: Path,
    *,
    start: int,
    stop: int,
    reject_empty: bool = True,
) -> tuple[list[int], list[int], list[list[Path]]]:
    frames = list(range(int(start), int(stop)))
    if not frames:
        raise ValueError("frame range is empty")
    flat_candidates = sorted(
        path for path in image_root.glob("c*-f*") if path.is_file()
    )
    if flat_candidates:
        requested_frames = set(frames)
        flat: dict[tuple[int, int], Path] = {}
        for path in flat_candidates:
            match = _FLAT_IMAGE_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(
                    f"invalid legacy flat image filename: {path.name!r}; "
                    "expected c<non-negative-camera-id>-f<six-digit-frame>"
                    ".<image-extension>"
                )
            frame = int(match.group("frame"))
            if frame not in requested_frames:
                continue
            camera = int(match.group("camera"))
            key = frame, camera
            if key in flat:
                raise ValueError(
                    f"duplicate legacy flat image for camera {camera} at "
                    f"frame {frame}: {flat[key].name!r} and {path.name!r}"
                )
            flat[key] = path
        cameras = sorted(camera for frame, camera in flat if frame == frames[0])
        if not cameras:
            raise FileNotFoundError(
                f"legacy flat dataset has no images for requested frame {frames[0]}"
            )
        expected = set(cameras)
        paths: list[list[Path]] = []
        for frame in frames:
            observed = {camera for value, camera in flat if value == frame}
            if observed != expected:
                raise ValueError(
                    "legacy flat image grid has inconsistent camera coverage "
                    f"at frame {frame}: missing={sorted(expected-observed)}, "
                    f"extra={sorted(observed-expected)}"
                )
            paths.append([flat[(frame, camera)] for camera in cameras])
    else:
        camera_dirs = sorted(
            (path for path in image_root.iterdir() if path.is_dir()),
            key=_camera_directory_key,
        )
        if not camera_dirs:
            raise FileNotFoundError(f"no multiview images found in {image_root}")
        per_camera: list[dict[int, Path]] = []
        for directory in camera_dirs:
            frame_paths: dict[int, Path] = {}
            for path in sorted(
                (
                    candidate for candidate in directory.iterdir()
                    if (
                        os.environ.get("FTGSPP_SYMLINK_PREFLIGHT") == "1"
                        and candidate.is_symlink()
                    ) or candidate.is_file()
                ),
                key=_numeric_stem_key,
            ):
                if not path.stem.isdigit():
                    continue
                frame = int(path.stem)
                if frame in frame_paths:
                    raise ValueError(
                        f"duplicate numeric frame {frame} in {directory}: "
                        f"{frame_paths[frame].name!r} and {path.name!r}"
                    )
                frame_paths[frame] = path
            if not frame_paths:
                raise FileNotFoundError(f"no numeric frame images found in {directory}")
            per_camera.append(frame_paths)
        source_numbers = sorted(per_camera[0])
        source_offset = 0 if frames[0] in set(source_numbers) else source_numbers[0] - frames[0]
        source_frames = [frame + source_offset for frame in frames]
        cameras = [
            _camera_directory_id(directory.name, index)
            for index, directory in enumerate(camera_dirs)
        ]
        if len(cameras) != len(set(cameras)):
            raise ValueError(
                "camera directories normalize to duplicate numeric IDs: "
                f"{[directory.name for directory in camera_dirs]}"
            )
        optional_paths = [
            [per_camera[index].get(source) for index in range(len(camera_dirs))]
            for source in source_frames
        ]
        missing = [
            (frames[fi], cameras[ci], source_frames[fi])
            for fi, row in enumerate(optional_paths)
            for ci, path in enumerate(row)
            if path is None
        ]
        if missing:
            raise FileNotFoundError(
                "directory image grid is incomplete; "
                "(logical_frame, camera_id, source_frame) examples="
                f"{missing[:8]}"
            )
        paths = [[path for path in row if path is not None] for row in optional_paths]
    if reject_empty:
        empty = [
            path for row in paths for path in row
            if (
                path.lstat().st_size
                if os.environ.get("FTGSPP_SYMLINK_PREFLIGHT") == "1"
                and path.is_symlink()
                else path.stat().st_size
            ) <= 0
        ]
        if empty:
            raise ValueError(f"multiview image grid contains empty files: {empty[:8]}")
    return frames, cameras, paths


def load_time_offsets(
    cameras: Sequence[int],
    search_roots: Sequence[Path],
    *,
    strict: bool = True,
) -> tuple[dict[int, float], Path | None]:
    source = next(
        (
            root / filename
            for root in search_roots
            for filename in TIME_OFFSET_FILENAMES
            if (root / filename).is_file()
        ),
        None,
    )
    if source is None:
        return {int(camera): 0.0 for camera in cameras}, None
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected a camera-to-seconds object in {source}")
    result: dict[int, float] = {}
    missing: list[int] = []
    for camera_value in cameras:
        camera = int(camera_value)
        aliases = [name for name in camera_aliases(camera) if name in raw]
        if not aliases:
            missing.append(camera)
            result[camera] = 0.0
            continue
        values: list[float] = []
        for name in aliases:
            raw_value = raw[name]
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"{source} has a non-numeric time offset for camera "
                    f"{camera} via key {name!r}: {raw_value!r}"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source} has a non-numeric time offset for camera "
                    f"{camera} via key {name!r}: {raw_value!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"{source} has a non-finite time offset for camera "
                    f"{camera} via key {name!r}: {raw_value!r}"
                )
            values.append(value)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"{source} has conflicting aliases for camera {camera}: "
                f"{dict(zip(aliases, values))}"
            )
        result[camera] = values[0]
    if missing:
        message = f"{source} has no time offset for cameras {missing}"
        if strict:
            raise ValueError(
                message + "; either fix the synchronization file or set "
                "data.strict_time_offsets=false to explicitly fill 0 seconds"
            )
        warnings.warn(
            message + "; using 0 seconds because strict_time_offsets=false",
            stacklevel=2,
        )
    return result, source
