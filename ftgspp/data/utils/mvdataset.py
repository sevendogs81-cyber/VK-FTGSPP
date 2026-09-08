# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional, Self, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from ftgspp.config.schema import DEFAULT_FPS
from ftgspp.data.multiview import (
    load_time_offsets,
    resolve_camera_name as _camera_name,
    scan_image_grid,
    select_camera_ids,
)
from ftgspp.data.utils.easy_utils import EasyCamera, read_camera
from ftgspp.data.utils.undistort import colmap_undistort, compute_undistorted_geometry
from ftgspp.utils import Interval, PathLike

def _load_time_offsets(
    *,
    cameras: Sequence[int],
    video_path: Optional[PathLike],
    image_path: Path,
    strict_time_offsets: bool = True,
) -> Tensor:
    search_paths = []
    if video_path is not None:
        search_paths.append(Path(video_path))
    search_paths.extend([image_path, image_path.parent])

    offset_map, offset_path = load_time_offsets(
        cameras, search_paths, strict=bool(strict_time_offsets),
    )
    values = [offset_map[int(camera)] for camera in cameras]
    offsets = torch.tensor(values, dtype=torch.float32).view(len(cameras), 1)

    if offset_path is not None:
        print(
            f"[mvdataset] loaded time offsets from {offset_path}; "
            f"using render_ftgs convention t = frame / fps - offset "
            f"(range {float(offsets.min()):.6f}s to {float(offsets.max()):.6f}s)"
        )
    return offsets


def _selection_to_camera_ids(selection, cameras: Sequence[int], label: str) -> list[int]:
    if selection is None:
        return []
    if isinstance(selection, Interval):
        selected = list(selection.into_range())
    else:
        selected = [int(camera) for camera in selection]

    return select_camera_ids(selected, cameras, label)


def _camera_ids_for_names(
    *,
    cameras_by_name: dict[str, EasyCamera],
    cameras: Sequence[int],
    names: set[str],
    label: str,
    allow_positional: bool,
) -> list[int]:
    selected = []
    matched_names = set()
    for camera in cameras:
        name = _camera_name(
            cameras_by_name,
            int(camera),
            allow_positional=allow_positional,
        )
        if name in names:
            selected.append(int(camera))
            matched_names.add(name)

    missing = sorted(names - matched_names)
    if missing:
        raise ValueError(f"{label} contains cameras without image directories: {missing}")
    return selected


def _read_camera_setup(
    *,
    calibration_path: Path,
    cameras: Sequence[int],
    configured_train_cameras,
    configured_eval_cameras,
    configured_camera_split_source,
) -> tuple[dict[str, EasyCamera], list[int], list[int], str]:
    allow_positional = list(cameras) == list(range(len(cameras)))
    full_intri = calibration_path / "intri.yml"
    full_extri = calibration_path / "extri.yml"
    train_intri = calibration_path / "train_intri.yml"
    train_extri = calibration_path / "train_extri.yml"
    test_intri = calibration_path / "test_intri.yml"
    test_extri = calibration_path / "test_extri.yml"

    split_paths = (train_intri, train_extri, test_intri, test_extri)
    split_exists = [path.is_file() for path in split_paths]
    full_exists = [full_intri.is_file(), full_extri.is_file()]

    if any(split_exists) and not all(split_exists):
        missing = [str(path) for path, exists in zip(split_paths, split_exists) if not exists]
        raise FileNotFoundError(
            "incomplete train/test calibration split; missing: " + ", ".join(missing)
        )

    if all(split_exists):
        train_by_name = read_camera(train_intri, train_extri)
        test_by_name = read_camera(test_intri, test_extri)
        overlap = sorted(set(train_by_name) & set(test_by_name))
        if overlap:
            raise ValueError(f"train/test calibration cameras overlap: {overlap}")

        if all(full_exists):
            cameras_by_name = read_camera(calibration_path)
        elif any(full_exists):
            missing = full_extri if full_intri.exists() else full_intri
            raise FileNotFoundError(f"incomplete full calibration pair; missing: {missing}")
        else:
            cameras_by_name = {**train_by_name, **test_by_name}

        unknown_split_names = sorted(
            (set(train_by_name) | set(test_by_name)) - set(cameras_by_name)
        )
        if unknown_split_names:
            raise ValueError(
                "train/test calibration cameras missing from full calibration: "
                f"{unknown_split_names}"
            )

        if (
            configured_camera_split_source is not None
            and configured_train_cameras is not None
            and configured_eval_cameras is not None
        ):
            train_cameras = _selection_to_camera_ids(
                configured_train_cameras, cameras, "resolved data.train_cameras"
            )
            eval_cameras = _selection_to_camera_ids(
                configured_eval_cameras, cameras, "resolved data.eval_cameras"
            )
            source = configured_camera_split_source
        else:
            train_cameras = _camera_ids_for_names(
                cameras_by_name=cameras_by_name,
                cameras=cameras,
                names=set(train_by_name),
                label="train calibration",
                allow_positional=allow_positional,
            )
            eval_cameras = _camera_ids_for_names(
                cameras_by_name=cameras_by_name,
                cameras=cameras,
                names=set(test_by_name),
                label="test calibration",
                allow_positional=allow_positional,
            )
            assigned = set(train_cameras) | set(eval_cameras)
            unassigned = [int(camera) for camera in cameras if int(camera) not in assigned]
            if unassigned:
                raise ValueError(
                    "image cameras are missing from train/test calibration files: "
                    f"{unassigned}"
                )
            source = "train/test calibration files"
    else:
        if not all(full_exists):
            missing = [
                str(path)
                for path, exists in zip((full_intri, full_extri), full_exists)
                if not exists
            ]
            raise FileNotFoundError(
                "full calibration pair is required when train/test calibration "
                "files are absent; missing: " + ", ".join(missing)
            )

        cameras_by_name = read_camera(calibration_path)
        if configured_eval_cameras is None:
            raise ValueError(
                "data.eval_cameras must be specified when train/test calibration "
                "files are absent; use [] for no hold-out (train on all cameras)"
            )
        eval_cameras = _selection_to_camera_ids(
            configured_eval_cameras, cameras, "data.eval_cameras"
        )
        if configured_train_cameras is None:
            eval_set = set(eval_cameras)
            train_cameras = [int(camera) for camera in cameras if int(camera) not in eval_set]
        else:
            train_cameras = _selection_to_camera_ids(
                configured_train_cameras, cameras, "data.train_cameras"
            )
        source = (
            "full calibration, no hold-out"
            if not eval_cameras
            else "full calibration plus config eval cameras"
        )

    if not train_cameras:
        raise ValueError("camera split contains no training cameras")
    overlap = sorted(set(train_cameras) & set(eval_cameras))
    if overlap:
        raise ValueError(f"training and evaluation cameras overlap: {overlap}")
    assigned = set(train_cameras) | set(eval_cameras)
    discovered = {int(camera) for camera in cameras}
    if assigned != discovered:
        raise ValueError(
            "camera split must assign every discovered camera exactly once: "
            f"unassigned={sorted(discovered - assigned)}, "
            f"unknown={sorted(assigned - discovered)}"
        )

    print(
        f"[mvdataset] camera split from {source}; "
        f"train={train_cameras}, eval={eval_cameras}"
    )
    return cameras_by_name, train_cameras, eval_cameras, source


def _scale_intrinsic_to_image(K: np.ndarray, cam: EasyCamera, image_h: int, image_w: int) -> np.ndarray:
    K = K.copy()
    if cam.H > 0 and cam.W > 0 and (cam.H != image_h or cam.W != image_w):
        sx = image_w / max(float(cam.W), 1.0)
        sy = image_h / max(float(cam.H), 1.0)
        K[0, :] *= sx
        K[1, :] *= sy
        K[2, 2] = 1.0
    return K


def _distortion_params(D: np.ndarray) -> tuple[float, float, float, float, float]:
    d = np.asarray(D, dtype=np.float64).reshape(-1)
    if len(d) < 5:
        d = np.pad(d, (0, 5 - len(d)))
    return tuple(float(x) for x in d[:5])  # type: ignore[return-value]


def _scaled_target_intrinsic(Kt: Tensor, src_h: int, src_w: int, scale: float) -> tuple[int, int, Tensor]:
    out_h = int(max(1.0, scale * src_h))
    out_w = int(max(1.0, scale * src_w))
    K = Kt.clone()
    K[0, :] *= out_w / max(float(src_w), 1.0)
    K[1, :] *= out_h / max(float(src_h), 1.0)
    K[2, 2] = 1.0
    return out_h, out_w, K


def _scan_mvdataset_images(
    image_path: Path, frame_range: Interval
) -> tuple[list[int], list[int], str, list[list[Path]]]:
    frames, cameras, paths = scan_image_grid(
        image_path,
        start=frame_range.start,
        stop=frame_range.stop,
        reject_empty=True,
    )
    return frames, cameras, paths[0][0].suffix, paths


def _read_easy_camera_tensors(
    *,
    cameras_by_name: dict[str, EasyCamera],
    cameras: Sequence[int],
    sample_paths: Sequence[Path],
    scale: float,
    blank_pixels: bool,
):
    w2cs = []
    source_intrinsics = []
    distortions = []
    target_intrinsics = []
    target_sizes = []
    allow_positional = list(cameras) == list(range(len(cameras)))

    for i, camera in enumerate(cameras):
        name = _camera_name(
            cameras_by_name,
            int(camera),
            allow_positional=allow_positional,
        )
        cam = cameras_by_name[name]
        sample = sample_paths[i]
        img_sample = cv2.imread(str(sample), cv2.IMREAD_COLOR)
        if img_sample is None:
            raise FileNotFoundError(f"failed to read sample image: {sample}")
        image_h, image_w = img_sample.shape[:2]

        K = _scale_intrinsic_to_image(cam.K, cam, int(image_h), int(image_w))
        if float(np.abs(cam.D).sum()) == 0.0:
            undist_h, undist_w = int(image_h), int(image_w)
            Kt = torch.tensor(K, dtype=torch.float32)
        else:
            k1, k2, p1, p2, k3 = _distortion_params(cam.D)
            undist_h, undist_w, Kt = compute_undistorted_geometry(
                int(image_h),
                int(image_w),
                float(K[0, 0]),
                float(K[1, 1]),
                float(K[0, 2]),
                float(K[1, 2]),
                k1,
                k2,
                p1,
                p2,
                k3,
                blank_pixels,
            )
        target_h, target_w, target_K = _scaled_target_intrinsic(Kt, undist_h, undist_w, scale)

        w2cs.append(torch.tensor(cam.w2c, dtype=torch.float32))
        source_intrinsics.append(torch.tensor(K, dtype=torch.float32))
        distortions.append(torch.tensor(cam.D, dtype=torch.float32).reshape(5, 1))
        target_intrinsics.append(target_K.float())
        target_sizes.append((target_h, target_w))

    return (
        torch.stack(w2cs, dim=0),
        torch.stack(source_intrinsics, dim=0),
        torch.stack(distortions, dim=0),
        torch.stack(target_intrinsics, dim=0),
        target_sizes,
    )


def _selector_to_positions(length: int, selector) -> tuple[list[int], bool]:
    if isinstance(selector, Tensor):
        selector = selector.detach().cpu().tolist()

    if isinstance(selector, int):
        return [selector], True
    if isinstance(selector, slice):
        return list(range(length))[selector], False
    if isinstance(selector, range):
        return list(selector), False
    if isinstance(selector, Iterable):
        return [int(i) for i in selector], False

    raise TypeError(f"unsupported index type: {type(selector)!r}")


def _index_tensor_by_axes(tensor: Tensor, selections: Sequence[tuple[list[int], bool]]) -> Tensor:
    out = tensor
    dim = 0
    for positions, remove_axis in selections:
        if remove_axis:
            out = out.select(dim, positions[0])
        else:
            idx = torch.tensor(positions, dtype=torch.long, device=out.device)
            out = out.index_select(dim, idx)
            dim += 1
    return out


def _metadata_grid(
    frame_values: Sequence,
    camera_values: Sequence,
    axes: tuple[str, ...],
    values: str,
):
    if values == "frame":
        if axes == ("frame", "camera"):
            return [[f for _ in camera_values] for f in frame_values]
        if axes == ("frame",):
            return list(frame_values)
        if axes == ("camera",):
            return [frame_values[0] for _ in camera_values]
        return frame_values[0]

    if axes == ("frame", "camera"):
        return [[c for c in camera_values] for _ in frame_values]
    if axes == ("frame",):
        return [camera_values[0] for _ in frame_values]
    if axes == ("camera",):
        return list(camera_values)
    return camera_values[0]


@dataclass
class MVBatch:
    rgb: Tensor
    w2c: Tensor
    intrinsic: Tensor
    time: Tensor
    _frame_values: list[int]
    _camera_values: list[int]
    _paths: list[list[Path]]
    _axes: tuple[str, ...]

    @cached_property
    def height(self) -> int:
        return int(self.rgb.shape[-3])

    @cached_property
    def width(self) -> int:
        return int(self.rgb.shape[-2])

    @property
    def frame(self):
        return _metadata_grid(self._frame_values, self._camera_values, self._axes, "frame")

    @property
    def camera(self):
        return _metadata_grid(self._frame_values, self._camera_values, self._axes, "camera")

    @property
    def path(self):
        if self._axes == ("frame", "camera"):
            return self._paths
        if self._axes == ("frame",):
            return [row[0] for row in self._paths]
        if self._axes == ("camera",):
            return self._paths[0]
        return self._paths[0][0]

    def __len__(self) -> int:
        if not self._axes:
            return 1
        if self._axes[0] == "frame":
            return len(self._frame_values)
        return len(self._camera_values)

    def _new(
        self,
        *,
        rgb: Tensor,
        w2c: Tensor,
        intrinsic: Tensor,
        time: Tensor,
        frame_values: list[int],
        camera_values: list[int],
        paths: list[list[Path]],
        axes: tuple[str, ...],
    ) -> Self:
        return self.__class__(
            rgb=rgb,
            w2c=w2c,
            intrinsic=intrinsic,
            time=time,
            _frame_values=frame_values,
            _camera_values=camera_values,
            _paths=paths,
            _axes=axes,
        )

    def __getitem__(self, index) -> Self:
        if not self._axes:
            raise IndexError("cannot index a scalar MVBatch")
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > len(self._axes):
            raise IndexError("too many indices for MVBatch")

        frame_positions = list(range(len(self._frame_values)))
        camera_positions = list(range(len(self._camera_values)))
        selections: list[tuple[list[int], bool]] = []
        axes: list[str] = []

        for axis_idx, axis in enumerate(self._axes):
            selector = index[axis_idx] if axis_idx < len(index) else slice(None)
            axis_len = len(self._frame_values) if axis == "frame" else len(self._camera_values)
            positions, remove_axis = _selector_to_positions(axis_len, selector)
            selections.append((positions, remove_axis))
            if axis == "frame":
                frame_positions = positions
            else:
                camera_positions = positions
            if not remove_axis:
                axes.append(axis)

        frame_values = [self._frame_values[i] for i in frame_positions]
        camera_values = [self._camera_values[i] for i in camera_positions]
        paths = [[self._paths[i][j] for j in camera_positions] for i in frame_positions]

        return self._new(
            rgb=_index_tensor_by_axes(self.rgb, selections),
            w2c=_index_tensor_by_axes(self.w2c, selections),
            intrinsic=_index_tensor_by_axes(self.intrinsic, selections),
            time=_index_tensor_by_axes(self.time, selections),
            frame_values=frame_values,
            camera_values=camera_values,
            paths=paths,
            axes=tuple(axes),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def flatten(self):
        if self._axes == ("frame", "camera"):
            for i in range(len(self._frame_values)):
                for j in range(len(self._camera_values)):
                    yield self[i, j]
            return
        for item in self:
            if item._axes:
                yield from item.flatten()
            else:
                yield item

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        return self._new(
            rgb=self.rgb.to(device, non_blocking=non_blocking),
            w2c=self.w2c.to(device, non_blocking=non_blocking),
            intrinsic=self.intrinsic.to(device, non_blocking=non_blocking),
            time=self.time.to(device, non_blocking=non_blocking),
            frame_values=list(self._frame_values),
            camera_values=list(self._camera_values),
            paths=[list(row) for row in self._paths],
            axes=self._axes,
        )

    def pin_memory(self) -> Self:
        return self._new(
            rgb=self.rgb.pin_memory(),
            w2c=self.w2c.pin_memory(),
            intrinsic=self.intrinsic.pin_memory(),
            time=self.time.pin_memory(),
            frame_values=list(self._frame_values),
            camera_values=list(self._camera_values),
            paths=[list(row) for row in self._paths],
            axes=self._axes,
        )

    def cuda(self, *, non_blocking: bool = False) -> Self:
        return self.to("cuda", non_blocking=non_blocking)


class MVDataset:
    """On-demand multiview dataset that decodes RGB views from disk."""

    def __init__(
        self,
        *,
        image_path: Path,
        suffix: str,
        frames: Sequence[int],
        cameras: Sequence[int],
        camera_names: Sequence[str],
        w2c: Tensor,
        source_intrinsic: Tensor,
        distortion: Tensor,
        intrinsic: Tensor,
        time: Tensor,
        paths: list[list[Path]],
        target_sizes: Sequence[tuple[int, int]],
        train_cameras: Sequence[int],
        eval_cameras: Sequence[int],
        camera_split_source: str,
        blank_pixels: bool = False,
        frame_indices: Optional[list[int]] = None,
        camera_indices: Optional[list[int]] = None,
        axes: tuple[str, ...] = ("frame", "camera"),
    ):
        self.image_path = image_path
        self.suffix = suffix
        self._frames = list(frames)
        self._cameras = list(cameras)
        self._camera_names = list(camera_names)
        if len(self._camera_names) != len(self._cameras):
            raise ValueError("camera_names must match cameras")
        self._w2c = w2c.float()
        self._source_intrinsic = source_intrinsic.float()
        self._distortion = distortion.float()
        self._needs_undistort = self._distortion.abs().sum(dim=(-2, -1)) != 0
        self._intrinsic = intrinsic.float()
        self._time = time.float()
        self._paths = paths
        self._target_sizes = list(target_sizes)
        self._train_cameras = [int(camera) for camera in train_cameras]
        self._eval_cameras = [int(camera) for camera in eval_cameras]
        self.camera_split_source = camera_split_source
        self._blank_pixels = blank_pixels
        self._frame_indices = (
            list(range(len(self._frames))) if frame_indices is None else frame_indices
        )
        self._camera_indices = (
            list(range(len(self._cameras))) if camera_indices is None else camera_indices
        )
        self._axes = axes

    @classmethod
    def from_config(cls, config) -> Self:
        return cls.new(
            image_path=config.data.extracted_path,
            calibration_path=config.data.calibration_path,
            frame_range=config.data.frames,
            fps=config.data.fps,
            scale=config.data.scale,
            video_path=config.data.video_path,
            configured_train_cameras=config.data.train_cameras,
            configured_eval_cameras=config.data.eval_cameras,
            configured_camera_split_source=config.data.camera_split_source,
            strict_time_offsets=bool(
                getattr(config.data, "strict_time_offsets", True)
            ),
        )

    @classmethod
    def new(
        cls,
        image_path: PathLike,
        calibration_path: PathLike,
        frame_range: Interval,
        fps: Optional[float] = DEFAULT_FPS,
        scale: float = 1,
        video_path: Optional[PathLike] = None,
        blank_pixels: bool = False,
        configured_train_cameras=None,
        configured_eval_cameras=None,
        configured_camera_split_source=None,
        strict_time_offsets: bool = True,
    ) -> Self:
        image_path = Path(image_path)
        calibration_path = Path(calibration_path)
        if fps is None:
            fps = DEFAULT_FPS
            warnings.warn(f"no fps provided, falling back to {fps}fps")
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {fps!r}")

        frames, cameras, suffix, paths = _scan_mvdataset_images(image_path, frame_range)
        time_offsets = _load_time_offsets(
            cameras=cameras,
            video_path=video_path,
            image_path=image_path,
            strict_time_offsets=bool(strict_time_offsets),
        )

        cameras_by_name, train_cameras, eval_cameras, split_source = _read_camera_setup(
            calibration_path=calibration_path,
            cameras=cameras,
            configured_train_cameras=configured_train_cameras,
            configured_eval_cameras=configured_eval_cameras,
            configured_camera_split_source=configured_camera_split_source,
        )

        w2c, source_intrinsic, distortion, intrinsic, target_sizes = _read_easy_camera_tensors(
            cameras_by_name=cameras_by_name,
            cameras=cameras,
            sample_paths=paths[0],
            scale=scale,
            blank_pixels=blank_pixels,
        )

        w2c_all = w2c.unsqueeze(0).repeat(len(frames), 1, 1, 1)
        intrinsic_all = intrinsic.unsqueeze(0).repeat(len(frames), 1, 1, 1)
        time = torch.empty((len(frames), len(cameras), 1), dtype=torch.float32)
        for i, frame in enumerate(frames):
            time[i] = frame / fps - time_offsets

        allow_positional = list(cameras) == list(range(len(cameras)))
        return cls(
            image_path=image_path,
            suffix=suffix,
            frames=frames,
            cameras=cameras,
            camera_names=[
                _camera_name(
                    cameras_by_name,
                    int(camera),
                    allow_positional=allow_positional,
                )
                for camera in cameras
            ],
            w2c=w2c_all,
            source_intrinsic=source_intrinsic,
            distortion=distortion,
            intrinsic=intrinsic_all,
            time=time,
            paths=paths,
            target_sizes=target_sizes,
            train_cameras=train_cameras,
            eval_cameras=eval_cameras,
            camera_split_source=split_source,
            blank_pixels=blank_pixels,
        )

    def _view(
        self,
        *,
        frame_indices: list[int],
        camera_indices: list[int],
        axes: tuple[str, ...],
    ) -> Self:
        return self.__class__(
            image_path=self.image_path,
            suffix=self.suffix,
            frames=self._frames,
            cameras=self._cameras,
            camera_names=self._camera_names,
            w2c=self._w2c,
            source_intrinsic=self._source_intrinsic,
            distortion=self._distortion,
            intrinsic=self._intrinsic,
            time=self._time,
            paths=self._paths,
            target_sizes=self._target_sizes,
            train_cameras=self._train_cameras,
            eval_cameras=self._eval_cameras,
            camera_split_source=self.camera_split_source,
            blank_pixels=self._blank_pixels,
            frame_indices=frame_indices,
            camera_indices=camera_indices,
            axes=axes,
        )

    @cached_property
    def num_frames(self) -> int:
        return len(self._frame_indices)

    @cached_property
    def num_cameras(self) -> int:
        return len(self._camera_indices)

    @property
    def train_cameras(self) -> list[int]:
        return list(self._train_cameras)

    @property
    def eval_cameras(self) -> list[int]:
        return list(self._eval_cameras)

    def camera_name(self, camera: int) -> str:
        try:
            index = self._cameras.index(int(camera))
        except ValueError as error:
            raise KeyError(f"unknown camera: {camera}") from error
        return self._camera_names[index]

    @cached_property
    def fps(self) -> float:
        if len(self._frame_indices) < 2:
            return 0.0
        t0 = self._time[self._frame_indices[0], self._camera_indices[0]]
        t1 = self._time[self._frame_indices[1], self._camera_indices[0]]
        return 1 / float(t1 - t0)

    @cached_property
    def duration(self) -> float:
        if len(self._frame_indices) < 2:
            return 0.0
        t0 = self._time[self._frame_indices[0], self._camera_indices[0]]
        t1 = self._time[self._frame_indices[1], self._camera_indices[0]]
        tn = self._time[self._frame_indices[-1], self._camera_indices[0]]
        return float(tn - t0 + (t1 - t0))

    @cached_property
    def height(self) -> int:
        return max(self._target_sizes[index][0] for index in self._camera_indices)

    @cached_property
    def width(self) -> int:
        return max(self._target_sizes[index][1] for index in self._camera_indices)

    @property
    def frame(self):
        return _metadata_grid(
            [self._frames[i] for i in self._frame_indices],
            [self._cameras[i] for i in self._camera_indices],
            self._axes,
            "frame",
        )

    @property
    def camera(self):
        return _metadata_grid(
            [self._frames[i] for i in self._frame_indices],
            [self._cameras[i] for i in self._camera_indices],
            self._axes,
            "camera",
        )

    @property
    def path(self):
        paths = [[self._paths[i][j] for j in self._camera_indices] for i in self._frame_indices]
        if self._axes == ("frame", "camera"):
            return paths
        if self._axes == ("frame",):
            return [row[0] for row in paths]
        if self._axes == ("camera",):
            return paths[0]
        return paths[0][0]

    def _squeeze_view_tensor(self, tensor: Tensor) -> Tensor:
        out = tensor
        if "camera" not in self._axes:
            out = out[:, 0]
        if "frame" not in self._axes:
            out = out[0]
        return out

    def _tensor_view(self, tensor: Tensor) -> Tensor:
        out = tensor[self._frame_indices]
        out = out[:, self._camera_indices]
        return self._squeeze_view_tensor(out)

    @property
    def w2c(self) -> Tensor:
        return self._tensor_view(self._w2c)

    @property
    def intrinsic(self) -> Tensor:
        return self._tensor_view(self._intrinsic)

    @property
    def time(self) -> Tensor:
        return self._tensor_view(self._time)

    def _read_image(self, frame_idx: int, camera_idx: int) -> Tensor:
        img_path = self._paths[frame_idx][camera_idx]
        # Prefer OpenCV over torchvision.io.decode_image: the latter can
        # segfault or hang inside forked DataLoader workers on some JPEGs.
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"failed to read image: {img_path}")
        img = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).contiguous()
        if bool(self._needs_undistort[camera_idx]):
            img, _ = colmap_undistort(
                img,
                self._source_intrinsic[camera_idx],
                self._distortion[camera_idx],
                blank_pixels=self._blank_pixels,
            )

        target_h, target_w = self._target_sizes[camera_idx]
        if (img.shape[0], img.shape[1]) != (target_h, target_w):
            img_np = img.numpy()
            img_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img = torch.from_numpy(img_np)
        return img.contiguous()

    @property
    def rgb(self) -> Tensor:
        rows = []
        for frame_idx in self._frame_indices:
            row = [self._read_image(frame_idx, cam_idx) for cam_idx in self._camera_indices]
            rows.append(torch.stack(row, dim=0))
        return self._squeeze_view_tensor(torch.stack(rows, dim=0))

    def load(self, device: torch.device | str | None = None) -> MVBatch:
        batch = MVBatch(
            rgb=self.rgb,
            w2c=self.w2c,
            intrinsic=self.intrinsic,
            time=self.time,
            _frame_values=[self._frames[i] for i in self._frame_indices],
            _camera_values=[self._cameras[i] for i in self._camera_indices],
            _paths=[[self._paths[i][j] for j in self._camera_indices] for i in self._frame_indices],
            _axes=self._axes,
        )
        if device is not None:
            return batch.to(device)
        return batch

    def to(self, device: torch.device | str) -> MVBatch:
        return self.load(device)

    def cuda(self) -> MVBatch:
        return self.to("cuda")

    def __len__(self) -> int:
        if not self._axes:
            return 1
        if self._axes[0] == "frame":
            return len(self._frame_indices)
        return len(self._camera_indices)

    def __getitem__(self, index) -> Self:
        if not self._axes:
            raise IndexError("cannot index a scalar MVDataset")
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > len(self._axes):
            raise IndexError("too many indices for MVDataset")

        frame_indices = list(self._frame_indices)
        camera_indices = list(self._camera_indices)
        axes: list[str] = []

        for axis_idx, axis in enumerate(self._axes):
            selector = index[axis_idx] if axis_idx < len(index) else slice(None)
            current = frame_indices if axis == "frame" else camera_indices
            positions, remove_axis = _selector_to_positions(len(current), selector)
            selected = [current[i] for i in positions]
            if axis == "frame":
                frame_indices = selected
            else:
                camera_indices = selected
            if not remove_axis:
                axes.append(axis)

        return self._view(
            frame_indices=frame_indices,
            camera_indices=camera_indices,
            axes=tuple(axes),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def flatten(self):
        for frame_idx in self._frame_indices:
            for camera_idx in self._camera_indices:
                yield self._view(
                    frame_indices=[frame_idx],
                    camera_indices=[camera_idx],
                    axes=(),
                )

    def at(
        self,
        frame: Optional[slice] = None,
        camera: Optional[slice | Sequence[int]] = None,
    ) -> Self:
        if frame is None:
            frame_indices = list(self._frame_indices)
        else:
            start = frame.start if frame.start is not None else min(self._frames)
            stop = frame.stop if frame.stop is not None else max(self._frames) + 1
            frame_indices = [
                idx for idx in self._frame_indices if start <= self._frames[idx] < stop
            ]

        if camera is None:
            camera_indices = list(self._camera_indices)
        elif isinstance(camera, slice):
            start = camera.start if camera.start is not None else min(self._cameras)
            stop = camera.stop if camera.stop is not None else max(self._cameras) + 1
            camera_indices = [
                idx for idx in self._camera_indices if start <= self._cameras[idx] < stop
            ]
        else:
            wanted = {int(c) for c in camera}
            camera_indices = [idx for idx in self._camera_indices if self._cameras[idx] in wanted]

        return self._view(
            frame_indices=frame_indices,
            camera_indices=camera_indices,
            axes=("frame", "camera"),
        )
