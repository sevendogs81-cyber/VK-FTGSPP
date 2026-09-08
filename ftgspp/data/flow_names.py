"""UFM/MEMFOF cache path naming (shared by preprocess and native consumers)."""

from __future__ import annotations

from ftgspp.data.utils.images import CAMERA_INDEX_WIDTH, FRAME_INDEX_WIDTH


def keyframe_indices(num_frames: int, keyframe_stride: int) -> list[int]:
    frames = list(range(0, num_frames, keyframe_stride))
    if not frames:
        return [0] if num_frames > 0 else []
    if frames[-1] != num_frames - 1:
        frames.append(num_frames - 1)
    return frames


def to_flow_pair_dir(frame_0: int, frame_1: int) -> str:
    return f"f{frame_0:0{FRAME_INDEX_WIDTH}d}_f{frame_1:0{FRAME_INDEX_WIDTH}d}"


def to_flow_cam_filename(camera: int) -> str:
    return f"c{camera:0{CAMERA_INDEX_WIDTH}d}.npz"
