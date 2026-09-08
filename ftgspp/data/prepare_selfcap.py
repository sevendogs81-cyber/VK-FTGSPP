# FreeTimeGS++
# SelfCap / EasyVolcap scene preparation for the FTGSPP EasyMocap workflow.

"""Prepare SelfCap-style scenes (e.g. bar-release) for FTGSPP.

SelfCap releases typically look like::

    scene/
      videos/02.mp4 ...
      pcds/000000.ply ...
      optimized/intri.yml
      optimized/extri.yml
      optimized/sync.json   # optional

This tool extracts frames into the EasyMocap image layout, stages calibration /
time offsets, and links keyframe PLYs::

    <output>/
      images/<camera>/<frame:06d>.jpg
      intri.yml
      extri.yml
      t_offsets.json
      points_stride10/f000000.ply ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for bare environments

    class _NullBar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            return None

    def tqdm(iterable=None, total=None, desc=None):  # type: ignore[misc]
        return iterable if iterable is not None else _NullBar()


FRAME_INDEX_WIDTH = 6
SYNC_FILENAMES = ("sync.json", "timecode.json", "t_offsets.json", "time_offsets.json")
CALIB_DIR_CANDIDATES = ("optimized", ".")
PCD_DIR_CANDIDATES = ("pcds", "dense_pcds")


def _camera_aliases(camera: int) -> tuple[str, ...]:
    aliases = (
        str(camera), f"{camera:02d}", f"{camera:03d}",
        f"{camera:04d}", f"{camera:06d}", f"c{camera}",
        f"c{camera:02d}", f"c{camera:03d}", f"c{camera:04d}",
        f"c{camera:06d}",
    )
    return tuple(dict.fromkeys(aliases))


def keyframe_offsets(num_frames: int, keyframe_stride: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero")
    if keyframe_stride <= 0:
        raise ValueError("keyframe_stride must be greater than zero")
    offsets = list(range(0, num_frames, keyframe_stride))
    if offsets[-1] != num_frames - 1:
        offsets.append(num_frames - 1)
    return offsets


def _resolve_pcds_dir(scene_root: Path, pcds_dir: Path | None) -> Path:
    if pcds_dir is not None:
        path = pcds_dir.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"pcds directory not found: {path}")
        return path
    for name in PCD_DIR_CANDIDATES:
        path = (scene_root / name).resolve()
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"no pcds directory under {scene_root}; expected one of {PCD_DIR_CANDIDATES} "
        "or pass --pcds"
    )


def link_keyframe_points(
    *,
    pcds_dir: Path,
    output_dir: Path,
    start: int,
    stop: int,
    keyframe_stride: int,
) -> Path:
    """Link SelfCap ``000000.ply`` keyframes to FTGSPP ``f000000.ply`` names."""
    num_frames = stop - start
    offsets = keyframe_offsets(num_frames, keyframe_stride)
    output_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for offset in offsets:
        frame = start + offset
        src = pcds_dir / f"{frame:0{FRAME_INDEX_WIDTH}d}.ply"
        dst = output_dir / f"f{frame:0{FRAME_INDEX_WIDTH}d}.ply"
        if not src.is_file():
            raise FileNotFoundError(f"missing keyframe point cloud: {src}")
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        linked += 1
    print(f"[prepare_selfcap] linked {linked} keyframe PLYs -> {output_dir}")
    return output_dir


def _default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 4)))


def _sorted_videos(video_dir: Path) -> list[Path]:
    videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.stem)
    if not videos:
        raise FileNotFoundError(f"no mp4 files found in {video_dir}")
    return videos


def _read_camera_names(intri_path: Path) -> list[str]:
    """Parse EasyMocap/OpenCV FileStorage ``names`` without requiring cv2."""
    if not intri_path.exists():
        raise FileNotFoundError(intri_path)
    text = intri_path.read_text(errors="ignore")
    match = re.search(r"(?m)^names:\s*\n((?:\s*-\s*.+\n)+)", text)
    if match:
        names = []
        for line in match.group(1).splitlines():
            item = re.sub(r"^\s*-\s*", "", line).strip().strip("\"'")
            if item and item != "none":
                names.append(item)
        if names:
            return names

    cams = sorted(set(re.findall(r"(?m)^K_([A-Za-z0-9_]+):", text)))
    if cams:
        return cams
    raise ValueError(f"could not parse camera names from {intri_path}")


def _resolve_calibration_dir(scene_root: Path, calibration_dir: Path | None) -> Path:
    if calibration_dir is not None:
        path = calibration_dir
        if not path.is_absolute():
            path = scene_root / path
        if not (path / "intri.yml").exists() or not (path / "extri.yml").exists():
            raise FileNotFoundError(
                f"calibration dir {path} must contain intri.yml and extri.yml"
            )
        return path.resolve()

    for name in CALIB_DIR_CANDIDATES:
        path = (scene_root / name).resolve() if name != "." else scene_root.resolve()
        if (path / "intri.yml").exists() and (path / "extri.yml").exists():
            return path
    raise FileNotFoundError(
        f"could not find intri.yml/extri.yml under {scene_root} "
        f"(tried: {', '.join(CALIB_DIR_CANDIDATES)})"
    )


def _find_sync_file(scene_root: Path, calibration_dir: Path) -> Path | None:
    for root in (calibration_dir, scene_root, scene_root / "videos"):
        for filename in SYNC_FILENAMES:
            path = root / filename
            if path.is_file():
                return path
    return None


def _camera_ids_from_names(names: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for name in names:
        try:
            ids.append(int(name))
        except ValueError as exc:
            raise ValueError(
                f"camera name {name!r} is not numeric; FTGSPP EasyMocap layout "
                "expects numeric camera directory names"
            ) from exc
    if len(ids) != len(set(ids)):
        raise ValueError(
            "camera names normalize to duplicate numeric IDs: "
            f"{list(names)}"
        )
    return ids


def _llffhold_eval_cameras(camera_ids: Sequence[int], hold: int) -> list[int]:
    if hold <= 0:
        raise ValueError(f"llffhold must be > 0, got {hold}")
    ordered = sorted(int(camera) for camera in camera_ids)
    return [camera for idx, camera in enumerate(ordered) if idx % hold == 0]


def _validate_video_calibration(
    videos: Sequence[Path], calib_names: Sequence[str]
) -> None:
    video_names = {video.stem for video in videos}
    calib_set = set(calib_names)
    missing_videos = sorted(calib_set - video_names)
    missing_calib = sorted(video_names - calib_set)
    if missing_videos or missing_calib:
        parts = []
        if missing_videos:
            parts.append(f"calibration cameras without videos: {missing_videos}")
        if missing_calib:
            parts.append(f"videos without calibration: {missing_calib}")
        raise ValueError("; ".join(parts))


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown stage mode: {mode}")


def stage_calibration(
    *,
    calibration_dir: Path,
    output_dir: Path,
    mode: str = "symlink",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    intri_src = calibration_dir / "intri.yml"
    extri_src = calibration_dir / "extri.yml"
    intri_dst = output_dir / "intri.yml"
    extri_dst = output_dir / "extri.yml"
    _link_or_copy(intri_src, intri_dst, mode)
    _link_or_copy(extri_src, extri_dst, mode)
    return intri_dst, extri_dst


def write_time_offsets(
    *,
    sync_path: Path,
    output_path: Path,
    camera_names: Sequence[str],
) -> Path:
    raw = json.loads(sync_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected object in {sync_path}")
    if len(camera_names) != len(set(camera_names)):
        raise ValueError(f"duplicate camera names: {list(camera_names)}")

    offsets: dict[str, float] = {}
    missing: list[str] = []
    for name in camera_names:
        aliases = _camera_aliases(int(name)) if name.isdigit() else (name,)
        matches = [alias for alias in aliases if alias in raw]
        if not matches:
            missing.append(name)
            continue
        values: list[float] = []
        for key in matches:
            if isinstance(raw[key], bool):
                raise ValueError(
                    f"{sync_path} has a non-numeric time offset for camera "
                    f"{name!r} via key {key!r}: {raw[key]!r}"
                )
            try:
                value = float(raw[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{sync_path} has a non-numeric time offset for camera "
                    f"{name!r} via key {key!r}: {raw[key]!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"{sync_path} has a non-finite time offset for camera "
                    f"{name!r} via key {key!r}: {raw[key]!r}"
                )
            values.append(value)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"{sync_path} has conflicting aliases for camera {name!r}: "
                f"{dict(zip(matches, values))}"
            )
        offsets[name] = values[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(offsets, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if missing:
        print(
            f"[prepare_selfcap] sync file {sync_path} missing {len(missing)} "
            f"camera(s) {missing}; left them absent in {output_path} so "
            "data.strict_time_offsets=true rejects training. Set that config "
            "field to false only to explicitly accept a warned 0-second fallback."
        )
    else:
        print(f"[prepare_selfcap] wrote time offsets to {output_path}")
    return output_path


def _frame_path(images_dir: Path, camera: str, frame: int, suffix: str) -> Path:
    return images_dir / camera / f"{frame:0{FRAME_INDEX_WIDTH}d}{suffix}"


def already_extracted(
    *,
    images_dir: Path,
    cameras: Sequence[str],
    start: int,
    stop: int,
    suffix: str,
) -> bool:
    if stop <= start:
        return False
    for camera in cameras:
        for frame in range(start, stop):
            path = _frame_path(images_dir, camera, frame, suffix)
            if not path.is_file() or path.stat().st_size <= 0:
                return False
    return True


async def _video_length(video: Path) -> int:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nw=1:nk=1",
        str(video),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {video}: {stderr.decode(errors='ignore')}"
        )
    text = stdout.decode().strip()
    if not text or text == "N/A":
        raise RuntimeError(f"could not read nb_frames for {video}")
    return int(text)


async def _probe_fps(video: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nw=1:nk=1",
        str(video),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe fps failed for {video}: {stderr.decode(errors='ignore')}"
        )
    num, den = map(int, stdout.decode().strip().split("/"))
    return num / max(den, 1)


async def _probe_shared_fps(videos: Sequence[Path]) -> float:
    values = await asyncio.gather(*(_probe_fps(video) for video in videos))
    reference = float(values[0])
    mismatches = {
        video.name: float(value)
        for video, value in zip(videos, values)
        if not math.isclose(
            float(value), reference, rel_tol=1e-6, abs_tol=1e-6,
        )
    }
    if mismatches:
        raise ValueError(
            "per-camera video FPS values differ; one shared logical timeline "
            f"cannot represent them: reference={reference}, "
            f"mismatches={mismatches}"
        )
    return reference


async def _extract_one(
    *,
    video: Path,
    camera: str,
    images_dir: Path,
    start: int,
    stop: int,
    suffix: str,
    quality: int,
    sem: asyncio.Semaphore,
    bar,
) -> None:
    async with sem:
        out_dir = images_dir / camera
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(out_dir / f"%0{FRAME_INDEX_WIDTH}d{suffix}")

        vf = f"trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video),
            "-vf",
            vf,
            "-vsync",
            "0",
            "-start_number",
            str(start),
        ]
        if suffix.lower() == ".webp":
            cmd.extend(
                [
                    "-vcodec",
                    "libwebp",
                    "-lossless",
                    "1",
                    "-quality",
                    str(quality),
                ]
            )
        else:
            cmd.extend(["-q:v", str(quality)])
        cmd.append(pattern)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            preexec_fn=os.setpgrp,
        )
        code = await proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed for {video} (exit {code})")

        for frame in range(start, stop):
            path = _frame_path(images_dir, camera, frame, suffix)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(
                    f"{camera}: missing/empty expected frame after extract: {path}"
                )
        bar.update(1)


async def extract_frames(
    *,
    videos: Sequence[Path],
    images_dir: Path,
    start: int,
    stop: int,
    suffix: str = ".jpg",
    quality: int = 2,
    processes: int | None = None,
) -> None:
    if stop <= start:
        raise ValueError(f"empty frame range: start={start}, stop={stop}")

    lengths = await asyncio.gather(*(_video_length(video) for video in videos))
    for video, length in zip(videos, lengths):
        if stop > length:
            raise ValueError(
                f"{video.name} has {length} frames, but stop={stop} exceeds it"
            )

    images_dir.mkdir(parents=True, exist_ok=True)
    workers = _default_workers() if processes is None else max(1, processes)
    sem = asyncio.Semaphore(workers)
    cameras = [video.stem for video in videos]

    with tqdm(total=len(videos), desc="extract cameras") as bar:
        await asyncio.gather(
            *(
                _extract_one(
                    video=video,
                    camera=camera,
                    images_dir=images_dir,
                    start=start,
                    stop=stop,
                    suffix=suffix,
                    quality=quality,
                    sem=sem,
                    bar=bar,
                )
                for video, camera in zip(videos, cameras)
            )
        )


def write_data_config(
    *,
    path: Path,
    name: str,
    root: Path,
    video_path: Path,
    calibration_path: Path,
    extracted_path: Path,
    fps: float,
    scale: float,
    start: int,
    stop: int,
    eval_cameras: Sequence[int],
) -> Path:
    eval_list = ", ".join(str(int(camera)) for camera in eval_cameras)
    text = f"""name: {name}
fps: {fps}
# Synchronization metadata is reconstruction-critical. If an offset file is
# present, require an explicit entry for every discovered camera.
strict_time_offsets: true
root: {root}
video_path: {video_path}
calibration_path: {calibration_path}
extracted_path: {extracted_path}
scale: {scale}
frames:
  start: {start}
  stop: {stop}

# Prepared from SelfCap; empty eval_cameras = train on all views (no hold-out).
train_cameras: null
eval_cameras: [{eval_list}]
camera_split_source: null
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[prepare_selfcap] wrote data config to {path}")
    return path


@dataclass(frozen=True)
class PrepareResult:
    output_root: Path
    images_dir: Path
    calibration_path: Path
    points_path: Path | None
    eval_cameras: list[int]
    config_path: Path | None
    time_offsets_path: Path | None


def prepare_selfcap(
    *,
    scene_root: Path,
    output_root: Path | None = None,
    video_dir: Path | None = None,
    calibration_dir: Path | None = None,
    pcds_dir: Path | None = None,
    start: int = 0,
    stop: int = 300,
    keyframe_stride: int = 10,
    suffix: str = ".jpg",
    quality: int = 2,
    processes: int | None = None,
    stage_mode: str = "symlink",
    llffhold: int = 0,
    no_holdout: bool = True,
    eval_cameras: Sequence[int] | None = None,
    fps: float | None = None,
    scale: float = 0.5,
    name: str | None = None,
    write_config: Path | None = None,
    skip_extract: bool = False,
    skip_points: bool = False,
    dry_run: bool = False,
) -> PrepareResult:
    scene_root = scene_root.resolve()
    output_root = (output_root or scene_root).resolve()
    video_dir = (
        video_dir.resolve()
        if video_dir is not None
        else (scene_root / "videos").resolve()
    )
    if not video_dir.exists():
        fallback = scene_root
        if list(fallback.glob("*.mp4")):
            video_dir = fallback
        else:
            raise FileNotFoundError(f"video directory not found: {video_dir}")

    calibration_dir = _resolve_calibration_dir(scene_root, calibration_dir)
    videos = _sorted_videos(video_dir)
    calib_names = _read_camera_names(calibration_dir / "intri.yml")
    _validate_video_calibration(videos, calib_names)

    camera_names = [video.stem for video in videos]
    camera_ids = _camera_ids_from_names(camera_names)

    if eval_cameras is not None:
        eval_ids = [int(camera) for camera in eval_cameras]
    elif no_holdout or llffhold <= 0:
        eval_ids = []
    else:
        eval_ids = _llffhold_eval_cameras(camera_ids, llffhold)
    if len(eval_ids) != len(set(eval_ids)):
        raise ValueError(f"eval_cameras contains duplicate camera IDs: {eval_ids}")
    unknown = sorted(set(eval_ids) - set(camera_ids))
    if unknown:
        raise ValueError(f"eval_cameras not present in scene: {unknown}")
    if set(eval_ids) == set(camera_ids) and eval_ids:
        raise ValueError(
            "eval_cameras covers all cameras; leave at least one for training "
            "or pass --no-holdout"
        )

    images_dir = output_root / "images"
    points_path = output_root / f"points_stride{keyframe_stride}"
    resolved_pcds: Path | None = None
    if not skip_points:
        resolved_pcds = _resolve_pcds_dir(scene_root, pcds_dir)

    if fps is None:
        fps = asyncio.run(_probe_shared_fps(videos))
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")

    scene_name = name or output_root.name.replace("-release", "").replace("_release", "")
    sync_path = _find_sync_file(scene_root, calibration_dir)

    print("[prepare_selfcap] plan:")
    print(f"  scene_root       = {scene_root}")
    print(f"  output_root      = {output_root}")
    print(f"  videos           = {video_dir} ({len(videos)} cameras)")
    print(f"  calibration_dir  = {calibration_dir}")
    print(f"  images_dir       = {images_dir}")
    print(f"  pcds_dir         = {resolved_pcds}")
    print(f"  points_path      = {points_path}")
    print(f"  frames           = [{start}, {stop})")
    print(f"  keyframe_stride  = {keyframe_stride}")
    print(f"  suffix/quality   = {suffix} / {quality}")
    print(f"  fps/scale        = {fps} / {scale}")
    print(f"  eval_cameras     = {eval_ids}")
    print(f"  sync             = {sync_path}")

    if dry_run:
        return PrepareResult(
            output_root=output_root,
            images_dir=images_dir,
            calibration_path=output_root,
            points_path=None if skip_points else points_path,
            eval_cameras=eval_ids,
            config_path=write_config,
            time_offsets_path=None,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    stage_calibration(
        calibration_dir=calibration_dir,
        output_dir=output_root,
        mode=stage_mode,
    )

    time_offsets_path = None
    if sync_path is not None:
        time_offsets_path = write_time_offsets(
            sync_path=sync_path,
            output_path=output_root / "t_offsets.json",
            camera_names=camera_names,
        )

    if skip_extract:
        print("[prepare_selfcap] skip_extract=True; not writing frames")
    elif already_extracted(
        images_dir=images_dir,
        cameras=camera_names,
        start=start,
        stop=stop,
        suffix=suffix,
    ):
        print(
            f"[prepare_selfcap] frames already present in {images_dir}; skipping extract"
        )
    else:
        asyncio.run(
            extract_frames(
                videos=videos,
                images_dir=images_dir,
                start=start,
                stop=stop,
                suffix=suffix,
                quality=quality,
                processes=processes,
            )
        )

    linked_points: Path | None = None
    if skip_points:
        print("[prepare_selfcap] skip_points=True; not linking keyframe PLYs")
    else:
        assert resolved_pcds is not None
        linked_points = link_keyframe_points(
            pcds_dir=resolved_pcds,
            output_dir=points_path,
            start=start,
            stop=stop,
            keyframe_stride=keyframe_stride,
        )

    config_path = None
    if write_config is not None:
        config_path = write_data_config(
            path=write_config,
            name=scene_name,
            root=output_root,
            video_path=video_dir,
            calibration_path=output_root,
            extracted_path=images_dir,
            fps=float(fps),
            scale=scale,
            start=start,
            stop=stop,
            eval_cameras=eval_ids,
        )

    print("[prepare_selfcap] done")
    if write_config is not None:
        print(f"  next: python -m ftgspp.run data={write_config.stem} pipeline=check")
    else:
        print(f"  prepared root: {output_root}")
    return PrepareResult(
        output_root=output_root,
        images_dir=images_dir,
        calibration_path=output_root,
        points_path=linked_points,
        eval_cameras=eval_ids,
        config_path=config_path,
        time_offsets_path=time_offsets_path,
    )


def _parse_int_list(values: Iterable[str] | None) -> list[int] | None:
    if values is None:
        return None
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a SelfCap-style scene for FTGSPP: extract EasyMocap-layout "
            "images, stage calibration/time offsets, and link keyframe point clouds."
        )
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        required=True,
        help="SelfCap scene root (e.g. .../SelfCap/bar-release)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prepared scene root (default: --scene-root). Contains images/ + calib + points.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory of per-camera mp4 files (default: <scene-root>/videos)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory with intri.yml/extri.yml (default: optimized/ or scene root)",
    )
    parser.add_argument(
        "--pcds",
        type=Path,
        default=None,
        help="Per-frame PLY directory (default: <scene-root>/pcds or dense_pcds)",
    )
    parser.add_argument("--start", type=int, default=0, help="start frame (inclusive)")
    parser.add_argument("--stop", type=int, default=300, help="stop frame (exclusive)")
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        default=10,
        help="keyframe stride for linking pcds -> points_strideN/fXXXXXX.ply",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "webp"),
        default="jpg",
        help="output image format",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="ffmpeg quality (jpg: -q:v, webp: -quality)",
    )
    parser.add_argument(
        "-j",
        "--processes",
        type=int,
        default=None,
        help="parallel ffmpeg workers (default: up to 8)",
    )
    parser.add_argument(
        "--stage-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="how to stage intri.yml/extri.yml into --output",
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=0,
        help="hold out every N-th camera as eval (0 = no hold-out, default)",
    )
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        default=True,
        help="train on all cameras; write eval_cameras: [] (default)",
    )
    parser.add_argument(
        "--holdout",
        action="store_false",
        dest="no_holdout",
        help="enable llffhold-based eval split",
    )
    parser.add_argument(
        "--eval-cameras",
        nargs="+",
        default=None,
        help="explicit eval camera ids, e.g. --eval-cameras 2 11",
    )
    parser.add_argument("--fps", type=float, default=None, help="override fps in config")
    parser.add_argument("--scale", type=float, default=0.5, help="data.scale for config")
    parser.add_argument("--name", type=str, default=None, help="Hydra data.name")
    parser.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="optional path to write configs/data/<name>.yaml",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="stage calibration/config/points only; do not extract frames",
    )
    parser.add_argument(
        "--skip-points",
        action="store_true",
        help="do not link keyframe PLYs from pcds/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the preparation plan without writing files",
    )
    return parser


def main(argv: list[str] | None = None) -> PrepareResult:
    parser = make_parser()
    ns = parser.parse_args(argv)
    suffix = ".jpg" if ns.format == "jpg" else ".webp"
    return prepare_selfcap(
        scene_root=ns.scene_root,
        output_root=ns.output,
        video_dir=ns.video_dir,
        calibration_dir=ns.calibration_dir,
        pcds_dir=ns.pcds,
        start=ns.start,
        stop=ns.stop,
        keyframe_stride=ns.keyframe_stride,
        suffix=suffix,
        quality=ns.quality,
        processes=ns.processes,
        stage_mode=ns.stage_mode,
        llffhold=ns.llffhold,
        no_holdout=ns.no_holdout,
        eval_cameras=_parse_int_list(ns.eval_cameras),
        fps=ns.fps,
        scale=ns.scale,
        name=ns.name,
        write_config=ns.write_config,
        skip_extract=ns.skip_extract,
        skip_points=ns.skip_points,
        dry_run=ns.dry_run,
    )


if __name__ == "__main__":
    main()
