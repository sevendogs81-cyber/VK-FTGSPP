#!/usr/bin/env python3
"""Build the VK-FTGSPP DNA-Rendering cover animation from completed runs.

The expensive rendering step remains explicit so it can be distributed across
available Vulkan devices. Typical usage from the repository root is:

    python dev/tools/make_dna_teaser.py paths
    ftgspp render --run runs/fastgs_eval_v3/0012_09 \
        --camera-path runs/dna_teaser_original_speed/paths/0012_09.json \
        --width 384 --fps 15 --duration 8.0666666667 --device 0 \
        --output runs/dna_teaser_original_speed/render/0012_09
    python dev/tools/make_dna_teaser.py compose

All ten render directories must contain the same numbered PNG frames before
``compose`` is run. The final GIF and MP4 are published atomically.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCENES = (
    "0012_09",
    "0019_06",
    "0025_11",
    "0034_04",
    "0094_02",
    "0124_03",
    "0152_01",
    "0165_08",
    "0188_02",
    "0219_07",
)

# The DNA capture has 16 azimuths and three camera heights. These IDs select
# the middle-height ring in capture order, including the separately ordered 22.
ORBIT_CAMERA_IDS = (25, 28, 31, 34, 37, 40, 43, 46, 1, 4, 7, 10, 13, 16, 19, 22)

FPS = 15
FRAME_COUNT = 121
PATH_DURATION = (FRAME_COUNT - 1) / FPS
EXPORT_DURATION = FRAME_COUNT / FPS
SOURCE_SIZE = 384
TILE_WIDTH = 256
IMAGE_HEIGHT = 288
LABEL_HEIGHT = 32
HEADER_HEIGHT = 64
CANVAS_SIZE = (TILE_WIDTH * 5, HEADER_HEIGHT + (IMAGE_HEIGHT + LABEL_HEIGHT) * 2)
WORK_NAME = "dna_teaser_original_speed"
FOCAL_SCALES = {"0019_06": 0.84}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def scene_camera_map(metadata: dict) -> dict[int, dict]:
    result = {}
    for camera in metadata["cameras"]:
        camera_id = int(camera["name"].split("/", 1)[0].strip())
        result[camera_id] = camera
    missing = set(ORBIT_CAMERA_IDS) - result.keys()
    if missing:
        raise ValueError(f"Missing DNA orbit cameras: {sorted(missing)}")
    return result


def make_path(manifest: Path) -> dict:
    metadata = json.loads(manifest.read_text())["metadata"]
    cameras = scene_camera_map(metadata)
    start = float(metadata["time_start"])
    end = float(metadata["time_end"])
    target = np.asarray(metadata["center"], dtype=np.float64)
    world_up = np.asarray(metadata["world_up"], dtype=np.float64)
    world_up /= np.linalg.norm(world_up)

    positions = {}
    for camera_id in ORBIT_CAMERA_IDS:
        positions[camera_id] = np.linalg.inv(
            np.asarray(cameras[camera_id]["w2c"], dtype=np.float64)
        )[:3, 3]
    offsets = np.stack([positions[camera_id] - target for camera_id in ORBIT_CAMERA_IDS])
    elevations = offsets @ world_up
    planar = offsets - elevations[:, None] * world_up
    orbit_radius = float(np.median(np.linalg.norm(planar, axis=1)))
    elevation = float(np.median(elevations))

    start_direction = planar[0] / np.linalg.norm(planar[0])
    next_direction = planar[1] / np.linalg.norm(planar[1])
    tangent = next_direction - np.dot(next_direction, start_direction) * start_direction
    tangent /= np.linalg.norm(tangent)
    base_camera = cameras[ORBIT_CAMERA_IDS[0]]
    focal_scale = FOCAL_SCALES.get(manifest.parent.name, 1.0)

    keyframes = []
    count = FRAME_COUNT - 1
    for i in range(count + 1):
        u = i / count
        angle = 2 * math.pi * u
        radial = math.cos(angle) * start_direction + math.sin(angle) * tangent
        position = target + elevation * world_up + orbit_radius * radial
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.column_stack((right, down, forward))
        c2w[:3, 3] = position

        camera = copy.deepcopy(base_camera)
        camera["K"][0][0] *= focal_scale
        camera["K"][1][1] *= focal_scale
        camera["w2c"] = np.linalg.inv(c2w).tolist()
        # One rendered frame per original DNA sample: 0..8 s inclusive at 15 FPS.
        camera["timestamp"] = start + (end - start) * u
        camera["name"] = "Smooth DNA orbit"
        keyframes.append({"path_time": PATH_DURATION * u, "camera": camera})
    return {"schema": "ftgspp-camera-path-v1", "keyframes": keyframes}


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def create_paths(repo: Path) -> None:
    run_root = repo / "runs/fastgs_eval_v3"
    output = repo / "runs" / WORK_NAME / "paths"
    for scene in SCENES:
        write_json_atomic(output / f"{scene}.json", make_path(run_root / scene / "viewer_manifest.json"))
    print(f"Wrote {len(SCENES)} closed orbit paths to {output}")


def frame_files(repo: Path) -> list[list[Path]]:
    roots = [repo / "runs" / WORK_NAME / "render" / scene for scene in SCENES]
    frames = [sorted(root.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png")) for root in roots]
    counts = {len(files) for files in frames}
    if counts != {FRAME_COUNT}:
        detail = ", ".join(f"{scene}={len(files)}" for scene, files in zip(SCENES, frames, strict=True))
        raise ValueError(f"Expected {FRAME_COUNT} frames per scene; found {detail}")
    return frames


def foreground_crop(files: list[Path]) -> tuple[int, int, int, int]:
    """Return one stable, maximally tight crop that contains every frame."""
    x0 = y0 = SOURCE_SIZE
    x1 = y1 = 0
    for file in files:
        with Image.open(file) as image:
            array = np.asarray(image.convert("RGB"))
        mask = np.max(array, axis=2) > 8
        columns = np.flatnonzero(np.count_nonzero(mask, axis=0) >= 3)
        rows = np.flatnonzero(np.count_nonzero(mask, axis=1) >= 3)
        if not len(columns) or not len(rows):
            continue
        x0 = min(x0, int(columns[0]))
        x1 = max(x1, int(columns[-1]) + 1)
        y0 = min(y0, int(rows[0]))
        y1 = max(y1, int(rows[-1]) + 1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, SOURCE_SIZE, SOURCE_SIZE)

    margin = round(SOURCE_SIZE * 0.035)
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(SOURCE_SIZE, x1 + margin), min(SOURCE_SIZE, y1 + margin)
    target_aspect = TILE_WIDTH / IMAGE_HEIGHT
    width, height = x1 - x0, y1 - y0
    if width / height < target_aspect:
        wanted = math.ceil(height * target_aspect)
        extra = wanted - width
        x0 -= extra // 2
        x1 += extra - extra // 2
    else:
        wanted = math.ceil(width / target_aspect)
        extra = wanted - height
        y0 -= extra // 2
        y1 += extra - extra // 2

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > SOURCE_SIZE:
        x0 -= x1 - SOURCE_SIZE
        x1 = SOURCE_SIZE
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > SOURCE_SIZE:
        y0 -= y1 - SOURCE_SIZE
        y1 = SOURCE_SIZE
    return (max(0, x0), max(0, y0), min(SOURCE_SIZE, x1), min(SOURCE_SIZE, y1))


def fit_tile(source: Image.Image, crop: tuple[int, int, int, int]) -> Image.Image:
    return source.convert("RGB").crop(crop).resize(
        (TILE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS
    )


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, face, fill) -> None:
    bounds = draw.textbbox((0, 0), text, font=face)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2 - bounds[1]
    draw.text((x, y), text, font=face, fill=fill)


def compose_frame(
    sources: list[Path], crops: list[tuple[int, int, int, int]] | None = None
) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, "#05070b")
    draw = ImageDraw.Draw(canvas)

    title = font(29, bold=True)
    subtitle = font(15)
    metadata = font(15, bold=True)
    draw.text((24, 8), "VK-", font=title, fill="#f7f9ff")
    prefix_width = draw.textlength("VK-", font=title)
    draw.text((24 + prefix_width, 8), "FTGSPP", font=title, fill="#39c6ff")
    draw.text((25, 40), "Native Vulkan 4D Gaussian Splatting", font=subtitle, fill="#9aa7bd")
    right = "DNA-Rendering  ·  10 dynamic scenes  ·  360° views"
    draw.text((CANVAS_SIZE[0] - 24, 32), right, font=metadata, fill="#dce5f5", anchor="rm")
    draw.line((0, HEADER_HEIGHT - 1, CANVAS_SIZE[0], HEADER_HEIGHT - 1), fill="#172033", width=1)

    label = font(16, bold=True)
    crops = crops or [(48, 0, 336, SOURCE_SIZE)] * len(SCENES)
    for index, (scene, source) in enumerate(zip(SCENES, sources, strict=True)):
        row, column = divmod(index, 5)
        x = column * TILE_WIDTH
        y = HEADER_HEIGHT + row * (IMAGE_HEIGHT + LABEL_HEIGHT)
        with Image.open(source) as image:
            canvas.paste(fit_tile(image, crops[index]), (x, y))
        label_y = y + IMAGE_HEIGHT
        draw.rectangle((x, label_y, x + TILE_WIDTH, label_y + LABEL_HEIGHT), fill="#070a10")
        draw.rectangle((x, label_y, x + TILE_WIDTH, label_y + 2), fill="#172033")
        centered_text(draw, (x, label_y, x + TILE_WIDTH, label_y + LABEL_HEIGHT), scene, label, "#f2f5fb")
        if column:
            draw.line((x, y, x, label_y + LABEL_HEIGHT), fill="#101725", width=2)
    return canvas


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def compose(repo: Path) -> None:
    frames = frame_files(repo)
    crops = [foreground_crop(scene_frames) for scene_frames in frames]
    for scene, crop in zip(SCENES, crops, strict=True):
        print(f"{scene} stable crop: {crop}")
    temporary = repo / "runs" / WORK_NAME / "composite.tmp"
    final_frames = repo / "runs" / WORK_NAME / "composite"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for i, sources in enumerate(zip(*frames, strict=True)):
        output = temporary / f"{i:06d}.png"
        compose_frame(list(sources), crops).save(output, optimize=True)
        print(f"Composed {i + 1}/{len(frames[0])}", end="\r", flush=True)
    print()
    if final_frames.exists():
        shutil.rmtree(final_frames)
    temporary.replace(final_frames)

    assets = repo / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    gif_temp = assets / "vk-ftgspp_dna_teaser.gif.tmp"
    mp4_temp = assets / "vk-ftgspp_dna_teaser.mp4.tmp"
    gif_final = assets / "vk-ftgspp_dna_teaser.gif"
    mp4_final = assets / "vk-ftgspp_dna_teaser.mp4"
    for output in (gif_temp, mp4_temp):
        output.unlink(missing_ok=True)
    pattern = str(final_frames / "%06d.png")
    run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-framerate", str(FPS), "-i", pattern,
            "-filter_complex",
            "split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];"
            "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
            "-loop", "0", "-f", "gif", str(gif_temp),
        ]
    )
    run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-framerate", str(FPS), "-i", pattern,
            "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-f", "mp4", str(mp4_temp),
        ]
    )
    gif_temp.replace(gif_final)
    mp4_temp.replace(mp4_final)
    print(f"Published {gif_final} and {mp4_final}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("paths", "compose"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.command == "paths":
        create_paths(args.repo.resolve())
    else:
        compose(args.repo.resolve())


if __name__ == "__main__":
    main()
