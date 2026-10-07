#!/usr/bin/env python3
"""Render the twelve v8 last checkpoints into a DNA-style 4DAnyone teaser.

Run with the repository's .venv/bin/python:
  dev/tools/make_4danyone_teaser.py render --device 3
  dev/tools/make_4danyone_teaser.py compose

Uses the installed native viewer renderer and its unchanged SH/threshold. Camera
paths follow the calibrated camera ring, including its orientation, rather than
inferred model bounds. All scene timelines span their full 4 or 5 seconds during
the same 5-second orbit (the 4-second clips therefore play at 0.8x). Composition
uses a constant subject scale and padded tiles. Scenes with large translation
use smoothed, containment-constrained recentering to maximize visible subject size.
No Gaussian filtering, pixel masking, retouching, exposure adjustment, or
anisotropic image scaling is applied.
Intermediate frames and provenance stay under runs; assets publish atomically.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from ftgspp.io.hashing import file_sha256
from ftgspp.viewer.asset import atomic_json, load_asset
from ftgspp.viewer.offline import native_module, save_rgba8_png, shader_dir
from ftgspp.viewer.path import CameraPath
from make_dna_teaser import centered_text, font


SCENES = (
    "spiderman", "karina", "aria_lina", "brucelee", "conormcgregor", "feifeili",
    "samaltman", "liangbo", "engineait800", "unitree", "wusan", "xiaoshua",
)
FPS = 30
FRAME_COUNT = 151
PATH_DURATION = (FRAME_COUNT - 1) / FPS
SOURCE_SIZE = (640, 768)
SOURCE_EXTENTS = {"spiderman": (1024, 768), "brucelee": (896, 768), "xiaoshua": (640, 896)}
COLUMNS = 4
TILE_SIZE = (480, 576)
TILE_MARGIN = 12
FOCAL_SCALES = {"xiaoshua": 0.86}
HEADER = 64
LABEL = 32
CANVAS_SIZE = (COLUMNS * TILE_SIZE[0], HEADER + 3 * (TILE_SIZE[1] + LABEL))
RUN_ROOT = Path("runs/fastgs_eval_v8_visual/final_200k")
WORK_ROOT = Path("runs/4danyone_teaser_v8_200k/tight_subject")
ASSET_STEM = "vk-ftgspp_4danyone_teaser"
# User-approved framing exception: keep the visible body and hands; the trailing
# cape may leave the tile. Reviewed around the full orbit and hand extrema.
SPIDERMAN_BODY_CROP = (224, 104, 776, 768)


def spiderman_body_framing() -> dict:
    extent = np.subtract(SPIDERMAN_BODY_CROP[2:], SPIDERMAN_BODY_CROP[:2])
    scale = float(np.min((np.asarray(TILE_SIZE) - 2 * TILE_MARGIN) / extent))
    size = np.floor(extent * scale).astype(int)
    return dict(crop=list(SPIDERMAN_BODY_CROP), fitted_size=size.tolist(),
                offset=((np.asarray(TILE_SIZE) - size) // 2).tolist(),
                scale=scale, relative_to_previous=scale / 0.5, frame_crops=None,
                framing_target="head, torso and arms; trailing cape may leave frame")


def make_path(metadata: dict, scene: str = "") -> dict:
    cameras = {int(c["name"].split("/")[0]): c for c in metadata["cameras"]}
    if set(cameras) != set(range(24)):
        raise ValueError("Expected the complete 24-camera 4DAnyone ring")
    poses = np.linalg.inv(np.asarray([cameras[i]["w2c"] for i in range(24)]))
    center = poses[:, :3, 3].mean(axis=0)
    radial = poses[22, :3, 3] - center
    radius = np.linalg.norm(radial)
    radial /= radius
    tangent = poses[23, :3, 3] - center
    tangent -= np.dot(tangent, radial) * radial
    tangent /= np.linalg.norm(tangent)
    axis = np.cross(radial, tangent)
    # Derive roll and vertical convention from calibration, not metadata.world_up.
    base = copy.deepcopy(cameras[22])
    width, height = SOURCE_EXTENTS.get(scene, SOURCE_SIZE)
    # Expand the aperture without changing focal length or scene magnification.
    scale = SOURCE_SIZE[1] / base["height"]
    k = np.asarray(base["K"], dtype=float)
    k[:2] *= scale
    k[0, 2] += (width - base["width"] * scale) / 2
    k[1, 2] += (height - SOURCE_SIZE[1]) / 2
    # Fixed per-scene framing throughout the orbit; no animated zoom/cropping.
    focal_scale = FOCAL_SCALES.get(scene, 0.94)
    k[0, 0] *= focal_scale
    k[1, 1] *= focal_scale
    base.update(width=width, height=height, K=k.tolist())
    start, end = metadata["time_start"], metadata["time_end"]
    keys = []
    for i in range(FRAME_COUNT):
        u = i / (FRAME_COUNT - 1)
        rotation = Rotation.from_rotvec(axis * (2 * np.pi * u)).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = rotation @ poses[22, :3, :3]
        pose[:3, 3] = center + radius * (rotation @ radial)
        c = dict(base, name="Smooth calibrated 4DAnyone orbit",
                 w2c=np.linalg.inv(pose).tolist(), timestamp=start + (end - start) * u)
        keys.append(dict(path_time=PATH_DURATION * u, camera=c))
    return dict(schema="ftgspp-camera-path-v1", keyframes=keys)


def render(repo: Path, device: int, scenes: list[str]) -> None:
    work = repo / WORK_ROOT
    work.mkdir(parents=True, exist_ok=True)
    renderer = native_module("_ftgspp_vk").ViewRenderer(shader_dir(), device)
    try:
        for scene in scenes:
            extent = SOURCE_EXTENTS.get(scene, SOURCE_SIZE)
            rgba = np.empty((extent[1], extent[0], 4), dtype=np.uint8)
            final = work / "render" / scene
            if final.exists():
                report = json.loads((final / "render.json").read_text())
                if (report["model_sha256"] != file_sha256(repo / RUN_ROOT / scene / "gaussians.npz")
                        or report["frame_count"] != FRAME_COUNT
                        or (report["frames"][0]["camera"]["width"],
                            report["frames"][0]["camera"]["height"]) != extent
                        or len(list(final.glob("[0-9]" * 6 + ".png"))) != FRAME_COUNT):
                    raise ValueError(f"Existing render is incomplete or stale: {final}")
                print(f"Already rendered: {scene}", flush=True)
                continue
            output = final.with_name(scene + ".partial")
            output.mkdir(parents=True, exist_ok=False)
            asset = load_asset(repo / RUN_ROOT / scene)
            data = make_path(asset.metadata, scene)
            atomic_json(work / "paths" / f"{scene}.json", data)
            path = CameraPath(data)
            renderer.upload(asset.model.arrays())
            asset.model = None
            records = []
            for i in range(FRAME_COUNT):
                c = path.sample(i / FPS)
                renderer.render(c, c["timestamp"], asset.metadata["sh_degree"],
                                asset.metadata["marginal_threshold"])
                renderer.readback_rgba8_into(rgba, background=(0, 0, 0))
                file = f"{i:06d}.png"
                save_rgba8_png(output / file, rgba)
                records.append(dict(file=file, path_time=i / FPS, camera=c))
                if i % 30 == 0 or i == FRAME_COUNT - 1:
                    print(f"{scene}: {i + 1}/{FRAME_COUNT}", flush=True)
            atomic_json(output / "render.json", dict(
                schema="ftgspp-render-v1", source_run=str(RUN_ROOT / scene),
                model_sha256=asset.sha256, fps=FPS, frame_count=FRAME_COUNT,
                sh_degree=asset.metadata["sh_degree"],
                marginal_threshold=asset.metadata["marginal_threshold"],
                display=dict(background=[0, 0, 0], exposure=0), frames=records,
            ))
            output.rename(final)
            print(f"Completed {scene}", flush=True)
    finally:
        renderer.close()


def smooth_crop_positions(lo: np.ndarray, hi: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Plan the entire pan in advance, with hard per-frame containment bounds."""
    target = (lo + hi - extent) / 2
    positions = np.empty_like(target)
    for axis in range(2):
        lower, upper = hi[:, axis] - extent[axis], lo[:, axis]

        def objective(x):
            delta, velocity, acceleration = x - target[:, axis], np.diff(x), np.diff(x, n=2)
            gradient = delta.copy()
            gradient[:-1] -= 100 * velocity
            gradient[1:] += 100 * velocity
            gradient[:-2] += 1000 * acceleration
            gradient[1:-1] -= 2000 * acceleration
            gradient[2:] += 1000 * acceleration
            value = (0.5 * np.dot(delta, delta) + 50 * np.dot(velocity, velocity)
                     + 500 * np.dot(acceleration, acceleration))
            return value, gradient

        result = minimize(objective, np.clip(target[:, axis], lower, upper), jac=True,
                          bounds=list(zip(lower, upper)), method="L-BFGS-B",
                          options={"maxiter": 2000, "ftol": 1e-10, "maxls": 30})
        if not result.success:
            raise RuntimeError(f"Unable to plan a smooth contained pan: {result.message}")
        positions[:, axis] = np.clip(np.rint(result.x), lower, upper)
    return positions.astype(int)


def measure_framing(repo: Path) -> dict:
    """Fit all frames at a constant scale, excluding isolated speckles.

    Connected components are only used to measure the framing rectangle; the
    output contains the original RGB pixels and never applies a pixel mask.
    Keep every component with >= 0.2% of the largest component's area, including
    detached hands and props. Add four source pixels before fitting the envelope.
    """
    import cv2

    cv2.setNumThreads(1)
    framing = {}
    for scene in SCENES:
        if scene == "spiderman":
            framing[scene] = spiderman_body_framing()
            continue
        source_extent = SOURCE_EXTENTS.get(scene, SOURCE_SIZE)
        boxes = []
        for i in range(FRAME_COUNT):
            source = repo / WORK_ROOT / "render" / scene / f"{i:06d}.png"
            rgb = cv2.imread(str(source))
            if rgb is None or (rgb.shape[1], rgb.shape[0]) != source_extent:
                raise ValueError(f"Missing or invalid render: {source}")
            mask = (rgb.max(axis=2) > 12).astype(np.uint8)
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            if not len(areas):
                raise ValueError(f"Empty foreground: {source}")
            kept = stats[1:][areas >= max(24, float(areas.max()) * 0.002)]
            lo = kept[:, :2].min(axis=0)
            hi = (kept[:, :2] + kept[:, 2:4]).max(axis=0)
            boxes.append([*lo.tolist(), *hi.tolist()])
        boxes_array = np.asarray(boxes)
        lo = np.maximum(0, boxes_array[:, :2].min(axis=0) - 4)
        hi = np.minimum(source_extent, boxes_array[:, 2:].max(axis=0) + 4)
        crop = [*lo.tolist(), *hi.tolist()]
        extent = hi - lo
        inner = np.asarray(TILE_SIZE) - 2 * TILE_MARGIN
        scale = float(np.min(inner / extent))
        frame_lo = np.maximum(0, boxes_array[:, :2] - 4)
        frame_hi = np.minimum(source_extent, boxes_array[:, 2:] + 4)
        moving_extent = (frame_hi - frame_lo).max(axis=0)
        moving_scale = float(np.min(inner / moving_extent))
        frame_crops = None
        if moving_scale > scale * 1.07:
            # Keep the same scale for the entire clip. The constrained global
            # plan anticipates motion extrema without sudden corrective jumps.
            extent, scale = moving_extent, moving_scale
            left_top = smooth_crop_positions(frame_lo, frame_hi, extent)
            frame_crops = np.column_stack((left_top, left_top + extent)).tolist()
        size = np.maximum(1, np.floor(extent * scale).astype(int))
        offset = (np.asarray(TILE_SIZE) - size) // 2
        framing[scene] = dict(crop=crop, fitted_size=size.tolist(), offset=offset.tolist(),
                              scale=float(scale), relative_to_previous=float(scale / 0.5),
                              frame_bounds=boxes, frame_crops=frame_crops)
        print(f"{scene}: crop {crop}; subject scale {scale / 0.5:.2f}x previous", flush=True)
    return framing


def compose_frame(repo: Path, i: int, framing: dict) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, "#05070b")
    draw = ImageDraw.Draw(canvas)
    title, subtitle, metadata = font(29, bold=True), font(15), font(15, bold=True)
    draw.text((24, 8), "VK-", font=title, fill="#f7f9ff")
    draw.text((24 + draw.textlength("VK-", font=title), 8), "FTGSPP", font=title, fill="#39c6ff")
    draw.text((25, 40), "Native Vulkan 4D Gaussian Splatting", font=subtitle, fill="#9aa7bd")
    draw.text((CANVAS_SIZE[0] - 24, 32), "4DAnyone  ·  12 dynamic scenes  ·  360° views",
              font=metadata, fill="#dce5f5", anchor="rm")
    draw.line((0, HEADER - 1, CANVAS_SIZE[0], HEADER - 1), fill="#172033", width=1)
    for index, scene in enumerate(SCENES):
        row, column = divmod(index, COLUMNS)
        x, y = column * TILE_SIZE[0], HEADER + row * (TILE_SIZE[1] + LABEL)
        with Image.open(repo / WORK_ROOT / "render" / scene / f"{i:06d}.png") as source:
            if source.size != SOURCE_EXTENTS.get(scene, SOURCE_SIZE):
                raise ValueError(f"Unexpected render extent: {source.size}")
            fit = framing[scene]
            tile = Image.new("RGB", TILE_SIZE, "black")
            crop = fit["frame_crops"][i] if fit.get("frame_crops") else fit["crop"]
            subject = source.convert("RGB").crop(crop).resize(
                fit["fitted_size"], Image.Resampling.LANCZOS)
            tile.paste(subject, tuple(fit["offset"][:2]))
            canvas.paste(tile, (x, y))
        label_y = y + TILE_SIZE[1]
        draw.rectangle((x, label_y, x + TILE_SIZE[0], label_y + LABEL), fill="#070a10")
        draw.line((x, label_y, x + TILE_SIZE[0], label_y), fill="#172033", width=2)
        centered_text(draw, (x, label_y, x + TILE_SIZE[0], label_y + LABEL),
                      scene, font(16, bold=True), "#f2f5fb")
        if column:
            draw.line((x, y, x, label_y + LABEL), fill="#101725", width=2)
    return canvas


def compose(repo: Path, framing: dict | None = None) -> None:
    work = repo / WORK_ROOT
    sources = {}
    for scene in SCENES:
        folder = work / "render" / scene
        report = json.loads((folder / "render.json").read_text())
        if report["frame_count"] != FRAME_COUNT or report["fps"] != FPS:
            raise ValueError(f"Inconsistent render: {scene}")
        if report["model_sha256"] != file_sha256(repo / RUN_ROOT / scene / "gaussians.npz"):
            raise ValueError(f"Model changed: {scene}")
        sources[scene] = dict(model_sha256=report["model_sha256"],
                              run=str(RUN_ROOT / scene),
                              scene_time_start=report["frames"][0]["camera"]["timestamp"],
                              scene_time_end=report["frames"][-1]["camera"]["timestamp"])
    framing = measure_framing(repo) if framing is None else framing
    framing["spiderman"] = spiderman_body_framing()
    atomic_json(work / "framing_subject_fit.json", framing)
    frames = work / "composite_vk_ftgspp.partial"
    frames.mkdir(exist_ok=False)
    for i in range(FRAME_COUNT):
        compose_frame(repo, i, framing).save(frames / f"{i:06d}.png", compress_level=1)
        if i % 30 == 0:
            print(f"Composed {i + 1}/{FRAME_COUNT}", flush=True)
    complete = work / "composite_vk_ftgspp"
    frames.rename(complete)
    assets = repo / "docs/assets"
    pattern = str(complete / "%06d.png")
    outputs = {}
    for suffix, options in (
        ("mp4", ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart"]),
        ("gif", ["-filter_complex", "fps=15,split[s0][s1];"
                 "[s0]palettegen=max_colors=256:stats_mode=diff[p];"
                 "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-loop", "0"]),
    ):
        temporary = assets / f".{ASSET_STEM}.partial.{suffix}"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", "4",
                        "-framerate", str(FPS), "-i", pattern, *options, str(temporary)], check=True)
        # Decode the complete encoded file before publishing it.
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(temporary),
                        "-f", "null", "-"], check=True)
        final = assets / f"{ASSET_STEM}.{suffix}"
        temporary.replace(final)
        outputs[suffix] = dict(file=final.name, bytes=final.stat().st_size, sha256=file_sha256(final))
        print(f"Published {final}", flush=True)
    poster = assets / f"{ASSET_STEM}.png"
    poster_temporary = poster.with_suffix(".png.partial")
    poster_temporary.write_bytes((complete / "000000.png").read_bytes())
    poster_temporary.replace(poster)
    atomic_json(assets / f"{ASSET_STEM}.json", dict(
        schema="vk-ftgspp-4danyone-teaser-v1", selection="v8 last / 200000 optimizer updates",
        scenes=sources, layout=dict(columns=COLUMNS, rows=3, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1]),
        fps=FPS, frame_count=FRAME_COUNT, duration_seconds=FRAME_COUNT / FPS,
        gif_fps=15, path_duration_seconds=PATH_DURATION, source_extent=list(SOURCE_SIZE),
        source_extents={s: list(SOURCE_EXTENTS.get(s, SOURCE_SIZE)) for s in SCENES},
        framing="calibrated orbit; constant scale per scene; smoothed containment-constrained recentering where needed",
        composition_revision="subject-fit-4col-v5-spiderman-body", tile_margin_pixels=TILE_MARGIN,
        framing_exceptions={"spiderman": "trailing cape excluded from containment requirement"},
        scene_framing={s: {k: v for k, v in fit.items() if k != "frame_bounds"}
                       for s, fit in framing.items()},
        focal_scales={scene: FOCAL_SCALES.get(scene, 0.94) for scene in SCENES},
        timeline="full source interval mapped linearly to 5 seconds; 4s clips 0.8x, 5s clips 1x",
        processing="native RGB on black, subject framing and uniform resize, no pixel masks or Gaussian filtering",
        work_directory=str(WORK_ROOT), outputs=outputs,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "compose"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args()
    if args.command == "render":
        render(args.repo.resolve(), args.device, args.scenes)
    else:
        compose(args.repo.resolve())


if __name__ == "__main__":
    main()
