"""Headless native rendering and explicit full-quality capture/export."""

from __future__ import annotations
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from .asset import atomic_bytes, atomic_json
from .path import CameraPath


def native_module(name):
    try:
        return importlib.import_module("ftgspp." + name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing {name}. Build with scripts/setup.sh"
            + (" --with-viewer" if name == "_ftgspp_viewer" else "")
            + f" ({exc})"
        ) from exc


def shader_dir():
    import ftgspp

    return str(
        Path(
            os.environ.get(
                "FTGSPP_VK_SHADER_DIR", Path(ftgspp.__file__).parent / "shaders"
            )
        )
        .expanduser()
        .resolve()
    )


def compose(
    ft, *, mode=0, background=(0, 0, 0), checker=False, exposure=0, transparent=False
):
    ft = np.asarray(ft, dtype=np.float32)
    if ft.ndim != 3 or ft.shape[2] != 4 or not np.isfinite(ft).all():
        raise ValueError("Expected finite HxWx4 foreground/transmittance")
    if not transparent and mode == 0:
        # Keep only one RGB float image alive for desktop/reference captures.
        rgb = ft[..., :3].copy()
        bg = np.asarray(background, dtype=np.float32)
        if checker:
            y = np.arange(ft.shape[0])[:, None] // 16
            x = np.arange(ft.shape[1])[None, :] // 16
            checks = np.where((x + y) & 1, 0.65, 0.35).astype(np.float32)
            contribution = np.clip(ft[..., 3], 0, 1) * checks
            for channel in range(3):
                np.add(rgb[..., channel], contribution, out=rgb[..., channel])
        elif np.any(bg):
            t_plane = np.clip(ft[..., 3], 0, 1)
            for channel in range(3):
                if bg[channel]:
                    np.add(rgb[..., channel], t_plane * bg[channel], out=rgb[..., channel])
        if exposure:
            np.multiply(rgb, 2.0**exposure, out=rgb)
        np.clip(rgb, 0, 1, out=rgb)
        np.multiply(rgb, 255, out=rgb)
        np.rint(rgb, out=rgb)
        return rgb.astype(np.uint8)
    t = np.clip(ft[..., 3:4], 0, 1)
    alpha = 1 - t
    if transparent:
        if mode != 0:
            raise ValueError("Transparent PNG is supported for RGB mode only")
        # F is premultiplied; PNG stores straight alpha. Never apply a second gamma curve.
        rgb = (
            np.divide(
                ft[..., :3], alpha, out=np.zeros_like(ft[..., :3]), where=alpha > 1e-6
            )
            * 2.0**exposure
        )
        value = np.concatenate((np.clip(rgb, 0, 1), alpha), axis=2)
    elif mode in (1, 2):
        value = np.repeat(alpha if mode == 1 else t, 3, axis=2)
    else:
        raise ValueError("Unknown display mode")
    return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)


def save_png(path, ft, **settings):
    image = compose(ft, **settings)
    image = image[..., [2, 1, 0, 3]] if image.shape[2] == 4 else image[..., ::-1]
    _write_png(path, image)


def save_rgba8_png(path, rgba, *, transparent=False):
    import cv2

    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA if transparent else cv2.COLOR_RGBA2BGR)
    _write_png(path, image)


def _write_png(path, image):
    import cv2

    path = Path(path)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    atomic_bytes(path, encoded.tobytes())


def display_options(state):
    return {
        k: state[k]
        for k in ("mode", "background", "checker", "exposure", "transparent")
        if k in state
    }


def export_frames(
    asset,
    path_data,
    output,
    *,
    device=0,
    width=None,
    height=None,
    fps=30.0,
    duration=None,
    display=None,
    video=False,
    desktop=None,
    progress=None,
):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive")
    path = CameraPath(path_data)
    seconds = path.duration if duration is None else float(duration)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid export duration")
    display = display or {}
    if video and display.get("transparent"):
        raise ValueError("MP4 does not preserve transparency; use PNG sequence")
    ffmpeg = shutil.which("ffmpeg") if video else None
    if video and not ffmpeg:
        raise ValueError("ffmpeg is required for --video; PNG export needs no ffmpeg")
    if display.get("transparent") and display.get("mode", 0) != 0:
        raise ValueError("Transparent PNG requires RGB mode")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count = max(1, int(math.ceil(seconds * fps - 1e-9)))
    saved = desktop.state() if desktop else None
    renderer = None
    rgba = None
    records, timings = [], []
    try:
        if desktop is None:
            renderer = native_module("_ftgspp_vk").ViewRenderer(shader_dir(), device)
            renderer.upload(asset.model.arrays())
            asset.model = None
        for i in range(count):
            c = path.sample(path.times[0] + i / fps, width, height)
            t = c["timestamp"]
            start = time.perf_counter()
            if desktop:
                desktop.restore_state(
                    dict(camera=c, time=t, playing=False, snap=False, lock_camera=True)
                )
                ft = desktop.capture(c["width"], c["height"])
            else:
                renderer.render(
                    c,
                    t,
                    asset.metadata["sh_degree"],
                    asset.metadata["marginal_threshold"],
                )
                shape = (c["height"], c["width"], 4)
                if rgba is None or rgba.shape != shape:
                    rgba = np.empty(shape, dtype=np.uint8)
                renderer.readback_rgba8_into(rgba, **display)
            timings.append((time.perf_counter() - start) * 1000)
            file = f"{i:06d}.png"
            if desktop:
                save_png(output / file, ft, **display)
            else:
                save_rgba8_png(output / file, rgba, transparent=display.get("transparent", False))
            records.append(dict(file=file, path_time=i / fps, camera=c))
            if progress:
                progress(i + 1, count)
        report = dict(
            schema="ftgspp-render-v1",
            model_sha256=asset.sha256,
            fps=fps,
            frame_count=count,
            sh_degree=asset.metadata["sh_degree"] if not saved else saved["sh_degree"],
            marginal_threshold=asset.metadata["marginal_threshold"]
            if not saved
            else saved["marginal_threshold"],
            display=display,
            frames=records,
            render_and_readback_ms=dict(
                p50=float(np.percentile(timings, 50)),
                p95=float(np.percentile(timings, 95)),
            ),
        )
        atomic_json(output / "render.json", report)
        if video:
            # Retain lossless PNGs. Encoded video is explicitly a lossy convenience artifact.
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(output / "%06d.png"),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(output / "video.mp4"),
                ],
                check=True,
            )
        return report
    finally:
        if renderer is not None:
            renderer.close()
        if desktop is not None:
            desktop.restore_state(saved)
