#!/usr/bin/env python3
"""Repeatable viewer CPU/GPU transport measurements; no desktop required."""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ftgspp.viewer.app import parser_for
from ftgspp.viewer.asset import atomic_json, load_asset, scaled_camera
from ftgspp.viewer.offline import compose
from ftgspp.viewer.remote import RemoteViewer, wire


def summary(values):
    return dict(p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="NPZ or run directory (recovers calibration)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--camera", type=int, help="Capture camera index")
    p.add_argument("--time", type=float, help="Initial scene time")
    p.add_argument("--time-step", type=float, default=.0001, help="Seconds per moving frame")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.samples < 5:
        p.error("Use at least five measured frames")
    cli = [
        "--model", str(args.model), "--device", str(args.device), "--remote", "--no-session",
    ]
    if args.camera is not None:
        cli += ["--camera", str(args.camera)]
    if args.time is not None:
        cli += ["--time", str(args.time)]
    options = parser_for("viewer").parse_args(cli)
    started = perf_counter()
    asset = load_asset(args.model)
    load_ms = (perf_counter() - started) * 1000
    viewer = RemoteViewer(options, asset)
    started = perf_counter()
    viewer.start_renderer()
    upload_ms = (perf_counter() - started) * 1000
    state = wire(viewer.state)
    gpu_pack = hasattr(viewer.renderer, "readback_rgba8_into")
    report = dict(model_sha256=asset.sha256, label=asset.metadata["label"],
                  device=args.device, samples=args.samples, warmup=5,
                  camera=state["camera"], initial_time=state["time"], time_step=args.time_step,
                  camera_source=asset.metadata.get("camera_source", "manifest or inferred"),
                  camera_count=len(asset.metadata["cameras"]),
                  load_ms=load_ms, initialize_upload_ms=upload_ms,
                  pixel_path="GPU RGBA8" if gpu_pack else "FP32 F/T + CPU compose",
                  scope="host timings; no network/browser; model-specific", extents={})
    try:
        for w, h in ((960, 540), (1440, 810), (1920, 1080)):
            camera = scaled_camera(state["camera"], w, h)
            stages = ("forward", "pack_readback", "jpeg") if gpu_pack else ("forward", "readback", "compose", "jpeg")
            phases = {name: [] for name in (*stages, "total")}
            rgba = np.empty((h, w, 4), dtype=np.uint8) if gpu_pack else None
            for i in range(args.samples + 5):
                times = [perf_counter()]
                viewer.renderer.render(camera, state["time"] + i * args.time_step,
                                       state["sh_degree"], state["marginal_threshold"])
                times.append(perf_counter())
                if gpu_pack:
                    viewer.renderer.readback_rgba8_into(rgba)
                    times.append(perf_counter())
                    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                else:
                    ft = viewer.renderer.readback()
                    times.append(perf_counter())
                    rgb = compose(ft)
                    times.append(perf_counter())
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, _ = cv2.imencode(".jpg", bgr,
                                     [cv2.IMWRITE_JPEG_QUALITY, 88])
                assert ok
                times.append(perf_counter())
                if i >= 5:
                    for j, name in enumerate(stages):
                        phases[name].append((times[j + 1] - times[j]) * 1000)
                    phases["total"].append((times[-1] - times[0]) * 1000)
            extent = {key: summary(value) for key, value in phases.items()}
            for scenario in ("moving", "identical", "display_only"):
                samples = []
                frames = viewer.renderer.stats()["frames"]
                for i in range(args.samples + 5):
                    current = dict(state)
                    if scenario == "moving":
                        current["time"] += i * args.time_step
                    if scenario == "display_only":
                        current["exposure"] = (i % 2) * .25
                    _, elapsed = viewer.render_image(current, w, h)
                    if i >= 5:
                        samples.append(elapsed)
                extent[scenario] = dict(**summary(samples),
                    forward_calls=viewer.renderer.stats()["frames"] - frames)
            report["extents"][f"{w}x{h}"] = extent
        report["renderer_stats"] = viewer.renderer.stats()
        report["control_update"] = {}
        for count in (0, 100, 256):
            viewer.apply_update(dict(state=dict(bookmarks=[state["camera"]] * count,
                                                path_times=list(range(count)))))
            values = []
            for _ in range(args.samples):
                start = perf_counter()
                viewer.apply_update(dict(state=dict(time=state["time"])))
                values.append((perf_counter() - start) * 1000)
            report["control_update"][str(count)] = summary(values)
    finally:
        viewer.renderer.close()
    atomic_json(args.output, report)
    print(f"Saved viewer timing report: {args.output}")


if __name__ == "__main__":
    main()
