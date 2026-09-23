#!/usr/bin/env python3
"""GPU RGBA8 vs NumPy composition, encoding and headless PNG on a real run/NPZ."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from ftgspp import _ftgspp_vk
from ftgspp.viewer.asset import atomic_json, load_asset, scaled_camera
from ftgspp.viewer.offline import compose, export_frames, save_rgba8_png, shader_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Run directory or NPZ")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    asset = load_asset(args.model)
    meta = asset.metadata
    cameras = meta["cameras"] or [meta["initial_camera"]]
    start, end = meta["time_start"], meta["time_end"]
    options = [{}, {"background": [1, 1, 1]}, {"background": [.15, .4, .8]},
               {"checker": True}, {"exposure": 1.3}, {"mode": 1}, {"mode": 2},
               {"transparent": True}, {"transparent": True, "exposure": -1.5}]
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    r = _ftgspp_vk.ViewRenderer(shader_dir(), args.device)
    try:
        r.upload(asset.model.arrays())
        for case, (ci, fraction, zoom) in enumerate(((0, .25, 1), (len(cameras)//2, .5, 1), (0, .5, 2))):
            t = start + fraction * (end - start)
            c = scaled_camera(cameras[ci], 1920, 1080)
            c["K"][0, 0] *= zoom
            c["K"][1, 1] *= zoom
            c["timestamp"] = t
            r.render(c, t, meta["sh_degree"], meta["marginal_threshold"])
            ft = r.readback()
            assert np.isfinite(ft).all() and np.max(ft[..., :3]) > .01
            forward_count = r.stats()["frames"]
            rgba = np.empty(ft.shape, dtype=np.uint8)
            for settings in options:
                r.readback_rgba8_into(rgba, **settings)
                reference = compose(ft, **settings)
                got = rgba if settings.get("transparent") else rgba[..., :3]
                error = int(np.abs(got.astype(np.int16) - reference.astype(np.int16)).max())
                assert error <= 1, (settings, error)
                if not settings.get("transparent"):
                    assert np.all(rgba[..., 3] == 255)
                records.append(dict(camera=ci, time=t, zoom=zoom, display=settings, max_unorm_error=error))
            assert r.stats()["frames"] == forward_count
            np.testing.assert_array_equal(r.readback(), ft)
            r.readback_rgba8_into(rgba, transparent=True)
            file = args.output / f"view_{case}.png"
            save_rgba8_png(file, rgba, transparent=True)
            decoded = cv2.cvtColor(cv2.imread(str(file), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            np.testing.assert_array_equal(decoded, rgba)
        stats = r.stats()
        export_reference = compose(ft, transparent=True)
    finally:
        r.close()
    # Exercise the actual headless export entry, including new RGBA8 readback.
    export_frames(asset, dict(schema="ftgspp-camera-path-v1", keyframes=[dict(path_time=0, camera=c)]),
                  args.output / "export", device=args.device, display=dict(transparent=True))
    image = cv2.imread(str(args.output / "export/000000.png"), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    np.testing.assert_allclose(image, export_reference, atol=1, rtol=0)
    report = dict(model_sha256=asset.sha256, label=meta["label"], checks=len(records),
                  max_unorm_error=max(r["max_unorm_error"] for r in records),
                  original_ft="unchanged", stats=stats, comparisons=records,
                  png_roundtrip="exact", headless_export="PASS")
    atomic_json(args.output / "report.json", report)
    print(f"RGBA8 vs CPU: {len(records)} cases, max error {report['max_unorm_error']}/255; PNG and export: PASS")


if __name__ == "__main__":
    main()
