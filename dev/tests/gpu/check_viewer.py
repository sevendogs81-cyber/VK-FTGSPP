#!/usr/bin/env python3
"""Synthetic GPU equivalence, replay and optional real-desktop lifecycle test."""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--desktop", action="store_true", help="requires a Vulkan-capable X11 display"
    )
    p.add_argument("--output", type=Path, default=Path("build/viewer/acceptance"))
    args = p.parse_args()
    from ftgspp import _ftgspp_vk as vk
    from ftgspp.viewer.asset import inferred_metadata, atomic_json
    from ftgspp.viewer.offline import shader_dir, save_png
    from ftgspp.io.model import NativeGaussians
    from ftgspp.io.viewer_manifest import write_manifest
    from ftgspp.io.hashing import file_sha256

    rng = np.random.default_rng(2026)
    n = 96
    means = rng.normal(0, 0.4, (n, 3)).astype("f4")
    means[:, 2] += 3
    model = NativeGaussians(
        means,
        rng.uniform(-3.8, -2, (n, 3)),
        np.tile([1, 0, 0, 0], (n, 1)),
        np.full((n, 1), 2),
        rng.normal(0, 0.2, (n, 16, 3)),
        rng.uniform(0, 1, (n, 1)),
        rng.uniform(-2, 0.5, (n, 1)),
        rng.normal(0, 0.2, (n, 3)),
        rng.uniform(-1, 1, (n, 1)),
        1.0,
    )
    c = dict(
        width=192,
        height=128,
        K=np.array([[160, 0, 93], [0, 155, 60], [0, 0, 1]], "f4"),
        w2c=np.eye(4, dtype="f4"),
        name="Synthetic",
        timestamp=0.4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "gaussians.npz", **model.arrays())
    metadata = inferred_metadata(model)
    metadata.update(
        cameras=[c],
        initial_camera=c,
        sample_times=[0, 0.25, 0.4, 0.7, 1],
        time_start=0,
        time_end=1,
        label="Synthetic validation",
        inferred=[],
    )
    write_manifest(
        args.output / "gaussians.npz",
        metadata,
        file_sha256(args.output / "gaussians.npz"),
    )
    r = vk.ViewRenderer(shader_dir(), args.device)
    old = vk.FTGSPPVulkan()
    old.initialize(shader_dir() + "/", args.device)
    differences, timing = [], []
    try:
        r.upload(model.arrays())
        old.set_dynamic_gaussians(
            *(
                model.arrays()[key]
                for key in (
                    "means",
                    "log_scales",
                    "quaternions",
                    "opacity_logits",
                    "sh_coefficients",
                    "times",
                    "duration_raw",
                    "velocities",
                    "gate_raw",
                )
            )
        )
        # Nonidentity pose catches transpose/convention errors; all SH variants and moving/gated model.
        for sh in range(4):
            for t in (0.0, 0.4, 1.0):
                c["w2c"][0, 3] = (t - 0.4) * 0.2
                old.configure_ftgspp(
                    dict(max_duration=1.0, marginal_threshold=0.05, max_sh_degree=sh)
                )
                old.clear_dynamic_views()
                old.add_dynamic_view_meta(
                    c["height"], c["width"], c["w2c"], c["K"], t, 0, 0, False
                )
                old.render_dynamic_val_view(0)
                reference = np.asarray(old.pixel_state).copy()
                r.render(c, t, sh, 0.05)
                result = r.readback()
                difference = float(np.max(np.abs(reference - result)))
                differences.append(difference)
                np.testing.assert_allclose(result, reference, atol=2e-6, rtol=2e-6)
        r.render(c, 0.4, 3, 0.05)
        reference = r.readback()
        rgba = np.empty(reference.shape, np.uint8)
        r.readback_rgba8_into(rgba)
        from ftgspp.viewer.offline import compose
        np.testing.assert_allclose(rgba[..., :3], compose(reference), atol=1, rtol=0)
        assert np.all(rgba[..., 3] == 255)
        for bad in (np.empty(reference.shape, np.float32), rgba[:, ::-1],
                    np.empty((1, 1, 4), np.uint8)):
            try:
                r.readback_rgba8_into(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("RGBA8 readback accepted invalid destination")
        rgba.flags.writeable = False
        try:
            r.readback_rgba8_into(rgba)
        except ValueError:
            pass
        else:
            raise AssertionError("RGBA8 readback accepted readonly memory")
        np.testing.assert_array_equal(r.readback(), reference)
        reusable = np.empty_like(reference)
        r.readback_into(reusable)
        np.testing.assert_array_equal(reusable, reference)
        for bad in (np.empty(reference.shape, dtype=np.float64), reusable[:, ::-1],
                    np.empty((1, 1, 4), dtype=np.float32)):
            try:
                r.readback_into(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("readback_into accepted an invalid destination")
        reusable.flags.writeable = False
        try:
            r.readback_into(reusable)
        except ValueError:
            pass
        else:
            raise AssertionError("readback_into accepted readonly memory")
        reusable.flags.writeable = True
        r.render(c, 0.0, 3, 0.05)
        r.readback_into(reusable)
        assert not np.array_equal(reusable, reference)
        r.render(c, 0.4, 3, 0.05)
        np.testing.assert_array_equal(r.readback(), reference)  # independent ownership
        r._set_tile_capacity(1)
        r.render(c, 0.4, 3, 0.05)
        np.testing.assert_array_equal(r.readback(), reference)
        assert r.stats()["replays"] >= 1, r.stats()
        assert r.stats()["tile_pairs"] > 0, r.stats()
        r.upload(model.arrays())
        r.render(c, 0.4)
        np.testing.assert_array_equal(r.readback(), reference)
        rgba.flags.writeable = True
        r.readback_rgba8_into(rgba, transparent=True)
        np.testing.assert_allclose(rgba, compose(reference, transparent=True), atol=1, rtol=0)
        for size in ((256, 144), (64, 96), (192, 128)):
            d = dict(c, width=size[0], height=size[1])
            r.render(d, 0.4)
            assert r.readback().shape == (size[1], size[0], 4)
        r.render(c, 0.4)
        save_png(args.output / "synthetic.png", r.readback(), transparent=True)
        for _ in range(40):
            r.render(c, 0.4)
            timing.append(r.stats()["render_ms"])
        stats = r.stats()
        empty = {
            k: (v[:0] if k != "max_duration" else v) for k, v in model.arrays().items()
        }
        r.upload(empty)
        r.render(c, 0.4)
        ft = r.readback()
        assert np.all(ft[..., :3] == 0) and np.all(ft[..., 3] == 1)
    finally:
        r.close()
        old.cleanup()
    report = dict(
        max_abs_ft_error=max(differences),
        comparisons=len(differences),
        replay=stats["replays"],
        stats=stats,
        forward_ms_p50=float(np.percentile(timing[5:], 50)),
        forward_ms_p95=float(np.percentile(timing[5:], 95)),
        desktop="not requested",
    )
    if args.desktop:
        from ftgspp import _ftgspp_viewer

        app = _ftgspp_viewer.Desktop(shader_dir(), args.device, 1000, 700, False)
        try:
            app.load_scene(model.arrays(), metadata)
            for _ in range(5):
                app.frame()
            before = app.state()["rendered_frames"]
            for _ in range(10):
                app.frame()
            assert app.state()["rendered_frames"] == before, (
                "paused viewer must reuse F/T"
            )
            app.restore_state(dict(background=[1, 1, 1], checker=False))
            for i in range(40):
                if i % 10 == 0:
                    app.resize_window(800 + i * 4, 600 + i * 2)
                app.frame()
            app.load_scene(model.arrays(), metadata)
            app.restore_state(dict(playing=True, interactive=True))
            for _ in range(60):
                app.frame()
            app.restore_state(
                dict(
                    playing=False,
                    interactive=False,
                    camera=c,
                    time=0.4,
                    lock_camera=True,
                )
            )
            np.testing.assert_array_equal(
                app.capture(c["width"], c["height"]), reference
            )
            report["desktop"] = app.state()
        finally:
            app.close()
    atomic_json(args.output / "gpu_report.json", report)
    print(json.dumps(report, default=lambda x: x.tolist(), indent=2))


if __name__ == "__main__":
    main()
