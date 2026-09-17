#!/usr/bin/env python3
"""Data-free smoke test for the complete native FTGSPP Vulkan contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--shader-dir",
        type=Path,
        default=__import__("ftgspp.training.backend", fromlist=["_shader_dir"])._shader_dir(),
    )
    args = parser.parse_args()

    from ftgspp._ftgspp_vk import FTGSPPVulkan

    native = FTGSPPVulkan()
    for key in ("K", "microsteps", "optimizer_batch_size", "batch_size", "typo"):
        try:
            native.configure_ftgspp({key: 1})
        except ValueError as error:
            assert "unknown FTGSPP config field" in str(error)
        else:
            raise AssertionError(f"removed config accepted: {key}")
    for name in ("finalize_local_gradient", "set_gradient_state_packed", "mean_packed_gradients_into"):
        assert not hasattr(native, name), name
    native.initialize(str(args.shader_dir.resolve()) + "/", args.device)
    try:
        n, height, width = 2, 32, 32
        means = np.array([[-0.05, 0.0, 3.0], [0.05, 0.0, 3.0]], np.float32)
        log_scales = np.tile(
            np.array([[-2.5, -5.5, -2.5]], np.float32), (n, 1)
        )
        quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (n, 1))
        # One live donor and one dead destination exercise native relocation.
        opacity = np.array([[2.0], [-100.0]], np.float32)
        sh = np.zeros((n, 16, 3), np.float32)
        times = np.full((n, 1), 0.5, np.float32)
        durations = np.zeros((n, 1), np.float32)
        velocities = np.tile(np.array([[0.05, 0.0, 0.0]], np.float32), (n, 1))
        gates = np.zeros((n, 1), np.float32)
        native.set_dynamic_gaussians(
            means,
            log_scales,
            quats,
            opacity,
            sh,
            times,
            durations,
            velocities,
            gates,
        )
        native.configure_ftgspp(
            {
                "max_steps": 2,
                "scene_scale": 1.0,
                "ssim_lambda": 0.2,
                "means_lr": 1.6e-4,
                "means_lr_final": 1.6e-6,
                "features_dc_lr": 2.5e-3,
                "features_rest_lr": 1.25e-4,
                "opacities_lr": 5e-2,
                "scales_lr": 5e-3,
                "quats_lr": 1e-3,
                "times_lr": 1.6e-4,
                "durations_lr": 5e-3,
                "velocities_lr": 1.6e-4,
                "velocities_lr_final": 1.6e-6,
                "gates_lr": 1e-3,
                "max_duration": -1.0,
                "marginal_threshold": 0.05,
            }
        )
        # Camera params live on registered views (set_uniforms was removed in
        # the split-backward multi_gpu ABI).
        w2c = np.eye(4, dtype=np.float32)
        reference = np.zeros((height, width, 3), np.uint8)
        intrinsic = np.asarray(
            [[24.0, 0.0, width/2], [0.0, 24.0, height/2], [0.0, 0.0, 1.0]],
            np.float32,
        )
        native.clear_dynamic_views()
        native.add_dynamic_view_meta(
            height, width, w2c, intrinsic, 0.5, 0, 0, True
        )
        native.add_dynamic_view_meta(
            height, width, w2c, intrinsic, 0.55, 1, 1, False
        )
        native.bind_dynamic_view_image(0, True, reference)
        native.bind_dynamic_view_image(0, False, reference)
        if native.num_train != 1 or native.num_val != 1:
            raise AssertionError("native dynamic view registration failed")

        # Color correction: enable after views so camera count is known.
        native.configure_ftgspp(
            {
                "color_correction": True,
                "color_num_cameras": 2,
                "color_lr": 1e-3,
                "color_reg": 1e-3,
                "color_correction_start": -1,
            }
        )

        # A split single-view update must consume exactly one backward once.
        for operation in (lambda: native.optimizer_step(0),
                          lambda: native.train_dynamic_view_from_loss(0, 0),
                          lambda: native.backward_dynamic_view_from_loss_raw(0, 0)):
            try:
                operation()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Adam accepted without backward")
        native.backward_dynamic_view(0, 0)
        for operation in (lambda: native.backward_dynamic_view(0, 0),
                          lambda: native.optimizer_step(1)):
            try:
                operation()
            except RuntimeError:
                pass
            else:
                raise AssertionError("pending generation misuse accepted")
        native.optimizer_step(0)
        try:
            native.optimizer_step(0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Adam consumed gradients twice")
        native.train_dynamic_view_to_loss(0, 1)
        native.train_dynamic_view_from_loss(0, 1)
        try:
            native.train_dynamic_view_from_loss(0, 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("fused Adam consumed a forward loss twice")
        # Reset Gaussian/Adam state for the original one-step fixture.
        native.set_dynamic_gaussians(means, log_scales, quats, opacity, sh,
                                     times, durations, velocities, gates)
        native.set_optimizer_state({key: np.zeros_like(value)
                                    for key, value in native.get_optimizer_state().items()})

        # VkSplat-style one-call path: native-owned image, camera, timestamp,
        # Full native loss scalars, raster backward and explicit FTGSPP optimizer.
        native.train_dynamic_view(0, 0)
        explicit_rgba = np.asarray(native.pixel_state)
        if explicit_rgba.shape != (height, width, 4):
            raise AssertionError(f"unexpected explicit image shape {explicit_rgba.shape}")
        loss = native.get_last_loss()
        loss_terms = (
            "l1", "ssim", "reg_opacity", "reg_scale", "gate", "color_cc",
            "image_loss", "total_loss",
        )
        if not all(np.isfinite(float(loss[key])) for key in loss_terms):
            raise AssertionError(f"native loss reduction returned {loss}")
        expected_total = sum(float(loss[key]) for key in (
            "l1", "ssim", "reg_opacity", "reg_scale", "gate", "color_cc",
        ))
        if not np.isclose(
            float(loss["total_loss"]), expected_total, rtol=1e-5, atol=1e-7,
        ):
            raise AssertionError(
                f"native total_loss does not equal its component sum: {loss}"
            )
        color_state = native.get_optimizer_state()
        if "color_params" not in color_state or np.asarray(color_state["color_params"]).size != 24:
            raise AssertionError("color correction optimizer state missing after train step")
        # Non-zero CC must affect train render only; eval stays on raw RGB.
        zero_params = np.zeros_like(
            np.asarray(color_state["color_params"], dtype=np.float32).reshape(-1)
        )
        nonzero_params = zero_params.copy()
        nonzero_params[0] = 0.5  # camera-0 matrix[0,0] delta
        color_m = np.asarray(color_state["color_m"], dtype=np.float32).reshape(-1).copy()
        color_v = np.asarray(color_state["color_v"], dtype=np.float32).reshape(-1).copy()

        def _set_cc(params: np.ndarray) -> None:
            native.set_optimizer_state(
                {
                    "color_params": params,
                    "color_m": color_m,
                    "color_v": color_v,
                    **{
                        key: value
                        for key, value in color_state.items()
                        if key not in {"color_params", "color_m", "color_v"}
                    },
                }
            )

        _set_cc(zero_params)
        native.render_dynamic_train_view(0)
        train_zero = np.asarray(native.pixel_state)[..., :3].copy()
        native.render_dynamic_val_view(0)
        eval_zero = np.asarray(native.pixel_state)[..., :3].copy()
        if eval_zero.shape != (height, width, 3):
            raise AssertionError("native validation view render failed")

        _set_cc(nonzero_params)
        native.render_dynamic_train_view(0)
        train_cc = np.asarray(native.pixel_state)[..., :3].copy()
        native.render_dynamic_val_view(0)
        eval_cc = np.asarray(native.pixel_state)[..., :3].copy()
        if np.allclose(train_cc, train_zero, atol=1e-6, rtol=0.0):
            raise AssertionError("non-zero color correction failed to affect train render")
        if not np.array_equal(eval_cc, eval_zero):
            raise AssertionError("eval/final render incorrectly applied color correction")

        # Vulkan projected feature + VkSplat tile/alpha blend optical flow.
        native.render_flow_dynamic(0.55, means + velocities * 0.1)
        flow_rgba = np.asarray(native.pixel_state)
        if flow_rgba.shape != (height, width, 4):
            raise AssertionError(f"unexpected flow image shape {flow_rgba.shape}")
        tile_status = int(native.get_tile_invariant_status())
        if tile_status != 0:
            raise AssertionError(
                f"GPU tile invariant failed with status {tile_status}"
            )

        def expect_relocation_rejected(
            mapping: np.ndarray, counts: np.ndarray, expected: str,
        ) -> None:
            try:
                native.relocate_dynamic(
                    mapping.astype(np.int32),
                    counts.astype(np.int32),
                    "partial_copy",
                )
            except Exception as error:
                if expected not in str(error):
                    raise AssertionError(
                        f"unexpected relocation rejection: {error}"
                    ) from error
            else:
                raise AssertionError(
                    f"invalid relocation replay was accepted ({expected})"
                )

        # Replay/broadcast mappings must fail before dispatch when their donor,
        # destination, count, or index contract is invalid.
        expect_relocation_rejected(
            np.array([1, -1]), np.array([0, 1]), "destination is not dead",
        )
        expect_relocation_rejected(
            np.array([-1, 0]), np.array([0, 0]), "counts do not match",
        )
        expect_relocation_rejected(
            np.array([-1, -1]), np.array([51, 0]), "outside [0, 50]",
        )
        expect_relocation_rejected(
            np.array([-1, 2]), np.array([0, 0]), "out-of-range donor",
        )

        relocation = native.relocate_dynamic_native(1, "partial_copy", False, 1234)
        if relocation["all_dead"] or int(relocation["alive"]) != 1:
            raise AssertionError(f"native relocation planning failed: {relocation}")

        model = native.get_dynamic_gaussians()
        optimizer = native.get_optimizer_state()
        if np.asarray(model["means"]).shape != (n, 3):
            raise AssertionError("dynamic model download failed")
        if not optimizer:
            raise AssertionError("optimizer state download failed")
        print(
            "FTGSPP Vulkan smoke passed:",
            f"image={explicit_rgba.shape}",
            f"image_loss={float(loss['image_loss']):.6g}",
            f"relocated={int(relocation['relocated'])}",
            f"optimizer_buffers={len(optimizer)}",
        )
    finally:
        native.cleanup()


if __name__ == "__main__":
    main()
