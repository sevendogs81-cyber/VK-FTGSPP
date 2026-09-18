"""Data-free Vulkan numerical acceptance for scene/foreground objectives."""

from __future__ import annotations

import os
import unittest

import numpy as np

from ftgspp.training.backend import _shader_dir
from tests.reference.composite_ssim import composite_l1_ssim_reference


def _native_class():
    try:
        from ftgspp._ftgspp_vk import FTGSPPVulkan
    except ImportError:
        return None
    return FTGSPPVulkan


def _model_arrays():
    means = np.asarray([
        [-0.07, 0.00, 3.00], [0.05, 0.025, 3.06], [0.00, -0.06, 2.94],
    ], np.float32)
    log_scales = np.asarray([
        [-2.25, -2.55, -2.35], [-2.45, -2.15, -2.60], [-2.35, -2.50, -2.20],
    ], np.float32)
    quaternions = np.asarray([
        [0.98, 0.10, 0.04, 0.02],
        [0.96, -0.05, 0.16, 0.08],
        [0.97, 0.12, -0.09, 0.05],
    ], np.float32)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    opacity = np.asarray([[0.8], [0.35], [0.6]], np.float32)
    sh = np.zeros((3, 16, 3), np.float32)
    sh[:, 0] = np.asarray([
        [0.22, 0.10, 0.18], [0.05, 0.25, 0.12], [0.18, 0.08, 0.28],
    ], np.float32)
    times = np.full((3, 1), 0.5, np.float32)
    durations = np.zeros((3, 1), np.float32)
    velocities = np.zeros((3, 3), np.float32)
    gates = np.full((3, 1), -100.0, np.float32)
    return [
        means, log_scales, quaternions, opacity, sh, times, durations,
        velocities, gates,
    ]


def _reference_rgba(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    rgb = np.stack((
        50 + (3 * x + y) % 80,
        70 + (x + 2 * y) % 90,
        90 + (2 * x + 3 * y) % 70,
    ), axis=-1).astype(np.uint8)
    alpha = np.where((x - width / 2) ** 2 + (y - height / 2) ** 2 < 11**2,
                     210, 35).astype(np.uint8)
    return np.concatenate((rgb, alpha[..., None]), axis=-1)


def _setup_native(
    *, ssim_lambda: float, color_correction: bool = False,
    image_objective: int = 1,
):
    cls = _native_class()
    if cls is None:
        raise unittest.SkipTest("native extension _ftgspp_vk is unavailable")
    native = cls()
    native.initialize(
        str(_shader_dir().resolve()) + os.sep,
        int(os.environ.get("FTGSPP_TEST_DEVICE", "0")),
    )
    arrays = _model_arrays()
    native.set_dynamic_gaussians(*arrays)
    native.configure_ftgspp({
        "max_steps": 8,
        "max_sh_degree": 3,
        "scene_scale": 1.0,
        "ssim_lambda": float(ssim_lambda),
        "image_objective": int(image_objective),
        "max_duration": -1.0,
        "marginal_threshold": 0.05,
    })
    height = width = 32
    intrinsic = np.asarray([
        [25.0, 0.0, width / 2], [0.0, 24.0, height / 2], [0.0, 0.0, 1.0],
    ], np.float32)
    native.clear_dynamic_views()
    native.add_dynamic_view_meta(
        height, width, np.eye(4, dtype=np.float32), intrinsic,
        0.5, 0, 0, True,
    )
    reference = _reference_rgba(height, width)
    native.bind_dynamic_view_image(0, True, reference)
    if color_correction:
        native.configure_ftgspp({
            "color_correction": True,
            "color_num_cameras": 1,
            "color_lr": 1e-3,
            "color_reg": 0.0,
            "color_correction_start": -1,
        })
    return native, arrays


def _packed_gradient(native, name: str) -> np.ndarray:
    layout = native.packed_gradient_layout()
    packed = np.empty(native.packed_gradient_size(), np.float32)
    native.gradient_state_packed_into(packed)
    offset, count = layout[name]
    return packed[int(offset):int(offset) + int(count)].copy()


def _zero_regularizers(native) -> None:
    native.configure_ftgspp({
        "scale_reg": 0.0,
        "opacity_reg": 0.0,
        "gate_reg": 0.0,
    })


class ForegroundGpuNumericTests(unittest.TestCase):
    def test_scene_rgb_ignores_reference_alpha(self):
        reference = _reference_rgba(32, 32)
        opaque = reference.copy()
        opaque[..., 3] = 255
        transparent = reference.copy()
        transparent[..., 3] = 0
        observed = []
        for value in (opaque, transparent):
            native, _arrays = _setup_native(
                ssim_lambda=0.2, image_objective=0,
            )
            try:
                native.bind_dynamic_view_image(0, True, value)
                native.train_dynamic_view_to_loss(
                    0, 0, True, [0.73, 0.31, 0.19],
                )
                observed.append((
                    native.get_last_loss(),
                    np.asarray(native.v_pixel_state, np.float32).copy(),
                ))
                native.backward_dynamic_view_from_loss_raw(0, 0)
            finally:
                native.cleanup()
        self.assertEqual(observed[0][0], observed[1][0])
        np.testing.assert_array_equal(observed[0][1], observed[1][1])

    def test_occluding_splats_full_backward_matches_multistep_finite_difference(self):
        """Check loss -> raster -> projection for every spatial/image group.

        The three anisotropic splats overlap in screen space at different
        depths.  Central differences use several stable step sizes and the
        independent FP64 loss, avoiding the L1/clamp/sort discontinuities.
        """

        background = np.asarray([0.21, 0.47, 0.68], np.float32)
        validity = np.ones((32, 32), np.uint8)
        analytic_native, arrays = _setup_native(ssim_lambda=0.2)
        try:
            _zero_regularizers(analytic_native)
            analytic_native.backward_dynamic_view(
                0, 0, False, background.tolist(),
            )
            means_grad = _packed_gradient(
                analytic_native, "grad_base_xyz_ws",
            ).reshape(3, 3)
            scale_opacity_grad = _packed_gradient(
                analytic_native, "grad_base_log_scales_opacity",
            ).reshape(3, 4)
            rotation_grad = _packed_gradient(
                analytic_native, "grad_rotations",
            ).reshape(3, 4)
            # At step zero only SH degree zero is active.  The native
            # reordered layout stores float4(di=0) for splats 0,1,2 at the
            # first three subgroup slots; xyz are the DC RGB derivatives.
            reordered_sh = _packed_gradient(
                analytic_native, "grad_sh_coeffs_reordered",
            )
            sh_dc_grad = np.stack([
                reordered_sh[4 * index:4 * index + 3]
                for index in range(3)
            ])
        finally:
            analytic_native.cleanup()

        numeric_native, _ = _setup_native(ssim_lambda=0.2)
        try:
            _zero_regularizers(numeric_native)

            def image_loss(values) -> float:
                numeric_native.set_dynamic_gaussians(*values)
                numeric_native.train_dynamic_view_to_loss(
                    0, 0, False, background.tolist(),
                )
                state = np.asarray(
                    numeric_native.pixel_state, np.float32,
                ).copy()
                terms, _gradient = composite_l1_ssim_reference(
                    state, _reference_rgba(32, 32), validity, background,
                    ssim_lambda=0.2,
                )
                # Consume the pending forward. The optimizer mutation is
                # irrelevant because the next call restores every model array.
                numeric_native.train_dynamic_view_from_loss(0, 0)
                return float(terms["image_loss"])

            probes = (
                ("xyz", 0, (2, 0), float(means_grad[2, 0])),
                ("log_scale", 1, (0, 1), float(scale_opacity_grad[0, 1])),
                ("opacity_logit", 3, (1, 0), float(scale_opacity_grad[1, 3])),
                ("sh_dc", 4, (2, 0, 1), float(sh_dc_grad[2, 1])),
                ("rotation", 2, (0, 3), float(rotation_grad[0, 3])),
            )
            for label, array_index, coordinate, expected in probes:
                estimates = []
                for epsilon in (4e-3, 2e-3, 1e-3):
                    plus = [value.copy() for value in arrays]
                    minus = [value.copy() for value in arrays]
                    plus[array_index][coordinate] += np.float32(epsilon)
                    minus[array_index][coordinate] -= np.float32(epsilon)
                    estimates.append(
                        (image_loss(plus) - image_loss(minus))
                        / (2.0 * epsilon)
                    )
                np.testing.assert_allclose(
                    estimates,
                    np.full(3, expected),
                    rtol=1e-2,
                    atol=2e-7,
                    err_msg=f"multi-step finite difference failed for {label}",
                )
        finally:
            numeric_native.cleanup()

    def test_fused_and_split_adam_match_every_array_with_same_background(self):
        fused, _ = _setup_native(ssim_lambda=0.2)
        split, _ = _setup_native(ssim_lambda=0.2)
        background = [0.19, 0.43, 0.73]
        try:
            fused.train_dynamic_view(0, 0, True, background)
            split.backward_dynamic_view(0, 0, True, background)
            split.optimizer_step(0)
            for label, getter in (
                ("model", "get_dynamic_gaussians"),
                ("optimizer", "get_optimizer_state"),
            ):
                expected = getattr(fused, getter)()
                actual = getattr(split, getter)()
                self.assertEqual(set(expected), set(actual))
                for name in expected:
                    np.testing.assert_allclose(
                        actual[name], expected[name], rtol=2e-6, atol=2e-7,
                        err_msg=f"{label}.{name}",
                    )
            self.assertEqual(fused.get_last_loss(), split.get_last_loss())
        finally:
            fused.cleanup()
            split.cleanup()

    def test_composite_color_correction_full_pixel_and_parameter_vjp(self):
        native, arrays = _setup_native(
            ssim_lambda=0.2, color_correction=True,
        )
        try:
            # Optimizer/CC buffers are allocated lazily by the first train
            # step.  Restore the exact fixture afterwards so this throwaway
            # allocation cannot affect the numerical comparison below.
            native.train_dynamic_view(0, 0)
            state = native.get_optimizer_state()
            native.set_dynamic_gaussians(*arrays)
            state = {
                key: np.zeros_like(np.asarray(value, dtype=np.float32))
                for key, value in state.items()
            }
            params = np.asarray([
                0.08, -0.03, 0.02,
                0.01, -0.06, 0.04,
                -0.02, 0.05, 0.03,
                0.025, -0.018, 0.012,
            ], np.float32)
            state["color_params"] = params.copy()
            state["color_m"] = np.zeros_like(params)
            state["color_v"] = np.zeros_like(params)
            native.set_optimizer_state(state)
            validity = np.ones((32, 32), np.uint8)
            native.bind_dynamic_view_validity(0, True, validity)
            background = np.asarray([0.21, 0.47, 0.68], np.float32)
            native.train_dynamic_view_to_loss(0, 0, True, background.tolist())
            corrected = np.asarray(native.pixel_state, np.float32).copy()
            raw = np.asarray(native.raw_rgb, np.float32).copy()
            matrix = params[:9].reshape(3, 3) + np.eye(3, dtype=np.float32)
            bias = params[9:]
            expected_corrected = raw.copy()
            expected_corrected[..., :3] = (
                raw[..., :3] @ matrix
                + (1.0 - raw[..., 3, None]) * bias
            )
            np.testing.assert_allclose(
                corrected, expected_corrected, rtol=2e-6, atol=1e-7,
            )
            _terms, direct_gradient = composite_l1_ssim_reference(
                corrected, _reference_rgba(32, 32), validity, background,
                ssim_lambda=0.2,
            )
            h = direct_gradient[..., :3]
            expected_input = direct_gradient.copy()
            expected_input[..., :3] = h @ matrix.T
            expected_input[..., 3] -= np.sum(h * bias, axis=-1)
            expected_color = np.concatenate((
                (raw[..., :3].reshape(-1, 3).T @ h.reshape(-1, 3)).reshape(-1),
                np.sum((1.0 - raw[..., 3, None]) * h, axis=(0, 1)),
            ))
            native.backward_dynamic_view_from_loss_raw(0, 0)
            gpu_input = np.asarray(native.v_pixel_state, np.float32).copy()
            gpu_color = np.asarray(native.color_grad, np.float32).reshape(-1).copy()
            np.testing.assert_allclose(
                gpu_input, expected_input.astype(np.float32),
                rtol=3e-4, atol=5e-6,
            )
            np.testing.assert_allclose(
                gpu_color, expected_color.astype(np.float32),
                rtol=4e-4, atol=8e-5,
            )
        finally:
            native.cleanup()

    def test_composite_ssim_scalar_and_pixel_gradient_match_fp64(self):
        native, _arrays = _setup_native(ssim_lambda=0.2)
        try:
            validity = np.ones((32, 32), np.uint8)
            validity[1:3, 1:4] = 0
            validity[15, 15] = 0
            native.bind_dynamic_view_validity(0, True, validity)
            background = np.asarray([0.17, 0.43, 0.71], np.float32)
            native.train_dynamic_view_to_loss(
                0, 0, True, background.tolist(),
            )
            state = np.asarray(native.pixel_state, np.float32).copy()
            gpu_gradient = np.asarray(native.v_pixel_state, np.float32).copy()
            gpu_loss = native.get_last_loss()
            expected_loss, expected_gradient = composite_l1_ssim_reference(
                state, _reference_rgba(32, 32), validity, background,
                ssim_lambda=0.2,
            )
            self.assertAlmostEqual(
                float(gpu_loss["l1"]), expected_loss["l1"], delta=5e-6,
            )
            self.assertAlmostEqual(
                float(gpu_loss["ssim"]), expected_loss["ssim"], delta=5e-6,
            )
            np.testing.assert_allclose(
                gpu_gradient, expected_gradient.astype(np.float32),
                rtol=2e-4, atol=4e-6,
            )
            native.backward_dynamic_view_from_loss_raw(0, 0)
        finally:
            native.cleanup()

    def test_validity_without_ssim_window_is_l1_only(self):
        native, _arrays = _setup_native(ssim_lambda=0.8)
        try:
            validity = np.zeros((32, 32), np.uint8)
            validity[14:18, 14:18] = 1
            native.bind_dynamic_view_validity(0, True, validity)
            background = np.asarray([0.2, 0.4, 0.6], np.float32)
            native.train_dynamic_view_to_loss(0, 0, True, background.tolist())
            state = np.asarray(native.pixel_state, np.float32).copy()
            gradient = np.asarray(native.v_pixel_state, np.float32).copy()
            loss = native.get_last_loss()
            expected_loss, expected_gradient = composite_l1_ssim_reference(
                state, _reference_rgba(32, 32), validity, background,
                ssim_lambda=0.8,
            )
            self.assertEqual(float(loss["ssim"]), 0.0)
            self.assertAlmostEqual(float(loss["l1"]), expected_loss["l1"], delta=2e-6)
            np.testing.assert_allclose(
                gradient, expected_gradient.astype(np.float32),
                rtol=2e-5, atol=3e-7,
            )
            active = np.any(np.abs(gradient) > 1e-10, axis=-1)
            np.testing.assert_array_equal(active, validity.astype(bool))
            native.backward_dynamic_view_from_loss_raw(0, 0)
        finally:
            native.cleanup()


if __name__ == "__main__":
    unittest.main()
