"""Small scene-RGB CC checks, independent of NAS data or training quality."""
from __future__ import annotations

import unittest

import numpy as np

from tests.reference.composite_ssim import composite_l1_ssim_reference
from tests.test_gpu_foreground_numeric import (
    _packed_gradient, _reference_rgba, _setup_native, _zero_regularizers,
)


def _fixture():
    native, arrays = _setup_native(
        ssim_lambda=0.2, color_correction=True, image_objective=0,
    )
    try:
        native.train_dynamic_view(0, 0)
        state = {key: np.zeros_like(value) for key, value
                 in native.get_optimizer_state().items()}
        native.set_dynamic_gaussians(*arrays)
        params = np.asarray([
            0.08, -0.03, 0.02, 0.01, -0.06, 0.04,
            -0.02, 0.05, 0.03, 0.025, 0.018, 0.012,
        ], np.float32)
        state.update(color_params=params.copy(), color_m=np.zeros_like(params),
                     color_v=np.zeros_like(params))
        native.set_optimizer_state(state)
        _zero_regularizers(native)
        reference = _reference_rgba(32, 32)
        reference[..., 3] = 255
        native.bind_dynamic_view_image(0, True, reference)
        return native, arrays, state, reference, params
    except BaseException:
        native.cleanup()
        raise


class SceneColorCorrectionNumericTests(unittest.TestCase):
    def test_scene_affine_forward_and_image_parameter_gradients(self):
        native, _, _, reference, params = _fixture()
        try:
            native.train_dynamic_view_to_loss(0, 0, True, [0, 0, 0])
            corrected = np.asarray(native.pixel_state, np.float32).copy()
            raw = np.asarray(native.raw_rgb, np.float32).copy()
            matrix = params[:9].reshape(3, 3) + np.eye(3, dtype=np.float32)
            rgb = np.clip(raw[..., :3], 0, 1)
            np.testing.assert_allclose(corrected[..., :3], rgb @ matrix + params[9:],
                                       rtol=2e-6, atol=1e-7)
            # Opaque target + zero background is the scene graph whenever the
            # affine output stays in range, which this fixture explicitly checks.
            self.assertGreater(float(corrected[..., :3].min()), 0)
            self.assertLess(float(corrected[..., :3].max()), 1)
            _, direct = composite_l1_ssim_reference(
                corrected, reference, np.ones((32, 32), np.uint8),
                np.zeros(3, np.float32), ssim_lambda=0.2,
            )
            h = direct[..., :3]
            expected_input = (h @ matrix.T) * ((raw[..., :3] >= 0) & (raw[..., :3] <= 1))
            expected_params = np.concatenate((
                (rgb.reshape(-1, 3).T @ h.reshape(-1, 3)).ravel(),
                h.sum(axis=(0, 1)),
            ))
            native.backward_dynamic_view_from_loss_raw(0, 0)
            np.testing.assert_allclose(np.asarray(native.v_pixel_state)[..., :3],
                                       expected_input, rtol=3e-4, atol=5e-6)
            np.testing.assert_allclose(np.asarray(native.color_grad).ravel(),
                                       expected_params, rtol=4e-4, atol=8e-5)
        finally:
            native.cleanup()

    def test_scene_cc_primitive_gradients_match_finite_differences(self):
        native, arrays, _, _, _ = _fixture()
        try:
            native.backward_dynamic_view(0, 0, False, [0, 0, 0])
            xyz = _packed_gradient(native, "grad_base_xyz_ws").reshape(3, 3)
            so = _packed_gradient(native, "grad_base_log_scales_opacity").reshape(3, 4)
            rot = _packed_gradient(native, "grad_rotations").reshape(3, 4)
            sh = _packed_gradient(native, "grad_sh_coeffs_reordered")
        finally:
            native.cleanup()
        numeric, _, state, reference, _ = _fixture()
        try:
            def loss(values):
                numeric.set_dynamic_gaussians(*values)
                numeric.set_optimizer_state(state)
                numeric.train_dynamic_view_to_loss(0, 0, False, [0, 0, 0])
                image = np.asarray(numeric.pixel_state, np.float32).copy()
                self.assertGreater(float(image[..., :3].min()), 0)
                self.assertLess(float(image[..., :3].max()), 1)
                terms, _ = composite_l1_ssim_reference(
                    image, reference, np.ones((32, 32), np.uint8),
                    np.zeros(3, np.float32), ssim_lambda=0.2,
                )
                numeric.train_dynamic_view_from_loss(0, 0)
                return terms["image_loss"]

            probes = (
                ("xyz", 0, (2, 0), xyz[2, 0]),
                ("log_scale", 1, (0, 1), so[0, 1]),
                ("opacity", 3, (1, 0), so[1, 3]),
                ("sh_dc", 4, (2, 0, 1), sh[4 * 2 + 1]),
                ("rotation", 2, (0, 3), rot[0, 3]),
            )
            for label, group, coordinate, expected in probes:
                estimates = []
                for epsilon in (4e-3, 2e-3, 1e-3):
                    plus = [value.copy() for value in arrays]
                    minus = [value.copy() for value in arrays]
                    plus[group][coordinate] += epsilon
                    minus[group][coordinate] -= epsilon
                    estimates.append((loss(plus) - loss(minus)) / (2 * epsilon))
                np.testing.assert_allclose(
                    estimates, np.full(3, expected), rtol=1e-2, atol=2e-7,
                    err_msg=f"CC primitive finite difference: {label}",
                )
        finally:
            numeric.cleanup()


if __name__ == "__main__":
    unittest.main()
