from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ftgspp.io.checkpoint import load_model, save_model
from ftgspp.training.backend import NativeVulkanBackend, _shader_dir
from tests.reference.gradients import GradientLayoutReference
from ftgspp.data.scene import route_camera_split
from ftgspp.io.model import NativeGaussians
from ftgspp.io.ply import export_4dgs
from ftgspp.training.runner import _EvalSsimWorkspace, _pair_exchange_log_fields


def make_model(n: int = 3) -> NativeGaussians:
    quaternions = np.zeros((n, 4), np.float32)
    quaternions[:, 0] = 1
    return NativeGaussians(
        means=np.arange(n*3, dtype=np.float32).reshape(n, 3),
        log_scales=np.zeros((n, 3), np.float32),
        quaternions=quaternions,
        opacity_logits=np.zeros((n, 1), np.float32),
        sh_coefficients=np.zeros((n, 16, 3), np.float32),
        times=np.zeros((n, 1), np.float32),
        duration_raw=np.zeros((n, 1), np.float32),
        velocities=np.ones((n, 3), np.float32),
        gate_raw=np.full((n, 1), -100, np.float32),
        max_duration=5.0,
    )


class NativeRuntimeTests(unittest.TestCase):
    def test_holdout_eval_uses_ssim2_data_range(self):
        prediction = np.zeros((13, 15, 3), dtype=np.float32)
        reference = np.ones_like(prediction)
        actual = _EvalSsimWorkspace().ssim(prediction, reference)
        c1 = (0.01 * 2.0) ** 2
        expected = c1 / (1.0 + c1)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_pair_exchange_codec_fields_are_ready_for_train_jsonl(self):
        timing, metadata = _pair_exchange_log_fields({
            "pair_exchange": {
                "payload_bytes": 120,
                "peer_payload_bytes": 240,
                "bucket_count": 2,
                "local_visible_count": 3,
                "peer_visible_count": 5,
                "codec_policy": "auto_exact",
                "gaussian_codec": "exact_sorted_sparse",
                "sparse": True,
                "active_sh_degree": 2,
                "active_sh_floats": 27,
            },
        })

        self.assertEqual(timing["pair_payload_bytes"], 120.0)
        self.assertEqual(timing["pair_peer_visible"], 5.0)
        self.assertEqual(metadata, {
            "pair_codec_policy": "auto_exact",
            "pair_gaussian_codec": "exact_sorted_sparse",
            "pair_sparse": True,
            "pair_active_sh_degree": 2,
            "pair_active_sh_floats": 27,
        })

    def test_canonical_gradient_round_trip_including_reordered_sh(self):
        num_gaussians = 19
        padded_gaussians = 32
        counts = {
            "grad_base_xyz_ws": 3 * num_gaussians,
            "grad_base_log_scales_opacity": 4 * num_gaussians,
            "grad_times_durations": 2 * num_gaussians,
            "grad_velocities_gates": 4 * num_gaussians,
            "grad_sh_coeffs_reordered": 48 * padded_gaussians,
            "grad_rotations": 4 * num_gaussians,
        }
        offset = 0
        layout: dict[str, object] = {
            "abi_version": 2,
            "sh_reorder_size": 16,
        }
        for name, count in counts.items():
            layout[name] = (offset, count)
            offset += count
        layout["total_floats"] = offset
        rng = np.random.default_rng(7)
        canonical = {
            "means": rng.standard_normal((num_gaussians, 3)).astype(np.float32),
            "log_scales": rng.standard_normal((num_gaussians, 3)).astype(np.float32),
            "opacity_logits": rng.standard_normal((num_gaussians, 1)).astype(np.float32),
            "times": rng.standard_normal((num_gaussians, 1)).astype(np.float32),
            "duration_raw": rng.standard_normal((num_gaussians, 1)).astype(np.float32),
            "velocities": rng.standard_normal((num_gaussians, 3)).astype(np.float32),
            "gate_raw": rng.standard_normal((num_gaussians, 1)).astype(np.float32),
            "sh_coefficients": rng.standard_normal(
                (num_gaussians, 16, 3),
            ).astype(np.float32),
            "quaternions": rng.standard_normal((num_gaussians, 4)).astype(np.float32),
        }
        reference = GradientLayoutReference(layout)
        packed = reference.canonical_gradient_to_packed(canonical)
        restored = GradientLayoutReference(layout, packed).canonical_gradient_state()

        for name, expected in canonical.items():
            np.testing.assert_array_equal(restored[name], expected, err_msg=name)

    def test_camera_split_routes_only_selected_ids(self):
        train_ids, eval_ids = {2}, {19}
        camera_ids = list(range(17)) + [19]
        frames = 300
        train_views = eval_views = skipped = 0
        for _ in range(frames):
            for camera_id in camera_ids:
                route = route_camera_split(camera_id, train_ids, eval_ids)
                if route == "train":
                    train_views += 1
                elif route == "eval":
                    eval_views += 1
                else:
                    skipped += 1
        self.assertEqual(train_views, 300)
        self.assertEqual(eval_views, 300)
        self.assertEqual(skipped, 300 * 16)
        # Full training: empty eval keeps every camera in train.
        self.assertEqual(route_camera_split(5, set(range(17)), set()), "train")
        self.assertIsNone(route_camera_split(5, {2}, {19}))

    def test_color_camera_map_is_dense(self):
        """Sparse camera IDs must map to 0..N-1 for ColorCorrectors compatibility."""
        cameras = [2, 5, 19]
        camera_ids = sorted({int(c) for c in cameras})
        camera_to_index = {cid: i for i, cid in enumerate(camera_ids)}
        self.assertEqual(camera_to_index, {2: 0, 5: 1, 19: 2})
        self.assertEqual(len(camera_ids), 3)

    def test_color_correct_numpy_matches_torch_affine(self):
        from tests.reference.color_correct import NativeColorCorrectors
        cc = NativeColorCorrectors.create(2)
        cc.params[1, :9] = np.array([0.1, 0, 0, 0, -0.05, 0, 0, 0, 0.02], np.float32)
        cc.params[1, 9:] = np.array([0.01, -0.02, 0.03], np.float32)
        rgb = np.linspace(0, 1, 30, dtype=np.float32).reshape(2, 5, 3)
        out = cc.apply_numpy(rgb, 1)
        mat = cc.params[1, :9].reshape(3, 3) + np.eye(3, dtype=np.float32)
        expected = rgb.reshape(-1, 3) @ mat + cc.params[1, 9:]
        np.testing.assert_allclose(out.reshape(-1, 3), expected, atol=1e-6)

    def test_color_correct_adam_uses_sum_of_squares_derivative(self):
        from tests.reference.color_correct import NativeColorCorrectors

        cc = NativeColorCorrectors.create(1, reg=0.25)
        cc.params[:] = 2.0
        cc.adam_step(np.zeros_like(cc.params), step=1)
        # d[reg * sum(params**2)]/dparams = 2 * reg * params = 1 here.
        np.testing.assert_allclose(cc.m, 0.1, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(cc.v, 0.001, rtol=0.0, atol=1e-7)

    def test_ftgspp_clamp_then_affine_reference_and_jacobian(self):
        """Reference the native order: clamp(renderer), then unbounded affine."""

        raw = np.array([[-0.4, 0.25, 1.4]], dtype=np.float64)
        delta = np.array(
            [[0.2, -0.1, 0.0], [0.0, 0.1, 0.3], [0.4, 0.0, -0.2]],
            dtype=np.float64,
        )
        matrix = delta + np.eye(3, dtype=np.float64)
        bias = np.array([0.7, -0.5, 0.4], dtype=np.float64)
        output_grad = np.array([[0.7, -0.2, 0.5]], dtype=np.float64)

        clamped = np.clip(raw, 0.0, 1.0)
        corrected = clamped @ matrix + bias
        # The corrector output is not clamped a second time.
        self.assertGreater(float(corrected[0, 0]), 1.0)
        expected_raw_grad = (output_grad @ matrix.T) * (
            (raw >= 0.0) & (raw <= 1.0)
        )
        expected_matrix_grad = clamped.T @ output_grad
        expected_bias_grad = output_grad.sum(axis=0)

        def objective(raw_value: np.ndarray) -> float:
            return float(np.sum((np.clip(raw_value, 0.0, 1.0) @ matrix + bias) * output_grad))

        epsilon = 1e-5
        numeric_raw_grad = np.zeros_like(raw)
        for index in range(raw.size):
            plus = raw.copy()
            minus = raw.copy()
            plus.flat[index] += epsilon
            minus.flat[index] -= epsilon
            numeric_raw_grad.flat[index] = (
                objective(plus) - objective(minus)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            numeric_raw_grad, expected_raw_grad, rtol=0.0, atol=1e-9
        )
        np.testing.assert_allclose(expected_matrix_grad, clamped.T @ output_grad)
        np.testing.assert_allclose(expected_bias_grad, output_grad[0])

    def test_rgba_l1_clamp_reference_includes_alpha_chain_rule(self):
        raw = np.array([-0.2, 0.4, 1.3], dtype=np.float64)
        reference = np.array([0.3, 0.8, 0.1], dtype=np.float64)
        alpha = 0.25
        clamped = np.clip(raw, 0.0, 1.0)
        scalar = np.abs(alpha * (clamped - reference)).sum()
        expected_grad = (
            alpha
            * np.sign(alpha * (clamped - reference))
            * ((raw >= 0.0) & (raw <= 1.0))
        )
        self.assertAlmostEqual(
            scalar, alpha * np.abs(clamped - reference).sum()
        )

        epsilon = 1e-5
        numeric = np.zeros_like(raw)
        for index in range(raw.size):
            plus = raw.copy()
            minus = raw.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_loss = np.abs(alpha * (np.clip(plus, 0.0, 1.0) - reference)).sum()
            minus_loss = np.abs(alpha * (np.clip(minus, 0.0, 1.0) - reference)).sum()
            numeric[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
        np.testing.assert_allclose(numeric, expected_grad, rtol=0.0, atol=1e-9)

    def test_rgba_ssim_clamp_reference_includes_alpha_chain_rule(self):
        """Finite-difference the same valid 11x11 SSIM graph as the shader."""

        coefficients = np.array(
            [
                0.001028380123898387,
                0.0075987582094967365,
                0.036000773310661316,
                0.10936068743467331,
                0.21300552785396576,
                0.26601171493530273,
                0.21300552785396576,
                0.10936068743467331,
                0.036000773310661316,
                0.0075987582094967365,
                0.001028380123898387,
            ],
            dtype=np.float64,
        )
        kernel = np.outer(coefficients, coefficients)
        raw = np.linspace(-0.1, 1.1, 121, dtype=np.float64).reshape(11, 11)
        reference = np.linspace(0.85, 0.15, 121, dtype=np.float64).reshape(11, 11)
        alpha = np.linspace(0.35, 1.0, 121, dtype=np.float64).reshape(11, 11)
        raw[5, 5] = 0.4
        raw[2, 7] = -0.3
        raw[8, 3] = 1.3

        def terms(raw_value: np.ndarray):
            prediction = np.clip(raw_value, 0.0, 1.0) * alpha
            target = reference * alpha
            mu_prediction = np.sum(kernel * prediction)
            mu_target = np.sum(kernel * target)
            prediction_second = np.sum(kernel * prediction * prediction)
            target_second = np.sum(kernel * target * target)
            cross = np.sum(kernel * prediction * target)
            variance_prediction = prediction_second - mu_prediction**2
            variance_target = target_second - mu_target**2
            covariance = cross - mu_prediction * mu_target
            a = mu_prediction**2 + mu_target**2 + 0.01**2
            b = variance_prediction + variance_target + 0.03**2
            c = 2.0 * mu_prediction * mu_target + 0.01**2
            d = 2.0 * covariance + 0.03**2
            return prediction, target, mu_prediction, mu_target, a, b, c, d

        def objective(raw_value: np.ndarray) -> float:
            _, _, _, _, a, b, c, d = terms(raw_value)
            return float(1.0 - (c * d) / (a * b))

        prediction, target, mu_prediction, mu_target, a, b, c, d = terms(raw)
        # These are the shader's partials with respect to the first moment,
        # second raw moment and cross raw moment, before Gaussian convolution.
        d_ssim_d_mu = (
            2.0 * mu_target * d / (a * b)
            - 2.0 * mu_target * c / (a * b)
            - 2.0 * mu_prediction * c * d / (a * a * b)
            + 2.0 * mu_prediction * c * d / (a * b * b)
        )
        d_ssim_d_second = -c * d / (a * b * b)
        d_ssim_d_cross = 2.0 * c / (a * b)
        clamp_jacobian = ((raw >= 0.0) & (raw <= 1.0)).astype(np.float64)
        expected = -kernel * (
            d_ssim_d_mu
            + 2.0 * prediction * d_ssim_d_second
            + target * d_ssim_d_cross
        ) * alpha * clamp_jacobian

        epsilon = 1e-6
        for index in ((5, 5), (2, 7), (8, 3)):
            plus = raw.copy()
            minus = raw.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric = (objective(plus) - objective(minus)) / (2.0 * epsilon)
            self.assertAlmostEqual(numeric, expected[index], places=7)

    def test_model_npz_roundtrip(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_model(path, model)
            restored = load_model(path)
        self.assertEqual(len(restored), len(model))
        np.testing.assert_array_equal(restored.means, model.means)
        np.testing.assert_array_equal(restored.velocities, model.velocities)

    def test_export_contains_ftgspp_temporal_and_motion_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = export_4dgs(Path(directory) / "gaussians.ply", make_model(1))
            header = path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
        self.assertIn("property float t", header)
        self.assertIn("property float t_scale", header)
        self.assertIn("property float motion_2", header)

    def test_native_runtime_sources_do_not_import_torch_or_cuda(self):
        root = Path(__file__).resolve().parents[1] / "src" / "ftgspp"
        # Optional LPIPS/DA3 plugins may import Torch lazily; the core must not.
        paths = [
            path for path in root.rglob("*.py")
            if "optional" not in path.relative_to(root).parts
        ]
        self.assertTrue(paths, f"no core Python sources found under {root}")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            relative = str(path.relative_to(root))
            self.assertNotIn("import torch", source, relative)
            self.assertNotIn("from torch", source, relative)
            self.assertNotIn(".cuda", source, relative)

    def test_nonzero_cc_affects_train_render_only(self):
        """GPU: non-zero CC changes train RGB, leaves eval RGB identical."""
        import os

        try:
            from ftgspp._ftgspp_vk import FTGSPPVulkan
        except ImportError:
            self.skipTest("native extension _ftgspp_vk is unavailable")

        device = int(os.environ.get("FTGSPP_TEST_DEVICE", "0"))
        shader_dir = _shader_dir()
        native = FTGSPPVulkan()
        native.initialize(str(shader_dir.resolve()) + "/", device)
        try:
            n, height, width = 2, 32, 32
            means = np.array([[-0.05, 0.0, 3.0], [0.05, 0.0, 3.0]], np.float32)
            log_scales = np.full((n, 3), -2.5, np.float32)
            quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (n, 1))
            opacity = np.array([[2.0], [2.0]], np.float32)
            sh = np.zeros((n, 16, 3), np.float32)
            sh[:, 0, :] = 0.5
            times = np.full((n, 1), 0.5, np.float32)
            durations = np.zeros((n, 1), np.float32)
            velocities = np.zeros((n, 3), np.float32)
            gates = np.full((n, 1), -100.0, np.float32)
            native.set_dynamic_gaussians(
                means, log_scales, quats, opacity, sh, times, durations,
                velocities, gates,
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
            w2c = np.eye(4, dtype=np.float32)
            reference = np.zeros((height, width, 3), np.uint8)
            intrinsic = np.asarray(
                [[24.0, 0.0, width / 2], [0.0, 24.0, height / 2], [0.0, 0.0, 1.0]],
                np.float32,
            )
            native.clear_dynamic_views()
            native.add_dynamic_view_meta(
                height, width, w2c, intrinsic, 0.5, 0, 0, True
            )
            native.add_dynamic_view_meta(
                height, width, w2c, intrinsic, 0.5, 0, 0, False
            )
            native.bind_dynamic_view_image(0, True, reference)
            native.bind_dynamic_view_image(0, False, reference)
            native.configure_ftgspp(
                {
                    "color_correction": True,
                    "color_num_cameras": 1,
                    "color_lr": 1e-3,
                    "color_reg": 1e-3,
                    "color_correction_start": -1,
                }
            )
            # Allocate CC buffers via a throwaway train step, then rewrite params.
            native.train_dynamic_view(0, 0)
            state = native.get_optimizer_state()
            self.assertIn("color_params", state)
            zero = np.zeros(12, np.float32)
            nonzero = zero.copy()
            nonzero[0] = 0.5
            moments = {
                key: np.asarray(state[key], dtype=np.float32).copy()
                for key in state
                if key not in {"color_params", "color_m", "color_v"}
            }
            m = np.zeros(12, np.float32)
            v = np.zeros(12, np.float32)

            def set_cc(params: np.ndarray) -> None:
                native.set_optimizer_state(
                    {
                        **moments,
                        "color_params": params,
                        "color_m": m,
                        "color_v": v,
                    }
                )

            set_cc(zero)
            native.render_dynamic_train_view(0)
            train_zero = np.asarray(native.pixel_state)[..., :3].copy()
            native.render_dynamic_val_view(0)
            eval_zero = np.asarray(native.pixel_state)[..., :3].copy()

            set_cc(nonzero)
            native.render_dynamic_train_view(0)
            train_cc = np.asarray(native.pixel_state)[..., :3].copy()
            native.render_dynamic_val_view(0)
            eval_cc = np.asarray(native.pixel_state)[..., :3].copy()

            native.render_dynamic_val_view(0, True)
            eval_cc_enabled = np.asarray(native.pixel_state)[..., :3].copy()

            self.assertFalse(np.allclose(train_cc, train_zero, atol=1e-6))
            np.testing.assert_array_equal(eval_cc, eval_zero)
            self.assertFalse(np.allclose(eval_cc_enabled, eval_zero, atol=1e-6))

            native.begin_eval_val_view(0, True)
            gpu = dict(native.complete_eval_metrics((0.0, 0.0, 0.0)))
            self.assertIn("squared_error_sum", gpu)
            self.assertIn("ssim_sum", gpu)
            self.assertIn("foreground_squared_error_sum", gpu)
            self.assertGreater(float(gpu["valid_pixels"]), 0.0)
            native.submit_eval_rgb_readback()
            eval_components = np.asarray(
                native.complete_eval_components(), dtype=np.float32,
            )
            self.assertEqual(eval_components.shape, (height, width, 4))
            np.testing.assert_allclose(
                eval_components[..., :3], eval_cc_enabled, rtol=0.0, atol=1e-6,
            )
        finally:
            native.cleanup()

    def test_composite_gpu_pixel_gradient_preserves_transmittance_chain(self):
        """GPU: native loss emits dT=sum(dS*background) for composite_v1."""
        import os

        try:
            from ftgspp._ftgspp_vk import FTGSPPVulkan
        except ImportError:
            self.skipTest("native extension _ftgspp_vk is unavailable")

        device = int(os.environ.get("FTGSPP_TEST_DEVICE", "0"))
        native = FTGSPPVulkan()
        native.initialize(str(_shader_dir().resolve()) + "/", device)
        try:
            height = width = 32
            native.set_dynamic_gaussians(
                np.array([[0.0, 0.0, 3.0]], np.float32),
                np.full((1, 3), -2.0, np.float32),
                np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
                np.array([[2.0]], np.float32),
                np.pad(
                    np.array([[[0.35, 0.2, 0.1]]], np.float32),
                    ((0, 0), (0, 15), (0, 0)),
                ),
                np.array([[0.5]], np.float32),
                np.zeros((1, 1), np.float32),
                np.zeros((1, 3), np.float32),
                np.full((1, 1), -100.0, np.float32),
            )
            native.configure_ftgspp(
                {
                    "max_steps": 1,
                    "scene_scale": 1.0,
                    "ssim_lambda": 0.0,
                    "image_objective": 1,
                    "max_duration": -1.0,
                    "marginal_threshold": 0.05,
                }
            )
            intrinsic = np.asarray(
                [
                    [24.0, 0.0, width / 2],
                    [0.0, 24.0, height / 2],
                    [0.0, 0.0, 1.0],
                ],
                np.float32,
            )
            native.clear_dynamic_views()
            native.add_dynamic_view_meta(
                height, width, np.eye(4, dtype=np.float32), intrinsic,
                0.5, 0, 0, True,
            )
            # An all-background target makes opacity leakage visible without
            # relying on a particular reference foreground color.
            native.bind_dynamic_view_image(
                0, True, np.zeros((height, width, 4), np.uint8)
            )
            validity = np.zeros((height, width), np.uint8)
            validity[height // 2, width // 2] = 1
            native.bind_dynamic_view_validity(0, True, validity)
            # Avoid exact sign-weight cancellation in dT=sum(dS*background).
            background = np.array([0.2, 0.4, 0.65], np.float32)
            native.train_dynamic_view_to_loss(0, 0, True, background.tolist())
            pixel_gradient = np.asarray(native.v_pixel_state).reshape(
                height, width, 4
            ).copy()

            self.assertGreater(float(np.max(np.abs(pixel_gradient[..., 3]))), 0.0)
            active_pixels = np.any(np.abs(pixel_gradient) > 1e-9, axis=-1)
            np.testing.assert_array_equal(active_pixels, validity.astype(bool))
            np.testing.assert_allclose(
                pixel_gradient[..., 3],
                np.sum(pixel_gradient[..., :3] * background, axis=-1),
                rtol=2e-5,
                atol=2e-7,
            )
            loss = native.get_last_loss()
            self.assertAlmostEqual(float(loss["ssim"]), 0.0, places=7)
            # Consume the pending loss state so cleanup also covers the real
            # raster backward path rather than abandoning an in-flight step.
            native.backward_dynamic_view_from_loss_raw(0, 0)
        finally:
            native.cleanup()


if __name__ == "__main__":
    unittest.main()
