from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from ftgspp.optional.da3 import _fuse_view_clouds
from ftgspp.optional.da3_schedule import (
    ForegroundCoverage,
    apply_schedule_to_options,
    foreground_coverage_for_scene,
    infer_quality,
    pre_fuse_per_view_cap,
    resolve_da3_schedule,
    resolve_da3_schedule_for_scene,
    resolve_max_points,
    resolve_process_res,
    select_representative_camera_ids,
)
from ftgspp.training.point_sampling import auto_voxel_size
from ftgspp.training.point_sampling import voxel_fuse_points


class DA3ScheduleTests(unittest.TestCase):
    def test_cuda_voxel_fusion_matches_cpu_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        rng = np.random.default_rng(8)
        xyz = rng.normal(size=(10_000, 3)).astype(np.float32)
        rgb = rng.integers(0, 256, (10_000, 3), dtype=np.uint8)
        cpu = voxel_fuse_points(xyz, rgb, voxel_size=0.1, device="cpu")
        cuda = voxel_fuse_points(xyz, rgb, voxel_size=0.1, device="cuda")
        default = voxel_fuse_points(xyz, rgb, voxel_size=0.1)
        for cuda_value, cpu_value in zip(cuda, cpu, strict=True):
            np.testing.assert_array_equal(cuda_value, cpu_value)
        for default_value, cuda_value in zip(default, cuda, strict=True):
            np.testing.assert_array_equal(default_value, cuda_value)

        report: dict[str, object] = {}
        _fuse_view_clouds(
            [(xyz, rgb)], voxel_size=0.1, max_points=len(xyz),
            rng=np.random.default_rng(1), report=report,
        )
        self.assertTrue(str(report["device"]).startswith("cuda"))

    def test_auto_voxel_subsample_always_queries_full_tree(self):
        xyz = np.stack(
            (
                np.arange(100, dtype=np.float32),
                np.zeros(100, np.float32),
                np.zeros(100, np.float32),
            ),
            axis=1,
        )
        diagnostics: dict[str, object] = {}
        size = auto_voxel_size(
            xyz,
            rng=np.random.default_rng(3),
            sample_size=1,
            diagnostics=diagnostics,
        )
        self.assertEqual(size, 0.5)
        self.assertEqual(diagnostics["sample_size"], 1)
        self.assertEqual(diagnostics["neighbor_count"], 1)

    def test_auto_never_upscales_below_4k(self):
        res, clamped = resolve_process_res(
            None, native_hw=(720, 1280), quality="balanced",
        )
        self.assertEqual(res, 1008)
        self.assertFalse(clamped)
        res, clamped = resolve_process_res(
            None, native_hw=(480, 640), quality="balanced",
        )
        self.assertEqual(res, 630)
        self.assertFalse(clamped)

    def test_auto_caps_4k_instead_of_native(self):
        res, _clamped = resolve_process_res(
            None, native_hw=(2160, 3840), quality="balanced",
        )
        self.assertEqual(res, 1008)
        high, _ = resolve_process_res(
            None, native_hw=(2160, 3840), quality="high",
        )
        self.assertEqual(high, 1512)

    def test_explicit_process_res_is_clamped_to_native(self):
        res, clamped = resolve_process_res(
            1512, native_hw=(1060, 1060), quality="high",
        )
        self.assertEqual(res, 1050)
        self.assertTrue(clamped)

    def test_max_points_tracks_gaussian_budget_not_one_million(self):
        # SelfCap Table S2: 500k / 60 keyframes.
        points = resolve_max_points(
            None, num_gaussians=500_000, n_keyframes=60, quality="balanced",
        )
        self.assertEqual(points, 8 * (500_000 // 60))
        self.assertLess(points, 100_000)
        high = resolve_max_points(
            None, num_gaussians=500_000, n_keyframes=60, quality="high",
        )
        self.assertEqual(high, 12 * (500_000 // 60))
        self.assertLess(high, 250_000)

    def test_max_points_cap_binds_on_huge_gaussian_budget(self):
        points = resolve_max_points(
            None, num_gaussians=2_000_000, n_keyframes=30, quality="balanced",
        )
        self.assertEqual(points, 160_000)

    def test_explicit_max_points_is_honored(self):
        self.assertEqual(
            resolve_max_points(
                1_000_000, num_gaussians=500_000, n_keyframes=60, quality="fast",
            ),
            1_000_000,
        )

    def test_schedule_writes_concrete_options(self):
        options = SimpleNamespace(
            quality="balanced", process_res=None, max_views=None, max_points=None,
        )
        schedule = resolve_da3_schedule(
            options,
            native_hw=(1052, 1872),
            num_gaussians=500_000,
            n_keyframes=60,
            n_cameras=23,
        )
        apply_schedule_to_options(options, schedule)
        self.assertEqual(options.process_res, 1008)
        self.assertEqual(options.max_views, 6)
        self.assertEqual(options.fusion_views, 23)
        self.assertEqual(options.max_points, 66_664)
        self.assertEqual(options.min_consistent_views, 2)
        self.assertFalse(options.use_sky_mask)

    def test_auto_balances_speed_and_quality(self):
        self.assertEqual(infer_quality(num_gaussians=500_000, n_keyframes=60), "balanced")
        # Large per-keyframe budget still stays balanced; high is opt-in.
        self.assertEqual(infer_quality(num_gaussians=2_000_000, n_keyframes=30), "balanced")
        self.assertEqual(infer_quality(num_gaussians=100_000, n_keyframes=60), "fast")

        options = SimpleNamespace(
            quality="auto", process_res=None, max_views=None, max_points=None,
        )
        four_k = resolve_da3_schedule(
            options,
            native_hw=(2160, 3840),
            num_gaussians=2_000_000,
            n_keyframes=30,
            n_cameras=8,
        )
        self.assertEqual(four_k.quality, "balanced")
        self.assertEqual(four_k.process_res, 1008)
        self.assertEqual(four_k.max_points, 120_000)
        self.assertEqual(four_k.max_views, 6)
        self.assertEqual(four_k.fusion_views, 8)
        self.assertEqual(four_k.min_consistent_views, 2)
        self.assertFalse(four_k.use_sky_mask)

        tiny = resolve_da3_schedule(
            options,
            native_hw=(2160, 3840),
            num_gaussians=100_000,
            n_keyframes=60,
            n_cameras=8,
        )
        self.assertEqual(tiny.quality, "fast")
        self.assertEqual(tiny.process_res, 392)
        self.assertEqual(tiny.max_points, 8 * (100_000 // 60))
        self.assertEqual(tiny.fusion_views, 8)
        self.assertEqual(tiny.min_consistent_views, 2)
        self.assertFalse(tiny.use_sky_mask)

    def test_auto_schedule_from_training_knobs_matches_selfcap(self):
        class _View:
            size = (1052, 1872)

        class _Scene:
            frames = list(range(60))
            cameras = list(range(23))
            train_cameras = list(range(22))
            calibration_available = True

            def view(self, _frame, _camera):
                return _View()

        config = SimpleNamespace(
            init=SimpleNamespace(
                num_gaussians=500_000,
                da3=SimpleNamespace(
                    quality="auto",
                    process_res=None,
                    max_views=None,
                    max_points=None,
                    input_cameras="train",
                ),
            ),
        )
        schedule = resolve_da3_schedule_for_scene(
            config, _Scene(), list(range(60)),
        )
        self.assertEqual(schedule.process_res, 504)
        self.assertEqual(schedule.max_views, 8)
        self.assertEqual(schedule.fusion_views, 8)
        self.assertEqual(schedule.max_points, 8 * (500_000 // 60))
        self.assertEqual(schedule.quality, "balanced")
        self.assertEqual(schedule.min_consistent_views, 2)
        self.assertFalse(schedule.use_sky_mask)
        self.assertEqual(schedule.native_hw, (1052, 1872))

    def test_foreground_schedule_uses_actual_mask_size_and_auto_support(self):
        class _View:
            size = (1052, 1872)

            def load_mask(self):
                result = np.zeros((20, 40), np.float32)
                result[:, :10] = 1.0
                return result

            def load_validity(self):
                return np.ones((20, 40), np.float32)

        class _Scene:
            frames = list(range(60))
            cameras = list(range(8))
            train_cameras = list(range(8))
            calibration_available = True

            def view(self, _frame, _camera):
                return _View()

        config = SimpleNamespace(init=SimpleNamespace(
            num_gaussians=500_000,
            da3=SimpleNamespace(
                quality="auto", process_res=None, max_views=None,
                fusion_views=None, max_points=None, input_cameras="train",
                foreground_only=True, mask_threshold=0.5,
                min_consistent_views=None,
            ),
        ))
        schedule = resolve_da3_schedule_for_scene(
            config, _Scene(), list(range(60)),
        )
        self.assertEqual(schedule.foreground_fraction, 0.25)
        self.assertEqual(schedule.foreground_visibility_fraction, 1.0)
        self.assertEqual(schedule.foreground_sampled_views, 96)
        self.assertEqual(schedule.process_res, 1008)
        self.assertEqual(schedule.min_consistent_views, 2)
        self.assertFalse(schedule.use_sky_mask)

    def test_explicit_foreground_resolution_remains_authoritative(self):
        coverage = ForegroundCoverage(
            median_visible_fraction=0.04, visibility_fraction=1.0,
            sampled_views=4, nonempty_views=4,
            foreground_pixels=400, valid_pixels=10_000,
        )
        options = SimpleNamespace(
            quality="auto", process_res=504, max_views=None,
            fusion_views=None, max_points=None, foreground_only=True,
            min_consistent_views=None,
        )
        schedule = resolve_da3_schedule(
            options, native_hw=(1052, 1872), num_gaussians=500_000,
            n_keyframes=60, n_cameras=8, foreground_coverage=coverage,
        )
        self.assertEqual(schedule.process_res, 504)
        self.assertEqual(schedule.foreground_fraction, 0.04)

    def test_auto_geometry_gates_follow_fused_camera_count(self):
        options = SimpleNamespace(
            quality="auto", process_res=None, max_views=None, max_points=None,
        )
        mono = resolve_da3_schedule(
            options, native_hw=(720, 1280), num_gaussians=500_000,
            n_keyframes=60, n_cameras=1,
        )
        self.assertEqual(mono.fusion_views, 1)
        self.assertEqual(mono.min_consistent_views, 1)
        self.assertTrue(mono.use_sky_mask)

        stereo = resolve_da3_schedule(
            options, native_hw=(720, 1280), num_gaussians=500_000,
            n_keyframes=60, n_cameras=2,
        )
        self.assertEqual(stereo.fusion_views, 2)
        self.assertEqual(stereo.min_consistent_views, 2)
        self.assertFalse(stereo.use_sky_mask)

        dense = resolve_da3_schedule(
            options, native_hw=(1052, 1872), num_gaussians=2_000_000,
            n_keyframes=30, n_cameras=24,
        )
        self.assertEqual(dense.fusion_views, 10)
        self.assertEqual(dense.min_consistent_views, 2)
        self.assertFalse(dense.use_sky_mask)

    def test_explicit_geometry_gates_remain_authoritative(self):
        options = SimpleNamespace(
            quality="auto", process_res=None, max_views=None, max_points=None,
            min_consistent_views=1, use_sky_mask=True,
        )
        schedule = resolve_da3_schedule(
            options, native_hw=(1052, 1872), num_gaussians=500_000,
            n_keyframes=60, n_cameras=23,
        )
        self.assertEqual(schedule.fusion_views, 8)
        self.assertEqual(schedule.min_consistent_views, 1)
        self.assertTrue(schedule.use_sky_mask)

    def test_foreground_coverage_rejects_all_empty_target(self):
        view = SimpleNamespace(
            load_mask=lambda: np.zeros((4, 4), np.float32),
            load_validity=lambda: np.ones((4, 4), np.float32),
        )
        scene = SimpleNamespace(frames=[0], view=lambda _frame, _camera: view)
        with self.assertRaisesRegex(ValueError, "empty target"):
            foreground_coverage_for_scene(
                scene, [0], [0, 1], mask_threshold=0.5,
            )

    def test_representative_camera_selection_covers_ring(self):
        cameras = list(range(12))
        w2c = {}
        for camera in cameras:
            angle = 2.0 * np.pi * camera / len(cameras)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = [np.cos(angle), np.sin(angle), 0.0]
            w2c[camera] = np.linalg.inv(c2w)
        selected = select_representative_camera_ids(cameras, w2c, 4)
        self.assertEqual(len(selected), 4)
        angles = np.sort(2.0 * np.pi * np.asarray(selected) / len(cameras))
        gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
        self.assertLessEqual(float(gaps.max()), np.pi / 2.0 + 1e-6)

    def test_fuse_pre_caps_before_concatenating_dense_views(self):
        rng = np.random.default_rng(0)
        clouds = []
        for offset in (0.0, 5.0):
            xyz = rng.normal(size=(20_000, 3)).astype(np.float32)
            xyz[:, 0] += offset
            rgb = np.zeros((20_000, 3), np.uint8)
            clouds.append((xyz, rgb))
        xyz, _rgb, raw, _voxel = _fuse_view_clouds(
            clouds, voxel_size=0.05, max_points=1_000, rng=np.random.default_rng(1),
        )
        self.assertEqual(len(xyz), 1_000)
        self.assertLessEqual(raw, 4 * 1_000 + 2)
        self.assertLess(raw, 40_000)
        per_view = pre_fuse_per_view_cap(1_000, 2)
        self.assertEqual(per_view, 2_000)

if __name__ == "__main__":
    unittest.main()
