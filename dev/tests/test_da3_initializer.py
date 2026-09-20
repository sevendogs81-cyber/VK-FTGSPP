from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ftgspp.data.scene import Scene, View
from ftgspp.io.ply import read_point_ply, write_point_ply
from ftgspp.optional.da3 import (
    _cache_namespace,
    _camera_chunks,
    _filter_prediction_bundles,
    _fuse_view_clouds,
    _intrinsic_to_source,
    _loaded_model_identity,
    _prediction_transform_matrix,
    _PredictionBundle,
    _synchronized_views,
    _validate_prediction_depth_type,
    depth_to_points,
    generate_da3_points,
    geometric_consistency_filter,
    pixel_center_to_index_matrix,
    resolve_da3_cache_root,
)
from ftgspp.training.init_plan import bind_schedule_to_options
from ftgspp.training.initialize import _valid_da3_cache


class _Prediction:
    def __init__(self, images, extrinsics, intrinsics):
        count = len(images)
        self.depth = np.ones((count, 4, 4), np.float32)
        self.conf = np.ones_like(self.depth)
        self.sky = None
        self.processed_images = np.stack(images)
        if extrinsics is None:
            self.extrinsics = np.repeat(
                np.eye(4, dtype=np.float32)[None], count, axis=0,
            )
            for index in range(count):
                self.extrinsics[index, 0, 3] = -float(index)
        else:
            self.extrinsics = np.asarray(extrinsics, np.float32)
        self.intrinsics = (
            np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
            if intrinsics is None else np.asarray(intrinsics, np.float32)
        )


class _FakeDA3:
    def inference(self, images, extrinsics=None, intrinsics=None, **_kwargs):
        return _Prediction(images, extrinsics, intrinsics)


class DA3InitializerTests(unittest.TestCase):
    def test_cache_namespace_changes_for_every_semantic_input_family(self):
        options = SimpleNamespace(
            process_res_method="upper_bound_resize", depth_type="camera_z",
            confidence_percentile=20.0, use_sky_mask=True,
            use_ray_pose=False, ref_view_strategy="saddle_balanced",
            input_cameras="train", foreground_only=True,
            mask_threshold=0.5, min_consistent_views=2,
            sync_tolerance_fraction=0.25, relative_depth_tolerance=0.02,
            absolute_depth_tolerance_ratio=0.002,
            fusion_voxel_ratio=0.002, voxel_size=None,
            target_bounds=[-1, -1, 0, 1, 1, 2],
        )
        dataset = SimpleNamespace(
            frames=[10, 20],
            checkpoint_metadata=lambda: {"fingerprint": "dataset-a"},
        )
        schedule = SimpleNamespace(
            quality="balanced", process_res=504, max_views=4,
            fusion_views=4, max_points=1000, min_consistent_views=2,
            foreground_fraction=0.25,
            foreground_visibility_fraction=1.0,
            foreground_sampled_views=8,
            foreground_nonempty_views=8,
            foreground_pixels=2000,
            foreground_valid_pixels=8000,
        )

        def namespace(*, changed_options=None, changed_schedule=None,
                      dataset_fingerprint="dataset-a", seed=7,
                      init_seed=None, model_hash="model-a"):
            selected_options = changed_options or options
            selected_schedule = changed_schedule or schedule
            selected_dataset = SimpleNamespace(
                frames=[10, 20],
                checkpoint_metadata=lambda: {
                    "fingerprint": dataset_fingerprint,
                },
            )
            config = SimpleNamespace(
                init=SimpleNamespace(da3=selected_options, seed=init_seed),
                train=SimpleNamespace(seed=seed),
            )
            with patch(
                "ftgspp.io.provenance.source_identity",
                return_value={"set_sha256": "source-a"},
            ):
                return _cache_namespace(
                    config, selected_dataset, [0, 1],
                    schedule=selected_schedule,
                    candidate_camera_ids=[0, 1, 2, 3],
                    loaded_model_identity={
                        "loaded_state_dict_sha256": model_hash,
                    },
                )

        baseline = namespace()
        variants = [
            namespace(changed_options=SimpleNamespace(
                **{**vars(options), "mask_threshold": 0.6},
            )),
            namespace(changed_options=SimpleNamespace(
                **{**vars(options), "target_bounds": [-2, -1, 0, 1, 1, 2]},
            )),
            namespace(changed_schedule=SimpleNamespace(
                **{**vars(schedule), "process_res": 756},
            )),
            namespace(changed_schedule=SimpleNamespace(
                **{**vars(schedule), "foreground_fraction": 0.5},
            )),
            namespace(dataset_fingerprint="dataset-b"),
            namespace(seed=8),
            namespace(model_hash="model-b"),
        ]
        self.assertTrue(all(value != baseline for value in variants))
        self.assertEqual(len(set(variants)), len(variants))
        fixed_seed0 = namespace(seed=0, init_seed=3)
        fixed_seed7 = namespace(seed=7, init_seed=3)
        self.assertEqual(fixed_seed0, fixed_seed7)
        self.assertRegex(fixed_seed0, r"^scene-init3-kf2-v4-r504-[0-9a-f]{12}$")

    def test_crop_transform_restores_source_intrinsic_and_cannot_be_guessed(self):
        source_k = np.asarray(
            [[400.0, 0.0, 103.0], [0.0, 420.0, 51.0], [0.0, 0.0, 1.0]],
            np.float32,
        )
        transform = np.asarray(
            [[0.5, 0.0, -11.0], [0.0, 0.5, -7.0], [0.0, 0.0, 1.0]],
            np.float64,
        )
        prediction = SimpleNamespace(source_to_processed=transform[None])
        actual = _prediction_transform_matrix(
            prediction, 0, (120, 220), (46, 88), "upper_bound_crop",
        )
        np.testing.assert_array_equal(actual, transform)
        np.testing.assert_allclose(
            _intrinsic_to_source(transform @ source_k, actual),
            source_k, rtol=0.0, atol=1e-5,
        )
        with self.assertRaisesRegex(RuntimeError, "did not return.*metadata"):
            _prediction_transform_matrix(
                SimpleNamespace(), 0, (120, 220), (46, 88),
                "upper_bound_crop",
            )

    def test_global_consistency_crosses_prediction_chunks(self):
        depth = np.ones((1, 1), np.float32)
        cloud = (np.asarray([[0, 0, 1]], np.float32), np.zeros((1, 3), np.uint8))
        bundles = [
            _PredictionBundle(
                clouds=[cloud], depths=[depth],
                extrinsics=np.eye(4, dtype=np.float32)[None],
                intrinsics=np.eye(3, dtype=np.float32)[None],
                masks=[np.ones_like(depth)], validities=[np.ones_like(depth)],
                transforms=[np.eye(3)],
            )
            for _ in range(2)
        ]
        clouds, diagnostics = _filter_prediction_bundles(
            bundles, SimpleNamespace(
                min_consistent_views=2, relative_depth_tolerance=0.01,
                absolute_depth_tolerance_ratio=0.01,
                target_bounds=[-1, -1, 0, 1, 1, 2],
            ),
        )
        self.assertEqual([len(value[0]) for value in clouds], [1, 1])
        self.assertEqual(diagnostics["global_consistency_views"], 2)
        self.assertEqual(diagnostics["tolerance_extent_source"], "target_bounds")
        self.assertEqual(diagnostics["low_support_points"], 0)

    def test_depth_type_contract_and_loaded_state_hash(self):
        with self.assertRaisesRegex(RuntimeError, "depth_type"):
            _validate_prediction_depth_type(
                SimpleNamespace(depth_type="ray_distance"),
                SimpleNamespace(depth_type="camera_z"),
            )

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value, np.float32)
            def detach(self):
                return self
            def cpu(self):
                return self
            def contiguous(self):
                return self
            def numpy(self):
                return self.value

        model = SimpleNamespace(
            state_dict=lambda: {"b": Tensor([2]), "a": Tensor([1])},
            config=SimpleNamespace(_name_or_path="resolved", _commit_hash="abc"),
        )
        identity = _loaded_model_identity(
            model, SimpleNamespace(model_name="remote", model_path=None),
        )
        self.assertEqual(identity["commit_hash"], "abc")
        self.assertEqual(len(identity["loaded_state_dict_sha256"]), 64)
    def test_scene_auto_detects_missing_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            for camera in (0, 1):
                directory = images / str(camera)
                directory.mkdir(parents=True)
                cv2.imwrite(
                    str(directory / "000000.png"),
                    np.full((8, 12, 3), 20 + camera, np.uint8),
                )
            config = SimpleNamespace(
                data=SimpleNamespace(
                    root=root, video_path=root, calibration_path=root,
                    extracted_path=images, scale=0.5, fps=30.0,
                    frames=SimpleNamespace(start=0, stop=1),
                    train_cameras=None, eval_cameras=[],
                    strict_time_offsets=True,
                ),
                init=SimpleNamespace(
                    keyframe_stride=1,
                    da3=SimpleNamespace(cache_path=root / "missing-cache"),
                ),
            )
            dataset = Scene.from_config(config)

            self.assertFalse(dataset.calibration_available)
            self.assertEqual(dataset.cameras, [0, 1])
            self.assertEqual(dataset.train_cameras, [0, 1])
            self.assertEqual(dataset.view(0, 0).size, (4, 6))

    def test_uncalibrated_named_camera_offsets_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            for camera in ("left", "right"):
                directory = images / camera
                directory.mkdir(parents=True)
                cv2.imwrite(
                    str(directory / "000000.png"),
                    np.full((8, 8, 3), 20, np.uint8),
                )
            (root / "t_offsets.json").write_text(
                json.dumps({"left": 0.1, "right": -0.2}), encoding="utf-8",
            )
            config = SimpleNamespace(
                data=SimpleNamespace(
                    root=root, video_path=root, calibration_path=root,
                    extracted_path=images, scale=0.5, fps=30.0,
                    frames=SimpleNamespace(start=0, stop=1),
                    train_cameras=None, eval_cameras=[],
                    strict_time_offsets=True,
                ),
                init=SimpleNamespace(
                    keyframe_stride=1,
                    da3=SimpleNamespace(cache_path=root / "missing-cache"),
                ),
            )
            dataset = Scene.from_config(config)
            self.assertEqual(dataset.time_offsets, {0: 0.1, 1: -0.2})

    def test_depth_to_points_uses_processed_intrinsics_and_rgb(self):
        depth = np.full((2, 2), 2.0, np.float32)
        intrinsic = np.eye(3, dtype=np.float32)
        rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        w2c = np.eye(4, dtype=np.float32)
        xyz, colors = depth_to_points(
            depth, intrinsic, w2c, rgb, confidence=np.ones((2, 2), np.float32),
        )
        np.testing.assert_allclose(
            xyz,
            np.asarray([[1.0, 1.0, 2.0], [3.0, 1.0, 2.0],
                        [1.0, 3.0, 2.0], [3.0, 3.0, 2.0]], np.float32),
        )
        np.testing.assert_array_equal(colors, rgb.reshape(-1, 3))

    def test_confidence_and_sky_masks_are_applied(self):
        depth = np.ones((2, 2), np.float32)
        rgb = np.zeros((2, 2, 3), np.uint8)
        conf = np.asarray([[0.1, 0.2], [0.9, 1.0]], np.float32)
        sky = np.asarray([[False, False], [True, False]])
        xyz, _colors = depth_to_points(
            depth, np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), rgb,
            confidence=conf, sky=sky, confidence_percentile=50.0,
        )
        self.assertEqual(len(xyz), 1)

    def test_depth_to_points_stratifies_before_unprojection(self):
        depth = np.ones((100, 200), np.float32)
        rgb = np.zeros((100, 200, 3), np.uint8)
        xyz, _colors = depth_to_points(
            depth, np.eye(3, dtype=np.float32),
            np.eye(4, dtype=np.float32), rgb,
            confidence=np.ones_like(depth),
            confidence_percentile=0.0,
            max_points=128,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(len(xyz), 128)
        # Pixel-space stratification must retain broad image coverage.
        self.assertLess(float(xyz[:, 0].min()), 25.0)
        self.assertGreater(float(xyz[:, 0].max()), 175.0)
        self.assertLess(float(xyz[:, 1].min()), 15.0)
        self.assertGreater(float(xyz[:, 1].max()), 85.0)

    def test_camera_chunks_keep_anchor(self):
        self.assertEqual(_camera_chunks([0, 1, 2, 3, 4], 3), [[0, 1, 2], [0, 3, 4]])

    def test_camera_chunks_are_sequential_without_alignment_anchors(self):
        self.assertEqual(
            _camera_chunks(list(range(8)), 6, anchor_count=0),
            [list(range(6)), [6, 7]],
        )

    def test_camera_chunks_share_one_time_reference(self):
        views = []
        intrinsic = np.eye(3, dtype=np.float32)
        for frame in (0, 1, 2):
            for camera, offset in ((0, 0.0), (1, 0.2)):
                views.append(View(
                    path=Path(f"/tmp/c{camera}-f{frame}.png"),
                    frame=frame, camera_id=camera,
                    timestamp=float(frame) + offset,
                    w2c=np.eye(4, dtype=np.float32),
                    source_intrinsic=intrinsic, intrinsic=intrinsic,
                    distortion=np.zeros(5, np.float32), size=(4, 4),
                ))
        scene = Scene(
            frames=[0, 1, 2], cameras=[0, 1], train_cameras=[0, 1],
            eval_cameras=[], camera_split_source="test",
            train_views=views, eval_views=[], fps=1.0,
            time_offsets={0: 0.0, 1: -0.2},
        )
        options = SimpleNamespace(sync_tolerance_fraction=0.5)
        selected, reference, residuals = _synchronized_views(
            scene, 1, [1], options, reference_timestamp=1.0,
        )
        self.assertEqual(selected[0].frame, 1)
        self.assertEqual(reference, 1.0)
        self.assertAlmostEqual(residuals[0], 0.2)

    def test_view_fusion_collapses_duplicates_and_keeps_unique_background(self):
        foreground = np.zeros((80, 3), np.float32)
        foreground_rgb = np.zeros((80, 3), np.uint8)
        background = np.array([[12.0, 0.0, 0.0]], np.float32)
        background_rgb = np.array([[9, 8, 7]], np.uint8)
        xyz, rgb, raw, voxel = _fuse_view_clouds(
            [
                (np.concatenate([foreground, background]), np.concatenate([foreground_rgb, background_rgb])),
                (foreground + 1e-4, foreground_rgb),
            ],
            voxel_size=0.05,
            max_points=32,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(raw, 161)
        self.assertLessEqual(len(xyz), 32)
        self.assertGreater(voxel, 0.0)
        self.assertTrue(np.any(np.linalg.norm(xyz - background, axis=1) < 0.1))
        self.assertTrue((rgb == np.array([9, 8, 7], np.uint8)).all(axis=1).any())

    def test_fused_cloud_uses_spatial_cap_not_per_view_quota(self):
        rng = np.random.default_rng(1)
        left = np.stack([rng.normal(size=200) * 0.01, np.zeros(200), np.zeros(200)], axis=1).astype(np.float32)
        right = left + np.array([8.0, 0.0, 0.0], np.float32)
        rgb = np.zeros((200, 3), np.uint8)
        xyz, _rgb, raw, _voxel = _fuse_view_clouds(
            [(left, rgb), (right, rgb)],
            voxel_size=0.002,
            max_points=24,
            rng=np.random.default_rng(2),
        )
        self.assertEqual(raw, 400)
        self.assertEqual(len(xyz), 24)
        self.assertGreater((xyz[:, 0] < 4.0).sum(), 0)
        self.assertGreater((xyz[:, 0] > 4.0).sum(), 0)

    def test_uncalibrated_generation_estimates_and_caches_camera_rig(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = []
            intrinsic = np.eye(3, dtype=np.float32)
            for frame in (0, 1):
                for camera in (0, 1):
                    path = root / f"c{camera}-f{frame}.png"
                    cv2.imwrite(str(path), np.full((4, 4, 3), 30 + camera, np.uint8))
                    views.append(View(
                        path=path, frame=frame, camera_id=camera,
                        timestamp=float(frame), w2c=np.eye(4, dtype=np.float32),
                        source_intrinsic=intrinsic, intrinsic=intrinsic,
                        distortion=np.zeros(5, np.float32), size=(4, 4),
                    ))
            dataset = Scene(
                frames=[0, 1], cameras=[0, 1], train_cameras=[0, 1],
                eval_cameras=[], camera_split_source="uncalibrated test",
                train_views=views, eval_views=[], fps=1.0, source_root=root,
                time_offsets={0: 0.0, 1: 0.0}, calibration_available=False,
            )
            config = SimpleNamespace(
                data=SimpleNamespace(root=root),
                run=SimpleNamespace(gpu=0),
                train=SimpleNamespace(seed=0),
                init=SimpleNamespace(
                    keyframe_stride=1,
                    da3=SimpleNamespace(
                        model_name="fake", model_path=None, device="cpu",
                        process_res=4, process_res_method="upper_bound_resize",
                        max_views=2, max_points=32, voxel_size=0.25,
                        confidence_percentile=0.0, use_sky_mask=True,
                        use_ray_pose=False, ref_view_strategy="first",
                        input_cameras="train", cache_path=root / "cache",
                    ),
                ),
            )
            with patch(
                "ftgspp.optional.da3._load_model",
                return_value=(_FakeDA3(), "cpu"),
            ):
                points = generate_da3_points(config, dataset, [0, 1])

            self.assertEqual(len(points), 2)
            self.assertTrue(dataset.calibration_available)
            self.assertNotEqual(
                float(dataset.view(0, 0).w2c[0, 3]),
                float(dataset.view(0, 1).w2c[0, 3]),
            )
            active_cache = resolve_da3_cache_root(root / "cache")
            self.assertNotEqual(active_cache, root / "cache")
            self.assertTrue((root / "cache" / "current.json").is_file())
            self.assertTrue((active_cache / "camera_manifest.npz").is_file())
            self.assertTrue((active_cache / "f000000.ply").is_file())
            self.assertTrue((active_cache / "f000001.ply").is_file())

    def test_da3_cache_validation_resolves_auto_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = []
            intrinsic = np.eye(3, dtype=np.float32)
            for frame in (0, 1):
                for camera in (0, 1):
                    path = root / f"c{camera}-f{frame}.png"
                    cv2.imwrite(str(path), np.full((4, 4, 3), 30 + camera, np.uint8))
                    views.append(View(
                        path=path, frame=frame, camera_id=camera,
                        timestamp=float(frame), w2c=np.eye(4, dtype=np.float32),
                        source_intrinsic=intrinsic, intrinsic=intrinsic,
                        distortion=np.zeros(5, np.float32), size=(4, 4),
                    ))
            dataset = Scene(
                frames=[0, 1], cameras=[0, 1], train_cameras=[0, 1],
                eval_cameras=[], camera_split_source="uncalibrated test",
                train_views=views, eval_views=[], fps=1.0, source_root=root,
                time_offsets={0: 0.0, 1: 0.0}, calibration_available=False,
            )
            da3 = SimpleNamespace(
                model_name="fake", model_path=None, device="cpu",
                quality="balanced",
                process_res=None, process_res_method="upper_bound_resize",
                max_views=None, max_points=None, voxel_size=0.25,
                confidence_percentile=0.0, use_sky_mask=True,
                use_ray_pose=False, ref_view_strategy="first",
                input_cameras="train", cache_path=root / "cache",
            )
            config = SimpleNamespace(
                data=SimpleNamespace(root=root),
                run=SimpleNamespace(gpu=0),
                train=SimpleNamespace(seed=0),
                init=SimpleNamespace(
                    keyframe_stride=1, num_gaussians=500_000, da3=da3,
                ),
            )
            with patch(
                "ftgspp.optional.da3._load_model",
                return_value=(_FakeDA3(), "cpu"),
            ):
                generate_da3_points(config, dataset, [0, 1])

            da3.process_res = None
            da3.max_views = None
            da3.max_points = None
            self.assertTrue(
                _valid_da3_cache(
                    config, dataset, root / "cache",
                    expected_frames=[0, 1], frame_positions=[0, 1],
                )
            )
            active_cache = resolve_da3_cache_root(root / "cache")
            (active_cache / "f000000.ply").write_bytes(b"corrupt")
            self.assertFalse(
                _valid_da3_cache(
                    config, dataset, root / "cache",
                    expected_frames=[0, 1], frame_positions=[0, 1],
                )
            )

    def test_generated_ply_round_trip(self):
        xyz = np.asarray([[0, 1, 2], [3, 4, 5]], np.float32)
        rgb = np.asarray([[10, 20, 30], [40, 50, 60]], np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_point_ply(Path(tmp) / "f000000.ply", xyz, rgb)
            loaded_xyz, loaded_rgb = read_point_ply(path)
        np.testing.assert_allclose(loaded_xyz, xyz)
        np.testing.assert_array_equal(loaded_rgb, rgb)

    def test_generated_ply_hash_is_reused_during_write_and_read(self):
        xyz = np.asarray([[0, 1, 2], [3, 4, 5]], np.float32)
        rgb = np.asarray([[10, 20, 30], [40, 50, 60]], np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path, write_digest = write_point_ply(
                Path(tmp) / "f000000.ply", xyz, rgb, return_sha256=True,
            )
            loaded_xyz, loaded_rgb, read_digest = read_point_ply(
                path, return_sha256=True,
            )
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(write_digest, expected)
        self.assertEqual(read_digest, expected)
        np.testing.assert_allclose(loaded_xyz, xyz)
        np.testing.assert_array_equal(loaded_rgb, rgb)

    def test_pixel_center_matrix_shifts_index_coordinates(self):
        scale = np.asarray(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], np.float64,
        )
        converted = pixel_center_to_index_matrix(scale)
        # Pixel-center 0.5 maps to processed 1.0, which is index 0.5.
        source = np.asarray([0.5, 0.5, 1.0])
        processed_center = scale @ source
        processed_index = converted @ np.asarray([0.0, 0.0, 1.0])
        np.testing.assert_allclose(processed_center[:2] - 0.5, processed_index[:2])

    def test_bind_schedule_does_not_mutate_user_sky_null(self):
        options = SimpleNamespace(
            quality="auto", process_res=None, max_views=None, max_points=None,
            use_sky_mask=None, fusion_views=None, min_consistent_views=None,
        )
        schedule = SimpleNamespace(
            quality="balanced", process_res=504, max_views=1, fusion_views=1,
            max_points=100, min_consistent_views=1, use_sky_mask=True,
        )
        bound = bind_schedule_to_options(options, schedule)
        self.assertIsNone(options.use_sky_mask)
        self.assertTrue(bound.use_sky_mask)
        self.assertEqual(bound.process_res, 504)

    def test_monocular_null_and_true_sky_share_cache_namespace(self):
        schedule = SimpleNamespace(
            quality="balanced", process_res=504, max_views=1,
            fusion_views=1, max_points=1000, min_consistent_views=1,
            use_sky_mask=True, upscale_clamped=False, native_hw=(4, 4),
            per_keyframe_gaussians=8, foreground_fraction=None,
        )
        dataset = SimpleNamespace(
            frames=[10],
            checkpoint_metadata=lambda: {"fingerprint": "dataset-a"},
        )
        config_null = SimpleNamespace(
            init=SimpleNamespace(da3=SimpleNamespace(
                process_res_method="upper_bound_resize", depth_type="camera_z",
                use_sky_mask=None, use_ray_pose=False,
                ref_view_strategy="saddle_balanced", input_cameras="train",
                foreground_only=False, mask_threshold=0.5,
            )),
            train=SimpleNamespace(seed=7),
        )
        config_true = SimpleNamespace(
            init=SimpleNamespace(da3=SimpleNamespace(
                **{**vars(config_null.init.da3), "use_sky_mask": True},
            )),
            train=SimpleNamespace(seed=7),
        )
        with patch(
            "ftgspp.io.provenance.source_identity",
            return_value={"set_sha256": "source-a"},
        ):
            null_ns = _cache_namespace(
                config_null, dataset, [0], schedule=schedule,
                candidate_camera_ids=[0],
                loaded_model_identity={"loaded_state_dict_sha256": "model-a"},
            )
            true_ns = _cache_namespace(
                config_true, dataset, [0], schedule=schedule,
                candidate_camera_ids=[0],
                loaded_model_identity={"loaded_state_dict_sha256": "model-a"},
            )
        self.assertEqual(null_ns, true_ns)

    def test_mask_threshold_is_shared_by_target_views(self):
        depth = np.ones((1, 1), np.float32)
        xyz = np.asarray([[0.0, 0.0, 1.0]], np.float32)
        k = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], np.float32)
        ext = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)
        keep, support, conflict = geometric_consistency_filter(
            xyz, source_index=0,
            depths=[depth, depth],
            extrinsics=ext, intrinsics=np.stack([k, k]),
            foreground_masks=[np.ones((1, 1), np.float32), np.full((1, 1), 0.55, np.float32)],
            validities=[np.ones((1, 1), np.float32), np.ones((1, 1), np.float32)],
            min_consistent_views=2, relative_tolerance=0.05, absolute_tolerance=0.05,
            mask_threshold=0.6,
        )
        self.assertFalse(bool(keep[0]))
        self.assertEqual(int(support[0]), 1)

    def test_v1_requires_support_and_rejects_any_depth_conflict(self):
        depth_ok = np.ones((1, 1), np.float32)
        depth_conflict = np.full((1, 1), 2.0, np.float32)
        xyz = np.asarray([[0.0, 0.0, 1.0]], np.float32)
        k = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], np.float32)
        ext = np.eye(4, dtype=np.float32)[None].repeat(3, axis=0)
        fg = np.ones((1, 1), np.float32)
        valid = np.ones((1, 1), np.float32)
        keep, support, conflict = geometric_consistency_filter(
            xyz,
            source_index=0,
            depths=[depth_ok, depth_ok, depth_conflict],
            extrinsics=ext,
            intrinsics=np.stack([k, k, k]),
            foreground_masks=[fg, fg, fg],
            validities=[valid, valid, valid],
            min_consistent_views=2,
            relative_tolerance=0.01,
            absolute_tolerance=0.01,
            mask_threshold=0.5,
        )
        self.assertFalse(bool(keep[0]))
        self.assertEqual(int(support[0]), 2)
        self.assertEqual(int(conflict[0]), 1)


if __name__ == "__main__":
    unittest.main()
