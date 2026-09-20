from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ftgspp.training.initialize import (
    KeyframeCloud,
    _initialize_explicit_frames,
    _keyframe_clouds,
    _select_point_indices,
    _spacing,
    _valid_prepared_point_manifest,
    allocate_keyframe_budgets,
    initialize_explicit,
)
from ftgspp.training.point_sampling import spatial_sample_indices


class _View:
    def __init__(self, timestamp: float):
        self.timestamp = timestamp


class _Dataset:
    frames = [0, 1]
    cameras = [0]
    train_cameras = [0]
    fps = 1.0

    @staticmethod
    def view(frame: int, _camera_id: int) -> _View:
        return _View(float(frame))

    @staticmethod
    def checkpoint_metadata() -> dict[str, str]:
        return {"fingerprint": "test-dataset"}


def _chunks() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in (
        "means", "log_scales", "quaternions", "opacity_logits",
        "sh_coefficients", "times", "duration_raw", "velocities",
        "gate_raw",
    )}


class InitializeSamplingTests(unittest.TestCase):
    def test_carving_requires_a_versioned_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(_valid_prepared_point_manifest(
                root, _Dataset(), [0],
            ))
            self.assertFalse(_valid_prepared_point_manifest(
                root, _Dataset(), [0], expected_method="carving",
            ))

    def test_carving_backend_requires_a_carving_manifest(self):
        from ftgspp.io.provenance import source_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "f000000.ply"
            ply.write_bytes(b"test-ply")
            manifest = {
                "schema": "ftgspp-visual-hull-ibr-v2",
                "method": "carving",
                "dataset_fingerprint": "test-dataset",
                "source_identity": {
                    "set_sha256": source_identity()["set_sha256"],
                },
                "rows": [{
                    "frame": 0,
                    "ply": ply.name,
                    "ply_sha256": hashlib.sha256(ply.read_bytes()).hexdigest(),
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "completed.json").write_text(json.dumps({
                "schema": "ftgspp-visual-hull-ibr-v2",
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes(),
                ).hexdigest(),
                "keyframes": 1,
            }), encoding="utf-8")

            self.assertTrue(_valid_prepared_point_manifest(
                root, _Dataset(), [0],
                expected_method="carving",
            ))
            self.assertFalse(_valid_prepared_point_manifest(
                root, _Dataset(), [0], expected_method="point_filtering",
            ))

    def test_adaptive_budget_is_exact_capacity_aware_and_linear_time(self):
        budgets = allocate_keyframe_budgets(
            [100, 4, 25], 60,
        )
        self.assertEqual(sum(budgets), 60)
        self.assertTrue(all(
            budget <= capacity
            for budget, capacity in zip(budgets, [100, 4, 25], strict=True)
        ))
        sparse = allocate_keyframe_budgets(
            [2, 3], 20,
        )
        self.assertEqual(sum(sparse), 20)
        self.assertGreaterEqual(sparse[0], 2)
        self.assertGreaterEqual(sparse[1], 3)
        self.assertEqual(
            allocate_keyframe_budgets(
                [100, 1, 100], 3,
            ),
            [1, 1, 1],
        )

    def test_sparse_jitter_fallback_does_not_duplicate_coordinates(self):
        config = SimpleNamespace(
            init=SimpleNamespace(
                num_velocity_nns=3,
                scale=0.1, opacity=0.1, duration=0.25,
            ),
            model=SimpleNamespace(
                max_duration=1.0, marginal_gating=False,
                gate_init_logit=-1.0,
            ),
        )
        chunks = _chunks()
        xyz = np.asarray([[0.0, 0.0, 1.0]], np.float32)
        rgb = np.zeros((1, 3), np.uint8)
        _initialize_explicit_frames(
            config, _Dataset(), [(xyz, rgb)], [0], [3],
            np.random.default_rng(4), chunks,
        )
        self.assertEqual(len(np.unique(chunks["means"][0], axis=0)), 3)
        np.testing.assert_array_equal(
            chunks["velocities"][0], np.zeros((3, 3), np.float32),
        )

    def test_user_cloud_without_manifest_uses_its_training_camera_timestamp(self):
        clouds = _keyframe_clouds(
            [(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8))],
            root=None, frames=[1], dataset=_Dataset(), source="user_points",
        )
        self.assertEqual(clouds[0].reference_timestamp, 1.0)
        self.assertEqual(
            clouds[0].provenance["timestamp_source"],
            "selected_source_training_camera",
        )

    def test_short_combined_population_uses_replicas_to_hit_exact_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            clouds = [
                KeyframeCloud(
                    xyz=np.asarray([[0, 0, 1]], np.float32),
                    rgb=np.zeros((1, 3), np.uint8),
                    reference_timestamp=0.0, frame=0,
                    provenance={"source": "test", "root": None},
                ),
                KeyframeCloud(
                    xyz=np.asarray([[0, 0, 1], [1, 0, 1]], np.float32),
                    rgb=np.zeros((2, 3), np.uint8),
                    reference_timestamp=1.0, frame=1,
                    provenance={"source": "test", "root": None},
                ),
            ]
            config = SimpleNamespace(
                init=SimpleNamespace(
                    keyframe_stride=1, num_gaussians=5,
                    num_velocity_nns=0, scale=0.1, opacity=0.1,
                    duration=0.25,
                ),
                train=SimpleNamespace(seed=0),
                model=SimpleNamespace(max_duration=2.0, marginal_gating=False),
                run=SimpleNamespace(output_path=output),
            )
            with patch(
                "ftgspp.training.initialize._initial_points",
                return_value=clouds,
            ):
                model = initialize_explicit(config, _Dataset())
            report = json.loads(
                (output / "initialization_manifest.json").read_text("utf-8")
            )
            self.assertEqual(len(model), 5)
            self.assertEqual(report["actual_gaussians"], 5)
            self.assertTrue(report["comparable_budget"])
            self.assertIsNone(report["incomparable_reason"])

    def test_returns_every_point_in_source_order_when_cloud_fits_budget(self):
        for budget in (5, 8):
            with self.subTest(budget=budget):
                rng = np.random.default_rng(7)
                selection = _select_point_indices(
                    rng, population_size=5, budget=budget,
                )

                np.testing.assert_array_equal(
                    selection, np.arange(5, dtype=np.int64),
                )

    def test_subsamples_without_replacement_when_cloud_exceeds_budget(self):
        xyz = np.random.default_rng(3).normal(size=(100, 3)).astype(np.float32)
        selection = _select_point_indices(
            np.random.default_rng(7), population_size=100, budget=40, xyz=xyz,
        )

        self.assertEqual(selection.shape, (40,))
        self.assertEqual(np.unique(selection).size, 40)
        self.assertGreaterEqual(int(selection.min()), 0)
        self.assertLess(int(selection.max()), 100)

    def test_spatial_sample_keeps_separated_clusters(self):
        rng = np.random.default_rng(0)
        left = rng.normal(size=(400, 3)).astype(np.float32) * 0.02
        right = left + np.array([10.0, 0.0, 0.0], np.float32)
        xyz = np.concatenate([left, right], axis=0)
        selection = _select_point_indices(
            np.random.default_rng(4), population_size=len(xyz), budget=20, xyz=xyz,
        )
        chosen = xyz[selection]
        self.assertGreater((chosen[:, 0] < 5.0).sum(), 0)
        self.assertGreater((chosen[:, 0] > 5.0).sum(), 0)

    def test_local_spacing_excludes_query_point(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        expected = np.sqrt(np.asarray([
            (1.0**2 + 3.0**2 + 6.0**2) / 3.0,
            (1.0**2 + 2.0**2 + 5.0**2) / 3.0,
            (3.0**2 + 2.0**2 + 3.0**2) / 3.0,
            (6.0**2 + 5.0**2 + 3.0**2) / 3.0,
        ], dtype=np.float32))
        np.testing.assert_allclose(_spacing(points), expected, rtol=1e-6)

    def test_surface_sampling_targets_occupied_cells(self):
        rng = np.random.default_rng(17)
        xy = rng.uniform(-10.0, 10.0, size=(12000, 2))
        z = rng.normal(0.0, 1e-4, size=(12000, 1))
        points = np.concatenate([xy, z], axis=1).astype(np.float32)
        diagnostics: dict[str, object] = {}
        selection = spatial_sample_indices(
            np.random.default_rng(23), points, 800, diagnostics=diagnostics,
        )
        self.assertEqual(len(selection), 800)
        self.assertEqual(np.unique(selection).size, 800)
        self.assertEqual(diagnostics["strategy"], "adaptive_surface_grid")
        self.assertGreaterEqual(float(diagnostics["occupied_fraction"]), 0.8)
        self.assertEqual(diagnostics["fallback_points"], 0)
        self.assertLessEqual(diagnostics["voxel_search_full_passes"], 7)
        self.assertLessEqual(diagnostics["voxel_search_sample_points"], 8192)

    def test_smaller_reference_cloud_does_not_reduce_current_frame_selection(self):
        xyz_current = np.stack(
            [np.arange(5, dtype=np.float32), np.zeros(5), np.ones(5)], axis=1,
        )
        xyz_reference = np.array(
            [[0.0, 0.0, 1.0], [4.0, 0.0, 1.0]], dtype=np.float32,
        )
        rgb_current = np.arange(15, dtype=np.uint8).reshape(5, 3)
        rgb_reference = np.arange(6, dtype=np.uint8).reshape(2, 3)
        config = SimpleNamespace(
            init=SimpleNamespace(
                num_velocity_nns=1,
                scale=0.1,
                opacity=0.1,
                duration=0.25,
            ),
            model=SimpleNamespace(max_duration=1.0, marginal_gating=False),
        )
        chunks = _chunks()

        _initialize_explicit_frames(
            config,
            _Dataset(),
            [(xyz_current, rgb_current), (xyz_reference, rgb_reference)],
            frame_positions=[0, 1],
            per_frame=8,
            rng=np.random.default_rng(7),
            chunks=chunks,
        )

        restored = (
            chunks["means"][0]
            - (chunks["times"][0] - np.float32(0.0)) * chunks["velocities"][0]
        )
        np.testing.assert_allclose(restored[:5], xyz_current, rtol=0.0, atol=1e-5)
        self.assertEqual(chunks["means"][0].shape[0], 8)
        self.assertEqual(chunks["velocities"][0].shape[0], 8)

    def test_time_anchor_restores_source_for_positive_and_negative_intervals(self):
        xyz0 = np.stack(
            [np.arange(4, dtype=np.float32), np.zeros(4), np.ones(4)], axis=1,
        )
        xyz1 = xyz0 + np.array([0.5, 0.0, 0.0], np.float32)
        rgb = np.zeros((4, 3), np.uint8)
        config = SimpleNamespace(
            init=SimpleNamespace(
                num_velocity_nns=1,
                scale=0.1, opacity=0.1, duration=0.25,
            ),
            model=SimpleNamespace(max_duration=1.0, marginal_gating=False),
        )
        chunks = _chunks()
        _initialize_explicit_frames(
            config, _Dataset(), [(xyz0, rgb), (xyz1, rgb)],
            [0, 1], [4, 4], np.random.default_rng(0), chunks,
        )
        for index, source, t_ref in ((0, xyz0, 0.0), (1, xyz1, 1.0)):
            restored = (
                chunks["means"][index]
                - (chunks["times"][index] - np.float32(t_ref))
                * chunks["velocities"][index]
            )
            np.testing.assert_allclose(restored, source, rtol=0.0, atol=1e-5)

    def test_replicas_inherit_source_scales_and_opacity(self):
        xyz = np.asarray([[0.0, 0.0, 1.0], [4.0, 0.0, 1.0]], np.float32)
        rgb = np.zeros((2, 3), np.uint8)
        config = SimpleNamespace(
            init=SimpleNamespace(
                num_velocity_nns=0,
                scale=0.1, opacity=0.1, duration=0.25,
            ),
            model=SimpleNamespace(max_duration=1.0, marginal_gating=False),
        )
        chunks = _chunks()
        _initialize_explicit_frames(
            config, _Dataset(), [(xyz, rgb)], [0], [8],
            np.random.default_rng(3), chunks,
        )
        scales = chunks["log_scales"][0]
        np.testing.assert_allclose(
            scales,
            np.repeat(scales[:1], len(scales), axis=0),
            rtol=0.0,
            atol=1e-6,
        )
        expected_logit = np.log(np.float32(0.1) / np.float32(0.9))
        np.testing.assert_allclose(
            chunks["opacity_logits"][0],
            np.full((8, 1), expected_logit, np.float32),
            rtol=0.0,
            atol=1e-6,
        )

    def test_empty_keyframe_is_skipped_instead_of_synthesizing_points(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            clouds = [
                KeyframeCloud(
                    xyz=np.empty((0, 3), np.float32),
                    rgb=np.empty((0, 3), np.uint8),
                    reference_timestamp=0.0, frame=0,
                    provenance={"source": "test", "root": None},
                ),
                KeyframeCloud(
                    xyz=np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float32),
                    rgb=np.zeros((3, 3), np.uint8),
                    reference_timestamp=1.0, frame=1,
                    provenance={"source": "test", "root": None},
                ),
            ]
            config = SimpleNamespace(
                init=SimpleNamespace(
                    keyframe_stride=1, num_gaussians=3,
                    num_velocity_nns=0, scale=0.1, opacity=0.1,
                    duration=0.25,
                ),
                train=SimpleNamespace(seed=0),
                model=SimpleNamespace(max_duration=2.0, marginal_gating=False),
                run=SimpleNamespace(output_path=output),
            )
            with patch(
                "ftgspp.training.initialize._initial_points",
                return_value=clouds,
            ):
                model = initialize_explicit(config, _Dataset())
            self.assertEqual(len(model), 3)
            report = json.loads(
                (output / "initialization_manifest.json").read_text("utf-8")
            )
            self.assertEqual(report["keyframes"][0]["replica_count"], 0)
            self.assertTrue(report["keyframes"][0].get("skipped"))


if __name__ == "__main__":
    unittest.main()
