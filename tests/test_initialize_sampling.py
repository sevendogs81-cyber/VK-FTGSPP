from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from ftgspp.native_runtime.initialize import (
    _initialize_explicit_frames,
    _select_point_indices,
)


class _View:
    def __init__(self, timestamp: float):
        self.timestamp = timestamp


class _Dataset:
    frames = [0, 1]
    cameras = [0]

    @staticmethod
    def view(frame: int, _camera_id: int) -> _View:
        return _View(float(frame))


def _chunks() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in (
        "means", "log_scales", "quaternions", "opacity_logits",
        "sh_coefficients", "times", "duration_raw", "velocities",
        "gate_raw",
    )}


class InitializeSamplingTests(unittest.TestCase):
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
        selection = _select_point_indices(
            np.random.default_rng(7), population_size=100, budget=40,
        )

        self.assertEqual(selection.shape, (40,))
        self.assertEqual(np.unique(selection).size, 40)
        self.assertGreaterEqual(int(selection.min()), 0)
        self.assertLess(int(selection.max()), 100)

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
                temporal_flow_path="unused",
                temporal_covis_thresh=0.3,
                num_velocity_nns=1,
                temporal_motion_adapted=False,
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
            executor=None,
        )

        np.testing.assert_array_equal(chunks["means"][0], xyz_current)
        self.assertEqual(chunks["means"][0].shape[0], 5)
        self.assertEqual(chunks["velocities"][0].shape[0], 5)


if __name__ == "__main__":
    unittest.main()
