from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ftgspp.native_runtime.initialize import _temporal_ufm_velocity


class _View:
    def __init__(self, w2c: np.ndarray, intrinsic: np.ndarray, size: tuple[int, int]):
        self.w2c = w2c
        self.intrinsic = intrinsic
        self.size = size


class _Dataset:
    def __init__(self, cameras: list[int], views: dict[tuple[int, int], _View]):
        self.cameras = cameras
        self._views = views

    def view(self, frame: int, camera_id: int) -> _View:
        return self._views[(frame, camera_id)]


def _write_pair(root: Path, frame_0: int, frame_1: int, cameras: int, height: int, width: int) -> None:
    pair = root / f"f{frame_0:06d}_f{frame_1:06d}"
    pair.mkdir(parents=True, exist_ok=True)
    flow = np.zeros((height, width, 2), np.float32)
    flow[..., 0] = 2.0
    covis = np.ones((height, width), np.float32)
    for camera_index in range(cameras):
        np.savez_compressed(pair / f"c{camera_index:03d}.npz", flow=flow, covis=covis)


class InitializeFlowTests(unittest.TestCase):
    def test_parallel_camera_flow_matches_serial(self):
        height, width, n_cameras = 32, 48, 6
        intrinsic = np.array([[24.0, 0.0, 24.0], [0.0, 16.0, 16.0], [0.0, 0.0, 1.0]], np.float32)
        w2c = np.eye(4, dtype=np.float32)
        cameras = list(range(n_cameras))
        views = {
            (frame, camera_id): _View(w2c, intrinsic, (height, width))
            for frame in (0, 5)
            for camera_id in cameras
        }
        dataset = _Dataset(cameras, views)
        query = np.stack([
            np.linspace(-0.4, 0.4, 64, dtype=np.float32),
            np.linspace(-0.3, 0.3, 64, dtype=np.float32),
            np.full(64, 1.5, np.float32),
        ], axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pair(root, 0, 5, n_cameras, height, width)
            serial, serial_valid = _temporal_ufm_velocity(
                query, dataset, 0, 5, 1.0 / 12.0, root, 0.3, executor=None,
            )
            with ThreadPoolExecutor(max_workers=4) as executor:
                parallel, parallel_valid = _temporal_ufm_velocity(
                    query, dataset, 0, 5, 1.0 / 12.0, root, 0.3, executor=executor,
                )
        self.assertTrue(np.array_equal(serial_valid, parallel_valid))
        self.assertTrue(np.allclose(serial, parallel, rtol=0, atol=0))
        self.assertGreater(int(serial_valid.sum()), 0)


if __name__ == "__main__":
    unittest.main()
