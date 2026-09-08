from __future__ import annotations

import json
import tempfile
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ftgspp.data.prepare_selfcap import write_time_offsets
from ftgspp.native_runtime.backend import SharedRgbViewCache
from ftgspp.native_runtime.dataset import (
    NativeDataset,
    NativeView,
    _colmap_undistort_maps,
    _colmap_undistorted_geometry,
    _scan_images,
    _time_offsets,
    clear_undistort_cache,
)


def _view(path: Path, *, frame: int, camera: int, timestamp: float) -> NativeView:
    return NativeView(
        path=path,
        frame=frame,
        camera_id=camera,
        timestamp=timestamp,
        w2c=np.eye(4, dtype=np.float32),
        source_intrinsic=np.eye(3, dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
        distortion=np.zeros(5, dtype=np.float32),
        size=(8, 12),
    )


def _rgba8(value: int) -> np.ndarray:
    rgb = np.full((2, 3, 3), value, dtype=np.uint8)
    rgba = np.empty((2, 3, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return rgba


class _DecodedView:
    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def load_rgb(self) -> np.ndarray:
        self.calls += 1
        return np.full((2, 3, 3), self.value, dtype=np.uint8)


class DatasetManifestAndCacheTests(unittest.TestCase):
    def test_native_view_preserves_png_alpha_as_rgba(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "masked.png"
            rgba = np.zeros((8, 12, 4), dtype=np.uint8)
            rgba[..., 0] = 11
            rgba[..., 1] = 22
            rgba[..., 2] = 33
            rgba[..., 3] = np.arange(12, dtype=np.uint8)[None, :]
            self.assertTrue(cv2.imwrite(
                str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
            ))

            decoded = _view(
                path, frame=0, camera=7, timestamp=0.0,
            ).load_rgb()

        np.testing.assert_array_equal(decoded, rgba)

    def test_native_view_uses_reference_undistort_then_area_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distorted.png"
            height, width = 24, 32
            yy, xx = np.indices((height, width), dtype=np.uint8)
            rgb = np.stack((
                xx * np.uint8(7),
                yy * np.uint8(9),
                (xx + yy) * np.uint8(5),
            ), axis=-1)
            self.assertTrue(cv2.imwrite(
                str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            ))
            source_k = np.asarray([
                [28.0, 0.0, 15.4],
                [0.0, 27.0, 11.7],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            distortion = np.asarray(
                [0.08, -0.025, 0.001, -0.002, 0.0], dtype=np.float32,
            )
            undist_h, undist_w, undist_k = _colmap_undistorted_geometry(
                height, width, source_k, distortion,
            )
            target_h, target_w = undist_h // 2, undist_w // 2
            target_k = undist_k.copy()
            target_k[0] *= np.float32(target_w / undist_w)
            target_k[1] *= np.float32(target_h / undist_h)
            view = NativeView(
                path=path,
                frame=0,
                camera_id=3,
                timestamp=0.0,
                w2c=np.eye(4, dtype=np.float32),
                source_intrinsic=source_k,
                intrinsic=target_k,
                distortion=distortion,
                size=(target_h, target_w),
            )

            map_x, map_y = _colmap_undistort_maps(
                height, width, source_k, distortion,
            )
            expected = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
            expected = cv2.resize(
                expected, (target_w, target_h), interpolation=cv2.INTER_AREA,
            )
            expected_rgba = np.empty(expected.shape[:2] + (4,), dtype=np.uint8)
            expected_rgba[..., :3] = expected
            expected_rgba[..., 3] = 255
            clear_undistort_cache()
            decoded = view.load_rgb()

        np.testing.assert_array_equal(decoded, expected_rgba)

    def test_legacy_flat_scan_preserves_sparse_ids_spelling_and_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame in range(2):
                (root / f"c007-f{frame:06d}.jpg").write_bytes(b"seven")
                (root / f"c21-f{frame:06d}.PNG").write_bytes(b"twenty-one")

            frames, cameras, paths = _scan_images(root, 0, 2)

        self.assertEqual(frames, [0, 1])
        self.assertEqual(cameras, [7, 21])
        self.assertEqual(
            [[path.name for path in row] for row in paths],
            [
                ["c007-f000000.jpg", "c21-f000000.PNG"],
                ["c007-f000001.jpg", "c21-f000001.PNG"],
            ],
        )

    def test_legacy_flat_scan_rejects_duplicate_normalized_camera_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "c7-f000000.jpg").write_bytes(b"a")
            (root / "c007-f000000.png").write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "duplicate legacy flat"):
                _scan_images(root, 0, 1)

    def test_legacy_flat_scan_rejects_illegal_camera_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ccenter-f000000.jpg").write_bytes(b"a")
            with self.assertRaisesRegex(ValueError, "invalid legacy flat"):
                _scan_images(root, 0, 1)

    def test_legacy_flat_scan_rejects_incomplete_camera_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "c007-f000000.jpg", "c021-f000000.png",
                "c007-f000001.jpg",
            ):
                (root / name).write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "inconsistent camera coverage"):
                _scan_images(root, 0, 2)

    def test_directory_scan_preserves_sparse_ids_real_paths_and_frame_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_7 = root / "c007"
            camera_21 = root / "21"
            camera_7.mkdir()
            camera_21.mkdir()
            for frame in (5, 6):
                (camera_7 / f"{frame:06d}.jpg").write_bytes(b"seven")
                (camera_21 / f"{frame:06d}.PNG").write_bytes(b"twenty-one")

            frames, cameras, paths = _scan_images(root, 0, 2)

        self.assertEqual(frames, [0, 1])
        self.assertEqual(cameras, [7, 21])
        self.assertEqual(
            [[path.name for path in row] for row in paths],
            [
                ["000005.jpg", "000005.PNG"],
                ["000006.jpg", "000006.PNG"],
            ],
        )

    def test_directory_scan_rejects_duplicate_normalized_camera_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("7", "c007"):
                camera = root / name
                camera.mkdir()
                (camera / "000000.jpg").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "duplicate numeric IDs"):
                _scan_images(root, 0, 1)

    def test_incomplete_time_offset_file_is_strict_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (root / "t_offsets.json").write_text(
                '{"7": 0.001}', encoding="utf-8",
            )
            config = SimpleNamespace(data=SimpleNamespace(video_path=root))
            with self.assertRaisesRegex(ValueError, "strict_time_offsets=false"):
                _time_offsets(config, [7, 21], images)

            config.data.strict_time_offsets = False
            with self.assertWarnsRegex(UserWarning, "using 0 seconds"):
                offsets = _time_offsets(config, [7, 21], images)
            self.assertEqual(offsets, {7: 0.001, 21: 0.0})

    def test_time_offsets_accept_c_aliases_and_reject_conflicts_or_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            config = SimpleNamespace(data=SimpleNamespace(
                video_path=root, strict_time_offsets=True,
            ))
            offsets_path = root / "t_offsets.json"
            offsets_path.write_text(
                '{"c007": 0.001, "c000021": -0.002}', encoding="utf-8",
            )
            self.assertEqual(
                _time_offsets(config, [7, 21], images),
                {7: 0.001, 21: -0.002},
            )

            offsets_path.write_text(
                '{"7": 0.001, "c007": 0.002, "21": 0.0}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting aliases"):
                _time_offsets(config, [7, 21], images)

            offsets_path.write_text(
                '{"7": 1e309, "21": 0.0}', encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                _time_offsets(config, [7, 21], images)

    def test_prepare_keeps_missing_offsets_absent_for_strict_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sync.json"
            destination = root / "t_offsets.json"
            source.write_text('{"7": 0.001}', encoding="utf-8")

            write_time_offsets(
                sync_path=source, output_path=destination,
                camera_names=["7", "21"],
            )

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"7": 0.001},
            )

    def test_prepare_normalizes_c_prefixed_offset_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sync.json"
            destination = root / "t_offsets.json"
            source.write_text('{"c007": 0.001}', encoding="utf-8")

            write_time_offsets(
                sync_path=source, output_path=destination,
                camera_names=["7"],
            )

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"7": 0.001},
            )

    def test_manifest_requires_exact_discovered_camera_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "000000.jpg"
            path.write_bytes(b"not-decoded")
            dataset = NativeDataset(
                frames=[0], cameras=[7, 21], train_cameras=[7],
                eval_cameras=[], camera_split_source="incomplete-test",
                train_views=[_view(path, frame=0, camera=7, timestamp=0.0)],
                eval_views=[], fps=60.0, source_root=root,
                time_offsets={7: 0.0, 21: 0.0},
            )
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                dataset.validate_view_manifest()

    def test_checkpoint_records_offset_range_and_same_frame_spread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "c007.jpg", root / "c021.jpg"]
            for path in paths:
                path.write_bytes(b"not-decoded")
            dataset = NativeDataset(
                frames=[0], cameras=[7, 21], train_cameras=[7],
                eval_cameras=[21], camera_split_source="test",
                train_views=[
                    _view(paths[0], frame=0, camera=7, timestamp=0.002),
                ],
                eval_views=[
                    _view(paths[1], frame=0, camera=21, timestamp=-0.003),
                ],
                fps=60.0, source_root=root,
                time_offsets={7: -0.002, 21: 0.003},
            )
            dataset.validate_view_manifest()
            metadata = dataset.checkpoint_metadata()

        offset_range = metadata["time_offset_range_seconds"]
        self.assertAlmostEqual(offset_range["minimum"], -0.002)
        self.assertAlmostEqual(offset_range["maximum"], 0.003)
        self.assertAlmostEqual(offset_range["span"], 0.005)
        spread = metadata["same_frame_adjusted_timestamp_spread_seconds"]
        self.assertAlmostEqual(spread["minimum"], 0.005)
        self.assertAlmostEqual(spread["maximum"], 0.005)
        self.assertAlmostEqual(spread["mean"], 0.005)

    def test_manifest_checks_all_paths_and_has_stable_timing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{index:06d}.jpg" for index in range(2)]
            for path in paths:
                path.write_bytes(b"not-decoded-by-manifest")
            dataset = NativeDataset(
                frames=[0, 1], cameras=[7], train_cameras=[7],
                eval_cameras=[], camera_split_source="test",
                train_views=[
                    _view(paths[0], frame=0, camera=7, timestamp=0.0),
                    _view(paths[1], frame=1, camera=7, timestamp=1.0 / 60.0),
                ],
                eval_views=[], fps=60.0, source_root=root,
                time_offsets={7: 0.125},
            )
            dataset.validate_view_manifest()
            first = dataset.checkpoint_metadata()
            second = dataset.checkpoint_metadata()
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(first["camera_dense_mapping"], {"7": 0})
            self.assertEqual(first["frames"], [0, 1])
            self.assertEqual(first["fps"], 60.0)
            self.assertEqual(first["time_offsets"], {"7": 0.125})
            first["frames"].append(99)
            self.assertEqual(dataset.checkpoint_metadata()["frames"], [0, 1])

            paths[1].unlink()
            with self.assertRaises(FileNotFoundError):
                dataset.validate_view_manifest()

    def test_shared_cache_prefetch_is_ordered_and_deduplicates_rank_views(self):
        cache = SharedRgbViewCache(capacity=4)
        left, right = _DecodedView(10), _DecodedView(20)
        cache.prefetch([
            (True, 2, left),
            (True, 2, left),  # same view selected by two ranks
            (True, 5, right),
        ], workers=2)
        np.testing.assert_array_equal(
            cache.get_or_load(is_train=True, index=2, view=left),
            _rgba8(10),
        )
        np.testing.assert_array_equal(
            cache.get_or_load(is_train=True, index=5, view=right),
            _rgba8(20),
        )
        self.assertEqual(left.calls, 1)
        self.assertEqual(right.calls, 1)
        self.assertEqual(list(cache._items), [(True, 2), (True, 5)])

    def test_rank_completion_order_does_not_change_prefetched_lru(self):
        cache = SharedRgbViewCache(capacity=2)
        first, second, third = (
            _DecodedView(10), _DecodedView(20), _DecodedView(30),
        )
        cache.prefetch([
            (True, 1, first),
            (True, 2, second),
        ], workers=2)

        # Simulate rank 1 finishing before rank 0. Prefetch already fixed the
        # global-batch LRU order, so these worker hits must not reorder it.
        cache.get_or_load(is_train=True, index=2, view=second)
        cache.get_or_load(is_train=True, index=1, view=first)
        cache.prefetch([(True, 3, third)], workers=1)

        self.assertEqual(list(cache._items), [(True, 2), (True, 3)])
        self.assertEqual((first.calls, second.calls, third.calls), (1, 1, 1))

    def test_lookahead_decode_does_not_touch_lru_until_consumed(self):
        cache = SharedRgbViewCache(capacity=2)
        current, future = _DecodedView(10), _DecodedView(20)
        cache.prefetch([(True, 1, current)], workers=1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            cache.schedule([(True, 2, future)], executor=pool)
            pool.shutdown(wait=True)
            self.assertEqual(list(cache._items), [(True, 1)])
            cache.prefetch([(True, 2, future)], workers=1)
        self.assertEqual(list(cache._items), [(True, 1), (True, 2)])
        self.assertEqual((current.calls, future.calls), (1, 1))

    def test_pinned_prefill_does_not_evict_and_is_once_only(self):
        cache = SharedRgbViewCache(capacity=2, pin=True)
        first = np.full((2, 3, 3), 1, np.uint8)
        second = np.full((2, 3, 3), 2, np.uint8)
        third = np.full((2, 3, 3), 3, np.uint8)
        self.assertTrue(cache.prefill_arrays([
            (True, 0, first),
            (True, 1, second),
            (False, 0, third),
        ]))
        self.assertTrue(cache.is_prefilled)
        self.assertEqual(cache.capacity, 3)
        self.assertFalse(cache.prefill_arrays([(True, 9, first)]))
        self.assertEqual(list(cache._items), [(True, 0), (True, 1), (False, 0)])
        # Extra stores under pin must not drop the resident set.
        cache._store((True, 2), np.full((2, 3, 3), 4, np.uint8))
        self.assertIn((True, 0), cache._items)
        self.assertIn((True, 1), cache._items)
        self.assertIn((False, 0), cache._items)
        self.assertIn((True, 2), cache._items)

    def test_shared_resident_upload_decodes_once_across_replicas(self):
        from unittest.mock import MagicMock, patch

        from ftgspp.native_runtime.backend import NativeVulkanBackend

        train = []
        for i in range(3):
            view = _DecodedView(10 + i)
            view.camera_id = 0
            train.append(view)
        eval_views = []
        for i in range(2):
            view = _DecodedView(100 + i)
            view.camera_id = 0
            eval_views.append(view)
        dataset = SimpleNamespace(
            train_views=train,
            eval_views=eval_views,
            cameras=[SimpleNamespace(camera_id=0)],
        )
        shared = SharedRgbViewCache(capacity=len(train) + len(eval_views), pin=True)
        decode_counts = {"n": 0}

        def fake_decode(self, views, *, workers, label):
            decode_counts["n"] += 1
            return [
                np.ascontiguousarray(view.load_rgb(), dtype=np.uint8)
                for view in views
            ]

        def fake_add_meta(self, view, is_train, *, camera_index):
            return None

        def fake_finish(self, camera_ids):
            return None

        def fake_camera_map(self, dataset_arg):
            return {0: 0}, [0]

        backends = []
        for _ in range(2):
            backend = NativeVulkanBackend.__new__(NativeVulkanBackend)
            backend.native = MagicMock()
            backend.native.num_train = len(train)
            backend.native.num_val = len(eval_views)
            backend.native.clear_dynamic_views = MagicMock()
            backend._dataset = None
            backend._bound_train_index = None
            backend._bound_train_images = OrderedDict()
            backend._device_image_cache_capacity = 0
            backend._view_load_mode = "lazy"
            backend._shared_host_resident = False
            backend._rgb_cache = SharedRgbViewCache(1)
            backend._view_load_workers = 0
            backends.append(backend)

        with (
            patch.object(NativeVulkanBackend, "_decode_views", fake_decode),
            patch.object(NativeVulkanBackend, "_add_view_meta", fake_add_meta),
            patch.object(NativeVulkanBackend, "_finish_color_cameras", fake_finish),
            patch.object(NativeVulkanBackend, "_camera_index_map", fake_camera_map),
        ):
            for backend in backends:
                backend.upload_dataset(
                    dataset,
                    view_load_mode="resident",
                    cache_on_device=False,
                    workers=2,
                    shared_rgb_cache=shared,
                )

        # One train decode + one eval decode on the first replica only.
        self.assertEqual(decode_counts["n"], 2)
        self.assertTrue(shared.is_prefilled)
        self.assertEqual(len(shared._items), 5)
        for backend in backends:
            self.assertTrue(backend._shared_host_resident)
            self.assertIs(backend._rgb_cache, shared)
            self.assertTrue(backend._binds_rgb_per_step())

        # Bind path must pull from the shared pin-set without re-decoding.
        before = [view.calls for view in train]
        backends[0]._dataset = dataset
        backends[0].native.bind_dynamic_view_image = MagicMock()
        backends[0].native.release_dynamic_view_image = MagicMock()
        rgb = backends[0]._bind_train_rgb(1)
        np.testing.assert_array_equal(rgb, _rgba8(11))
        backends[0].native.bind_dynamic_view_image.assert_called_once()
        self.assertEqual([view.calls for view in train], before)


if __name__ == "__main__":
    unittest.main()
