from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ftgspp.data.prepare_selfcap import write_time_offsets
from ftgspp.training.backend import SharedRgbViewCache
from ftgspp.data.scene import (
    Scene,
    VIEW_HASH_CACHE_RELATIVE,
    View,
    _colmap_undistort_maps,
    _colmap_undistorted_geometry,
    _file_sha256,
    _file_sha256_many,
    _scan_images,
    _time_offsets,
    clear_undistort_cache,
)


def _view(path: Path, *, frame: int, camera: int, timestamp: float, **kwargs) -> View:
    return View(
        path=path,
        frame=frame,
        camera_id=camera,
        timestamp=timestamp,
        w2c=np.eye(4, dtype=np.float32),
        source_intrinsic=np.eye(3, dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
        distortion=np.zeros(5, dtype=np.float32),
        size=(8, 12),
        **kwargs,
    )


def _tiny_scene(root: Path, paths: list[Path], *, masks: list[Path] | None = None) -> Scene:
    views = []
    for index, path in enumerate(paths):
        extra = {}
        if masks is not None:
            extra["mask_path"] = masks[index]
            extra["mask_source"] = "external"
        views.append(
            _view(path, frame=index, camera=7, timestamp=index / 60.0, **extra)
        )
    return Scene(
        frames=list(range(len(paths))),
        cameras=[7],
        train_cameras=[7],
        eval_cameras=[],
        camera_split_source="test",
        train_views=views,
        eval_views=[],
        fps=60.0,
        source_root=root,
        time_offsets={7: 0.0},
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
            view = View(
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

    def test_directory_scan_ignores_hidden_sidecar_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "c007"
            camera.mkdir()
            (camera / "000000.jpg").write_bytes(b"seven")
            sidecar = root / ".ftgspp"
            sidecar.mkdir()
            (sidecar / "view_sha256_v1.json").write_text("{}", encoding="utf-8")

            frames, cameras, paths = _scan_images(root, 0, 1)

        self.assertEqual(frames, [0])
        self.assertEqual(cameras, [7])
        self.assertEqual(paths[0][0].name, "000000.jpg")

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
            dataset = Scene(
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
            dataset = Scene(
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
            dataset = Scene(
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

    def test_parallel_view_hashes_match_serial_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{index:06d}.jpg" for index in range(4)]
            masks = [root / f"{index:06d}_mask.png" for index in range(4)]
            for index, path in enumerate(paths):
                path.write_bytes(f"rgb-{index}".encode())
                masks[index].write_bytes(f"mask-{index}".encode())
            with patch.dict(os.environ, {"FTGSPP_HASH_WORKERS": "4"}):
                hashes = _file_sha256_many(
                    [*paths, *masks], cache_root=root,
                )
            for path in (*paths, *masks):
                self.assertEqual(hashes[path.as_posix()], _file_sha256(path))
            cache = json.loads(
                (root / VIEW_HASH_CACHE_RELATIVE).read_text(encoding="utf-8"),
            )
            self.assertEqual(cache["schema"], "ftgspp-view-sha256-v1")
            self.assertEqual(len(cache["entries"]), 8)

    def test_view_hash_cache_sits_beside_images_not_inside_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            images = scene / "images"
            images.mkdir()
            path = images / "000000.jpg"
            path.write_bytes(b"payload")
            _tiny_scene(images, [path]).checkpoint_metadata()
            self.assertTrue((scene / VIEW_HASH_CACHE_RELATIVE).is_file())
            self.assertFalse((images / ".ftgspp").exists())

    def test_view_hash_disk_cache_avoids_reread_and_keeps_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{index:06d}.jpg" for index in range(3)]
            for index, path in enumerate(paths):
                path.write_bytes(f"payload-{index}".encode())
            with patch.dict(os.environ, {"FTGSPP_HASH_WORKERS": "3"}):
                first = _tiny_scene(root, paths).checkpoint_metadata()
            cache_path = root / VIEW_HASH_CACHE_RELATIVE
            self.assertTrue(cache_path.is_file())
            with (
                patch.dict(os.environ, {"FTGSPP_HASH_WORKERS": "8"}),
                patch("ftgspp.data.scene._file_sha256") as hashed,
            ):
                hashed.side_effect = AssertionError("disk cache should not reread bytes")
                second = _tiny_scene(root, paths).checkpoint_metadata()
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            hashed.assert_not_called()
            cache_path.unlink()
            third = _tiny_scene(root, paths).checkpoint_metadata()
            self.assertEqual(first["fingerprint"], third["fingerprint"])

    def test_verified_view_hash_avoids_checkpoint_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "000000.jpg"
            path.write_bytes(b"manifest-verified-payload")
            digest = _file_sha256(path)
            view = _view(
                path, frame=0, camera=7, timestamp=0.0,
                rgb_sha256=digest,
            )
            scene = Scene(
                frames=[0], cameras=[7], train_cameras=[7],
                eval_cameras=[], camera_split_source="test",
                train_views=[view], eval_views=[], fps=60.0,
                source_root=root, time_offsets={7: 0.0},
            )
            with patch("ftgspp.data.scene._file_sha256") as hashed:
                hashed.side_effect = AssertionError(
                    "verified manifest content must not be hashed twice"
                )
                metadata = scene.checkpoint_metadata()
            hashed.assert_not_called()
            self.assertEqual(scene._content_hashes[path.as_posix()], digest)
            self.assertEqual(metadata["manifest_entries"], 1)

    def test_view_content_change_invalidates_cached_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "000000.jpg"
            path.write_bytes(b"payload-a")
            first = _tiny_scene(root, [path]).checkpoint_metadata()
            path.write_bytes(b"payload-b")
            second = _tiny_scene(root, [path]).checkpoint_metadata()
            self.assertNotEqual(first["fingerprint"], second["fingerprint"])
            cached = json.loads(
                (root / VIEW_HASH_CACHE_RELATIVE).read_text(encoding="utf-8"),
            )
            self.assertEqual(
                cached["entries"][path.resolve().as_posix()]["sha256"],
                _file_sha256(path),
            )

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

if __name__ == "__main__":
    unittest.main()
