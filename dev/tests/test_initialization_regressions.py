"""Behavioral regressions for cache publication, camera recovery and L3 replay."""
from __future__ import annotations

import copy
import importlib.util
import json
import multiprocessing
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from ftgspp.config.compose import resolve_config
from ftgspp.data.scene import Scene
from ftgspp.io import provenance
from ftgspp.io.init_cache import publish_cache_pointer, resolve_published_cache_root
from ftgspp.io.ply import write_point_ply
from ftgspp.optional import da3
from ftgspp.training.init_plan import L2_SOURCE_FILES
from ftgspp.training.initialize import (
    KeyframeCloud, _initial_points, _valid_da3_cache, initialize_explicit,
)
from tools.audit_selfcap_initialization import audit


class TinyDA3:
    """Predict known cameras at the actual input size, without Torch/GPU."""

    def __init__(self):
        self.intrinsics = []

    def inference(self, images, *, extrinsics=None, intrinsics=None, **kwargs):
        self.intrinsics.append(None if intrinsics is None else intrinsics.copy())
        count = len(images)
        h, w = images[0].shape[:2]
        if extrinsics is None:
            extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
            extrinsics[:, 0, 3] = np.arange(count) * -.1
        if intrinsics is None:
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
            intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 7.0
        depth = np.ones((count, h, w), np.float32)
        return SimpleNamespace(
            depth=depth, conf=np.ones_like(depth), sky=None,
            processed_images=np.stack(images),
            extrinsics=extrinsics, intrinsics=intrinsics,
        )


def tiny_scene(tmp_path, *, calibrated=True, heldout=False):
    images = tmp_path / "images"
    images.mkdir()
    for frame in (0, 1):
        for camera in (0, 1):
            assert cv2.imwrite(
                str(images / f"c{camera}-f{frame:06d}.png"),
                np.full((4, 4, 3), 30 + camera, np.uint8),
            )
    config = resolve_config(OmegaConf.create({
        "data": {
            "root": str(tmp_path), "extracted_path": str(images),
            "scale": 1.0, "fps": 1.0, "frames": {"start": 0, "stop": 2},
            "train_cameras": [0] if heldout else [0, 1],
            "eval_cameras": [1] if heldout else [],
        },
        "init": {
            "backend": "da3", "keyframe_stride": 1, "num_gaussians": 8,
            "da3": {"model_name": "fake", "device": "cpu", "process_res": 14,
                    "max_views": 2, "max_points": 32, "voxel_size": .01,
                    "confidence_percentile": 0.0, "min_consistent_views": 1},
        },
        "run": {"output_path": str(tmp_path)},
    }), base_dir=tmp_path).config
    scene = Scene.from_config(config)
    if calibrated:
        scene.calibration_available = True
    return config, scene


@pytest.fixture
def fake_torch(monkeypatch):
    cuda = SimpleNamespace(
        is_available=lambda: True, device=lambda *_args: nullcontext(),
        max_memory_allocated=lambda: 0, empty_cache=Mock(),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    return cuda


def test_recovered_intrinsics_condition_every_following_frame(tmp_path, monkeypatch, fake_torch):
    config, scene = tiny_scene(tmp_path, calibrated=False)
    model = TinyDA3()
    monkeypatch.setattr(da3, "_load_model", lambda *_args, **_kwargs: (model, "cpu"))
    result = da3.generate_da3_points(config, scene, [0, 1])
    assert model.intrinsics[0] is None
    np.testing.assert_array_equal(model.intrinsics[1], np.stack([
        scene.view(1, camera).intrinsic for camera in scene.cameras
    ]))
    assert scene.view(1, 0).intrinsic[0, 0] == 7.0
    assert not list(result.cache_root.glob("*.attributes.*"))


def test_cache_switch_avoids_model_load_and_pins_provenance(tmp_path, monkeypatch, fake_torch):
    config, scene = tiny_scene(tmp_path)
    loader = Mock(side_effect=lambda *_args, **_kwargs: (TinyDA3(), "cuda:0"))
    monkeypatch.setattr(da3, "_load_model", loader)
    a = da3.generate_da3_points(config, scene, [0, 1])
    config.train.seed = 1
    b = da3.generate_da3_points(config, scene, [0, 1])
    config.train.seed = 0
    clouds = _initial_points(config, scene, [0, 1])
    assert Path(clouds[0].provenance["root"]) == a.cache_root
    assert clouds[0].provenance["source"] == "da3_cache"
    assert da3.resolve_da3_cache_root(a.cache_root.parent) == a.cache_root
    assert loader.call_count == fake_torch.empty_cache.call_count == 2

    # Another configuration can publish its pointer between generation and
    # the caller recording provenance. The returned root must stay bound to A.
    base = a.cache_root.parent
    b_manifest = json.loads((b.cache_root / "manifest.json").read_text())
    def publish_b():
        publish_cache_pointer(base, b.cache_root, b.cache_root / "manifest.json", b_manifest["fingerprint"])
    publish_b()
    original = da3.generate_da3_points
    def switch_after_return(*args, **kwargs):
        result = original(*args, **kwargs)
        publish_b()
        return result
    monkeypatch.setattr(da3, "generate_da3_points", switch_after_return)
    clouds = _initial_points(config, scene, [0, 1])
    assert da3.resolve_da3_cache_root(base) == b.cache_root
    assert Path(clouds[0].provenance["root"]) == a.cache_root
    assert loader.call_count == 2


@pytest.mark.parametrize("heldout", [False, True])
def test_uncalibrated_cache_restores_rig_without_inference(tmp_path, monkeypatch, fake_torch, heldout):
    config, scene = tiny_scene(tmp_path, calibrated=False, heldout=heldout)
    provisional = copy.deepcopy(scene)
    loader = Mock(side_effect=lambda *_args, **_kwargs: (TinyDA3(), "cpu"))
    monkeypatch.setattr(da3, "_load_model", loader)
    first = da3.generate_da3_points(config, scene, [0, 1])
    # Includes the waiting-process case: its Scene predates camera publication.
    second = da3.generate_da3_points(config, provisional, [0, 1])
    assert not second.generated
    assert provisional.calibration_available
    reloaded = Scene.from_config(config)
    assert reloaded.calibration_available
    third = da3.generate_da3_points(config, reloaded, [0, 1])
    assert first.cache_root == second.cache_root == third.cache_root
    assert loader.call_count == 1
    for camera in scene.cameras:
        np.testing.assert_array_equal(reloaded.view(0, camera).intrinsic, scene.view(0, camera).intrinsic)
        np.testing.assert_array_equal(reloaded.view(0, camera).w2c, scene.view(0, camera).w2c)


@pytest.mark.parametrize("change", ["scale", "image", "model"])
def test_stale_camera_inputs_are_rejected(tmp_path, monkeypatch, fake_torch, change):
    config, scene = tiny_scene(tmp_path, calibrated=False)
    monkeypatch.setattr(da3, "_load_model", lambda *_args, **_kwargs: (TinyDA3(), "cpu"))
    old = da3.generate_da3_points(config, scene, [0, 1])
    if change == "scale":
        config.data.scale = .5
    elif change == "image":
        assert cv2.imwrite(str(scene.view(0, 0).path), np.full((4, 4, 3), 120, np.uint8))
    else:
        config.init.da3.model_name = "different-weights"
    requested = Scene.from_config(config)
    assert not requested.calibration_available
    assert requested.view(0, 0).size == ((2, 2) if change == "scale" else (4, 4))
    assert not _valid_da3_cache(config, requested, old.cache_root, [0, 1], [0, 1])
    new = da3.generate_da3_points(config, requested, [0, 1])
    assert new.generated and new.cache_root != old.cache_root
    assert Scene.from_config(config).calibration_available


@pytest.mark.parametrize("fail_at", ["identity", "inference"])
def test_failure_releases_cuda_and_preserves_published_cache(tmp_path, monkeypatch, fake_torch, fail_at):
    config, scene = tiny_scene(tmp_path)
    monkeypatch.setattr(da3, "_load_model", lambda *_args, **_kwargs: (TinyDA3(), "cuda:0"))
    old = da3.generate_da3_points(config, scene, [0, 1])
    base = old.cache_root.parent
    pointer = (base / "current.json").read_bytes()
    original_ply = (old.cache_root / "f000000.ply").read_bytes()
    config.train.seed = 1
    if fail_at == "identity":
        monkeypatch.setattr(da3, "_loaded_model_identity", Mock(side_effect=RuntimeError("injected identity failure")))
    else:
        original = da3._run_inference
        def fail_second(*args, **kwargs):
            if kwargs["frame"] == 1:
                raise RuntimeError("injected inference failure")
            return original(*args, **kwargs)
        monkeypatch.setattr(da3, "_run_inference", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        da3.generate_da3_points(config, scene, [0, 1])
    assert fake_torch.empty_cache.call_count == 2
    assert (base / "current.json").read_bytes() == pointer
    assert (old.cache_root / "f000000.ply").read_bytes() == original_ply
    assert not list(base.glob(".building-*"))
    config.train.seed = 0
    assert _valid_da3_cache(config, scene, old.cache_root, [0, 1], [0, 1])


def test_damaged_generation_is_repaired_without_overwriting_old_directory(tmp_path, monkeypatch, fake_torch):
    config, scene = tiny_scene(tmp_path)
    monkeypatch.setattr(da3, "_load_model", lambda *_args, **_kwargs: (TinyDA3(), "cpu"))
    old = da3.generate_da3_points(config, scene, [0, 1])
    (old.cache_root / "f000000.ply").write_bytes(b"truncated")
    repaired = da3.generate_da3_points(config, scene, [0, 1])
    assert repaired.cache_root != old.cache_root
    assert (old.cache_root / "f000000.ply").read_bytes() == b"truncated"
    assert _valid_da3_cache(config, scene, repaired.cache_root, [0, 1], [0, 1])


def test_failed_camera_publication_can_retry_without_mutating_scene(tmp_path, monkeypatch, fake_torch):
    config, scene = tiny_scene(tmp_path, calibrated=False)
    original_intrinsic = scene.view(0, 0).intrinsic.copy()
    loader = Mock(side_effect=lambda *_args, **_kwargs: (TinyDA3(), "cuda:0"))
    monkeypatch.setattr(da3, "_load_model", loader)
    with patch.object(da3, "publish_cache_pointer", side_effect=OSError("injected publish failure")):
        with pytest.raises(OSError, match="injected publish"):
            da3.generate_da3_points(config, scene, [0, 1])
    assert not scene.calibration_available
    np.testing.assert_array_equal(scene.view(0, 0).intrinsic, original_intrinsic)
    assert fake_torch.empty_cache.call_count == 1
    recovered = da3.generate_da3_points(config, scene, [0, 1])
    assert not recovered.generated
    assert scene.calibration_available
    assert scene.view(0, 0).intrinsic[0, 0] == 7.0
    assert loader.call_count == 1


def _concurrent_generation(config, scene, first, entered, release, started, loaded, result):
    class BlockingDA3(TinyDA3):
        def inference(self, *args, **kwargs):
            if first:
                entered.set()
                if not release.wait(15):
                    raise RuntimeError("test timed out waiting to release first writer")
            return super().inference(*args, **kwargs)
    def load(*args, **kwargs):
        loaded.set()
        return BlockingDA3(), "cpu"
    try:
        with patch.object(da3, "_load_model", load):
            started.set()
            clouds = da3.generate_da3_points(config, scene, [0, 1])
        result.put((str(clouds.cache_root), clouds.generated, None))
    except BaseException as exc:
        result.put((None, None, repr(exc)))


@pytest.mark.parametrize("calibrated", [False, True])
def test_concurrent_processes_share_one_completed_generation(tmp_path, fake_torch, calibrated):
    config, scene = tiny_scene(tmp_path, calibrated=calibrated)
    ctx = multiprocessing.get_context("fork")
    entered, release, started_a, started_b, loaded_a, loaded_b = [ctx.Event() for _ in range(6)]
    result = ctx.Queue()
    a = ctx.Process(target=_concurrent_generation, args=(config, scene, True, entered, release, started_a, loaded_a, result))
    b = ctx.Process(target=_concurrent_generation, args=(config, scene, False, entered, release, started_b, loaded_b, result))
    a.start()
    try:
        assert entered.wait(10)
        b.start()
        assert started_b.wait(10)
        assert not loaded_b.wait(.3)
        release.set()
        a.join(15)
        b.join(15)
        assert not a.is_alive() and not b.is_alive()
        rows = [result.get(timeout=2), result.get(timeout=2)]
        assert all(row[2] is None for row in rows), rows
        assert rows[0][0] == rows[1][0]
        assert sorted(row[1] for row in rows) == [False, True]
        assert not loaded_b.is_set()
        assert _valid_da3_cache(config, scene, Path(rows[0][0]), [0, 1], [0, 1])
    finally:
        release.set()
        for proc in (a, b):
            if proc.pid is not None:
                if proc.is_alive():
                    proc.terminate()
                proc.join(5)
        result.close()


def test_wheel_stage_identity_tracks_real_packaged_source(tmp_path):
    package = tmp_path / "site-packages" / "ftgspp"
    (package / "io").mkdir(parents=True)
    for relative in L2_SOURCE_FILES:
        path = package / Path(relative).relative_to("src/ftgspp")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version = 1\n")
    shutil.copyfile(provenance.__file__, package / "io/provenance.py")
    spec = importlib.util.spec_from_file_location("wheel_provenance", package / "io/provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = module.stage_identity(L2_SOURCE_FILES)
    assert set(before["files"].values()).isdisjoint({"missing", "unknown"})
    (package / "optional/da3.py").write_text("version = 2\n")
    assert module.stage_identity(L2_SOURCE_FILES)["set_sha256"] != before["set_sha256"]


@pytest.mark.parametrize("payload", [[], None, {"schema": "ftgspp-da3-cache-pointer-v1", "directory": ".."}])
def test_malformed_cache_pointer_is_a_miss(tmp_path, payload):
    (tmp_path / "current.json").write_text(json.dumps(payload))
    assert resolve_published_cache_root(tmp_path) == tmp_path


@pytest.mark.parametrize("capacities,target", [([0, 3, 0], 8), ([3, 0, 3], 10), ([100, 100], 60)])
@pytest.mark.parametrize("init_seed", [0, None])
def test_sparse_and_empty_keyframes_replay_actual_model(tmp_path, capacities, target, init_seed):
    config, _ = tiny_scene(tmp_path)
    config.init.seed = init_seed
    config.train.seed = 7
    config.init.num_gaussians = target
    config.init.num_velocity_nns = 3
    config.init.duration = .25
    config.model.max_duration = 3.0
    points_root = tmp_path / "points"
    points_root.mkdir()
    clouds = []
    rng = np.random.default_rng(3)
    xyz = rng.normal(size=(max(capacities), 3)).astype(np.float32)
    if capacities == [3, 0, 3]:
        xyz = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
    for index, count in enumerate(capacities):
        positions = xyz[:count] + np.float32(index * .1)
        rgb = np.zeros((count, 3), np.uint8)
        write_point_ply(points_root / f"f{index:06d}.ply", positions, rgb, allow_empty=True)
        clouds.append(KeyframeCloud(positions, rgb, float(index), index, {"root": str(points_root)}))
    dataset = SimpleNamespace(frames=list(range(len(clouds))), cameras=[0], train_cameras=[0], fps=1.)
    with patch("ftgspp.training.initialize._initial_points", return_value=clouds):
        model = initialize_explicit(config, dataset)
    assert len(model) == target
    assert np.isfinite(model.velocities).all()
    manifest = json.loads((tmp_path / "initialization_manifest.json").read_text())
    reference = np.concatenate([
        np.full(row["selected_points"], row["reference_timestamp"])
        for row in manifest["keyframes"]
    ])
    if sum(count > 0 for count in capacities) == 1:
        np.testing.assert_array_equal(model.velocities, 0)
        np.testing.assert_array_equal(model.times[:, 0], reference)
    elif capacities == [3, 0, 3]:
        # The original samples on both sides of the empty frame use the true
        # two-second interval, rather than matching against the empty cloud.
        np.testing.assert_allclose(model.velocities[:3], [
            [.6, .1, .1], [.1, .1, .1], [-.4, .1, .1],
        ], atol=1e-6)
    saved = {"init": {"seed": init_seed, "scale": config.init.scale, "num_velocity_nns": 3}, "train": {"seed": 7}}
    (tmp_path / "config.resolved.yaml").write_text(json.dumps(saved))
    replay = audit(tmp_path)
    sigma = np.exp(model.log_scales[:, 0])
    actual = np.mean(np.abs(model.times[:, 0] - reference) * np.linalg.norm(model.velocities, axis=1) / sigma)
    assert replay["seed"] == (7 if init_seed is None else init_seed)
    assert replay["train_seed"] == 7
    assert replay["statistics"]["jitter_shift_sigma"]["mean"] == pytest.approx(actual, rel=2e-6, abs=1e-7)
    assert replay["statistics"]["sigma"]["mean"] == pytest.approx(float(sigma.mean()), rel=2e-6)
