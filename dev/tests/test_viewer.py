"""Viewer CPU contracts; no window, Torch, or GPU required."""

import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from ftgspp.viewer.asset import (
    camera,
    atomic_json,
    load_asset,
    inferred_metadata,
    camera_hash,
)
from ftgspp.viewer.path import CameraPath, PATH_SCHEMA, validate_path
from ftgspp.viewer.offline import compose
from ftgspp.viewer.session import validate_state, save_session, load_session
from ftgspp.io.model import NativeGaussians
from ftgspp.io.viewer_manifest import write_manifest, write_training_manifest


def fixture_model(n=2):
    return NativeGaussians(
        np.array([[0, 0, 3]] * n, np.float32).reshape(n, 3),
        np.full((n, 3), -2, np.float32),
        np.tile([1, 0, 0, 0], (n, 1)),
        np.full((n, 1), 2.0),
        np.zeros((n, 16, 3)),
        np.arange(n).reshape(n, 1),
        np.zeros((n, 1)),
        np.zeros((n, 3)),
        np.ones((n, 1)),
        999.0,
    )


def test_raw_model_inference_not_max_duration(tmp_path):
    model = fixture_model()
    np.savez(tmp_path / "gaussians.npz", **model.arrays())
    with pytest.warns(UserWarning):
        a = load_asset(tmp_path)
    assert a.metadata["time_end"] == 1
    assert a.model.max_duration == 999
    meta = a.metadata
    write_manifest(a.model_path, meta, a.sha256)
    with pytest.warns(UserWarning):
        b = load_asset(tmp_path)
    np.testing.assert_array_equal(
        b.metadata["initial_camera"]["K"], meta["initial_camera"]["K"]
    )
    content = json.loads((tmp_path / "viewer_manifest.json").read_text())
    content["model"]["sha256"] = "bad"
    atomic_json(tmp_path / "viewer_manifest.json", content)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_asset(tmp_path)


def test_camera_validation_and_resize():
    c = inferred_metadata(fixture_model())["initial_camera"]
    from ftgspp.viewer.asset import scaled_camera

    resized = scaled_camera(c, 640, 360)
    np.testing.assert_allclose(resized["K"][:2], c["K"][:2] * 0.5)
    bad = dict(c, w2c=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="rigid"):
        camera(bad)
    for key, value in (("width", 0), ("timestamp", float("nan"))):
        with pytest.raises(ValueError):
            camera(dict(c, **{key: value}))


def test_straight_alpha_and_no_gamma():
    ft = np.array([[[0.1, 0.2, 0.3, 0.5], [0, 0, 0, 1]]], np.float32)
    np.testing.assert_array_equal(
        compose(ft, transparent=True), [[[51, 102, 153, 128], [0, 0, 0, 0]]]
    )
    np.testing.assert_array_equal(
        compose(ft, background=(1, 1, 1)), [[[153, 178, 204], [255, 255, 255]]]
    )
    assert np.all(compose(ft, mode=1)[0, 1] == 0)
    with pytest.raises(ValueError):
        compose(ft, mode=2, transparent=True)


def test_rgb_composition_matches_full_precision_equation():
    rng = np.random.default_rng(47)
    ft = rng.uniform(-0.1, 1.1, (33, 37, 4)).astype(np.float32)
    t = np.clip(ft[..., 3:4], 0, 1)
    y, x = np.indices(ft.shape[:2])
    checker = np.where(((x // 16 + y // 16) % 2)[..., None], 0.65, 0.35)
    for background, checks in (([0, 0, 0], False), ([1, 1, 1], False),
                               ([0.23, 0.56, 0.81], False), ([0, 0, 0], True)):
        bg = checker if checks else np.asarray(background, dtype=np.float32)
        for exposure in (-2.0, 0.0, 1.5):
            expected = np.rint(np.clip(
                (ft[..., :3] + t * bg) * 2.0**exposure, 0, 1
            ) * 255).astype(np.uint8)
            np.testing.assert_array_equal(
                compose(ft, background=background, checker=checks, exposure=exposure),
                expected,
            )


def test_path_slerp_and_separate_time():
    from scipy.spatial.transform import Rotation

    c = inferred_metadata(fixture_model())["initial_camera"]
    c = dict(c, w2c=np.eye(4), timestamp=8.0)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    pose[0, 3] = 4
    d = dict(c, w2c=np.linalg.inv(pose), timestamp=2.0)
    data = dict(
        schema=PATH_SCHEMA,
        keyframes=[dict(path_time=0, camera=c), dict(path_time=4, camera=d)],
    )
    p = CameraPath(data)
    middle = p.sample(2)
    assert middle["timestamp"] == 5
    np.testing.assert_allclose(
        np.linalg.inv(middle["w2c"])[:3, 3], [2, 0, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        np.linalg.inv(middle["w2c"])[:3, :3],
        Rotation.from_euler("y", 45, degrees=True).as_matrix(),
        atol=1e-6,
    )
    np.testing.assert_allclose(p.sample(100)["w2c"], d["w2c"], atol=1e-6)
    data["keyframes"][1]["path_time"] = 0
    with pytest.raises(ValueError, match="increasing"):
        validate_path(data)


def test_session_hash_and_validation(tmp_path):
    a = SimpleNamespace(sha256="abc", model_path=tmp_path / "m.npz")
    save_session(tmp_path / "session.json", a, dict(time=1.0, playing=True))
    assert load_session(tmp_path / "session.json", a)["playing"] is False
    a.sha256 = "other"
    with pytest.raises(ValueError, match="mismatch"):
        load_session(tmp_path / "session.json", a)
    for value in (
        {"time": float("nan")},
        {"world_up": [0, 0, 0]},
        {"sh_degree": 4},
        {"background": [-1, 0, 0]},
    ):
        with pytest.raises(ValueError):
            validate_state(value)


def test_atomic_no_clobber(tmp_path):
    file = tmp_path / "state.json"
    atomic_json(file, {"value": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_json(file, {"value": 2}, overwrite=False)
    assert json.loads(file.read_text())["value"] == 1


def test_manifest_uses_actual_views(tmp_path):
    model = fixture_model()
    c = inferred_metadata(model)["initial_camera"]
    view = SimpleNamespace(
        camera_id=9,
        frame=42,
        timestamp=0.75,
        size=(720, 1280),
        w2c=c["w2c"],
        intrinsic=c["K"],
    )
    ds = SimpleNamespace(train_views=[view], eval_views=[view], fps=24)
    cfg = SimpleNamespace(
        init=SimpleNamespace(sh_degree=2),
        model=SimpleNamespace(marginal_threshold=0.02),
    )
    write_training_manifest(tmp_path / "gaussians.npz", model, ds, cfg, "hash")
    data = json.loads((tmp_path / "viewer_manifest.json").read_text())
    assert len(data["metadata"]["cameras"]) == 1
    assert data["metadata"]["sample_times"] == [0.75]
    assert data["metadata"]["sh_degree"] == 2
    assert camera_hash(data["metadata"]["cameras"]) == data["camera_sha256"]


def test_cli_help_lazy():
    import os
    from pathlib import Path
    import subprocess

    code = 'import sys; from ftgspp.cli import main;\ntry: main(["viewer","--help"])\nexcept SystemExit: pass\nassert "torch" not in sys.modules; assert "ftgspp.training.runner" not in sys.modules; assert "ftgspp._ftgspp_viewer" not in sys.modules'
    env = os.environ.copy()
    source_root = str(Path(__file__).parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, env.get("PYTHONPATH")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_prepare_legacy_run_and_portable_metadata(tmp_path):
    import cv2
    from omegaconf import OmegaConf
    from ftgspp.viewer.app import prepare_main

    run = tmp_path / "legacy"
    root = tmp_path / "capture"
    images = root / "images" / "0"
    calibration = root / "calibration"
    images.mkdir(parents=True)
    calibration.mkdir()
    run.mkdir()
    k = np.array([[40, 0, 32], [0, 42, 24], [0, 0, 1]], dtype=float)
    for name in ("intri.yml", "extri.yml"):
        fs = cv2.FileStorage(str(calibration / name), cv2.FILE_STORAGE_WRITE)
        fs.startWriteStruct("names", cv2.FileNode_SEQ)
        fs.write("", "0")
        fs.endWriteStruct()
        if name == "intri.yml":
            fs.write("K_0", k)
            fs.write("H_0", 48)
            fs.write("W_0", 64)
        else:
            fs.write("Rot_0", np.eye(3))
            fs.write("T_0", np.array([[0.1], [0], [0]]))
        fs.release()
    for i in range(5, 8):
        assert cv2.imwrite(
            str(images / f"{i:06d}.png"), np.zeros((48, 64, 3), np.uint8)
        )
    cfg = dict(
        data=dict(
            extracted_path=str(root / "images"),
            calibration_path=str(calibration),
            root=str(root),
            frames=dict(start=5, stop=8),
            fps=24.0,
            scale=0.5,
            eval_cameras=[],
            train_cameras=None,
            strict_time_offsets=False,
        ),
        init=dict(sh_degree=2),
        model=dict(marginal_threshold=0.02),
    )
    OmegaConf.save(OmegaConf.create(dict(config=cfg)), run / "config.resolved.yaml")
    np.savez(run / "gaussians.npz", **fixture_model().arrays())
    prepare_main(["--run", str(run)])
    import shutil

    shutil.rmtree(root)  # Portable opening must not depend on source captures.
    asset = load_asset(run)
    c = asset.metadata["cameras"][0]
    assert (c["height"], c["width"]) == (24, 32)
    np.testing.assert_allclose(c["K"][:2], k[:2] * 0.5)
    np.testing.assert_allclose(asset.metadata["sample_times"], np.arange(5, 8) / 24)
    assert asset.metadata["sh_degree"] == 2
