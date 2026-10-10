"""Exercise preparation → saved artifacts → initialization without model downloads."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from ftgspp.config.compose import load_config
from ftgspp.data.prepare_dataset import main as prepare_dataset
from ftgspp.data.scene import Scene
from ftgspp.io.ply import read_point_ply
from ftgspp.optional import birefnet, da3, visual_hull
from ftgspp.training.initialize import initialize_explicit
from tests.gpu.run_pipeline_acceptance import scene as calibrated_scene


def _forbid_generation(*args, **kwargs):
    raise AssertionError("prepared inputs must not invoke model inference")


@pytest.mark.parametrize("pipeline,method,color_mode", [
    ("foreground", "colorize", "visibility_robust"), ("foreground", "colorize", "gray"),
    ("foreground", "point_filtering", "visibility_robust"),
    ("scene", "colorize", "visibility_robust"), ("scene", "colorize", "gray"),
])
def test_xyz_only_cloud_can_be_colored_or_kept_gray(tmp_path, monkeypatch, capsys, pipeline, method, color_mode):
    root = calibrated_scene(tmp_path / "capture")
    source = root / "xyz"
    source.mkdir()
    xyz, _ = read_point_ply(root / "points_stride1/f000000.ply")
    # An invisible point is preserved by colorize and removed by point_filtering.
    xyz = np.concatenate([xyz, [[0, 0, -2]]]).astype(np.float32)
    for frame in (0, 1):
        (source / f"f{frame:06d}.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 33\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
            + "".join(f"{x} {y} {z}\n" for x, y, z in xyz)
        )
    original = (source / "f000000.ply").read_bytes()
    prepare_dataset([
        "--images", str(root / "images"), "--pipeline", pipeline,
        "--fps", "30", "--points", str(source),
        *(["--mask-source", "alpha"] if pipeline == "foreground" else []),
    ])
    config_path = root / "ftgspp.yaml"
    assert load_config(config_path).init.num_gaussians == 66
    with pytest.raises(ValueError, match="colorize"):
        read_point_ply(source / "f000000.ply")
    monkeypatch.setattr(da3, "generate_da3_points", _forbid_generation)
    if color_mode == "gray":
        monkeypatch.setattr(visual_hull, "color_points_ibr", _forbid_generation)
    output = root / "colored"
    visual_hull.main([
        "--config", str(config_path), "--output", str(output),
        "--method", method, "--input-points", str(source),
        "--color-mode", color_mode, "--device", "cpu",
    ])
    printed = capsys.readouterr().out
    assert f"--method {'point_filtering' if pipeline == 'foreground' else 'colorize'}" in printed
    assert "--set init.backend=points --set init.num_gaussians=null" in printed
    result_xyz, rgb = read_point_ply(output / "f000000.ply")
    if method == "point_filtering":
        assert 0 < len(result_xyz) < len(xyz)
        assert np.all(result_xyz[:, 2] > 0)
        assert set(map(tuple, result_xyz)) <= set(map(tuple, xyz))
    else:
        np.testing.assert_array_equal(result_xyz, xyz)
        assert np.all(rgb[-1] == 128)
    if color_mode == "gray":
        assert np.all(rgb == 128)
    else:
        assert np.any(rgb[:-1] != 128)
    assert (source / "f000000.ply").read_bytes() == original
    node = OmegaConf.load(config_path)
    node.init.points_path = str(output)
    OmegaConf.save(node, config_path)
    config = load_config(config_path)
    config.run.output_path.mkdir(parents=True)
    model = initialize_explicit(config, Scene.from_config(config))
    expected_count = sum(len(read_point_ply(path)[0]) for path in output.glob("f*.ply"))
    assert len(model) == expected_count
    assert config.init.num_gaussians == expected_count
    with pytest.raises(FileExistsError, match="empty output"):
        visual_hull.prepare_visual_hull_points(
            config_path, output, None, method="colorize", input_points=source,
        )


@pytest.mark.parametrize("pipeline", ["foreground", "scene"])
def test_complete_inputs_skip_models_and_geometry_preparation(tmp_path, monkeypatch, capsys, pipeline):
    root = calibrated_scene(tmp_path / "capture")
    prepare_dataset([
        "--images", str(root / "images"), "--pipeline", pipeline,
        "--fps", "30",
        *(["--mask-source", "alpha"] if pipeline == "foreground" else []),
        "--points", str(root / "points_stride1"),
    ])
    assert "--method carving" not in capsys.readouterr().out
    monkeypatch.setattr(da3, "generate_da3_points", _forbid_generation)
    monkeypatch.setattr(birefnet, "_load_model", _forbid_generation)
    config = load_config(root / "ftgspp.yaml")
    config.run.output_path.mkdir(parents=True)
    model = initialize_explicit(config, Scene.from_config(config))
    assert len(model) == 64


@pytest.mark.parametrize("holdout", [False, True])
@pytest.mark.parametrize("pipeline", ["foreground", "scene"])
def test_raw_rgb_estimates_cameras_and_reloads_prepared_cache(tmp_path, monkeypatch, holdout, pipeline):
    root = tmp_path / "raw"
    for camera in (0, 1, 2):
        folder = root / "images" / str(camera)
        folder.mkdir(parents=True)
        for frame in (0, 1):
            cv2.imwrite(str(folder / f"{frame:06d}.png"), np.full((16, 16, 3), 150, np.uint8))
    prepare_dataset([
        "--images", str(root / "images"), "--pipeline", pipeline, "--fps", "30",
        *(["--eval-cameras", "2"] if holdout else []),
    ])
    # Only model predictions are fixtures. Real mask/cache/calibration/PLY I/O,
    # geometric preparation, manifest validation and initialization run below.
    mask_model = SimpleNamespace(state_dict=lambda: {})
    monkeypatch.setattr(birefnet, "_load_model", lambda *a, **k: (mask_model, None, "cpu", "float32"))
    monkeypatch.setattr(birefnet, "_infer_probabilities", lambda model, torch, device, images, **k:
                        [np.ones(image.shape[:2], np.float32) for image in images])
    if pipeline == "foreground":
        birefnet.prepare_birefnet_masks(
            images_root=root / "images", output_root=root / "masks/birefnet_hr", device="cpu", qa_count=0,
            view_manifest=root / "manifests/dataset.json",
        )
    else:
        monkeypatch.setattr(birefnet, "_load_model", _forbid_generation)

    class Model:
        def inference(self, images, extrinsics=None, intrinsics=None, **kwargs):
            count = len(images)
            h, w = images[0].shape[:2]
            poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
            poses[:, 0, 3] = np.arange(count) * 0.1
            k = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], np.float32)
            return SimpleNamespace(
                depth=np.ones((count, h, w), np.float32), conf=np.ones((count, h, w), np.float32),
                sky=None, processed_images=np.stack(images),
                extrinsics=poses if extrinsics is None else extrinsics,
                intrinsics=np.repeat(k[None], count, axis=0) if intrinsics is None else intrinsics,
            )

    monkeypatch.setattr(da3, "_load_model", lambda *a, **k: (Model(), "cpu"))
    config_path = root / "ftgspp.yaml"
    node = OmegaConf.load(config_path)
    node.init.num_gaussians = 64
    node.init.da3 = {"device": "cpu", "process_res": 16, "max_points": 256, "min_consistent_views": 1}
    OmegaConf.save(node, config_path)
    assert not Scene.from_config(load_config(config_path)).calibration_available
    if pipeline == "foreground":
        output = root / "points_vhull_ibr_stride1"
        visual_hull.prepare_visual_hull_points(
            config_path, output, None, resolution=8, coarse_resolution=4,
            min_views=1, color_mask_threshold=0.5, device="cpu",
        )
    else:
        config = load_config(config_path)
        config.run.output_path.mkdir(parents=True)
        assert len(initialize_explicit(config, Scene.from_config(config))) == 64
        assert not (root / "masks").exists()
    monkeypatch.setattr(da3, "_load_model", _forbid_generation)
    monkeypatch.setattr(birefnet, "_load_model", _forbid_generation)
    config = load_config(config_path)
    restored = Scene.from_config(config)
    assert restored.calibration_available
    assert restored.eval_cameras == ([2] if holdout else [])
    config.run.output_path.mkdir(parents=True, exist_ok=True)
    assert len(initialize_explicit(config, restored)) == 64
