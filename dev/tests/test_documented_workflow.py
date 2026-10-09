"""Keep the user example's data and generated-config handoff executable."""
import importlib.util
from pathlib import Path

import cv2
import pytest

from ftgspp.config.compose import load_config
from ftgspp.data.prepare_dataset import main as prepare_dataset
from ftgspp.data.scene import Scene


def _generator():
    path = Path(__file__).resolve().parents[2] / "examples/make_tiny_scene.py"
    spec = importlib.util.spec_from_file_location("tiny_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_scene


def test_tiny_scene_is_complete_and_refuses_overwrite(tmp_path):
    output = tmp_path / "tiny capture"
    config_path = _generator()(output)
    config = load_config(config_path)
    scene = Scene.from_config(config)
    assert config.run.output_path == output / "run"
    assert scene.train_cameras == [0, 1]
    assert scene.eval_cameras == [2]
    assert len(scene.frames) == 2
    assert scene.view(0, 0).load_mask().max() == 1
    assert scene.view(0, 0).load_rgb().shape == (32, 32, 4)
    for frame in (0, 1):
        assert (config.init.points_path / f"f{frame:06d}.ply").is_file()
    before = config_path.read_bytes()
    with pytest.raises(FileExistsError, match="new output"):
        _generator()(output)
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("mask_kind", ["alpha", "external"])
def test_indexing_preserves_capture_settings_in_same_config(tmp_path, mask_kind):
    capture = tmp_path / "capture"
    _generator()(capture)
    config = capture / "indexed.yaml"
    mask_args = ["--mask-source", "alpha"]
    if mask_kind == "external":
        mask_root = capture / "fmasks"
        for image in (capture / "images").glob("*/*.png"):
            target = mask_root / image.relative_to(capture / "images")
            target.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(target), cv2.imread(str(image), cv2.IMREAD_UNCHANGED)[..., 3])
        mask_args = ["--mask-root", str(mask_root)]
    prepare_dataset([
        "--images", str(capture / "images"), "--fps", "15",
        "--eval-cameras", "2", "--config", str(config),
        "--pipeline", "foreground", *mask_args,
    ])
    resolved = load_config(config)
    assert resolved.data.fps == 15
    assert resolved.data.frames.stop == 2
    assert resolved.data.masks.source == mask_kind
    assert resolved.data.eval_cameras == [2]
    assert resolved.init.points_path == capture / "points_vhull_ibr_stride2"
    assert Scene.from_config(resolved).view(0, 0).load_mask().max() == 1
